"""
Build the loan-month panel in start/stop/event interval format required by
lifelines' CoxTimeVaryingFitter.

Unlike the Gold `loan_features` table (one row per loan, macro exposure
averaged over the loan's life), this table keeps ONE ROW PER LOAN-MONTH,
with macro covariates as they actually were *at that month* -- which is
what lets the stress-testing engine shock a macro path and get a
correctly time-localized hazard response.

Interval format (lifelines convention):
    id, start, stop, event, <covariates as of [start, stop)>
    - event = 1 only on the row where the loan reaches serious delinquency
      (and only on that loan's LAST row)
    - all other rows (including the final row of censored loans) have event = 0
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

WRITE_FORMAT = "parquet"
SERIOUS_DELINQ_THRESHOLD = 3


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("cre-risk-time-varying-panel")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        # macOS fix: without this, Spark tries to bind its driver to
        # whatever your hostname resolves to, which fails with
        # BindException on many Mac networking setups (VPN, no active
        # network, etc.). Forcing localhost sidesteps that entirely.
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def build_panel(orig: DataFrame, perf: DataFrame, macro_national: DataFrame,
                 macro_state_hpi: DataFrame) -> DataFrame:
    # join macro at loan-month grain (same joins as Gold, but we DON'T
    # aggregate away the time dimension afterward)
    perf = perf.join(
        orig.select("loan_id", "property_state", "credit_score", "orig_ltv",
                    "orig_dti", "orig_interest_rate", "orig_upb", "mi_pct",
                    "occupancy_status", "loan_purpose"),
        on="loan_id", how="inner"
    )
    perf = perf.join(macro_national, on="reporting_period", how="left")
    perf = perf.join(macro_state_hpi, on=["reporting_period", "property_state"], how="left")

    # event flag per loan-month, then push "1" only onto the last row per loan
    perf = perf.withColumn(
        "is_serious_delinquent_month",
        (F.col("current_loan_delinquency_status") >= SERIOUS_DELINQ_THRESHOLD).cast("int")
    )

    w_loan = Window.partitionBy("loan_id").orderBy("loan_age")
    w_loan_desc = Window.partitionBy("loan_id").orderBy(F.col("loan_age").desc())

    # a loan "has" an event if it EVER hit serious delinquency; the event row
    # is the first month it happened (we truncate the panel there, since
    # months after default aren't meaningful risk-set observations)
    first_default = perf.groupBy("loan_id").agg(
        F.min(F.when(F.col("is_serious_delinquent_month") == 1, F.col("loan_age"))).alias("default_month")
    )
    perf = perf.join(first_default, on="loan_id", how="left")
    perf = perf.filter(
        F.col("default_month").isNull() | (F.col("loan_age") <= F.col("default_month"))
    )

    perf = perf.withColumn(
        "event",
        F.when(
            (F.col("default_month").isNotNull()) & (F.col("loan_age") == F.col("default_month")), 1
        ).otherwise(0)
    )

    # current mark-to-market LTV at this specific month (this is the whole point)
    perf = perf.withColumn(
        "current_ltv_regional_month",
        F.round(F.col("orig_ltv") * (F.col("current_upb") / F.col("orig_upb")), 2)
        # note: without an origination-time state HPI baseline this is paydown-only;
        # a fuller version would also divide by (hpi_index_state / hpi_index_state_orig)
        # as in the Gold table -- omitted here for panel-build simplicity, documented
        # as a known simplification.
    )
    perf = perf.withColumn(
        "refi_incentive_month",
        F.round(F.col("orig_interest_rate") - F.col("mortgage_rate_30yr"), 3)
    )

    panel = (
        perf
        .withColumn("start", F.col("loan_age") - 1)
        .withColumn("stop", F.col("loan_age"))
        .select(
            "loan_id", "start", "stop", "event",
            "credit_score", "orig_ltv", "orig_dti", "orig_interest_rate", "mi_pct",
            "occupancy_status", "loan_purpose",
            "current_ltv_regional_month", "refi_incentive_month",
            F.col("unemployment_rate").alias("unemployment_rate_month"),
            F.col("macro_stress_index").alias("macro_stress_month"),
        )
    )
    return panel


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    orig = spark.read.format(WRITE_FORMAT).load("data/silver/origination")
    perf = spark.read.format(WRITE_FORMAT).load("data/silver/performance")
    macro_national = spark.read.format(WRITE_FORMAT).load("data/bronze/macro_national")
    macro_state_hpi = spark.read.format(WRITE_FORMAT).load("data/bronze/macro_state_hpi")

    panel = build_panel(orig, perf, macro_national, macro_state_hpi)

    n_rows = panel.count()
    n_loans = panel.select("loan_id").distinct().count()
    n_events = panel.filter(F.col("event") == 1).count()
    print(f"Panel: {n_rows:,} loan-month rows, {n_loans:,} loans, {n_events} events")

    panel.write.mode("overwrite").format(WRITE_FORMAT).save("data/gold/loan_month_panel")
    print("Written -> data/gold/loan_month_panel")

    spark.stop()


if __name__ == "__main__":
    main()
