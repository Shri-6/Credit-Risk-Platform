"""
Gold layer: model-ready feature table.

Produces ONE ROW PER LOAN with:
  - static origination features
  - time-varying features observed as of a chosen snapshot / "as-of" date
  - the outcome label: did the loan reach 90+ DPD within the observation
    window (for the classifier), plus duration + event fields (for the
    Cox survival model).

This is the table that both src/models/train_classifier.py and
src/models/train_survival.py read.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

WRITE_FORMAT = "parquet"
SERIOUS_DELINQ_THRESHOLD = 3  # 90+ days past due, matches generator's default proxy


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("cre-risk-gold-features")
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


def build_gold_table(orig: DataFrame, perf: DataFrame, macro_national: DataFrame = None,
                      macro_state_hpi: DataFrame = None) -> DataFrame:
    # --- Macro joins (loan-month grain, before aggregating to loan grain) ---
    if macro_national is not None:
        perf = perf.join(macro_national, on="reporting_period", how="left")
    if macro_state_hpi is not None:
        perf = perf.join(
            orig.select("loan_id", "property_state"), on="loan_id", how="left"
        ).join(
            macro_state_hpi, on=["reporting_period", "property_state"], how="left"
        )

    # --- Outcome / label engineering ---
    w = Window.partitionBy("loan_id").orderBy("reporting_period")

    perf_labeled = perf.withColumn(
        "ever_serious_delinquent",
        F.max(
            (F.col("current_loan_delinquency_status") >= SERIOUS_DELINQ_THRESHOLD).cast("int")
        ).over(Window.partitionBy("loan_id"))
    )

    # first month a loan hits serious delinquency (for survival "time to event")
    first_default_month = (
        perf.filter(F.col("current_loan_delinquency_status") >= SERIOUS_DELINQ_THRESHOLD)
        .groupBy("loan_id")
        .agg(F.min("loan_age").alias("months_to_default"))
    )

    # last observed month per loan (censoring time if never defaulted)
    agg_exprs = [
        F.max("loan_age").alias("months_observed"),
        F.avg("macro_stress_index").alias("avg_macro_stress"),
        F.max("macro_stress_index").alias("max_macro_stress"),
        F.last("current_upb").alias("latest_upb"),
        F.last("current_interest_rate").alias("latest_rate"),
    ]
    if macro_national is not None:
        agg_exprs += [
            F.avg("unemployment_rate").alias("avg_unemployment_exposure"),
            F.last("unemployment_rate").alias("latest_unemployment_rate"),
            F.last("mortgage_rate_30yr").alias("latest_market_mortgage_rate"),
        ]
    if macro_state_hpi is not None:
        agg_exprs += [F.last("hpi_index_state").alias("latest_hpi_index_state")]

    last_observed = perf.groupBy("loan_id").agg(*agg_exprs)

    outcome = (
        last_observed
        .join(first_default_month, on="loan_id", how="left")
        .withColumn("event_observed", F.col("months_to_default").isNotNull().cast("int"))
        .withColumn(
            "duration_months",
            F.coalesce(F.col("months_to_default"), F.col("months_observed"))
        )
    )

    orig_with_hpi = orig
    if macro_state_hpi is not None:
        # HPI at origination, for regional mark-to-market LTV
        hpi_at_orig = macro_state_hpi.withColumnRenamed("hpi_index_state", "hpi_index_state_orig") \
            .withColumnRenamed("reporting_period", "origination_date")
        orig_with_hpi = orig.join(hpi_at_orig, on=["origination_date", "property_state"], how="left")

    gold = (
        orig_with_hpi
        .join(outcome, on="loan_id", how="inner")
        .withColumn("orig_year", F.year("origination_date"))
        .withColumn(
            "current_ltv_proxy",
            # naive proxy: UPB paydown only, ignores home price movement
            F.round(F.col("orig_ltv") * (F.col("latest_upb") / F.col("orig_upb")), 2)
        )
        .withColumn(
            "rate_delta_since_orig",
            F.round(F.col("latest_rate") - F.col("orig_interest_rate"), 3)
        )
        .fillna({"months_to_default": -1})
    )

    if macro_state_hpi is not None:
        gold = gold.withColumn(
            "current_ltv_regional",
            # true mark-to-market: paydown adjusted for regional home price change
            F.round(
                F.col("orig_ltv") * (F.col("latest_upb") / F.col("orig_upb"))
                * (F.col("hpi_index_state_orig") / F.col("latest_hpi_index_state")),
                2
            )
        )
    if macro_national is not None:
        gold = gold.withColumn(
            "refi_incentive",
            # positive => market rate is below the loan's own rate => incentive to refinance
            F.round(F.col("orig_interest_rate") - F.col("latest_market_mortgage_rate"), 3)
        )

    return gold


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    orig = spark.read.format(WRITE_FORMAT).load("data/silver/origination")
    perf = spark.read.format(WRITE_FORMAT).load("data/silver/performance")
    macro_national = spark.read.format(WRITE_FORMAT).load("data/bronze/macro_national")
    macro_state_hpi = spark.read.format(WRITE_FORMAT).load("data/bronze/macro_state_hpi")

    gold = build_gold_table(orig, perf, macro_national, macro_state_hpi)

    n = gold.count()
    n_events = gold.filter(F.col("event_observed") == 1).count()
    print(f"Gold table: {n:,} loans, {n_events:,} reached serious delinquency "
          f"({100*n_events/n:.2f}%)")

    gold.write.mode("overwrite").format(WRITE_FORMAT).save("data/gold/loan_features")
    print("Gold layer written -> data/gold/loan_features")

    spark.stop()


if __name__ == "__main__":
    main()
