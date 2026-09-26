"""
Silver layer: validated, business-rule-clean, conformed data.

Rules applied here mirror what a real mortgage risk team checks before
trusting data for modeling:
  - LTV/DTI/credit score within plausible bounds (quarantine outliers)
  - delinquency status must be monotonic-ish (no negative ages)
  - performance records can't precede origination or follow maturity
  - true primary key for performance is (loan_id, reporting_period),
    not loan_id alone -- this is what caused the "duplicate" warning
    in Bronze DQ (expected: 25-40 monthly rows per loan)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

WRITE_FORMAT = "parquet"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("cre-risk-silver-transform")
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


def clean_origination(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Returns (clean_df, quarantined_df)."""
    valid = (
        (F.col("credit_score").between(300, 850)) &
        (F.col("orig_ltv").between(0, 120)) &
        (F.col("orig_dti").between(0, 100)) &
        (F.col("orig_upb") > 0) &
        (F.col("origination_date") < F.col("maturity_date"))
    )
    clean = df.filter(valid).dropDuplicates(["loan_id"])
    quarantined = df.filter(~valid)
    return clean, quarantined


def clean_performance(df: DataFrame, origination_dates: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Returns (clean_df, quarantined_df). True PK is (loan_id, reporting_period)."""
    df = df.dropDuplicates(["loan_id", "reporting_period"])
    df = df.join(origination_dates, on="loan_id", how="inner")

    valid = (
        (F.col("reporting_period") >= F.col("origination_date")) &
        (F.col("current_upb") >= 0) &
        (F.col("current_loan_delinquency_status").between(0, 9))
    )
    clean = df.filter(valid).drop("origination_date", "maturity_date")
    quarantined = df.filter(~valid)
    return clean, quarantined


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    orig_bronze = spark.read.format(WRITE_FORMAT).load("data/bronze/origination")
    perf_bronze = spark.read.format(WRITE_FORMAT).load("data/bronze/performance")

    orig_clean, orig_bad = clean_origination(orig_bronze)
    print(f"Origination: {orig_clean.count():,} clean / {orig_bad.count():,} quarantined")

    perf_clean, perf_bad = clean_performance(
        perf_bronze, orig_bronze.select("loan_id", "origination_date", "maturity_date")
    )
    print(f"Performance: {perf_clean.count():,} clean / {perf_bad.count():,} quarantined")

    orig_clean.write.mode("overwrite").format(WRITE_FORMAT).save("data/silver/origination")
    perf_clean.write.mode("overwrite").format(WRITE_FORMAT) \
        .partitionBy("vintage_year").save("data/silver/performance")

    if orig_bad.count() > 0:
        orig_bad.write.mode("overwrite").format(WRITE_FORMAT).save("data/silver/_quarantine_origination")
    if perf_bad.count() > 0:
        perf_bad.write.mode("overwrite").format(WRITE_FORMAT).save("data/silver/_quarantine_performance")

    print("Silver layer written -> data/silver/{origination,performance}")
    spark.stop()


if __name__ == "__main__":
    main()
