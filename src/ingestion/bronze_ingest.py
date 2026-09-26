"""
Bronze layer: raw ingestion with schema enforcement + data quality gating.

Medallion architecture, stage 1 of 3:
    Bronze  -> raw data, typed, minimally cleaned, quarantine bad rows
    Silver  -> deduplicated, validated, business-rule-clean, conformed types
    Gold    -> feature-engineered, model-ready, aggregated where needed

Run:
    python src/ingestion/bronze_ingest.py

Note on Delta Lake: this pipeline writes partitioned Parquet. On Databricks
(or any environment with Maven Central access) change WRITE_FORMAT below to
"delta" -- every write call in this pipeline already uses the WRITE_FORMAT
constant, so it's a one-line change to get ACID transactions, time travel,
and MERGE support on top of the exact same pipeline logic.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, DateType
)

WRITE_FORMAT = "parquet"  # change to "delta" on Databricks / with Maven access

ORIGINATION_SCHEMA = StructType([
    StructField("loan_id", StringType(), False),
    StructField("origination_date", DateType(), True),
    StructField("credit_score", IntegerType(), True),
    StructField("first_time_homebuyer_flag", StringType(), True),
    StructField("maturity_date", DateType(), True),
    StructField("property_state", StringType(), True),
    StructField("zip3", IntegerType(), True),
    StructField("num_units", IntegerType(), True),
    StructField("occupancy_status", StringType(), True),
    StructField("orig_cltv", DoubleType(), True),
    StructField("orig_dti", DoubleType(), True),
    StructField("orig_upb", DoubleType(), True),
    StructField("orig_ltv", DoubleType(), True),
    StructField("orig_interest_rate", DoubleType(), True),
    StructField("channel", StringType(), True),
    StructField("loan_purpose", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("loan_term", IntegerType(), True),
    StructField("num_borrowers", IntegerType(), True),
    StructField("mi_pct", DoubleType(), True),
])

PERFORMANCE_SCHEMA = StructType([
    StructField("loan_id", StringType(), False),
    StructField("reporting_period", DateType(), True),
    StructField("current_upb", DoubleType(), True),
    StructField("current_loan_delinquency_status", IntegerType(), True),
    StructField("loan_age", IntegerType(), True),
    StructField("remaining_months_to_maturity", IntegerType(), True),
    StructField("current_interest_rate", DoubleType(), True),
    StructField("zero_balance_code", StringType(), True),
    StructField("macro_stress_index", DoubleType(), True),
])


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("cre-risk-bronze-ingest")
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


def data_quality_report(df: DataFrame, name: str, key_col: str) -> dict:
    """
    Automated data quality checks. In a real pipeline this would push metrics
    to a monitoring table / Great Expectations checkpoint; here we compute
    and print them, and RAISE if a hard gate fails, so bad data never
    silently reaches Silver.
    """
    total = df.count()
    null_keys = df.filter(F.col(key_col).isNull()).count()
    dupe_keys = total - df.select(key_col).distinct().count()

    report = {
        "table": name,
        "row_count": total,
        "null_primary_keys": null_keys,
        "duplicate_primary_keys": dupe_keys,
    }
    print(f"[DQ] {name}: {report}")

    # Hard gates -- fail fast rather than propagate bad data downstream
    assert null_keys == 0, f"DQ GATE FAILED: {name} has {null_keys} null {key_col} values"
    return report


def ingest_origination(spark: SparkSession, src_csv: str, bronze_path: str) -> DataFrame:
    df = (
        spark.read
        .option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(ORIGINATION_SCHEMA)
        .csv(src_csv)
    )
    df = df.withColumn("_ingested_at", F.current_timestamp()) \
           .withColumn("_source_file", F.input_file_name())

    data_quality_report(df, "origination", "loan_id")

    df.write.mode("overwrite").format(WRITE_FORMAT).save(bronze_path)
    print(f"Bronze origination written -> {bronze_path} ({df.count():,} rows)")
    return df


def ingest_performance(spark: SparkSession, src_csv: str, bronze_path: str) -> DataFrame:
    df = (
        spark.read
        .option("header", True)
        .option("dateFormat", "yyyy-MM-dd")
        .schema(PERFORMANCE_SCHEMA)
        .csv(src_csv)
    )
    df = df.withColumn("_ingested_at", F.current_timestamp()) \
           .withColumn("vintage_year", F.year("reporting_period"))

    data_quality_report(df, "performance", "loan_id")

    (
        df.write.mode("overwrite").format(WRITE_FORMAT)
        .partitionBy("vintage_year")
        .save(bronze_path)
    )
    print(f"Bronze performance written -> {bronze_path} ({df.count():,} rows, partitioned by vintage_year)")
    return df


def ingest_macro(spark: SparkSession, national_csv: str, state_hpi_csv: str, bronze_dir: str):
    national = spark.read.option("header", True).option("inferSchema", True).csv(national_csv)
    state_hpi = spark.read.option("header", True).option("inferSchema", True).csv(state_hpi_csv)

    data_quality_report(national, "macro_national", "reporting_period")

    national.write.mode("overwrite").format(WRITE_FORMAT).save(f"{bronze_dir}/macro_national")
    state_hpi.write.mode("overwrite").format(WRITE_FORMAT).save(f"{bronze_dir}/macro_state_hpi")
    print(f"Bronze macro written -> {bronze_dir}/macro_national, {bronze_dir}/macro_state_hpi")


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    ingest_origination(spark, "data/raw/origination.csv", "data/bronze/origination")
    ingest_performance(spark, "data/raw/performance.csv", "data/bronze/performance")
    ingest_macro(spark, "data/raw/macro_national.csv", "data/raw/macro_state_hpi.csv", "data/bronze")

    spark.stop()


if __name__ == "__main__":
    main()
