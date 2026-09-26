"""
Maps real Freddie Mac Single-Family Loan-Level Dataset files (pipe-delimited,
no header row) to the column names/types this pipeline's Bronze ingestion
expects (see ORIGINATION_SCHEMA / PERFORMANCE_SCHEMA in
src/ingestion/bronze_ingest.py).

Column positions below are taken from Freddie Mac's "Single-Family
Loan-Level Dataset General User Guide," January 2026 edition (32 fields
per file). Freddie Mac has changed this layout before (a field was added
in a prior release) and may again -- if your download doesn't parse
cleanly, re-check the current user guide at
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
(via the Clarity Data Intelligence portal) before assuming this script
is still correct. Column COUNT is the fastest sanity check: this script
asserts 32 origination fields and 32 performance fields; a mismatch means
the layout has changed and the position mapping below needs updating.

ACCESS NOTE: as of this writing, Freddie Mac requires registration and
sign-in via Clarity Data Intelligence (not the older freddiemac.embs.com
portal referenced in some older tutorials) to download these files.
See docs/SWAP_IN_REAL_DATA.md for the download steps.

Usage:
    python src/ingestion/map_freddie_columns.py \
        --orig_file /path/to/historical_data_2021Q1.txt \
        --perf_file /path/to/historical_data_time_2021Q1.txt \
        --out_dir data/raw

Handles ONE quarterly file pair per run; concatenate multiple quarters
by running this once per quarter and appending, or by passing multiple
--orig_file/--perf_file pairs (see --help).
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# ---- Origination file: 32 pipe-delimited fields, no header ----
# Position (1-indexed in Freddie Mac's docs) -> raw column name we assign
ORIG_RAW_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag",
    "maturity_date", "msa", "mi_pct", "num_units", "occupancy_status",
    "orig_cltv", "orig_dti", "orig_upb", "orig_ltv", "orig_interest_rate",
    "channel", "ppm_flag", "amortization_type", "property_state",
    "property_type", "postal_code", "loan_id", "loan_purpose", "loan_term",
    "num_borrowers", "seller_name", "servicer_name", "super_conforming_flag",
    "pre_relief_refi_loan_id", "special_eligibility_program",
    "relief_refinance_indicator", "property_valuation_method",
    "interest_only_indicator", "mi_cancellation_indicator",
]
assert len(ORIG_RAW_COLUMNS) == 32, (
    f"Expected 32 origination fields per the Jan 2026 layout, got "
    f"{len(ORIG_RAW_COLUMNS)}. The layout may have changed -- verify "
    f"against the current user guide before proceeding."
)

# ---- Monthly performance file: 32 pipe-delimited fields, no header ----
PERF_RAW_COLUMNS = [
    "loan_id", "reporting_period_raw", "current_upb", "delinquency_status_raw",
    "loan_age", "remaining_months_to_maturity", "defect_settlement_date",
    "modification_flag", "zero_balance_code", "zero_balance_effective_date",
    "current_interest_rate", "current_non_interest_bearing_upb", "ddlpi",
    "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries",
    "total_expenses", "legal_costs", "maintenance_preservation_costs",
    "taxes_and_insurance", "misc_expenses", "actual_loss_calculation",
    "cumulative_modification_cost", "interest_rate_step_indicator",
    "payment_deferral_flag", "eltv", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "delinquency_due_to_disaster",
    "borrower_assistance_status_code", "current_month_modification_cost",
    "interest_bearing_upb",
]
assert len(PERF_RAW_COLUMNS) == 32, (
    f"Expected 32 performance fields per the Jan 2026 layout, got "
    f"{len(PERF_RAW_COLUMNS)}. The layout may have changed -- verify "
    f"against the current user guide before proceeding."
)


def get_spark():
    return (
        SparkSession.builder
        .appName("freddie-mac-column-mapping")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def yyyymm_to_date(col):
    """Freddie Mac dates are YYYYMM strings (e.g. '202101'). Convert to
    the first of that month as a proper date, matching this pipeline's
    synthetic generator convention."""
    return F.to_date(F.concat(col.substr(1, 4), F.lit("-"), col.substr(5, 2), F.lit("-01")))


def map_origination(spark, path: str):
    raw_schema = StructType([StructField(c, StringType(), True) for c in ORIG_RAW_COLUMNS])
    df = spark.read.option("delimiter", "|").option("header", False).schema(raw_schema).csv(path)

    n_cols_actual = len(df.columns)
    assert n_cols_actual == 32, (
        f"Raw origination file has {n_cols_actual} columns, expected 32. "
        f"The file layout has likely changed -- check the current user guide "
        f"before trusting any downstream output."
    )

    mapped = df.select(
        F.col("loan_id"),
        yyyymm_to_date(F.col("first_payment_date")).alias("origination_date"),
        F.col("credit_score").cast("int"),
        F.col("first_time_homebuyer_flag"),
        yyyymm_to_date(F.col("maturity_date")).alias("maturity_date"),
        F.col("property_state"),
        # postal_code is "###00" (first 3 digits + 00); take first 3 chars as zip3
        F.col("postal_code").substr(1, 3).cast("int").alias("zip3"),
        F.col("num_units").cast("int"),
        F.col("occupancy_status"),
        F.col("orig_cltv").cast("double"),
        F.col("orig_dti").cast("double"),
        # Freddie Mac rounds original UPB to nearest $1,000, already in dollars
        F.col("orig_upb").cast("double"),
        F.col("orig_ltv").cast("double"),
        F.col("orig_interest_rate").cast("double"),
        F.col("channel"),
        F.col("loan_purpose"),
        F.col("property_type"),
        F.col("loan_term").cast("int"),
        F.col("num_borrowers").cast("int"),
        F.col("mi_pct").cast("double"),
    )

    # Freddie Mac's documented sentinel "not available" codes -> null,
    # so they don't get treated as real values downstream (e.g. credit
    # score 9999, LTV/DTI/MI% 999, num_units/num_borrowers 99)
    mapped = (
        mapped
        .withColumn("credit_score", F.when(F.col("credit_score") == 9999, None).otherwise(F.col("credit_score")))
        .withColumn("orig_cltv", F.when(F.col("orig_cltv") == 999, None).otherwise(F.col("orig_cltv")))
        .withColumn("orig_dti", F.when(F.col("orig_dti") == 999, None).otherwise(F.col("orig_dti")))
        .withColumn("orig_ltv", F.when(F.col("orig_ltv") == 999, None).otherwise(F.col("orig_ltv")))
        .withColumn("mi_pct", F.when(F.col("mi_pct") == 999, None).otherwise(F.col("mi_pct")))
        .withColumn("num_units", F.when(F.col("num_units") == 99, None).otherwise(F.col("num_units")))
        .withColumn("num_borrowers", F.when(F.col("num_borrowers") == 99, None).otherwise(F.col("num_borrowers")))
    )
    return mapped


def map_performance(spark, path: str):
    raw_schema = StructType([StructField(c, StringType(), True) for c in PERF_RAW_COLUMNS])
    df = spark.read.option("delimiter", "|").option("header", False).schema(raw_schema).csv(path)

    n_cols_actual = len(df.columns)
    assert n_cols_actual == 32, (
        f"Raw performance file has {n_cols_actual} columns, expected 32. "
        f"The file layout has likely changed -- check the current user guide "
        f"before trusting any downstream output."
    )

    # CURRENT LOAN DELINQUENCY STATUS is alphanumeric: usually a numeric
    # string ("0", "1", "2", ...) but "RA" (REO Acquisition) is a valid
    # non-numeric value. Map RA to a high status code so it's correctly
    # treated as severe/terminal by this pipeline's SERIOUS_DELINQ_THRESHOLD
    # logic, rather than crashing an int cast or silently becoming null.
    delinquency_status = (
        F.when(F.col("delinquency_status_raw") == "RA", F.lit(9))
        .otherwise(F.col("delinquency_status_raw").cast("int"))
    )

    mapped = df.select(
        F.col("loan_id"),
        yyyymm_to_date(F.col("reporting_period_raw")).alias("reporting_period"),
        F.col("current_upb").cast("double"),
        delinquency_status.alias("current_loan_delinquency_status"),
        F.col("loan_age").cast("int"),
        F.col("remaining_months_to_maturity").cast("int"),
        F.col("current_interest_rate").cast("double"),
        F.col("zero_balance_code"),
    )
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_file", required=True, help="Path to raw Freddie Mac origination .txt file")
    parser.add_argument("--perf_file", required=True, help="Path to raw Freddie Mac performance .txt file")
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Mapping origination file: {args.orig_file}")
    orig = map_origination(spark, args.orig_file)
    print(f"Mapped {orig.count():,} origination records")

    print(f"Mapping performance file: {args.perf_file}")
    perf = map_performance(spark, args.perf_file)
    print(f"Mapped {perf.count():,} performance records")

    # write as single CSV files matching what bronze_ingest.py expects
    (orig.coalesce(1).write.mode("overwrite").option("header", True)
     .csv(f"{args.out_dir}/_origination_tmp"))
    (perf.coalesce(1).write.mode("overwrite").option("header", True)
     .csv(f"{args.out_dir}/_performance_tmp"))

    import glob
    import shutil
    orig_part = glob.glob(f"{args.out_dir}/_origination_tmp/part-*.csv")[0]
    perf_part = glob.glob(f"{args.out_dir}/_performance_tmp/part-*.csv")[0]
    shutil.move(orig_part, f"{args.out_dir}/origination.csv")
    shutil.move(perf_part, f"{args.out_dir}/performance.csv")
    shutil.rmtree(f"{args.out_dir}/_origination_tmp")
    shutil.rmtree(f"{args.out_dir}/_performance_tmp")

    print(f"\nWrote {args.out_dir}/origination.csv and {args.out_dir}/performance.csv")
    print("NOTE: real data has no macro_stress_index column (that was a "
          "synthetic-only simulation input). Macro features come from "
          "generate_macro_data.py's FRED-equivalent joins in the Gold "
          "layer, not from a per-row field -- no pipeline change needed.")
    print("Now run: bronze_ingest.py -> silver_transform.py -> gold_features.py")

    spark.stop()


if __name__ == "__main__":
    main()
