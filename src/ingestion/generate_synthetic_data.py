"""
Synthetic loan-level data generator.

Mimics the ACTUAL Freddie Mac Single-Family Loan-Level Dataset schema
(origination file + monthly performance file), so the ingestion / feature /
modeling pipeline built against this synthetic data will work unchanged
against the real download.

Real data: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
(free account required, ~5-10 min signup). See docs/SWAP_IN_REAL_DATA.md.

Column reference (subset used here, real file has ~30 origination fields
and ~30 performance fields -- we use the fields that actually drive risk
modeling so the pipeline stays realistic without being bloated):

ORIGINATION FILE (one row per loan):
    loan_id, origination_date, credit_score, first_time_homebuyer_flag,
    maturity_date, msa, mi_pct, num_units, occupancy_status,
    orig_cltv, orig_dti, orig_upb, orig_ltv, orig_interest_rate,
    channel, loan_purpose, property_type, property_state, zip3,
    loan_term, num_borrowers

MONTHLY PERFORMANCE FILE (one row per loan per month):
    loan_id, reporting_period, current_upb, current_loan_delinquency_status,
    loan_age, remaining_months_to_maturity, modification_flag,
    zero_balance_code, zero_balance_effective_date, current_interest_rate,
    current_deferred_upb

We generate loans with realistic joint distributions (credit score vs.
default risk, LTV vs. default risk, vintage effects) so that the models
you train downstream produce non-trivial, defensible metrics -- not just
noise.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

US_STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
             "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]

PROPERTY_TYPES = ["SF", "PU", "CO", "MH", "CP"]  # single fam, PUD, condo, manufactured, co-op
LOAN_PURPOSES = ["P", "C", "N"]  # Purchase, Cash-out refi, No cash-out refi
OCCUPANCY = ["P", "I", "S"]  # Primary, Investment, Second home
CHANNELS = ["R", "B", "C", "T"]  # Retail, Broker, Correspondent, TPO


def generate_origination_data(n_loans: int, start_year: int = 2018, end_year: int = 2024) -> pd.DataFrame:
    """Generate the origination (loan-level static) file."""
    orig_dates = pd.to_datetime(
        RNG.choice(pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"), size=n_loans)
    )

    # Credit score: roughly normal, clipped to FICO range, with a realistic mean/std
    credit_score = np.clip(RNG.normal(740, 55, n_loans), 500, 850).round().astype(int)

    # Original LTV: beta-shaped, most loans 60-97%
    orig_ltv = np.clip(RNG.beta(5, 2, n_loans) * 100, 15, 97).round(1)
    orig_cltv = np.clip(orig_ltv + RNG.uniform(0, 8, n_loans), 15, 105).round(1)

    # DTI: normal around 36
    orig_dti = np.clip(RNG.normal(36, 9, n_loans), 5, 65).round(1)

    orig_upb = np.clip(RNG.lognormal(mean=12.3, sigma=0.5, size=n_loans), 50_000, 2_000_000).round(-2)

    # Rate loosely tracks vintage year (rate cycle 2018-2024) plus idiosyncratic spread
    year_base_rate = {2018: 4.6, 2019: 3.9, 2020: 3.1, 2021: 3.0, 2022: 5.3, 2023: 6.8, 2024: 6.7}
    base_rates = np.array([year_base_rate[d.year] for d in orig_dates])
    orig_interest_rate = np.clip(base_rates + RNG.normal(0, 0.4, n_loans), 2.0, 9.5).round(3)

    loan_term = RNG.choice([180, 240, 360], size=n_loans, p=[0.08, 0.07, 0.85])

    df = pd.DataFrame({
        "loan_id": [f"L{100000000 + i}" for i in range(n_loans)],
        "origination_date": orig_dates,
        "credit_score": credit_score,
        "first_time_homebuyer_flag": RNG.choice(["Y", "N"], n_loans, p=[0.25, 0.75]),
        "maturity_date": [d + pd.DateOffset(months=int(t)) for d, t in zip(orig_dates, loan_term)],
        "property_state": RNG.choice(US_STATES, n_loans),
        "zip3": RNG.integers(100, 999, n_loans),
        "num_units": RNG.choice([1, 2, 3, 4], n_loans, p=[0.92, 0.05, 0.02, 0.01]),
        "occupancy_status": RNG.choice(OCCUPANCY, n_loans, p=[0.82, 0.12, 0.06]),
        "orig_cltv": orig_cltv,
        "orig_dti": orig_dti,
        "orig_upb": orig_upb,
        "orig_ltv": orig_ltv,
        "orig_interest_rate": orig_interest_rate,
        "channel": RNG.choice(CHANNELS, n_loans, p=[0.55, 0.20, 0.20, 0.05]),
        "loan_purpose": RNG.choice(LOAN_PURPOSES, n_loans, p=[0.55, 0.25, 0.20]),
        "property_type": RNG.choice(PROPERTY_TYPES, n_loans, p=[0.75, 0.12, 0.10, 0.02, 0.01]),
        "loan_term": loan_term,
        "num_borrowers": RNG.choice([1, 2], n_loans, p=[0.35, 0.65]),
        "mi_pct": np.where(orig_ltv > 80, RNG.uniform(12, 35, n_loans).round(1), 0.0),
    })
    return df


def _default_hazard(row_credit, row_ltv, row_dti, macro_stress, months_on_book):
    """
    Latent monthly hazard of transitioning to serious delinquency.
    Built from realistic risk-factor directions:
      - lower credit score -> higher hazard
      - higher LTV -> higher hazard
      - higher DTI -> higher hazard
      - macro_stress (0=calm, 1=severe) scales hazard up
      - seasoning curve: hazard rises then falls (typical mortgage seasoning ramp)
    """
    credit_term = (750 - row_credit) / 100.0          # higher when credit is low
    ltv_term = (row_ltv - 70) / 30.0                    # higher when LTV is high
    dti_term = (row_dti - 30) / 20.0                    # higher when DTI is high
    seasoning = np.exp(-((months_on_book - 30) ** 2) / (2 * 24 ** 2))  # peaks ~month 30

    base = -7.5  # log-odds intercept -> low baseline monthly hazard
    logit = (base
             + 1.8 * credit_term
             + 1.3 * ltv_term
             + 0.9 * dti_term
             + 2.5 * macro_stress
             + 1.5 * seasoning)
    return 1 / (1 + np.exp(-logit))


def generate_performance_data(orig_df: pd.DataFrame, as_of_date: str = "2024-12-01") -> pd.DataFrame:
    """
    Simulate monthly performance history for every loan from origination
    up to as_of_date (or until it defaults / prepays / matures).
    """
    as_of = pd.Timestamp(as_of_date)
    records = []

    # crude macro stress path: 2020 COVID blip, 2022-2023 rate-shock stress
    def macro_stress_for(date: pd.Timestamp) -> float:
        if pd.Timestamp("2020-03-01") <= date <= pd.Timestamp("2020-08-01"):
            return 0.6
        if pd.Timestamp("2022-06-01") <= date <= pd.Timestamp("2023-12-01"):
            return 0.35
        return 0.05

    for row in orig_df.itertuples(index=False):
        current_upb = row.orig_upb
        rate = row.orig_interest_rate
        month = row.origination_date
        months_on_book = 0
        delinquency_status = 0  # 0 = current
        status_history_streak = 0
        defaulted = False
        prepaid = False

        while month <= as_of and month < row.maturity_date:
            months_on_book += 1
            stress = macro_stress_for(month)
            hazard = _default_hazard(row.credit_score, row.orig_ltv, row.orig_dti, stress, months_on_book)

            # amortization (very simplified, interest-only-ish decay for speed)
            current_upb = max(current_upb * (1 - 0.0028), 0)

            # small chance of voluntary prepayment (refi), higher if rates fell since origination
            prepay_hazard = 0.003 + (0.01 if rate > 6.0 and month.year >= 2020 and month.year <= 2021 else 0)

            roll = RNG.random()
            if roll < hazard and not defaulted and not prepaid:
                delinquency_status = min(delinquency_status + 1, 9)
                if delinquency_status >= 3:  # 90+ DPD -> serious delinquency / default proxy
                    defaulted = True
            elif RNG.random() < 0.35 and delinquency_status > 0:
                delinquency_status = max(delinquency_status - 1, 0)  # cures

            if not defaulted and RNG.random() < prepay_hazard:
                prepaid = True

            records.append({
                "loan_id": row.loan_id,
                "reporting_period": month,
                "current_upb": round(current_upb, 2),
                "current_loan_delinquency_status": delinquency_status,
                "loan_age": months_on_book,
                "remaining_months_to_maturity": max(row.loan_term - months_on_book, 0),
                "current_interest_rate": rate,
                "zero_balance_code": (
                    "01" if prepaid else ("03" if defaulted else None)
                ),
                "macro_stress_index": stress,
            })

            if defaulted or prepaid:
                break
            month = month + pd.DateOffset(months=1)

    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_loans", type=int, default=25000)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    print(f"Generating {args.n_loans:,} synthetic loans (Freddie Mac schema)...")
    orig_df = generate_origination_data(args.n_loans)
    print("Simulating monthly performance history (this is the slow part)...")
    perf_df = generate_performance_data(orig_df)

    orig_path = f"{args.out_dir}/origination.csv"
    perf_path = f"{args.out_dir}/performance.csv"
    orig_df.to_csv(orig_path, index=False)
    perf_df.to_csv(perf_path, index=False)

    n_defaults = perf_df.groupby("loan_id")["current_loan_delinquency_status"].max().ge(3).sum()
    print(f"Wrote {len(orig_df):,} loans -> {orig_path}")
    print(f"Wrote {len(perf_df):,} performance records -> {perf_path}")
    print(f"Loans reaching serious delinquency (90+ DPD): {n_defaults:,} "
          f"({100*n_defaults/len(orig_df):.2f}% cumulative default rate)")


if __name__ == "__main__":
    main()
