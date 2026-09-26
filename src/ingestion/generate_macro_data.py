"""
Synthetic macro-economic time series generator.

Mimics three real FRED series at monthly granularity, state-level where
FRED itself only has national series (we add state dispersion since real
regional risk analysis needs it -- see docs/SWAP_IN_REAL_DATA.md for how
to replace this with real FRED national series + real state-level HPI
from the FHFA / Zillow, which is a straightforward swap):

  - unemployment_rate   (national series: FRED UNRATE)
  - hpi_index           (national: FRED CSUSHPISA; state dispersion added)
  - mortgage_rate_30yr   (national series: FRED MORTGAGE30US)

Stress periods deliberately match the macro_stress_index already encoded
in generate_synthetic_data.py's performance simulation, so joining this
macro table onto the loan panel produces a coherent, non-circular signal
(macro stress in the label-generating process and macro features in the
model are the same underlying economic story, not two unrelated things).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

US_STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
             "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]

# Rough regional volatility multipliers (CA/FL/AZ swing harder in housing cycles
# than e.g. TX/OH) -- loosely realistic, not precise historical calibration.
STATE_HPI_BETA = {
    "CA": 1.4, "FL": 1.5, "AZ": 1.4, "NY": 1.1, "NJ": 1.1, "MA": 1.2,
    "WA": 1.3, "IL": 0.9, "PA": 0.9, "OH": 0.7, "GA": 1.0, "NC": 1.0,
    "MI": 0.8, "VA": 0.9, "TN": 1.0, "IN": 0.7, "MO": 0.7, "MD": 1.0, "WI": 0.8,
    "TX": 1.1,
}
assert set(US_STATES) == set(STATE_HPI_BETA.keys()), (
    "STATE_HPI_BETA must cover every state in US_STATES, or the Gold-layer "
    "regional HPI join will silently null out rows for the missing state(s)."
)


def generate_national_macro(start="2015-01-01", end="2024-12-01") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)

    unemployment = np.full(n, 4.5)
    hpi = np.zeros(n)
    hpi[0] = 100.0
    mortgage_rate = np.full(n, 4.0)

    for i, d in enumerate(dates):
        if i == 0:
            continue
        covid = pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2021-06-01")
        rate_shock = pd.Timestamp("2022-03-01") <= d <= pd.Timestamp("2023-12-01")

        # unemployment: spikes hard in COVID, slow decay after; mild rise in rate-shock period
        target = 4.5
        if covid and d <= pd.Timestamp("2020-06-01"):
            target = 13.0
        elif covid:
            target = 6.5
        elif rate_shock:
            target = 4.2
        unemployment[i] = unemployment[i-1] + 0.35 * (target - unemployment[i-1]) + RNG.normal(0, 0.1)
        unemployment[i] = np.clip(unemployment[i], 3.0, 15.0)

        # HPI: strong growth 2015-2019 and 2021 (cheap money), flat/dip in COVID onset,
        # deceleration/slight decline in rate-shock period
        monthly_growth = 0.004  # ~5%/yr baseline
        if covid and d <= pd.Timestamp("2020-06-01"):
            monthly_growth = -0.002
        elif pd.Timestamp("2020-07-01") <= d <= pd.Timestamp("2022-02-01"):
            monthly_growth = 0.012  # pandemic housing boom
        elif rate_shock:
            monthly_growth = -0.001
        hpi[i] = hpi[i-1] * (1 + monthly_growth + RNG.normal(0, 0.003))

        # 30yr mortgage rate: low/falling through 2021, sharp rise 2022-2023, plateau
        target_rate = 4.0
        if d <= pd.Timestamp("2021-12-01"):
            target_rate = 3.0
        elif rate_shock:
            target_rate = 7.0
        else:
            target_rate = 6.7
        mortgage_rate[i] = mortgage_rate[i-1] + 0.15 * (target_rate - mortgage_rate[i-1]) + RNG.normal(0, 0.05)
        mortgage_rate[i] = np.clip(mortgage_rate[i], 2.0, 9.0)

    return pd.DataFrame({
        "reporting_period": dates,
        "unemployment_rate": unemployment.round(2),
        "hpi_index_national": hpi.round(2),
        "mortgage_rate_30yr": mortgage_rate.round(3),
    })


def generate_state_hpi(national_hpi: pd.DataFrame) -> pd.DataFrame:
    """Expand national HPI into state-level series using per-state volatility betas."""
    rows = []
    base = national_hpi.set_index("reporting_period")["hpi_index_national"]
    growth = base.pct_change().fillna(0)

    for state, beta in STATE_HPI_BETA.items():
        state_growth = growth * beta + RNG.normal(0, 0.002, len(growth))
        state_index = (1 + state_growth).cumprod() * 100
        for date, val in zip(national_hpi["reporting_period"], state_index):
            rows.append({"reporting_period": date, "property_state": state, "hpi_index_state": round(val, 2)})

    return pd.DataFrame(rows)


def main(out_dir="data/raw"):
    national = generate_national_macro()
    state_hpi = generate_state_hpi(national)

    national.to_csv(f"{out_dir}/macro_national.csv", index=False)
    state_hpi.to_csv(f"{out_dir}/macro_state_hpi.csv", index=False)

    print(f"Wrote {len(national)} national macro rows -> {out_dir}/macro_national.csv")
    print(f"Wrote {len(state_hpi)} state HPI rows -> {out_dir}/macro_state_hpi.csv")
    print("\nSample national macro (stress periods should be visible):")
    print(national[national["reporting_period"].isin(
        pd.to_datetime(["2019-06-01", "2020-05-01", "2021-06-01", "2022-10-01", "2023-06-01"])
    )].to_string(index=False))


if __name__ == "__main__":
    main()
