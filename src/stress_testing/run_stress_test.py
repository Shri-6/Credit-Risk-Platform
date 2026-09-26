"""
CCAR-style macro stress testing engine.

Design choice worth explaining plainly: rather than computing absolute
default probabilities directly from the time-varying Cox model's baseline
cumulative hazard (which requires centering assumptions about covariate
means that add complexity without adding real rigor at this dataset's
scale), we use the Cox model for what it's uniquely good at -- estimating
the RELATIVE hazard multiplier a macro shock produces -- and apply that
multiplier to the XGBoost classifier's calibrated baseline probability of
default. This is a standard practical pattern in credit risk (a
statistical PD model with a macro overlay) and avoids overstating
precision the time-varying model's baseline hazard doesn't actually have
at only 133 training events.

Scenarios loosely follow Federal Reserve CCAR/DFAST conventions
(baseline / adverse / severely adverse), calibrated to shocks of a
magnitude similar to 2008 and COVID-era swings in the synthetic macro
series.
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

LGD = 0.35  # loss given default assumption -- typical range 25-45% for residential


SCENARIOS = {
    "baseline": {
        "unemployment_shock": 0.0,
        "macro_stress_shock": 0.0,
        "ltv_shock_pts": 0.0,
        "description": "Current conditions held flat.",
    },
    "moderate_stress": {
        "unemployment_shock": 2.0,
        "macro_stress_shock": 0.15,
        "ltv_shock_pts": 5.0,
        "description": "Unemployment +2pts, moderate home price softening.",
    },
    "severe_stress": {
        "unemployment_shock": 5.0,
        "macro_stress_shock": 0.40,
        "ltv_shock_pts": 12.0,
        "description": "Unemployment +5pts, sharp home price decline "
                        "(2008-style combined shock).",
    },
    "severely_adverse_ccar_style": {
        "unemployment_shock": 6.5,
        "macro_stress_shock": 0.55,
        "ltv_shock_pts": 18.0,
        "description": "Fed CCAR-style severely adverse: unemployment "
                        "peaks near 10%, sharp national HPI decline.",
    },
}


def load_time_varying_coefs(path="data/gold/cox_time_varying_summary.csv") -> dict:
    df = pd.read_csv(path, index_col=0)
    return df["coef"].to_dict()


def load_classifier_and_score(gold_path="data/gold/loan_features",
                               calibrated_path="data/gold/calibrated_predictions.parquet"):
    """
    Uses the PLATT-CALIBRATED probability (src/models/calibrate_classifier.py),
    not raw model.predict_proba(). The raw classifier output is inflated
    ~15x by scale_pos_weight (needed for ranking performance on the
    imbalanced label, but wrong for dollar-denominated expected loss) --
    see that script's docstring. Run calibrate_classifier.py before this.
    """
    df = pd.read_parquet(gold_path)
    calib = pd.read_parquet(calibrated_path)
    df = df.merge(calib[["loan_id", "baseline_pd_calibrated"]], on="loan_id", how="left")
    df = df.rename(columns={"baseline_pd_calibrated": "baseline_pd"})
    return df


def apply_scenario(df: pd.DataFrame, coefs: dict, scenario: dict) -> pd.Series:
    """
    Relative hazard multiplier = exp(sum of coef * shock) for the shocked
    covariates. Loans' own static risk factors (credit score, DTI, etc.)
    are already reflected in baseline_pd, so we only shock the
    macro-linked covariates here to avoid double-counting.
    """
    log_multiplier = (
        coefs.get("unemployment_rate_month", 0.0) * scenario["unemployment_shock"]
        + coefs.get("macro_stress_month", 0.0) * scenario["macro_stress_shock"]
        + coefs.get("current_ltv_regional_month", 0.0) * scenario["ltv_shock_pts"]
    )
    return np.exp(log_multiplier)


def run_stress_test(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    # only currently-active (non-defaulted) loans are exposed to forward stress
    active = df[df["event_observed"] == 0].copy()
    print(f"Active (non-defaulted) portfolio: {len(active):,} loans, "
          f"${active['latest_upb'].sum()/1e6:.1f}M UPB")

    results = []
    for name, scenario in SCENARIOS.items():
        multiplier = apply_scenario(active, coefs, scenario)
        stressed_pd = np.clip(active["baseline_pd"] * multiplier, 0, 1)
        expected_loss = (stressed_pd * active["latest_upb"] * LGD).sum()
        results.append({
            "scenario": name,
            "description": scenario["description"],
            "hazard_multiplier": round(float(multiplier if np.isscalar(multiplier) else multiplier.mean()), 3),
            "portfolio_mean_pd": round(float(stressed_pd.mean()), 5),
            "expected_loss_usd": round(float(expected_loss), 2),
            "expected_loss_pct_of_upb": round(float(expected_loss / active["latest_upb"].sum() * 100), 3),
        })

    return pd.DataFrame(results)


def main():
    coefs = load_time_varying_coefs()
    df = load_classifier_and_score()

    print(f"\nBaseline classifier PD stats: mean={df['baseline_pd'].mean():.5f}, "
          f"max={df['baseline_pd'].max():.5f}")

    results = run_stress_test(df, coefs)
    print("\n=== Stress test results ===")
    print(results.to_string(index=False))

    results.to_csv("data/gold/stress_test_results.csv", index=False)
    with open("data/gold/stress_test_results.json", "w") as f:
        json.dump(results.to_dict("records"), f, indent=2)

    print("\nSaved: data/gold/stress_test_results.csv, stress_test_results.json")


if __name__ == "__main__":
    main()
