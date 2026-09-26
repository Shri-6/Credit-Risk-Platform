"""
Cox Proportional Hazards survival model: time-to-default.

Unlike the XGBoost classifier (which answers "will this loan default in
the observation window?"), the Cox model answers "how long until this
loan defaults, and which factors accelerate or delay that?" -- this is
the differentiating, more sophisticated half of the project.

Uses `duration_months` and `event_observed` already computed in the Gold
layer (event_observed=1 means the loan reached 90+ DPD; 0 means censored
-- prepaid, matured, or still current as of the data snapshot).

We report the concordance index (equivalent to AUC for survival models)
and hazard ratios (exp(coef)) for each covariate, which is what makes
Cox models more interpretable than a black-box classifier for regulators
and risk committees -- a hazard ratio of 1.5 for high-LTV loans means
"49% higher instantaneous risk of default at any given time," a concrete,
explainable statement a classifier's SHAP value doesn't give you as
cleanly.
"""

import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

NUMERIC_COVARIATES = [
    "credit_score", "orig_ltv", "orig_dti", "orig_interest_rate",
    "current_ltv_regional", "refi_incentive", "avg_macro_stress",
    "avg_unemployment_exposure", "mi_pct",
]
CATEGORICAL_COVARIATES = ["occupancy_status", "loan_purpose"]
VINTAGE_SPLIT_YEAR = 2022


def load_data(path="data/gold/loan_features") -> pd.DataFrame:
    df = pd.read_parquet(path)
    keep = ["loan_id", "orig_year", "duration_months", "event_observed"] + \
           NUMERIC_COVARIATES + CATEGORICAL_COVARIATES
    df = df[keep].copy()
    # Cox requires strictly positive durations
    df["duration_months"] = df["duration_months"].clip(lower=1)
    # one-hot encode categoricals, drop first to avoid collinearity
    df = pd.get_dummies(df, columns=CATEGORICAL_COVARIATES, drop_first=True)
    return df


def main():
    df = load_data()
    train_df = df[df["orig_year"] < VINTAGE_SPLIT_YEAR].drop(columns=["orig_year", "loan_id"])
    test_df = df[df["orig_year"] >= VINTAGE_SPLIT_YEAR].drop(columns=["orig_year", "loan_id"])

    print(f"Train: {len(train_df):,} loans ({train_df['event_observed'].sum()} events)")
    print(f"Test:  {len(test_df):,} loans ({test_df['event_observed'].sum()} events)")

    cph = CoxPHFitter(penalizer=0.1)  # L2 penalty for stability given rare events
    cph.fit(train_df, duration_col="duration_months", event_col="event_observed")

    print("\n=== Cox PH model summary (train) ===")
    print(cph.summary[["coef", "exp(coef)", "p"]].round(4).to_string())

    # concordance on held-out test set
    test_risk_scores = cph.predict_partial_hazard(test_df)
    c_index = concordance_index(
        test_df["duration_months"], -test_risk_scores, test_df["event_observed"]
    )
    print(f"\nTest set concordance index: {c_index:.4f}")

    # save results
    summary_df = cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].round(5)
    summary_df.to_csv("data/gold/cox_model_summary.csv")

    metrics = {
        "test_concordance_index": round(c_index, 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_events": int(train_df["event_observed"].sum()),
        "n_test_events": int(test_df["event_observed"].sum()),
        "penalizer": 0.1,
        "top_hazard_ratios": summary_df.sort_values("p")[["exp(coef)", "p"]].head(5).to_dict("index"),
    }
    with open("data/gold/survival_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\nSaved: data/gold/cox_model_summary.csv, survival_metrics.json")

    # interpretability sentence generator -- useful directly in your write-up
    print("\n=== Plain-language hazard ratio interpretation ===")
    for feat in ["credit_score", "orig_ltv", "current_ltv_regional", "avg_macro_stress"]:
        if feat in summary_df.index:
            hr = summary_df.loc[feat, "exp(coef)"]
            pct = (hr - 1) * 100
            direction = "higher" if pct > 0 else "lower"
            print(f"  {feat}: HR={hr:.3f} -> a 1-unit increase is associated with "
                  f"{abs(pct):.1f}% {direction} instantaneous default hazard")


if __name__ == "__main__":
    main()
