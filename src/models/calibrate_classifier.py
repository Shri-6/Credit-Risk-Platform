"""
Calibrate the XGBoost classifier's predicted probabilities.

WHY THIS EXISTS: training with `scale_pos_weight` (necessary to get the
model to learn anything useful from a ~0.6% positive rate) systematically
inflates predicted probabilities -- the model optimizes ranking (AUC,
PR-AUC) at the cost of calibration. Raw predict_proba() output should
NOT be used directly for dollar-denominated expected-loss calculations;
doing so overstates portfolio expected loss by roughly the same factor
`scale_pos_weight` was set to.

Fix: Platt scaling (a 1D logistic regression of true labels on the raw
predicted probability) maps the inflated, well-RANKED probabilities back
onto the true ~0.6% base rate while preserving rank order (so AUC/PR-AUC,
which are rank-based, are completely unaffected by this step).

HONEST LIMITATION: with only 144 total positive events in the whole
dataset, we calibrate on the FULL labeled set (train + test) rather than
a further held-out fold, since splitting the already-rare positives again
would make the calibration itself unstable. This trades a small amount of
calibration overfitting risk for a calibration curve that isn't dominated
by noise from a single-digit number of events. A stricter version would
use nested cross-validation; noted here as a limitation, not hidden.
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.train_classifier import FEATURE_COLS, CATEGORICAL_COLS, LABEL_COL


def main():
    df = pd.read_parquet("data/gold/loan_features")
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")

    model = xgb.XGBClassifier()
    model.load_model("data/gold/xgb_classifier.json")

    X = df[FEATURE_COLS + CATEGORICAL_COLS]
    y = df[LABEL_COL]
    raw_pd = model.predict_proba(X)[:, 1]

    print(f"Raw predicted PD: mean={raw_pd.mean():.5f}, true base rate={y.mean():.5f}")
    print(f"Inflation factor: {raw_pd.mean() / y.mean():.1f}x")

    platt = LogisticRegression()
    platt.fit(raw_pd.reshape(-1, 1), y)
    calibrated_pd = platt.predict_proba(raw_pd.reshape(-1, 1))[:, 1]

    print(f"Calibrated PD: mean={calibrated_pd.mean():.5f} "
          f"(target: {y.mean():.5f})")

    # calibration curve check (binned observed vs. predicted)
    prob_true, prob_pred = calibration_curve(y, calibrated_pd, n_bins=5, strategy="quantile")
    print("\nCalibration curve (5 quantile bins, observed vs. predicted):")
    for t, p in zip(prob_true, prob_pred):
        print(f"  observed={t:.5f}  predicted={p:.5f}")

    df["baseline_pd_raw"] = raw_pd
    df["baseline_pd_calibrated"] = calibrated_pd
    df[["loan_id", "baseline_pd_raw", "baseline_pd_calibrated"]].to_parquet(
        "data/gold/calibrated_predictions.parquet"
    )

    calib_params = {"coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0])}
    with open("data/gold/platt_calibration_params.json", "w") as f:
        json.dump(calib_params, f, indent=2)

    print("\nSaved: data/gold/calibrated_predictions.parquet, platt_calibration_params.json")


if __name__ == "__main__":
    main()
