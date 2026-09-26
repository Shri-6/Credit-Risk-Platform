"""
XGBoost delinquency classifier.

Predicts P(loan reaches 90+ DPD) using the Gold-layer feature table.
Pipeline:
    1. Train/test split (time-based, not random -- see note below)
    2. Optuna hyperparameter search (maximizing PR-AUC, not accuracy --
       see note on class imbalance below)
    3. Train final model on best params, log run + metrics to MLflow
    4. SHAP explainability on the test set
    5. Save model + metrics to disk for the dashboard / write-up phase

NOTE ON TRAIN/TEST SPLIT: a random split would leak information across
time (a loan originated in 2023 telling the model about 2023 macro
conditions that a model deployed in 2022 could never have seen). We
split by origination vintage instead: train on loans originated before
a cutoff, test on loans originated after it. This is what a real risk
team would insist on, and it's a detail worth being able to explain.

NOTE ON CLASS IMBALANCE: the label is ~0.6% positive. We do NOT use
accuracy (a model that always predicts "no default" gets 99.4% accuracy
and is useless). We optimize and report PR-AUC and ROC-AUC, use
`scale_pos_weight` to rebalance the loss function, and report a
calibration check -- because for expected-loss calculations downstream,
a well-calibrated probability matters more than a raw ranking metric.
"""

import json
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import mlflow
import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_upb",
    "orig_interest_rate", "num_units", "num_borrowers", "loan_term",
    "mi_pct", "current_ltv_proxy", "current_ltv_regional",
    "rate_delta_since_orig", "refi_incentive", "avg_macro_stress",
    "max_macro_stress", "avg_unemployment_exposure",
]
CATEGORICAL_COLS = ["occupancy_status", "channel", "loan_purpose", "property_type",
                     "first_time_homebuyer_flag"]
LABEL_COL = "event_observed"
VINTAGE_SPLIT_YEAR = 2022  # train on loans originated before this, test on after


def load_data(path="data/gold/loan_features") -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return df


def time_based_split(df: pd.DataFrame):
    train = df[df["orig_year"] < VINTAGE_SPLIT_YEAR]
    test = df[df["orig_year"] >= VINTAGE_SPLIT_YEAR]
    return train, test


def make_matrices(df: pd.DataFrame):
    X = df[FEATURE_COLS + CATEGORICAL_COLS]
    y = df[LABEL_COL]
    return X, y


def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
        "scale_pos_weight": scale_pos_weight,
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)


def run_optuna(X_train, y_train, X_val, y_val, scale_pos_weight, n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=n_trials, show_progress_bar=False
    )
    return study.best_params, study.best_value


def main():
    mlflow.set_experiment("cre-risk-classifier")
    df = load_data()
    train_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df):,} loans ({train_df[LABEL_COL].sum()} events) | "
          f"Test: {len(test_df):,} loans ({test_df[LABEL_COL].sum()} events)")

    # further split train into train/val for Optuna (still time-respecting:
    # val is the most recent slice of the train period)
    train_df = train_df.sort_values("origination_date")
    val_cutoff = int(len(train_df) * 0.85)
    fit_df, val_df = train_df.iloc[:val_cutoff], train_df.iloc[val_cutoff:]

    X_fit, y_fit = make_matrices(fit_df)
    X_val, y_val = make_matrices(val_df)
    X_test, y_test = make_matrices(test_df)

    n_pos, n_neg = y_fit.sum(), len(y_fit) - y_fit.sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.1f} (class imbalance correction)")

    print("Running Optuna hyperparameter search (optimizing PR-AUC)...")
    best_params, best_val_prauc = run_optuna(X_fit, y_fit, X_val, y_val, scale_pos_weight)
    print(f"Best val PR-AUC: {best_val_prauc:.4f}")
    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name="xgb_classifier_final"):
        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("vintage_split_year", VINTAGE_SPLIT_YEAR)

        final_model = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            enable_categorical=True,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        # refit on ALL train data (fit + val) with the tuned hyperparameters
        X_train_full, y_train_full = make_matrices(train_df)
        final_model.fit(X_train_full, y_train_full)

        test_preds = final_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_preds)
        pr_auc = average_precision_score(y_test, test_preds)
        brier = brier_score_loss(y_test, test_preds)

        print(f"\n=== Test set (held-out, {VINTAGE_SPLIT_YEAR}+ vintage) ===")
        print(f"ROC-AUC:  {auc:.4f}")
        print(f"PR-AUC:   {pr_auc:.4f}  (baseline / random = {y_test.mean():.4f})")
        print(f"Brier score: {brier:.5f}  (lower = better calibrated)")

        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("test_brier_score", brier)
        mlflow.xgboost.log_model(final_model, "model")

        # --- SHAP explainability ---
        print("\nComputing SHAP values on test set...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer(X_test)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
        print("\nTop 10 features by mean |SHAP value|:")
        print(importance.head(10).round(4).to_string())

        importance.to_csv("data/gold/shap_feature_importance.csv")
        mlflow.log_artifact("data/gold/shap_feature_importance.csv")

        # save metrics summary for the write-up / dashboard
        metrics_summary = {
            "test_roc_auc": round(auc, 4),
            "test_pr_auc": round(pr_auc, 4),
            "test_brier_score": round(brier, 5),
            "baseline_default_rate": round(float(y_test.mean()), 4),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_train_events": int(train_df[LABEL_COL].sum()),
            "n_test_events": int(test_df[LABEL_COL].sum()),
            "vintage_split_year": VINTAGE_SPLIT_YEAR,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "best_params": best_params,
        }
        with open("data/gold/classifier_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)
        mlflow.log_artifact("data/gold/classifier_metrics.json")

        final_model.save_model("data/gold/xgb_classifier.json")

    print("\nSaved: data/gold/xgb_classifier.json, classifier_metrics.json, "
          "shap_feature_importance.csv")
    print("Run `mlflow ui` in this directory to view the tracked experiment.")


if __name__ == "__main__":
    main()
