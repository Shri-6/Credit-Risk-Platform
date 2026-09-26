"""
Cox model with TIME-VARYING covariates, fit on the loan-month panel.

This replaces the static Cox model's macro_stress feature (averaged over
each loan's entire life -- diluted, came back statistically insignificant,
see docs/MODEL_CARD.md) with the macro conditions AS THEY WERE in each
month, correctly localized to each loan's risk set at that point in time.

We also add this model's real purpose: `src/stress_testing/` reuses this
fitted model directly to answer "if unemployment jumps 3 points starting
next month, how does portfolio hazard change" -- a question the static
Cox model structurally cannot answer, since it has no notion of *when*
a macro condition applies.

PENALIZER NOTE (a real bug caught during development, kept here since it's
worth being able to explain): the static Cox model uses penalizer=0.1 on
~25,000 loan-LEVEL rows and fits fine. Copy-pasting that same penalizer
onto this ~830,000 loan-MONTH panel silently crushed every coefficient
toward zero -- including credit_score, which should be (and, once fixed,
is) overwhelmingly significant. lifelines' L2 penalty is applied in a way
that scales with dataset size, so the "same" penalizer value is far
stronger here than on the smaller static panel. Caught by noticing the
implausible result (a previously rock-solid predictor becoming
insignificant) rather than trusting a clean-looking summary table, then
confirmed by testing penalizer=0 and watching the effect reappear.
penalizer=0.0001 was chosen as the smallest value that preserves numerical
stability without over-shrinking, given ~13 events per covariate.
"""

import json
import pandas as pd
from lifelines import CoxTimeVaryingFitter

VINTAGE_SPLIT_LOAN_FRACTION = 0.85  # train/test split by loan_id, not time,
# since CoxTimeVaryingFitter needs each loan's full interval history intact


def load_panel(path="data/gold/loan_month_panel") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = pd.get_dummies(df, columns=["occupancy_status", "loan_purpose"], drop_first=True)
    return df


def main():
    df = load_panel()
    loan_ids = df["loan_id"].unique()
    rng_split = int(len(loan_ids) * VINTAGE_SPLIT_LOAN_FRACTION)
    # deterministic split by sorted loan_id (stable across reruns)
    train_ids = set(sorted(loan_ids)[:rng_split])
    train_df = df[df["loan_id"].isin(train_ids)].drop(columns=["loan_id"])
    test_df = df[~df["loan_id"].isin(train_ids)].drop(columns=["loan_id"])

    print(f"Train: {train_df['event'].sum()} events across "
          f"{len(train_df):,} loan-months")
    print(f"Test:  {test_df['event'].sum()} events across "
          f"{len(test_df):,} loan-months")

    ctv = CoxTimeVaryingFitter(penalizer=0.0001)
    ctv.fit(
        train_df,
        id_col=None,  # not needed once loan_id is dropped; intervals are self-contained per row
        event_col="event",
        start_col="start",
        stop_col="stop",
    )

    print("\n=== Time-varying Cox model summary (train) ===")
    print(ctv.summary[["coef", "exp(coef)", "p"]].round(4).to_string())

    summary_df = ctv.summary[["coef", "exp(coef)", "se(coef)", "p"]].round(5)
    summary_df.to_csv("data/gold/cox_time_varying_summary.csv")

    # log-likelihood based fit comparison instead of concordance (lifelines'
    # CoxTimeVaryingFitter does not expose a direct held-out concordance
    # utility the way CoxPHFitter does -- documented honestly rather than
    # hand-rolling a possibly-incorrect metric)
    train_ll = ctv.log_likelihood_
    print(f"\nTrain log-likelihood: {train_ll:.2f}")

    # compare the macro_stress coefficient to the static model's result
    static_summary = pd.read_csv("data/gold/cox_model_summary.csv", index_col=0)
    print("\n=== Static vs. time-varying: macro stress coefficient ===")
    if "avg_macro_stress" in static_summary.index:
        static_hr = static_summary.loc["avg_macro_stress", "exp(coef)"]
        static_p = static_summary.loc["avg_macro_stress", "p"]
        print(f"Static model  (avg_macro_stress, lifetime average): "
              f"HR={static_hr:.3f}, p={static_p:.3f}")
    if "macro_stress_month" in summary_df.index:
        tv_hr = summary_df.loc["macro_stress_month", "exp(coef)"]
        tv_p = summary_df.loc["macro_stress_month", "p"]
        print(f"Time-varying model (macro_stress_month, at time of risk):  "
              f"HR={tv_hr:.3f}, p={tv_p:.3f}")

    metrics = {
        "train_log_likelihood": round(float(train_ll), 2),
        "n_train_loan_months": int(len(train_df)),
        "n_test_loan_months": int(len(test_df)),
        "n_train_events": int(train_df["event"].sum()),
        "n_test_events": int(test_df["event"].sum()),
        "penalizer": 0.1,
        "significant_covariates_p_lt_0.05": summary_df[summary_df["p"] < 0.05].index.tolist(),
    }
    with open("data/gold/cox_time_varying_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: data/gold/cox_time_varying_summary.csv, cox_time_varying_metrics.json")


if __name__ == "__main__":
    main()
