"""
Portfolio risk dashboard.

Run with: streamlit run src/dashboard/app.py

Reads only from data/gold/ -- run the full pipeline (Bronze -> Silver ->
Gold -> models -> stress test) before launching this, or it will show
"file not found" errors for whichever stage hasn't been run yet.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CRE/Residential Loan Risk Platform", layout="wide")

GOLD = "data/gold"


@st.cache_data
def load_data():
    loans = pd.read_parquet(f"{GOLD}/loan_features")
    calib = pd.read_parquet(f"{GOLD}/calibrated_predictions.parquet")
    loans = loans.merge(calib[["loan_id", "baseline_pd_calibrated"]], on="loan_id", how="left")

    stress = pd.read_csv(f"{GOLD}/stress_test_results.csv")
    shap_importance = pd.read_csv(f"{GOLD}/shap_feature_importance.csv", index_col=0).squeeze("columns")
    shap_importance.name = "mean_abs_shap"

    with open(f"{GOLD}/classifier_metrics.json") as f:
        clf_metrics = json.load(f)
    with open(f"{GOLD}/survival_metrics.json") as f:
        surv_metrics = json.load(f)

    return loans, stress, shap_importance, clf_metrics, surv_metrics


try:
    loans, stress, shap_importance, clf_metrics, surv_metrics = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing pipeline output: {e}\n\n"
        "Run the full pipeline first (see README.md steps 1-11) before "
        "launching this dashboard."
    )
    st.stop()

st.title("CRE / Residential Loan Portfolio Risk Platform")
st.caption(
    "Built on synthetic data schema-matched to the Freddie Mac Single-Family "
    "Loan-Level Dataset. See docs/MODEL_CARD.md for full methodology, "
    "validation results, and known limitations."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Overview", "Stress Testing", "Model Explainability", "Loan Lookup"
])

# ---------------- Tab 1: Portfolio Overview ----------------
with tab1:
    n_loans = len(loans)
    total_upb = loans["latest_upb"].sum()
    default_rate = loans["event_observed"].mean()
    active_upb = loans.loc[loans["event_observed"] == 0, "latest_upb"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total loans", f"{n_loans:,}")
    c2.metric("Total UPB", f"${total_upb/1e6:,.1f}M")
    c3.metric("Cumulative default rate", f"{default_rate*100:.2f}%")
    c4.metric("Active (non-defaulted) UPB", f"${active_upb/1e6:,.1f}M")

    col1, col2 = st.columns(2)

    with col1:
        by_vintage = loans.groupby("orig_year").agg(
            loan_count=("loan_id", "count"),
            default_rate=("event_observed", "mean"),
        ).reset_index()
        fig = px.bar(by_vintage, x="orig_year", y="default_rate",
                     title="Default Rate by Origination Vintage",
                     labels={"orig_year": "Origination Year", "default_rate": "Default Rate"})
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_state = loans.groupby("property_state").agg(
            loan_count=("loan_id", "count"),
            avg_upb=("latest_upb", "mean"),
            default_rate=("event_observed", "mean"),
        ).reset_index().sort_values("loan_count", ascending=False)
        fig = px.bar(by_state, x="property_state", y="loan_count",
                     color="default_rate", color_continuous_scale="Reds",
                     title="Loan Count by State (colored by default rate)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(loans, x="credit_score", nbins=40,
                            color=loans["event_observed"].map({0: "Current", 1: "Defaulted"}),
                            title="Credit Score Distribution by Outcome",
                            labels={"color": "Outcome"}, barmode="overlay", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(loans, x="current_ltv_regional", nbins=40,
                            color=loans["event_observed"].map({0: "Current", 1: "Defaulted"}),
                            title="Regional Mark-to-Market LTV by Outcome",
                            labels={"color": "Outcome"}, barmode="overlay", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 2: Stress Testing ----------------
with tab2:
    st.subheader("CCAR-Style Macro Stress Scenarios")
    st.caption(
        "Hazard multiplier from the time-varying Cox model's macro coefficients, "
        "applied to the Platt-calibrated XGBoost baseline PD. "
        "Assumes 35% loss given default (LGD)."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stress["scenario"], y=stress["expected_loss_pct_of_upb"],
        text=stress["expected_loss_pct_of_upb"].map(lambda v: f"{v:.2f}%"),
        textposition="outside",
        marker_color=["#2ca02c", "#ff7f0e", "#d62728", "#8b0000"][:len(stress)],
    ))
    fig.update_layout(
        title="Expected Loss (% of UPB) by Scenario",
        yaxis_title="Expected Loss (% of UPB)", xaxis_title="Scenario",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        stress[["scenario", "description", "hazard_multiplier",
                "portfolio_mean_pd", "expected_loss_usd", "expected_loss_pct_of_upb"]],
        use_container_width=True, hide_index=True,
    )

    st.info(
        "**Design note:** absolute default probabilities come from the "
        "calibrated XGBoost classifier, not the Cox model's baseline "
        "hazard directly -- the time-varying Cox model is used only for "
        "its macro *sensitivity* (the relative hazard multiplier). "
        "See `src/stress_testing/run_stress_test.py` docstring for the "
        "full reasoning."
    )

# ---------------- Tab 3: Model Explainability ----------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classifier Performance")
        st.metric("Test ROC-AUC", clf_metrics["test_roc_auc"])
        st.metric("Test PR-AUC", clf_metrics["test_pr_auc"])
        st.metric("Test Brier Score", clf_metrics["test_brier_score"])
        st.caption(
            f"Test set: {clf_metrics['n_test']:,} loans, "
            f"{clf_metrics['n_test_events']} events. See MODEL_CARD.md -- "
            "with this few events, treat these as directionally strong, "
            "not precise point estimates."
        )

    with col2:
        st.subheader("Survival Model Performance")
        st.metric("Test Concordance Index", surv_metrics["test_concordance_index"])
        st.caption(
            f"Test set: {surv_metrics['n_test']:,} loans, "
            f"{surv_metrics['n_test_events']} events."
        )

    st.subheader("SHAP Feature Importance (Classifier)")
    top_shap = shap_importance.sort_values(ascending=True).tail(15)
    fig = px.bar(x=top_shap.values, y=top_shap.index, orientation="h",
                 labels={"x": "Mean |SHAP value|", "y": "Feature"},
                 title="Top 15 Features by Mean Absolute SHAP Value")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 4: Loan Lookup ----------------
with tab4:
    st.subheader("Individual Loan Risk Profile")
    loan_id = st.selectbox("Select a loan", options=loans["loan_id"].sort_values().tolist())

    row = loans[loans["loan_id"] == loan_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Credit Score", int(row["credit_score"]))
    c2.metric("Original LTV", f"{row['orig_ltv']:.1f}%")
    c3.metric("Current Regional LTV", f"{row['current_ltv_regional']:.1f}%")
    c4.metric("Calibrated PD", f"{row['baseline_pd_calibrated']*100:.3f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Origination Year", int(row["orig_year"]))
    c6.metric("Property State", row["property_state"])
    c7.metric("Current UPB", f"${row['latest_upb']:,.0f}")
    c8.metric(
        "Status",
        "Defaulted" if row["event_observed"] == 1 else "Current",
    )

    st.caption(
        "PD shown is the Platt-calibrated XGBoost prediction. Compare against "
        f"the portfolio-wide baseline mean PD of {loans['baseline_pd_calibrated'].mean()*100:.3f}% "
        "to gauge relative risk."
    )
