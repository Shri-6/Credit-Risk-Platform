# Model Risk Management Documentation

Prepared in the structure of a model risk document under **SR 11-7**
("Guidance on Model Risk Management," Federal Reserve / OCC, 2011). This
is a personal/portfolio project, not a submission to any actual regulator
or institution — the goal is to demonstrate the *documentation discipline*
SR 11-7 requires, not to claim formal regulatory compliance.

---

## 1. Model Overview and Purpose

| | |
|---|---|
| **Model name** | Residential Loan Portfolio Credit Risk & Early-Warning System |
| **Model owner** | [Your name] |
| **Business purpose** | Estimate probability of default (PD) and time-to-default for a residential mortgage portfolio; project portfolio expected loss under macroeconomic stress scenarios |
| **Model components** | (1) XGBoost binary classifier — 12-month PD; (2) Cox PH survival model (static and time-varying) — time-to-default and macro sensitivity; (3) Platt-scaling calibration layer; (4) CCAR-style stress testing engine |
| **Intended use** | Portfolio-level risk monitoring, early-warning flagging, stress-loss estimation |
| **Out of scope / not intended for** | Individual loan underwriting or adverse-action decisions (the model was not validated for fair-lending disparate-impact testing, a requirement for any use in origination decisions) |

## 2. Data

- **Source**: Synthetic data schema-matched to the Freddie Mac Single-Family
  Loan-Level Dataset (see `docs/SWAP_IN_REAL_DATA.md` for the real-data
  swap procedure and exact column mapping).
- **Population**: 25,000 loans, origination years 2018–2024, 20 U.S. states.
- **Lineage**: Raw CSV -> Bronze (schema-enforced, DQ-gated) -> Silver
  (deduplicated, business-rule validated) -> Gold (feature-engineered,
  macro-joined). Full lineage code in `src/ingestion/` and `src/features/`.
- **Known data limitation**: synthetic data, not the real Freddie Mac
  population. Directionally realistic (validated risk-factor separation
  by outcome — see Section 4) but not a substitute for real loan
  performance history for any actual risk decision.

## 3. Methodology

### 3.1 XGBoost Classifier
- Binary classification: P(90+ DPD within observation window).
- Time-based (vintage) train/test split — loans originated before 2022
  train the model; loans originated 2022+ are held out. This avoids the
  look-ahead bias a random split would introduce.
- Hyperparameters tuned via Optuna (30 trials, maximizing PR-AUC).
- Class imbalance (0.58% positive rate) addressed via `scale_pos_weight`.
- Explainability via SHAP (TreeExplainer).

### 3.2 Cox Proportional Hazards — Static
- Time-to-default model with proper right-censoring (prepayment,
  maturity, and end-of-observation-window all treated as censoring
  events, not defaults).
- **Known limitation, addressed in 3.3**: macro exposure computed as a
  lifetime average per loan, which dilutes the true time-of-event macro
  effect. `avg_macro_stress` was not statistically significant in this
  specification (p=0.93).

### 3.3 Cox Proportional Hazards — Time-Varying
- Same model family, refit on a loan-month panel (start/stop/event
  interval format) so macro covariates reflect conditions AS OF each
  month, not a lifetime average.
- Recovers a highly significant, correctly-signed macro effect
  (`macro_stress_month` HR=7.99, p<0.0001) that the static specification
  could not detect.
- **Model development issue found and corrected**: an L2 penalizer value
  copied from the static model (`penalizer=0.1`) over-shrank every
  coefficient toward zero on this much larger panel, due to how the
  penalty scales with dataset size. Corrected to `penalizer=0.0001`.
  Full detail in `docs/MODEL_CARD.md`.

### 3.4 Probability Calibration
- Raw XGBoost output is NOT used for expected-loss dollar calculations.
  `scale_pos_weight` inflates predicted probabilities by ~15x (mean
  predicted PD 8.7% vs. true base rate 0.58%) — a direct, expected
  consequence of correcting for class imbalance via loss reweighting,
  and a known failure mode worth checking for whenever this technique
  is used.
- Corrected via Platt scaling (1D logistic regression mapping raw score
  to calibrated probability), calibrated on the full labeled dataset
  given how few positive events (144) are available to calibrate from.

### 3.5 Stress Testing
- Combines the time-varying Cox model's macro *sensitivity* (relative
  hazard multiplier under a shocked macro path) with the calibrated
  classifier's *absolute* baseline PD — see
  `src/stress_testing/run_stress_test.py` docstring for why these two
  models are combined this way rather than using the Cox model's
  absolute baseline hazard directly.
- Four scenarios, loosely CCAR-styled: baseline, moderate stress, severe
  stress, severely adverse.
- 35% loss-given-default (LGD) assumption, a standard planning figure
  for first-lien residential mortgages (not empirically estimated from
  this dataset, which has too few defaults to estimate LGD directly).

## 4. Validation Results

| Model | Metric | Value |
|---|---|---|
| XGBoost classifier | Test ROC-AUC | 0.965 |
| XGBoost classifier | Test PR-AUC | 0.077 (baseline 0.0022) |
| XGBoost classifier | Test Brier score | 0.054 |
| Cox PH (static) | Test concordance | 0.965 |
| Cox PH (time-varying) | Train log-likelihood | -1004.85 |

**Directional validity check**: defaulted loans in this dataset average a
638 credit score vs. 741 for current loans, 85% original LTV vs. 71%,
82% mark-to-market regional LTV vs. 69%, and higher average macro-stress
and unemployment exposure — all in the expected direction, confirming the
models are learning genuine risk signal rather than artifacts.

## 5. Known Limitations (do not omit these when presenting this project)

1. **Small event count.** Test set: 23 default events. Any
   threshold-based metric (precision/recall) computed from this many
   events has high variance; ROC-AUC/concordance are more stable but
   still directional, not precise. This resolves on the real
   millions-of-loans dataset.
2. **Static Cox macro blind spot**, addressed by the time-varying
   refit (Section 3.3) — kept in this document because knowing *why* a
   simpler model failed, and what fixed it, is itself part of validation.
3. **LGD is an assumption (35%), not empirically estimated** — too few
   defaults in this dataset to estimate loss severity directly.
4. **Synthetic data** — see Section 2.
5. **Not validated for fair lending / disparate impact** — required
   before any use in an actual underwriting or adverse-action context;
   explicitly out of scope here.

## 6. Ongoing Monitoring Plan (proposed, not yet implemented)

- **Population stability index (PSI)** on key features (credit_score,
  LTV, DTI) computed monthly against the training distribution; alert if
  PSI > 0.2 on any monitored feature.
- **Rolling AUC/PR-AUC** recomputed quarterly as new performance data
  arrives, compared against the validation baseline in Section 4.
- **Recalibration trigger**: if the realized default rate deviates from
  the calibrated PD by more than [threshold] over a rolling 12-month
  window, trigger a full recalibration (Section 3.4 procedure).
- **Annual model revalidation**, including a full backtest against the
  most recent 12 months of realized outcomes.

## 7. Governance

- **Model owner**: [Your name]
- **Independent validation**: not performed (solo project) — in a real
  institution, SR 11-7 requires this modeling work to be validated by a
  team independent of model development before production use.
- **Change log**: tracked via git history in this repository; MLflow
  experiment tracking (`mlruns/`) records every training run's
  hyperparameters and metrics for auditability.
