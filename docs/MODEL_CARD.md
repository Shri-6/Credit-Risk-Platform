# Model Card: Delinquency Classifier + Survival Model

## XGBoost Classifier

- **Task**: P(loan reaches 90+ DPD within observation window)
- **Test set**: 10,613 loans (2022+ vintage, time-based holdout), 23 events
- **ROC-AUC**: 0.965 | **PR-AUC**: 0.077 (baseline 0.0022, ~35x lift) | **Brier**: 0.054
- **Top SHAP drivers**: credit_score, orig_dti, avg_unemployment_exposure,
  avg_macro_stress, orig_ltv

### Honest limitation
The test set has only **23 positive events**. PR-AUC and any threshold-based
metric (precision/recall at a cutoff) computed from 23 events has enormous
variance — a handful of borderline cases going the other way would move
these numbers substantially. ROC-AUC is somewhat more stable since it's
rank-based over all negatives too, but should still be read as "strong
separation, not a precise point estimate." **On the real Freddie Mac
dataset (millions of loans), this problem disappears** — flagged here so
it's not a surprise, and so you can speak to it directly if asked ("how
confident are you in that AUC?" — answer: directionally very, precisely
no, and here's why).

## Cox Proportional Hazards Survival Model

- **Task**: time-to-default, with prepayment/maturity/censoring handled correctly
- **Test set concordance index**: 0.965 (same caveat as above — 23 events)
- **Hazard ratios** (train set, statistically significant at p<0.05):
  - `credit_score`: HR = 0.997 (each +1 point ≈ 0.3% lower instantaneous hazard)
  - `orig_ltv`: HR = 1.004 (each +1 LTV point ≈ 0.4% higher hazard)
  - `current_ltv_regional`: HR = 1.004 (regional mark-to-market LTV, same direction)
  - `mi_pct`: HR = 1.006 (mortgage insurance presence correlates with risk, as expected —
    MI is more common on high-LTV loans, so this is partly proxying for LTV)

### Honest limitation
`avg_macro_stress` came back **statistically insignificant** (p = 0.93) with
the "wrong" sign in this static Cox specification. This is a real modeling
issue, not a bug: `avg_macro_stress` is a single number averaged over each
loan's *entire* observed life, but a loan that defaults early only
experienced a few months of whatever stress regime was active then — the
lifetime average dilutes the signal that a **time-varying Cox model**
would capture properly (stress *at the time of each risk-set evaluation*,
not averaged over history). This is the natural "Week 8+" extension:
refit as a Cox model with time-varying covariates (`lifelines`'
`CoxTimeVaryingFitter`), which is also more defensible for the stress-testing
phase, since it lets you literally re-run the hazard function under a
shocked macro path instead of re-averaging a static feature.

**This is worth stating plainly in an interview** rather than hiding:
knowing *why* a static Cox model understates a time-varying effect, and
what the fix is, is a stronger signal of understanding survival analysis
than a clean p-value would have been.

## Update: Time-Varying Cox Model (fixes the macro_stress issue above)

Refit as `CoxTimeVaryingFitter` on the loan-month panel
(`src/features/build_time_varying_panel.py`, `src/models/train_survival_time_varying.py`),
so macro conditions are evaluated at the correct point in time relative to
each loan's risk set, instead of averaged over its whole life.

**Result: `macro_stress_month` HR = 7.99, p < 0.0001** (vs. the static
model's HR = 0.944, p = 0.934 — statistically indistinguishable from no
effect). `credit_score`, `orig_dti`, `mi_pct`, and `current_ltv_regional_month`
are all significant at p < 0.001 as well. This is the result the static
model's design structurally couldn't produce.

### Bug caught during this upgrade #1: penalizer scaling
Copying the static model's `penalizer=0.1` onto the ~830,000-row loan-month
panel silently crushed every coefficient toward zero — including
`credit_score`, which should be (and, once fixed, is) overwhelmingly
significant. lifelines' L2 penalty scales with dataset size, so the same
penalizer value is far stronger on a much larger panel. Caught by noticing
an implausible result (a rock-solid predictor becoming insignificant)
rather than trusting a clean-looking summary table; confirmed by testing
`penalizer=0` and watching the effect reappear, then settling on
`penalizer=0.0001` as the smallest value that's numerically stable without
over-shrinking.

## Bug caught during stress testing #2: probability calibration
`scale_pos_weight` (used in the XGBoost classifier to fix ranking
performance on a ~0.6% positive rate) inflates predicted probabilities by
roughly **15x** — mean predicted PD was 8.7% against a true base rate of
0.58%. This doesn't affect ROC-AUC/PR-AUC (both are rank-based, unaffected
by any monotonic transformation of the scores), but it would badly
overstate any dollar-denominated expected-loss figure. Fixed with Platt
scaling (`src/models/calibrate_classifier.py`): a 1D logistic regression
mapping the inflated raw score back onto the true base rate, calibrated on
the full labeled set rather than a further held-out fold (144 total events
is already thin; splitting again would make the calibration itself
unstable — a real, stated limitation, not hidden). Post-calibration mean
PD: 0.58%, matching the true rate almost exactly.

## Stress Test Results (CCAR-style scenarios)

Using the time-varying Cox model's macro coefficients as a relative hazard
multiplier applied to the calibrated XGBoost baseline PD (see
`src/stress_testing/run_stress_test.py` docstring for why this
combination, rather than the Cox model's absolute baseline hazard, was
used):

| Scenario | Hazard multiplier | Portfolio mean PD | Expected loss (% of UPB) |
|---|---|---|---|
| Baseline | 1.00x | 0.47% | 0.17% |
| Moderate stress | 1.70x | 0.80% | 0.29% |
| Severe stress | 3.92x | 1.85% | 0.67% |
| Severely adverse (CCAR-style) | 6.61x | 2.79% | 1.00% |

Assumes 35% loss given default (LGD), a standard planning assumption for
residential first-lien mortgages.
