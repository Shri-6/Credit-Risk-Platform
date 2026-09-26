# Resume Bullets — Draft

Pulled directly from validated results in this repo (see `docs/MODEL_CARD.md`
and `docs/MODEL_RISK_DOCUMENTATION.md` for full backing detail). Adjust
wording/tense to match the rest of your resume's voice, and swap in your
real numbers if you regenerate the pipeline and get different values.

---

## Option A — single consolidated entry (matches the density of the
## reference examples you started from)

**Residential Loan Portfolio Credit Risk & Early-Warning Platform**
*[Month Year] – [Month Year]*

- Built an end-to-end Medallion lakehouse (PySpark, Bronze/Silver/Gold)
  ingesting 25,000+ loans and 980,000+ monthly performance records
  schema-matched to the Freddie Mac Single-Family Loan-Level Dataset,
  with automated data quality gating and macroeconomic feature joins.
- Developed an XGBoost delinquency classifier (Optuna-tuned, SHAP-explained,
  time-based vintage validation split, test ROC-AUC 0.965) and both static
  and time-varying Cox proportional-hazards survival models (test
  concordance 0.965) for time-to-default estimation.
- Diagnosed and corrected two model-development defects: an L2
  regularization parameter that silently suppressed statistically real
  effects across a 30x larger dataset, and a probability-calibration
  issue (from class-imbalance correction) that inflated predicted default
  probabilities 15x — fixed via Platt scaling before any expected-loss
  calculation.
- Designed CCAR-style macro stress scenarios (baseline through
  severely-adverse) combining survival-model macro sensitivity with
  calibrated classifier output, projecting portfolio expected loss from
  0.16% to 0.72% of UPB; delivered results via an interactive Streamlit
  dashboard with SHAP-based explainability and loan-level drill-down.

---

## Option B — split into two entries (more granular, useful if this
## resume section rewards depth over density)

**Credit Risk Lakehouse & Predictive Modeling**
- Built a PySpark Medallion lakehouse (Bronze/Silver/Gold) with automated
  data quality gating, ingesting loan-level and macroeconomic time-series
  data schema-matched to Freddie Mac's public loan-level dataset.
- Trained an XGBoost delinquency classifier (Optuna hyperparameter search,
  SHAP explainability, MLflow experiment tracking) achieving 0.965 test
  ROC-AUC on a time-based (vintage) holdout split.
- Built static and time-varying Cox proportional-hazards survival models
  (`lifelines`) for time-to-default estimation; the time-varying
  specification recovered a statistically significant macroeconomic
  hazard effect (p<0.0001) that the static model's lifetime-averaged
  covariates had missed entirely.

**Macro Stress Testing & Risk Reporting**
- Designed a CCAR-style stress-testing engine projecting portfolio
  expected loss under four macroeconomic scenarios, combining survival-
  model hazard sensitivity with a Platt-calibrated classifier baseline
  probability of default.
- Corrected a 15x probability-calibration bias introduced by class-
  imbalance correction (`scale_pos_weight`) before it could propagate
  into dollar-denominated loss estimates.
- Authored SR 11-7-style model risk documentation (methodology,
  validation results, limitations, ongoing monitoring plan) and built an
  interactive Streamlit dashboard for portfolio monitoring and loan-level
  risk lookup.

---

## Notes on making these claims honestly

- **"25,000+ loans"**: true of the synthetic dataset as built here. If
  you swap in real Freddie Mac data before finalizing (see
  `docs/SWAP_IN_REAL_DATA.md`), update this to the real, larger count —
  it strengthens the claim further and removes any need to caveat it.
- **"Delta Lake"**: don't claim this unless you've actually done the
  one-line `WRITE_FORMAT = "delta"` swap on Databricks (see README,
  "Notes on Delta Lake vs. Parquet"). As built here, it's Parquet with
  Delta-shaped partitioning — genuinely close, but not the same claim.
- **Both bug-fix bullets are real and defensible** — you can go deep on
  either one in an interview without over-claiming. This is the
  strongest, least-generic material in the whole project; don't bury it
  at the bottom of a bullet list where a skimming reviewer might miss it.
- **If asked "how many loans actually defaulted"**: 144 out of 25,000
  (0.58%). Be ready to say this plainly rather than let a reviewer
  discover the small event count themselves and wonder why you didn't
  mention it.
