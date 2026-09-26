# CRE/Residential Loan Portfolio Credit Risk & Early-Warning Platform

End-to-end credit risk platform: Medallion lakehouse ingestion → delinquency
classification (XGBoost) → time-to-default survival modeling (Cox PH,
static and time-varying) → macro stress testing → Streamlit risk dashboard.

## 60-second summary

**What it does**: ingests loan-level and macroeconomic data through a
Bronze/Silver/Gold lakehouse, trains an XGBoost classifier and a Cox
proportional-hazards survival model to estimate default risk, then runs
CCAR-style macro stress scenarios to project portfolio expected loss —
all surfaced in an interactive dashboard.

**Key results**:
| | |
|---|---|
| Classifier test ROC-AUC | 0.965 |
| Survival model test concordance | 0.965 |
| Time-varying macro effect (fixed from insignificant to) | HR=7.99, p<0.0001 |
| Expected loss, baseline → severely-adverse stress | 0.16% → 0.72% of UPB |

**What makes this more than a tutorial project**: two real bugs found and
fixed during development, not glossed over —
1. a regularization parameter that scaled incorrectly across a 30x larger
   dataset and silently zeroed out a known-real predictor (caught by
   noticing an *implausible* result, not a crash);
2. a probability-calibration issue where the technique used to fix class
   imbalance inflated predicted default probabilities 15x, which would
   have overstated every dollar figure in the stress test had it gone
   unchecked.

Full narrative in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — worth
reading before you talk about this project in an interview, since "how
did you catch that" is exactly what gets asked.

**Stack**: PySpark, XGBoost, lifelines (Cox PH), Optuna, SHAP, MLflow,
scikit-learn, Streamlit, Plotly.

**A note on data**: this was built against a schema-accurate synthetic
dataset (matched to the real Freddie Mac Single-Family Loan-Level
Dataset) because the build environment couldn't reach Freddie Mac's
data portal or the FRED API. [`docs/SWAP_IN_REAL_DATA.md`](docs/SWAP_IN_REAL_DATA.md)
has the exact ~15-minute procedure to point this same pipeline at the
real download — worth doing before finalizing if you have the time, and
worth stating plainly if you haven't yet.

---

## Status: all planned phases (Weeks 1–11) complete

- [x] Synthetic loan-level data generator, schema-matched to the real
      **Freddie Mac Single-Family Loan-Level Dataset**
- [x] Bronze layer: schema-enforced raw ingestion + automated DQ gating
- [x] Silver layer: dedup, business-rule validation, quarantine of bad rows
- [x] Gold layer: model-ready feature table (25,000 loans, 40 features,
      0.58% cumulative default rate, realistic risk-factor separation)
- [x] Synthetic macro time series (unemployment, national + state-level
      HPI, 30yr mortgage rate) matching FRED series structure, joined
      onto the loan panel
- [x] Regional mark-to-market LTV (`current_ltv_regional`), refi
      incentive, and unemployment-exposure features
- [x] XGBoost delinquency classifier: Optuna-tuned, MLflow-tracked,
      SHAP-explained, time-based (vintage) train/test split, class-imbalance
      corrected. Test ROC-AUC 0.965.
- [x] Cox PH survival model (`lifelines`): time-to-default with proper
      censoring, concordance 0.965, hazard ratios for every covariate.
      See `docs/MODEL_CARD.md` for honest limitations (small event count,
      static vs. time-varying macro effect).
- [x] Time-varying Cox model on the loan-month panel: fixes the static
      model's macro_stress blind spot (HR 7.99, p<0.0001 vs. previously
      insignificant). Two real bugs found and fixed during this phase —
      see `docs/MODEL_CARD.md`.
- [x] Platt-scaling probability calibration (raw classifier PD was
      inflated ~15x by `scale_pos_weight`; fixed before any dollar-based
      calculation).
- [x] CCAR-style macro stress testing: baseline / moderate / severe /
      severely-adverse scenarios, expected loss 0.16% -> 0.72% of UPB.
- [x] Streamlit risk dashboard (`src/dashboard/app.py`): portfolio
      overview, stress test visualization, SHAP explainability, loan-level
      lookup.
- [x] SR 11-7-style model risk documentation
      (`docs/MODEL_RISK_DOCUMENTATION.md`): methodology, validation
      results, known limitations, monitoring plan, governance.
- [ ] Week 12 (in progress): repo cleanup (`.gitignore` added), push to
      GitHub with a real commit history, record a 2-minute demo walkthrough
      of the dashboard, finalize resume bullets (see
      `docs/RESUME_BULLETS.md`)

## Why synthetic data, and how to swap in the real dataset

This sandbox environment can't reach the Freddie Mac data portal (requires
a free account + login) or the FRED API (needs an API key), so Weeks 1-2
were built against a **schema-accurate synthetic dataset** — see
`docs/SWAP_IN_REAL_DATA.md` for the exact, ~15-minute steps to point this
same pipeline at the real download. Nothing in `src/` needs to change;
only the CSVs in `data/raw/` are swapped.

## Architecture

```
data/raw/*.csv
      │
      ▼
┌─────────────┐   schema enforcement        ┌─────────────┐
│   BRONZE    │──  + DQ gating (asserts)  ──▶│   SILVER    │
│  (raw types)│                              │ (validated, │
└─────────────┘                              │  deduped)   │
                                              └──────┬──────┘
                                                     │ feature engineering
                                                     ▼
                                              ┌─────────────┐
                                              │    GOLD     │
                                              │ (model-ready│
                                              │  1 row/loan)│
                                              └──────┬──────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    ▼                ▼                ▼
                            XGBoost classifier  Cox PH survival   Stress tests
                                    │                │                │
                                    └────────────────┴────────────────┘
                                                     │
                                                     ▼
                                          Streamlit risk dashboard
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline (Weeks 1–11)

```bash
# 1. Generate synthetic loan-level data (schema-matched to Freddie Mac)
python src/ingestion/generate_synthetic_data.py --n_loans 25000 --out_dir data/raw

# 2. Generate synthetic macro time series (unemployment, HPI, mortgage rate)
python src/ingestion/generate_macro_data.py

# 3. Bronze: raw ingest + schema enforcement + DQ gates
python src/ingestion/bronze_ingest.py

# 4. Silver: dedup + business-rule validation
python src/ingestion/silver_transform.py

# 5. Gold: feature engineering + macro joins -> model-ready table
python src/features/gold_features.py

# 6. Train XGBoost classifier (Optuna tuning + MLflow tracking + SHAP)
python src/models/train_classifier.py

# 7. Train Cox PH survival model (static)
python src/models/train_survival.py

# 8. Build the loan-month panel for time-varying survival modeling
python src/features/build_time_varying_panel.py

# 9. Train the time-varying Cox model (fixes the macro-stress blind spot above)
python src/models/train_survival_time_varying.py

# 10. Calibrate the classifier's probabilities (must run before stress testing)
python src/models/calibrate_classifier.py

# 11. Run the CCAR-style stress testing engine
python src/stress_testing/run_stress_test.py

# 12. Launch the risk dashboard (after all of the above have been run at least once)
streamlit run src/dashboard/app.py
# opens at http://localhost:8501

# View tracked experiments (after step 6)
mlflow ui   # then open http://localhost:5001 if 5000 is taken (macOS AirPlay)
```

See `docs/MODEL_RISK_DOCUMENTATION.md` for the full SR 11-7-style writeup
(methodology, validation results, limitations, monitoring plan) and
`docs/MODEL_CARD.md` for full results and — importantly — the honest
limitations of both models given this dataset's size. Don't skip reading
that before you write resume bullets off these numbers.

Each stage prints row counts and DQ results so you can see the pipeline
actually gating bad data, not just passing everything through.

**A real bug worth knowing about (and worth mentioning if asked about
this project in an interview):** the first version of the macro state-HPI
generator silently omitted Texas from its volatility-parameter dict. Since
the Gold-layer join is `on=["reporting_period", "property_state"]`, every
Texas loan got a `null` for `current_ltv_regional` instead of an error —
a classic silent-join-failure. Caught it by checking null counts on the
new columns post-join (`df[cols].isnull().sum()`) rather than assuming the
join worked because the job didn't crash. Fixed by asserting
`set(US_STATES) == set(STATE_HPI_BETA.keys())` at generation time, so this
class of bug fails loudly next time instead of silently nulling rows.

## Repo structure

```
data/            raw -> bronze -> silver -> gold, Medallion architecture
src/ingestion/   loan + macro data generation, Bronze ingest, Silver transform
src/features/    Gold feature engineering (loan + macro joins)
src/models/      XGBoost classifier, static + time-varying Cox, calibration
src/stress_testing/  CCAR-style macro scenario engine
src/dashboard/   Streamlit risk dashboard
docs/            model card, model risk documentation, real-data swap instructions
notebooks/       exploratory analysis
```

## Notes on Delta Lake vs. Parquet

This pipeline writes partitioned Parquet (`WRITE_FORMAT = "parquet"` at the
top of each ingestion script) because this build environment can't reach
Maven Central to download the Delta Lake JARs. On Databricks Community
Edition (free, and what the original resume-style project targets), change
`WRITE_FORMAT` to `"delta"` — every read/write call already goes through
that constant, so the entire pipeline gets ACID transactions, schema
evolution, time travel, and `MERGE INTO` support with a one-line change.
This is worth doing before you finalize the project, since "Delta Lake" is
a specific, checked resume claim.
