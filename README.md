# CRE / Residential Loan Portfolio Credit Risk & Early-Warning Platform

An end-to-end credit-risk project built around a synthetic residential mortgage portfolio. The pipeline follows a Bronze/Silver/Gold data flow, trains an XGBoost delinquency classifier and Cox proportional-hazards survival models, applies macroeconomic stress scenarios, and presents the results through a Streamlit dashboard.

> **Important data note:** This project uses synthetic data throughout. We did not obtain or analyze the actual Freddie Mac loan-level dataset or live FRED data. The synthetic loan data was generated to follow the structure and schema of the Freddie Mac Single-Family Loan-Level Dataset, and the synthetic macro data follows the structure of relevant FRED series. The model results below therefore describe experiments on synthetic data and should not be presented as results obtained from real Freddie Mac or FRED observations.

## 60-second summary

### What the project does

The project takes synthetic loan-level and macroeconomic data through a Bronze/Silver/Gold pipeline, creates risk features, and trains two types of models:

- **XGBoost classifier** for estimating delinquency/default risk.
- **Cox proportional-hazards models** for looking at time-to-default while accounting for censored observations.

The pipeline also includes macroeconomic stress scenarios and a Streamlit dashboard where the portfolio results, stress-test output, SHAP explanations, and individual loan information can be explored.

### Results from the synthetic dataset

| Metric | Result |
|---|---:|
| Classifier test ROC-AUC | 0.965 |
| Survival model test concordance | 0.965 |
| Time-varying macro effect | HR = 7.99, p < 0.0001 |
| Expected loss, baseline → severely adverse scenario | 0.16% → 0.72% of UPB |

These numbers should be treated as **results from the synthetic experiment**, not as evidence about the performance of a model on a real mortgage portfolio.

## Development issues that were found

One useful part of the project was finding problems that did not cause the pipeline to crash.

The first was a regularization parameter that was scaled incorrectly after the dataset became roughly 30 times larger. The result was that a predictor that was expected to contribute to the model was effectively regularized away. The problem was identified because the resulting model behavior looked implausible, rather than because the code produced an exception.

The second involved probability calibration. The class-imbalance setting used during XGBoost training caused the raw predicted probabilities to be substantially inflated. In the experiment, the raw default probabilities were roughly 15 times higher than the calibrated values. This was important because those probabilities feed into the stress-testing calculations. Using the uncalibrated values would have materially overstated the dollar-based loss estimates.

These were development and modeling issues in this synthetic-data project. They should not be described as bugs discovered in a production banking system or in real Freddie Mac data.

## Data and scope

The project was built with a **schema-accurate synthetic loan dataset** based on the structure of the Freddie Mac Single-Family Loan-Level Dataset.

The synthetic dataset contains:

- 25,000 loans
- 40 model features
- A cumulative default rate of approximately 0.58%
- Synthetic unemployment data
- Synthetic national and state-level HPI data
- Synthetic 30-year mortgage-rate data
- Regional mark-to-market LTV
- Refinance incentive features
- Unemployment-exposure features

Because the data is synthetic, the realistic-looking distributions and relationships are assumptions built into the data generator. They are useful for testing the pipeline, but they are not empirical findings about the mortgage market.

## Project status

The core data, modeling, stress-testing, and dashboard pipeline is complete.

- [x] Synthetic loan-level data generator with a schema modeled on the **Freddie Mac Single-Family Loan-Level Dataset**
- [x] Bronze layer with schema enforcement and automated data-quality checks
- [x] Silver layer with deduplication, business-rule validation, and bad-row quarantine
- [x] Gold layer with model-ready features
- [x] Synthetic macroeconomic time series for unemployment, HPI, and mortgage rates
- [x] Regional mark-to-market LTV (`current_ltv_regional`)
- [x] Refinance-incentive and unemployment-exposure features
- [x] XGBoost delinquency classifier with Optuna tuning, MLflow tracking, SHAP explanations, a time-based vintage split, and class-imbalance handling
- [x] Cox PH survival model using `lifelines`, including censoring and hazard ratios
- [x] Time-varying Cox model using the loan-month panel
- [x] Probability calibration using Platt scaling
- [x] Macro stress-testing engine with baseline, moderate, severe, and severely adverse scenarios
- [x] Streamlit dashboard with portfolio overview, stress-test visualization, SHAP explanations, and loan-level lookup
- [x] Model-risk documentation covering methodology, validation, limitations, monitoring, and governance
- [ ] Final repository cleanup, GitHub history, dashboard walkthrough, and final resume bullets

## Why synthetic data?

The main reason for using synthetic data was practical: the goal of this project was to build and test the complete pipeline without depending on access to external datasets or APIs.

The synthetic data was designed to resemble the structure of the target datasets so that the ingestion, feature engineering, modeling, and dashboard components could be developed end to end.

This distinction matters:

**What we did:**  
Built and tested the complete pipeline using synthetic data that follows the target schema.

**What we did not do:**  
Download, clean, or model the actual Freddie Mac loan-level dataset or live FRED observations.

The code is structured so that the input files can be replaced later if access to the real datasets is obtained. However, that should be described as a planned extension rather than something already completed.

## Architecture

```text
Synthetic raw CSV files
        │
        ▼
┌─────────────────┐
│     BRONZE      │
│ raw data types  │
│ schema + DQ     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     SILVER      │
│ validated       │
│ deduplicated    │
└────────┬────────┘
         │
         │ feature engineering
         ▼
┌─────────────────┐
│      GOLD       │
│ model-ready     │
│ loan features   │
└────────┬────────┘
         │
         ├───────────────┐
         ▼               ▼
  XGBoost classifier   Cox PH models
         │               │
         └───────┬───────┘
                 │
                 ▼
          Stress testing
                 │
                 ▼
        Streamlit dashboard
```

## Pipeline

The complete workflow is:

```bash
# 1. Generate synthetic loan-level data
python src/ingestion/generate_synthetic_data.py --n_loans 25000 --out_dir data/raw

# 2. Generate synthetic macro time series
python src/ingestion/generate_macro_data.py

# 3. Bronze: raw ingest + schema enforcement + DQ checks
python src/ingestion/bronze_ingest.py

# 4. Silver: deduplication + business-rule validation
python src/ingestion/silver_transform.py

# 5. Gold: feature engineering + macro joins
python src/features/gold_features.py

# 6. Train XGBoost classifier
python src/models/train_classifier.py

# 7. Train static Cox PH survival model
python src/models/train_survival.py

# 8. Build the loan-month panel
python src/features/build_time_varying_panel.py

# 9. Train the time-varying Cox model
python src/models/train_survival_time_varying.py

# 10. Calibrate classifier probabilities
python src/models/calibrate_classifier.py

# 11. Run macro stress testing
python src/stress_testing/run_stress_test.py

# 12. Launch the Streamlit dashboard
streamlit run src/dashboard/app.py
```

The dashboard runs locally at:

```text
http://localhost:8501
```

MLflow can be started with:

```bash
mlflow ui
```

## Model development

### XGBoost delinquency classifier

The classifier uses a time-based vintage split rather than a completely random train/test split. This was intended to make the evaluation more representative of a forward-looking modeling setup.

The training process includes:

- Optuna hyperparameter tuning
- MLflow experiment tracking
- SHAP-based feature explanations
- Class-imbalance handling
- Probability calibration before using the predictions in the stress-testing calculations

The final test ROC-AUC on the synthetic dataset was **0.965**.

### Cox proportional-hazards survival model

The survival component models time-to-default rather than treating default as only a binary outcome.

The static Cox model uses:

- Time-to-event information
- Censoring
- Hazard ratios for model covariates

A time-varying Cox model was then added using a loan-month panel. This allowed macroeconomic variables to change over time rather than being treated as fixed loan-level values.

The time-varying model produced a macro-stress hazard ratio of **7.99 (p < 0.0001)** in the synthetic experiment.

Because the underlying data is synthetic and the event count is limited, this result should not be interpreted as an estimate of the actual effect of macroeconomic stress on real mortgage defaults.

## Probability calibration

One of the more important modeling checks was comparing the classifier's raw probabilities with calibrated probabilities.

The class-imbalance correction used during XGBoost training affected the probability estimates. In this experiment, the raw probabilities were approximately 15 times higher than the calibrated values.

That distinction matters because the classifier output is later used in expected-loss calculations. Using the raw probabilities without checking calibration would have produced materially larger loss estimates.

Platt scaling was therefore applied before the stress-testing stage.

## Macro stress testing

The stress-testing component uses four scenarios:

1. Baseline
2. Moderate
3. Severe
4. Severely adverse

The scenarios modify the synthetic macroeconomic variables and use the resulting risk estimates to calculate portfolio expected loss.

In the synthetic experiment, expected loss increased from approximately:

**0.16% → 0.72% of UPB**

between the baseline and severely adverse scenarios.

These figures are scenario outputs from the synthetic portfolio. They are not regulatory stress-test results and should not be described as actual CCAR results.

The term **"CCAR-style"** refers only to the general structure of scenario-based portfolio stress testing. This project was not submitted to or validated by the Federal Reserve or another regulator.

## Data-quality issue found during development

A particularly useful data-engineering issue occurred in the synthetic state-level HPI generation.

The state HPI generator initially omitted Texas from its volatility-parameter dictionary. Because the Gold-layer join used:

```python
["reporting_period", "property_state"]
```

Texas loans received null values for `current_ltv_regional` instead of causing the pipeline to fail.

This is a good example of a silent join failure: the job completed successfully, but the output was wrong.

The issue was found by checking null counts in the newly joined columns:

```python
df[cols].isnull().sum()
```

The generator was then changed to validate that the set of states in the data matched the set of states represented in the HPI parameter dictionary:

```python
set(US_STATES) == set(STATE_HPI_BETA.keys())
```

This makes the pipeline fail earlier if a state is accidentally omitted.

Again, this was a bug in the synthetic-data development workflow, not a data-quality issue found in the actual Freddie Mac dataset.

## Technology stack

- Python
- PySpark
- XGBoost
- `lifelines`
- Optuna
- SHAP
- MLflow
- scikit-learn
- Streamlit
- Plotly
- Parquet

## Repository structure

```text
data/
    raw/              synthetic input data
    bronze/           schema-enforced data
    silver/           validated and deduplicated data
    gold/             model-ready feature table

src/
    ingestion/        synthetic data generation and ingestion
    features/         feature engineering and macro joins
    models/           XGBoost, Cox models, calibration
    stress_testing/   macro scenario engine
    dashboard/        Streamlit application

docs/
    model card
    model-risk documentation
    real-data integration notes

notebooks/
    exploratory analysis
```

## Parquet vs. Delta Lake

The current implementation writes **partitioned Parquet**, not Delta Lake.

This was a deliberate environment limitation: the build environment could not download the required Delta Lake JARs.

The ingestion code keeps the write format behind a configuration constant, so the same pipeline can be adapted to Delta Lake in an environment where the required dependencies are available.

For that reason, the project should currently be described as a **Parquet-based Bronze/Silver/Gold pipeline** rather than claiming that it already runs on Delta Lake.

If the project is later moved to Databricks and validated with Delta Lake, the implementation can be updated to use Delta features such as ACID transactions, schema evolution, time travel, and `MERGE INTO`.

## Important limitations

There are several limitations to keep in mind when presenting this project:

1. **The loan and macro datasets are synthetic.** No real Freddie Mac or FRED data was used.
2. **The model metrics are therefore experimental results on generated data.**
3. **The synthetic generator controls the relationships in the data.** High model performance does not necessarily mean the models would perform similarly on real mortgage data.
4. **The survival models have a limited number of default events**, which limits how confidently the estimated hazard relationships can be interpreted.
5. **The stress scenarios are illustrative.** They are not official regulatory scenarios.
6. **The current storage layer uses Parquet rather than Delta Lake.**
7. **The project has not been validated against an external real-world holdout dataset.**

These limitations are important because they separate what was actually demonstrated in the project from what would need to be tested before using the system for real credit-risk decision-making.
