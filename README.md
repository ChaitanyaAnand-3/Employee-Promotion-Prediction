# Employee Promotion Prediction

A machine learning project that uses historical employee information to estimate whether an employee may be promoted. It compares classification algorithms, tunes CatBoost with stratified cross-validation, and includes a Streamlit app for individual and CSV predictions.

> **Responsible use:** Treat predictions as decision support for human review. Do not use this model as the sole basis for employment or promotion decisions. Historical data may reflect past bias, and model performance should be checked across employee groups.

## Project contents

- `Untitled16.ipynb` — notebook with data exploration, preprocessing, feature engineering, model comparison, and prediction workflow.
- `train_LZdllcl.csv` — labeled employee training dataset (`is_promoted` is the target).
- `test_2umaH9m.csv` — separate employee dataset without promotion labels, used to generate predictions.
- `algorithm_f1_scores_cv_optuna.csv` — model comparison scores, including the five-fold CatBoost Optuna result.
- `promotion_predictions_full_train_cv_optuna.csv` — predictions generated for the separate unlabeled test file.
- `promotion_best_model_cv_optuna.joblib` — saved tuned CatBoost model.
- `streamlit_promotion_app/` — Streamlit application, model, configuration, and app dependencies.

## Dataset

The labeled training file contains **54,808 employee records** and 14 columns: an employee ID, 12 predictor columns, and the target `is_promoted`.

- `is_promoted = 1`: promoted
- `is_promoted = 0`: not promoted
- 4,668 records (about **8.5%**) are promoted, so the target classes are imbalanced.
- `education` and `previous_year_rating` contain missing values.

## Feature engineering

Four additional features are calculated from the original employee columns:

| Feature | Formula | Description |
|---|---|---|
| `training_per_service` | `no_of_trainings / (length_of_service + 1)` | Training count relative to years of service; adding 1 avoids division by zero. |
| `score_x_kpi` | `avg_training_score * (KPIs_met >80%)` | Combines training score with the KPI indicator. |
| `rating_x_kpi` | `previous_year_rating * (KPIs_met >80%)` | Combines prior-year rating with the KPI indicator. |
| `age_service_gap` | `age - length_of_service` | Difference between age and years of service. |

## Modeling and results

The project compared Logistic Regression, KNN, SVC, Naive Bayes, Decision Tree, Random Forest, ExtraTrees, Gradient Boosting, AdaBoost, HistGradientBoosting, XGBoost, LightGBM, and CatBoost. The selected CatBoost configuration was tuned with Optuna using **five-fold stratified cross-validation**. F1-score was used to select the model and probability cutoff because the promoted class is the minority class.

| Model | Training F1 | Validation / CV F1 | Held-out test F1 |
|---|---:|---:|---:|
| CatBoost Optuna (5-fold CV) | 0.5540 | **0.5233** | 0.5241 |
| CatBoost baseline | 0.5765 | 0.5114 | **0.5295** |
| XGBoost | 0.6417 | 0.4935 | 0.5069 |
| HistGradientBoosting | 0.5522 | 0.4922 | 0.5141 |
| Random Forest | 0.8844 | 0.4777 | 0.4454 |
| Decision Tree | 0.4860 | 0.4419 | 0.4534 |
| ExtraTrees | 0.8110 | 0.4328 | 0.4094 |
| Logistic Regression | 0.4253 | 0.4054 | 0.4283 |
| Gradient Boosting | 0.3772 | 0.3762 | 0.3662 |
| SVC | 0.3812 | 0.3621 | 0.3570 |
| KNN Classifier | 0.9993 | 0.3455 | 0.3208 |
| AdaBoost | 0.3085 | 0.3120 | 0.3042 |
| Naive Bayes | 0.2727 | 0.2715 | 0.2703 |

The Optuna model was selected using its cross-validation F1-score, **not** the held-out test score. The baseline CatBoost model scored higher on the held-out test set in this run; that test result was not used to tune the Optuna model. Scores are from this experiment and may differ if you rerun it or use other data.

The Streamlit app uses the saved CatBoost model refit on all labeled rows, with a default probability cutoff of **0.3028** selected from five-fold out-of-fold predictions. The separate `test_2umaH9m.csv` file has no labels, so its predictions cannot be scored with F1 until true outcomes are available.

## Run the Streamlit app

### Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- pip

### Windows / VS Code

1. Clone or download this repository and open its folder in VS Code.
2. Open a terminal in the repository folder and create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r streamlit_promotion_app/requirements.txt
   ```

   If PowerShell blocks activation in the current terminal, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

3. Start the app:

   ```powershell
   streamlit run streamlit_promotion_app/app.py
   ```

4. Streamlit opens the app in your browser. Enter one employee’s details manually or upload a CSV and download the resulting predictions.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r streamlit_promotion_app/requirements.txt
streamlit run streamlit_promotion_app/app.py
```

## Input columns for CSV predictions

The app expects the original employee fields below. `employee_id` is optional, and an `is_promoted` column is ignored if supplied. The app calculates its four engineered features automatically.

```text
department
region
education
gender
recruitment_channel
no_of_trainings
age
previous_year_rating
length_of_service
KPIs_met >80%
awards_won?
avg_training_score
```

The batch upload should contain the original columns, not the engineered columns. Missing categorical entries are filled with a missing-value label, and numeric values are converted before prediction.

## Notes

- The labeled holdout data used to report test F1 is different from the unlabeled `test_2umaH9m.csv` prediction file.
- The application returns a promotion probability and a class prediction based on the cutoff. Changing the cutoff changes the classification trade-off.
- Check for data drift, model performance changes, and fairness concerns before considering any operational use.
