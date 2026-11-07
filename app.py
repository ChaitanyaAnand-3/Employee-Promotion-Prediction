from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "promotion_model.joblib"
CONFIG_PATH = APP_DIR / "model_config.json"

st.set_page_config(page_title="Employee Promotion Predictor", page_icon="📈", layout="wide")

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

@st.cache_resource
def load_model(model_path: str):
    """Load the fitted CatBoost model once per Streamlit session."""
    return joblib.load(model_path)


def prepare_features(input_frame: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Build the exact raw and engineered features used when training the model."""
    frame = input_frame.copy()
    frame = frame.drop(columns=["employee_id", "is_promoted"], errors="ignore")

    required_raw_columns = [
        "department", "region", "education", "gender", "recruitment_channel",
        "no_of_trainings", "age", "previous_year_rating", "length_of_service",
        "KPIs_met >80%", "awards_won?", "avg_training_score",
    ]
    missing = [column for column in required_raw_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    numeric_raw_columns = [
        "no_of_trainings", "age", "previous_year_rating", "length_of_service",
        "KPIs_met >80%", "awards_won?", "avg_training_score",
    ]
    for column in numeric_raw_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Calculate engineered columns from the original employee inputs so the
    # formulas are applied consistently for manual entries and CSV rows.
    frame["training_per_service"] = frame["no_of_trainings"] / (frame["length_of_service"] + 1)
    frame["score_x_kpi"] = frame["avg_training_score"] * frame["KPIs_met >80%"]
    frame["rating_x_kpi"] = frame["previous_year_rating"] * frame["KPIs_met >80%"]
    frame["age_service_gap"] = frame["age"] - frame["length_of_service"]

    for column in CONFIG["categorical_features"]:
        frame[column] = frame[column].fillna("__MISSING__").astype(str)

    # Match the model's learned feature names and order exactly.
    return frame.reindex(columns=expected_features)


def make_prediction(model, raw_features: pd.DataFrame, expected_features: list[str], cutoff: float):
    model_features = prepare_features(raw_features, expected_features)
    probability = float(model.predict_proba(model_features)[0, 1])
    predicted_class = int(probability >= cutoff)
    return probability, predicted_class, model_features


hero_col, intro_col = st.columns([1, 2])
with hero_col:
    st.image(
        str(APP_DIR / "hero.jpg"),
        use_container_width=True,
        caption="Photo by Alena Darmel via Pexels",
    )
with intro_col:
    st.title("Employee Promotion Predictor")
    st.write(
        "Enter one employee's details manually, or upload a CSV to score multiple employees. "
        "The app calculates the model's four engineered features automatically."
    )
st.info(
    f"Default cutoff: {CONFIG['threshold']:.4f}, selected using "
    f"{CONFIG['threshold_source']}. Predictions at or above the cutoff are classified as promoted."
)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model at {MODEL_PATH}: {exc}")
    st.stop()

model_features = list(getattr(model, "feature_names_", [])) or CONFIG["expected_features"]
threshold = st.slider(
    "Promotion probability cutoff",
    min_value=0.01,
    max_value=0.99,
    value=float(CONFIG["threshold"]),
    step=0.01,
    help="Change this if you want a stricter or more lenient promotion classification.",
)

manual_tab, csv_tab = st.tabs(["Enter one employee", "Upload CSV"])

with manual_tab:
    st.subheader("Employee details")
    st.caption("Enter the original employee fields. The app calculates the four engineered model features when you press Predict.")
    with st.form("employee_details_form"):
        employee_id = st.text_input("Employee ID (optional)")
        col1, col2, col3 = st.columns(3)
        with col1:
            department = st.selectbox("Department", ["Analytics", "Finance", "HR", "Legal", "Operations", "Procurement", "R&D", "Sales & Marketing", "Technology"])
            region = st.selectbox("Region", [f"region_{i}" for i in range(1, 35)])
            education = st.selectbox("Education", ["Below Secondary", "Bachelor's", "Master's & above"])
            gender = st.selectbox("Gender", ["f", "m"])
            recruitment_channel = st.selectbox("Recruitment channel", ["other", "referred", "sourcing"])
        with col2:
            no_of_trainings = st.number_input("Number of trainings", min_value=1, max_value=10, value=1, step=1)
            age = st.number_input("Age", min_value=20, max_value=60, value=30, step=1)
            previous_year_rating = st.selectbox("Previous year rating", [1, 2, 3, 4, 5])
            length_of_service = st.number_input("Length of service (years)", min_value=1, max_value=37, value=5, step=1)
        with col3:
            kpi_met = st.selectbox("Met more than 80% of KPIs?", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
            awards_won = st.selectbox("Won an award?", [0, 1], format_func=lambda value: "Yes" if value == 1 else "No")
            avg_training_score = st.number_input("Average training score", min_value=39, max_value=99, value=60, step=1)

        predict_clicked = st.form_submit_button("Predict promotion")

    if predict_clicked:
        manual_row = pd.DataFrame([{
            "employee_id": employee_id if employee_id.strip() else "Manual entry",
            "department": department,
            "region": region,
            "education": education,
            "gender": gender,
            "recruitment_channel": recruitment_channel,
            "no_of_trainings": no_of_trainings,
            "age": age,
            "previous_year_rating": previous_year_rating,
            "length_of_service": length_of_service,
            "KPIs_met >80%": kpi_met,
            "awards_won?": awards_won,
            "avg_training_score": avg_training_score,
        }])
        try:
            probability, predicted_class, model_input = make_prediction(model, manual_row, model_features, threshold)
            label = "Promoted" if predicted_class == 1 else "Not promoted"
            result_col, probability_col = st.columns(2)
            result_col.metric("Prediction", label)
            probability_col.metric("Promotion probability", f"{probability:.1%}")
            with st.expander("See all features passed to the model"):
                st.dataframe(model_input, use_container_width=True, hide_index=True)
            manual_output = pd.DataFrame([{
                "employee_id": manual_row.loc[0, "employee_id"],
                "promotion_probability": probability,
                "is_promoted": predicted_class,
                "prediction_label": label,
            }])
            st.download_button(
                "Download this prediction as CSV",
                data=manual_output.to_csv(index=False).encode("utf-8"),
                file_name="employee_promotion_prediction.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not make prediction: {exc}")

with csv_tab:
    st.subheader("Score multiple employees")
    st.caption("Upload a CSV with the same original employee fields. employee_id is optional; is_promoted is ignored if present.")
    uploaded_file = st.file_uploader("Choose an employee CSV file", type=["csv"], key="batch_csv")
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            if input_data.empty:
                st.warning("The uploaded CSV has a header but no employee rows.")
            else:
                features = prepare_features(input_data, model_features)
                probabilities = model.predict_proba(features)[:, 1]
                predictions = (probabilities >= threshold).astype("int8")
                output = pd.DataFrame({
                    "employee_id": input_data["employee_id"] if "employee_id" in input_data.columns else np.arange(1, len(input_data) + 1),
                    "promotion_probability": probabilities,
                    "is_promoted": predictions,
                    "prediction_label": np.where(predictions == 1, "Promoted", "Not promoted"),
                })
                first, second, third = st.columns(3)
                first.metric("Employees scored", f"{len(output):,}")
                second.metric("Predicted promoted", f"{int(predictions.sum()):,}")
                third.metric("Predicted not promoted", f"{len(output) - int(predictions.sum()):,}")
                st.dataframe(output.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download all predictions as CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="employee_promotion_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not score this CSV: {exc}")

st.caption("The app calculates training_per_service, score_x_kpi, rating_x_kpi, and age_service_gap from the original employee details. The supplied test file has no true labels, so F1 cannot be calculated for it.")
