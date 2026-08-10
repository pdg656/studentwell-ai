import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "studentwell_best_model.pkl"


FEATURES = [
    "age",
    "academic_workload",
    "sleep_hours",
    "exercise_days",
    "nutrition_score",
    "study_hours",
    "social_activity",
    "stress_score",
    "anxiety_score",
    "wellness_score",
]


def load_model():
    loaded_model = joblib.load(MODEL_PATH)

    if isinstance(loaded_model, dict):
        return loaded_model["model"]

    return loaded_model


def create_test_input():
    return pd.DataFrame({
        "age": [21],
        "academic_workload": [5],
        "sleep_hours": [7.0],
        "exercise_days": [3],
        "nutrition_score": [5],
        "study_hours": [4.0],
        "social_activity": [5],
        "stress_score": [5.0],
        "anxiety_score": [3.0],
        "wellness_score": [4.0],
    })


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_loads():
    model = load_model()
    assert model is not None


def test_expected_features_exist():
    data = create_test_input()

    assert list(data.columns) == FEATURES


def test_prediction_works():
    model = load_model()
    data = create_test_input()

    prediction = model.predict(data)

    assert len(prediction) == 1


def test_prediction_is_valid_risk_level():
    model = load_model()
    data = create_test_input()

    prediction = model.predict(data)[0]

    assert prediction in ["Low", "Moderate", "High"]


def test_extreme_input_works():
    model = load_model()

    extreme_data = pd.DataFrame({
        "age": [18],
        "academic_workload": [10],
        "sleep_hours": [0.0],
        "exercise_days": [0],
        "nutrition_score": [0],
        "study_hours": [12.0],
        "social_activity": [0],
        "stress_score": [10.0],
        "anxiety_score": [10.0],
        "wellness_score": [0.0],
    })

    prediction = model.predict(extreme_data)

    assert len(prediction) == 1