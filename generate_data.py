import pandas as pd
import numpy as np

# --------------------------------------------------
# Generate synthetic student wellness data
# --------------------------------------------------

np.random.seed(42)

n_students = 5000

# Student identifiers
student_id = np.arange(1, n_students + 1)

# Demographic / academic variables
age = np.random.randint(18, 30, n_students)

academic_workload = np.random.randint(
    1, 11, n_students
)

# Lifestyle variables
sleep_hours = np.round(
    np.random.normal(6.8, 1.2, n_students),
    1
)

sleep_hours = np.clip(
    sleep_hours,
    3,
    10
)

exercise_days = np.random.randint(
    0,
    8,
    n_students
)

nutrition_score = np.random.randint(
    1,
    11,
    n_students
)

study_hours = np.round(
    np.random.normal(5, 2, n_students),
    1
)

study_hours = np.clip(
    study_hours,
    0,
    14
)

social_activity = np.random.randint(
    0,
    11,
    n_students
)

# --------------------------------------------------
# Wellness variables
# --------------------------------------------------

stress_score = (
    0.35 * academic_workload
    + 0.25 * (10 - sleep_hours)
    + 0.10 * (7 - exercise_days)
    + np.random.normal(
        0,
        1.5,
        n_students
    )
)

stress_score = np.clip(
    stress_score,
    0,
    10
)

stress_score = np.round(
    stress_score,
    1
)

anxiety_score = (
    0.45 * stress_score
    + 0.15 * academic_workload
    + np.random.normal(
        0,
        1.5,
        n_students
    )
)

anxiety_score = np.clip(
    anxiety_score,
    0,
    10
)

anxiety_score = np.round(
    anxiety_score,
    1
)

wellness_score = (
    0.25 * sleep_hours
    + 0.15 * exercise_days
    + 0.20 * nutrition_score
    + 0.20 * social_activity
    - 0.20 * stress_score
)

wellness_score = np.clip(
    wellness_score,
    0,
    10
)

wellness_score = np.round(
    wellness_score,
    1
)

# --------------------------------------------------
# Target variable
# --------------------------------------------------

risk_level = pd.cut(
    stress_score,
    bins=[-1, 3.5, 6.5, 10],
    labels=[
        "Low",
        "Moderate",
        "High"
    ]
)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

data = pd.DataFrame({
    "student_id": student_id,
    "age": age,
    "academic_workload": academic_workload,
    "sleep_hours": sleep_hours,
    "exercise_days": exercise_days,
    "nutrition_score": nutrition_score,
    "study_hours": study_hours,
    "social_activity": social_activity,
    "stress_score": stress_score,
    "anxiety_score": anxiety_score,
    "wellness_score": wellness_score,
    "risk_level": risk_level
})

# --------------------------------------------------
# Save dataset
# --------------------------------------------------

data.to_csv(
    "data/student_wellness.csv",
    index=False
)

print("Dataset created successfully!")

print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nRisk distribution:")
print(data["risk_level"].value_counts())