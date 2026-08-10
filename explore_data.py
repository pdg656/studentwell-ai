import pandas as pd

data = pd.read_csv(
    "data/student_wellness.csv"
)

print(data.head())

print("\nDataset information:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())

print("\nRisk distribution:")
print(data["risk_level"].value_counts())