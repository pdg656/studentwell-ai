
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/student_wellness.csv")

print("Shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nData types:")
print(data.dtypes)

print("\nSummary statistics:")
print(data.describe())

print("\nRisk distribution:")
print(data["risk_level"].value_counts())

# Correlation analysis
numeric_data = data.select_dtypes(include="number")

plt.figure(figsize=(10, 8))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Student Wellness Feature Correlations")
plt.tight_layout()
plt.show()