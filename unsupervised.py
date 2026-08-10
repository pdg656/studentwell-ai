import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/student_wellness.csv"

data = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STUDENTWELL AI - UNSUPERVISED LEARNING")
print("=" * 60)

print("\nDataset shape:")
print(data.shape)

# ============================================================
# 2. SELECT FEATURES
# ============================================================

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
    "wellness_score"
]

X = data[FEATURES]

# ============================================================
# 3. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")

# ============================================================
# 4. PCA
# ============================================================

pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA completed.")

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)

# ============================================================
# 5. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

data["cluster"] = clusters

# ============================================================
# 6. ADD PCA VALUES
# ============================================================

data["pca_1"] = X_pca[:, 0]
data["pca_2"] = X_pca[:, 1]

# ============================================================
# 7. CLUSTER DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER DISTRIBUTION")
print("=" * 60)

print(
    data["cluster"].value_counts().sort_index()
)

# ============================================================
# 8. CLUSTER PROFILES
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER PROFILES")
print("=" * 60)

cluster_profiles = data.groupby("cluster")[FEATURES].mean()

print(
    cluster_profiles.round(2).to_string()
)

# ============================================================
# 9. RISK DISTRIBUTION BY CLUSTER
# ============================================================

print("\n" + "=" * 60)
print("RISK LEVEL BY CLUSTER")
print("=" * 60)

risk_by_cluster = pd.crosstab(
    data["cluster"],
    data["risk_level"],
    normalize="index"
) * 100

print(
    risk_by_cluster.round(2).to_string()
)

# ============================================================
# 10. SAVE RESULTS
# ============================================================

data.to_csv(
    "data/student_wellness_clusters.csv",
    index=False
)

cluster_profiles.to_csv(
    "data/cluster_profiles.csv"
)

# ============================================================
# 11. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("UNSUPERVISED LEARNING COMPLETE")
print("=" * 60)

print(
    "\nSaved:"
)

print(
    "data/student_wellness_clusters.csv"
)

print(
    "data/cluster_profiles.csv"
)
joblib.dump(scaler, "studentwell_scaler.pkl")
joblib.dump(kmeans, "studentwell_kmeans.pkl")

print("Saved:")
print("studentwell_scaler.pkl")
print("studentwell_kmeans.pkl")