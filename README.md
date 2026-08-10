# 🎓 StudentWell AI

StudentWell AI is an educational machine-learning decision-support prototype that analyzes synthetic student wellness data to explore patterns associated with academic workload, sleep, exercise, nutrition, stress, anxiety, and overall wellness.

## 🚀 Live Demo

[**Try StudentWell AI →**](https://studentwell-ai.streamlit.app/)

An interactive Streamlit application for exploring student wellness risk prediction, wellness profiles, and machine-learning insights.

## 📌 Project Overview

StudentWell AI combines supervised machine learning, unsupervised learning, explainability, and an interactive Streamlit dashboard.

The application allows a user to complete a wellness check-in and receive:

- Estimated wellness risk level
- Risk probability estimates
- Wellness profile/cluster
- Key areas identified from the check-in
- General wellness recommendations
- Model-based feature insights

## 🧠 Machine Learning

### Supervised Learning

The project uses classification models to predict one of three wellness risk categories:

- Low
- Moderate
- High

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- High-risk recall

The final model uses the project's V2 feature set.

### Unsupervised Learning

The project also uses:

- StandardScaler
- PCA
- K-means clustering

The clustering pipeline identifies groups of students with similar wellness-related characteristics.

These clusters are interpreted based on their observed feature profiles rather than being assigned predetermined meanings.

## 📊 Dataset

The project uses **synthetic student wellness data**.

No real student health records, counseling records, or personally identifiable health information are used.

The synthetic dataset contains features including:

- Age
- Academic workload
- Sleep hours
- Exercise frequency
- Nutrition
- Study hours
- Social activity
- Stress
- Anxiety
- Overall wellness
- Risk level

## 🖥️ Application

The Streamlit application provides an interactive wellness check-in.

Users enter wellness-related information using sliders and receive a model-generated assessment.

The application is intended as an educational prototype and not as a medical or mental-health diagnostic system.

## 🔎 Explainability

The application presents key wellness areas associated with the user's responses and provides general recommendations.

Feature importance from the trained model is also used to understand which variables contribute most strongly to model predictions.

## 🔐 Privacy and Ethics

This project intentionally uses synthetic data.

The prototype does not connect to:

- University counseling records
- Medical records
- Student health systems
- Real CAPS data

Any future integration with real student-support systems would require appropriate authorization, authentication, privacy controls, security review, and institutional oversight.

## ⚠️ Limitations

This project is an educational machine-learning prototype.

Model predictions should not be interpreted as medical diagnoses, clinical assessments, or definitive measures of mental health.

The synthetic dataset may not represent real student populations.

The model would require extensive validation before any real-world deployment.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- K-means
- PCA
- Random Forest

## 📁 Project Structure

```text
studentwell-ai/
│
├── data/
├── tests/
│
├── streamlit_app.py
├── generate_data.py
├── explore_data.py
├── eda.py
├── unsupervised.py
│
├── studentwell_best_model.pkl
├── studentwell_model_v1.pkl
├── studentwell_model_v2.pkl
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE