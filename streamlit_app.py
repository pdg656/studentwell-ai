
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StudentWell AI",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "studentwell_best_model.pkl"
SCALER_PATH = BASE_DIR / "studentwell_scaler.pkl"
KMEANS_PATH = BASE_DIR / "studentwell_kmeans.pkl"


# ============================================================
# LOAD SUPERVISED MODEL
# ============================================================

try:
    loaded_model = joblib.load(MODEL_PATH)

    # The saved file may contain a dictionary.
    # If so, extract the actual trained model.
    if isinstance(loaded_model, dict):
        if "model" in loaded_model:
            model = loaded_model["model"]
        elif "best_model" in loaded_model:
            model = loaded_model["best_model"]
        else:
            st.error(
                "The saved model dictionary does not contain "
                "a recognized model key."
            )
            model = None
    else:
        model = loaded_model

    model_loaded = model is not None

except Exception as e:
    model = None
    model_loaded = False

    st.error(
        f"Could not load the trained model: {e}"
    )


# ============================================================
# LOAD UNSUPERVISED LEARNING MODELS
# ============================================================

try:
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    clustering_loaded = True

except Exception as e:
    scaler = None
    kmeans = None
    clustering_loaded = False

    st.warning(
        "The clustering models could not be loaded. "
        "Risk prediction will still work, but the wellness "
        f"profile cannot be calculated yet. Details: {e}"
    )


# ============================================================
# TITLE / INTRODUCTION
# ============================================================

st.title("🎓 StudentWell AI")

st.write(
    "StudentWell AI is an ML-powered student wellness "
    "decision-support prototype."
)

st.info(
    "Complete the check-in below to receive an estimated "
    "wellness risk level, a wellness profile, model "
    "probabilities, and general wellness recommendations."
)


# ============================================================
# ABOUT YOU
# ============================================================

st.header("Student Wellness Check-In")

st.subheader("👤 About You")

age = st.slider(
    "What is your age?",
    min_value=18,
    max_value=29,
    value=21
)


# ============================================================
# WELLNESS QUESTIONS
# ============================================================

st.subheader("🧠 Your Wellness")

stress_score = st.slider(
    "How stressed do you feel?",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

sleep_hours = st.slider(
    "How many hours do you usually sleep?",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

academic_workload = st.slider(
    "How would you rate your academic workload?",
    min_value=0,
    max_value=10,
    value=5
)

exercise_days = st.slider(
    "How many days per week do you exercise?",
    min_value=0,
    max_value=7,
    value=3
)

nutrition_score = st.slider(
    "How would you rate your overall nutrition?",
    min_value=0,
    max_value=10,
    value=5
)


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

st.subheader("📚 Additional Information")

study_hours = st.slider(
    "How many hours do you study per day?",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

social_activity = st.slider(
    "How would you rate your social activity?",
    min_value=0,
    max_value=10,
    value=5
)

anxiety_score = st.slider(
    "How would you rate your anxiety?",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1
)

wellness_score = st.slider(
    "How would you rate your overall wellness?",
    min_value=0.0,
    max_value=7.0,
    value=4.0,
    step=0.1
)


# ============================================================
# CHECK-IN SUMMARY
# ============================================================

with st.expander("📋 Review Your Check-In"):

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Stress score:** {stress_score:.1f}")
        st.write(f"**Sleep hours:** {sleep_hours:.1f}")
        st.write(
            f"**Academic workload:** {academic_workload}/10"
        )
        st.write(
            f"**Exercise days:** {exercise_days}/7"
        )

    with summary_col2:
        st.write(
            f"**Nutrition score:** {nutrition_score}/10"
        )
        st.write(
            f"**Study hours:** {study_hours:.1f}"
        )
        st.write(
            f"**Social activity:** {social_activity}/10"
        )
        st.write(
            f"**Anxiety score:** {anxiety_score:.1f}"
        )
        st.write(
            f"**Wellness score:** {wellness_score:.1f}/7"
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze My Wellness",
    type="primary",
    width="stretch"
)


if analyze:

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not model_loaded:

        st.error(
            "The trained model could not be loaded. "
            "Make sure studentwell_best_model.pkl exists."
        )

        st.stop()


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({
        "age": [age],
        "academic_workload": [academic_workload],
        "sleep_hours": [sleep_hours],
        "exercise_days": [exercise_days],
        "nutrition_score": [nutrition_score],
        "study_hours": [study_hours],
        "social_activity": [social_activity],
        "stress_score": [stress_score],
        "anxiety_score": [anxiety_score],
        "wellness_score": [wellness_score]
    })


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction = model.predict(input_data)[0]

        # ----------------------------------------------------
        # CLUSTER PREDICTION
        # ----------------------------------------------------

        student_cluster = None

        if clustering_loaded:

            cluster_features = input_data[
                [
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
            ]

            scaled_features = scaler.transform(
                cluster_features
            )

            student_cluster = kmeans.predict(
                scaled_features
            )[0]


        # ====================================================
        # RESULTS HEADER
        # ====================================================

        st.divider()

        st.header("🎯 Your Wellness Assessment")


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if prediction == "High":

            st.error(
                "⚠️ Your estimated wellness risk level is HIGH."
            )

        elif prediction == "Moderate":

            st.warning(
                "⚠️ Your estimated wellness risk level is MODERATE."
            )

        else:

            st.success(
                "✅ Your estimated wellness risk level is LOW."
            )


        st.metric(
            "Predicted Risk Level",
            str(prediction)
        )


        st.caption(
            "This is a machine-learning estimate based on "
            "the information entered above. It is not a "
            "clinical or medical assessment."
        )


        # ====================================================
        # WELLNESS PROFILE
        # ====================================================

        st.subheader("📊 Your Wellness Profile")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Sleep",
                f"{sleep_hours:.1f} hrs"
            )

        with col2:

            st.metric(
                "Workload",
                f"{academic_workload}/10"
            )

        with col3:

            st.metric(
                "Exercise",
                f"{exercise_days} days"
            )

        with col4:

            st.metric(
                "Nutrition",
                f"{nutrition_score}/10"
            )


        # ====================================================
        # WELLNESS CLUSTER
        # ====================================================

        if student_cluster is not None:

            st.subheader("🧩 Your Wellness Profile Type")

            cluster_names = {
                0: "Lower Stress / Moderate Wellness",
                1: "Higher Workload / Elevated Stress",
                2: "Healthier Lifestyle / Higher Wellness"
            }

            cluster_descriptions = {
                0: (
                    "This cluster generally showed lower stress "
                    "and moderate wellness characteristics."
                ),
                1: (
                    "This cluster generally showed higher academic "
                    "workload and elevated stress characteristics."
                ),
                2: (
                    "This cluster generally showed stronger lifestyle "
                    "and higher wellness characteristics."
                )
            }

            cluster_name = cluster_names.get(
                int(student_cluster),
                f"Wellness Profile {student_cluster}"
            )

            cluster_description = cluster_descriptions.get(
                int(student_cluster),
                "This profile represents a pattern identified "
                "by the unsupervised learning model."
            )

            st.info(
                f"**{cluster_name}**\n\n"
                f"{cluster_description}"
            )

            st.caption(
                "K-means clustering identifies patterns in the "
                "synthetic dataset. The cluster names above are "
                "human interpretations of those patterns and "
                "are not diagnoses or fixed student categories."
            )


        # ====================================================
        # RISK PROBABILITIES
        # ====================================================

        st.subheader("📈 Risk Probabilities")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_

            probability_data = pd.DataFrame({
                "Risk Level": classes,
                "Probability": probabilities
            })

            probability_data["Percentage"] = (
                probability_data["Probability"] * 100
            )

            display_data = probability_data[
                ["Risk Level", "Percentage"]
            ].copy()

            display_data["Percentage"] = display_data[
                "Percentage"
            ].map(
                lambda x: f"{x:.1f}%"
            )

            st.dataframe(
                display_data,
                hide_index=True,
                width="stretch"
            )

            st.bar_chart(
                probability_data.set_index(
                    "Risk Level"
                )["Probability"]
            )

            st.caption(
                "These values represent the model's estimated "
                "class probabilities for this synthetic-data "
                "prototype. They should not be interpreted as "
                "clinical probabilities."
            )

        else:

            st.info(
                "Probability estimates are not available "
                "for this model."
            )


        # ====================================================
        # KEY AREAS FROM CHECK-IN
        # ====================================================

        st.subheader("🔎 Key Areas From Your Check-In")

        key_areas = []

        if stress_score >= 7:

            key_areas.append(
                "Your reported stress level is relatively high."
            )

        if academic_workload >= 8:

            key_areas.append(
                "Your reported academic workload is relatively high."
            )

        if anxiety_score >= 7:

            key_areas.append(
                "Your reported anxiety score is relatively high."
            )

        if sleep_hours < 6:

            key_areas.append(
                "Your reported sleep duration is relatively low."
            )

        if exercise_days < 2:

            key_areas.append(
                "Your reported exercise frequency is relatively low."
            )

        if nutrition_score < 5:

            key_areas.append(
                "Your nutrition rating may be an area worth "
                "paying attention to."
            )

        if social_activity < 3:

            key_areas.append(
                "Your reported social activity is relatively low."
            )

        if study_hours >= 8:

            key_areas.append(
                "Your reported daily study time is relatively high."
            )

        if wellness_score < 3:

            key_areas.append(
                "Your reported overall wellness score is relatively low."
            )

        if not key_areas:

            st.success(
                "No particularly elevated areas were identified "
                "using the check-in thresholds."
            )

        else:

            for area in key_areas:

                st.write(
                    f"• {area}"
                )


        # ====================================================
        # MODEL EXPLAINABILITY
        # ====================================================

        st.subheader("🧠 Model Feature Importance")

        if hasattr(model, "feature_importances_"):

            feature_names = [
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

            importance_data = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            importance_data = importance_data.sort_values(
                "Importance",
                ascending=False
            )

            importance_data["Importance (%)"] = (
                importance_data["Importance"] * 100
            )

            st.bar_chart(
                importance_data.set_index(
                    "Feature"
                )["Importance (%)"]
            )

            st.dataframe(
                importance_data[
                    ["Feature", "Importance (%)"]
                ].style.format(
                    {"Importance (%)": "{:.2f}%"}
                ),
                hide_index=True,
                width="stretch"
            )

            st.caption(
                "Feature importance describes how much each "
                "feature contributed to the trained model's "
                "decision process. It does not establish that "
                "a feature causes a particular wellness outcome."
            )

        else:

            st.info(
                "Feature importance is not available for this model."
            )


        # ====================================================
        # PERSONALIZED GENERAL WELLNESS RECOMMENDATIONS
        # ====================================================

        st.subheader("💡 General Wellness Suggestions")

        recommendations = []

        if sleep_hours < 6:

            recommendations.append(
                "Your responses suggest that sleep may be an "
                "area worth paying attention to. Consider "
                "prioritizing consistent sleep and recovery."
            )

        elif sleep_hours >= 7:

            recommendations.append(
                "Your reported sleep duration is a positive "
                "part of your current routine."
            )

        if stress_score >= 8:

            recommendations.append(
                "Your reported stress level is high. Consider "
                "using stress-management strategies, taking "
                "regular breaks, and exploring appropriate "
                "campus support resources."
            )

        elif stress_score >= 5:

            recommendations.append(
                "Your reported stress level is moderate. "
                "Consider scheduling regular breaks and "
                "stress-management activities."
            )

        if academic_workload >= 8:

            recommendations.append(
                "Your reported academic workload is high. "
                "Consider breaking larger tasks into smaller "
                "steps and planning your week."
            )

        if exercise_days < 2:

            recommendations.append(
                "Your reported exercise frequency is low. "
                "If appropriate for you, consider incorporating "
                "regular physical activity into your routine."
            )

        if nutrition_score < 5:

            recommendations.append(
                "Your nutrition rating may be an area worth "
                "paying attention to. Consider maintaining "
                "regular, balanced meals."
            )

        if anxiety_score >= 7:

            recommendations.append(
                "Your reported anxiety score is elevated. "
                "Consider connecting with an appropriate "
                "qualified professional or student-support service "
                "if you would find that helpful."
            )

        if social_activity < 3:

            recommendations.append(
                "You reported relatively low social activity. "
                "Consider maintaining supportive social "
                "connections and activities that help you "
                "feel connected."
            )

        if study_hours >= 8:

            recommendations.append(
                "Your reported study time is relatively high. "
                "Consider building regular breaks and recovery "
                "time into your study schedule."
            )

        if not recommendations:

            recommendations.append(
                "Keep maintaining your current wellness habits "
                "and continue checking in with yourself."
            )

        for recommendation in recommendations:

            st.write(
                f"• {recommendation}"
            )


        # ====================================================
        # RESULTS SUMMARY
        # ====================================================

        st.subheader("📝 Results Summary")

        summary_text = (
            f"Your model-estimated risk level is **{prediction}**. "
            f"Your check-in included a stress score of "
            f"{stress_score:.1f}/10, approximately "
            f"{sleep_hours:.1f} hours of sleep, an academic "
            f"workload rating of {academic_workload}/10, and "
            f"{exercise_days} exercise days per week."
        )

        st.write(summary_text)

        if student_cluster is not None:

            st.write(
                f"The unsupervised learning model placed your "
                f"responses into **{cluster_name}**."
            )


        # ====================================================
        # SAFETY / RESPONSIBLE AI NOTICE
        # ====================================================

        st.divider()

        st.subheader("⚠️ Important Information")

        st.warning(
            "StudentWell AI is an educational machine-learning "
            "prototype and is NOT a medical or mental-health "
            "diagnostic tool."
        )

        st.caption(
            "Results are estimates based on user-provided "
            "information and should not replace professional "
            "evaluation, medical advice, or emergency support."
        )

        st.caption(
            "This prototype uses synthetic student data and "
            "does not collect, access, or connect to real "
            "student health or counseling records."
        )

        st.caption(
            "The clustering profiles are patterns identified "
            "from synthetic data and should not be interpreted "
            "as clinical or psychological categories."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.write(
            "The model expects these features:"
        )

        st.code(
            [
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
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "StudentWell AI • Educational ML Prototype • "
    "Synthetic Data"
)

