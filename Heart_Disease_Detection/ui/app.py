import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ============ PAGE CONFIGURATION ============
st.set_page_config(page_title="Heart Disease Prediction")

# ============ SIDEBAR NAVIGATION ============
with st.sidebar:
    selected = option_menu(
        menu_title="Heart Disease Prediction System",
        options=["Data Visualization", "Heart Disease Prediction"],
        icons=["bar-chart", "heart-pulse"],
        menu_icon="hospital",
        default_index=0
    )

# ============ SIMULATED EXAMPLE DATA ============
df = pd.DataFrame({
    "Age": np.random.randint(30, 80, 100),
    "Cholesterol": np.random.randint(100, 400, 100),
    "Blood Pressure": np.random.randint(80, 180, 100),
    "Heart Rate": np.random.randint(60, 120, 100),
    "Diabetes": np.random.choice(["Yes", "No"], 100),
    "Risk Level": np.random.choice(["Low", "Moderate", "High"], 100)
})

df_numeric = df.copy()
df_numeric["Risk Level"] = df_numeric["Risk Level"].map({"Low": 0, "Moderate": 1, "High": 2})

# ============ HOME PAGE ============
if selected == "Data Visualization":
    st.title("📊 Data Visualization & Insights")

    # Show dataset preview
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head())

    # Correlation Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Cholesterol Distribution
    st.subheader("📈 Cholesterol Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Cholesterol"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.success("✅ Use this analysis to understand risk factors!")

# ============ HEART DISEASE PREDICTION ============
elif selected == "Heart Disease Prediction":
    st.title("🩺 Heart Disease Risk Prediction")
    st.markdown("### Enter Patient Details Below")

    # Improved Input Layout using st.columns()
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("📅 Age", 20, 80, 50)
        cholesterol = st.number_input("🩸 Cholesterol Level", 100, 400, 200)

    with col2:
        blood_pressure = st.number_input("❤️ Blood Pressure", 80, 200, 120)
        exercise = st.selectbox("🏃‍♂️ Do you exercise regularly?", ["Yes", "No"])

    smoking = st.selectbox("🚬 Do you smoke?", ["Yes", "No"])

    # Fake rule-based expert system
    def rule_based_prediction(age, cholesterol, blood_pressure, exercise, smoking):
        if cholesterol > 240 and age > 50:
            return "High Risk"
        elif blood_pressure > 140 and smoking == "Yes":
            return "High Risk"
        elif exercise == "Yes" and cholesterol < 180:
            return "Low Risk"
        else:
            return "Moderate Risk"

    # Fake ML-based prediction
    def ml_prediction():
        return np.random.choice(["Low Risk", "Moderate Risk", "High Risk"], p=[0.3, 0.5, 0.2])

    # Predict Button
    st.markdown("---")  # Add a separator for better visuals

    if st.button("🔍 Predict Risk"):
        rule_pred = rule_based_prediction(age, cholesterol, blood_pressure, exercise, smoking)
        ml_pred = ml_prediction()

        st.subheader("🩺 Prediction Results")
        st.success(f"📜 **Rule-Based Expert System:** {rule_pred}")
        st.info(f"🤖 **Machine Learning Model:** {ml_pred}")

        st.markdown("✅ This is a test UI! Integrate real models next. 🚀")
