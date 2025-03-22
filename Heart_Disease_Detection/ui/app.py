import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import joblib
import math
import sys
import os

expert_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rule_based_system"))

sys.path.append(expert_system_path)

from rules import HeartDiseaseExpert

st.set_page_config(page_title="Heart Disease Prediction")

with st.sidebar:
    selected = option_menu(
        menu_title="Heart Disease Prediction System",
        options=["Data Visualization", "Heart Disease Prediction"],
        icons=["bar-chart", "heart-pulse"],
        menu_icon="hospital",
        default_index=0
    )
    
data_path = os.path.join(os.path.dirname(__file__), "../data/raw_data.csv")
data = pd.read_csv(data_path)

#   Data Visualization Page
if selected == "Data Visualization":

    st.title("📊 Data Visualization & Insights")

    st.markdown("Statistical summaries and visualizations.")

# 🔹 Section 1: Statistical Summary
    st.divider()
    st.header("Statistical Summary")
    st.dataframe(data.describe())

# 🔹 Section 2: Feature Correlation
    st.divider()
    st.header("Feature Correlation Heatmap")
    correlation = data.corr()['target'].drop('target').abs().sort_values(ascending=False)
    corr_df = correlation.to_frame().reset_index()
    corr_df.columns = ['Feature', 'Correlation with Target']

    fig, ax = plt.subplots(figsize=(5, len(corr_df) * 0.3))
    sns.heatmap(corr_df.set_index("Feature"), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    selected_features = correlation[correlation > 0.15]
    st.markdown("### Selected Features (Correlation > 0.15):")
    st.success(", ".join(selected_features.index))

# 🔹 Section 3: Boxplots for Numerical Features
    st.divider()
    st.header("Boxplots for Numerical Features")

    numerical_features = ["age", "thalach", "oldpeak", "ca"]
    cols = 2
    rows = math.ceil(len(numerical_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 3))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        sns.boxplot(y=data[feature], ax=axes[i])
        axes[i].set_title(f"Boxplot of {feature}")

    plt.tight_layout()
    st.pyplot(fig)

# 🔹 Section 4: Histograms for Categorical Features
    st.divider()
    st.header("Histograms for Categorical Features")

    categorical_features = ["sex", "cp", "exang", "slope", "thal"]
    cols = 3
    rows = math.ceil(len(categorical_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, rows * 3))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        sns.countplot(x=data[feature], edgecolor="black", ax=axes[i])
        axes[i].set_title(f"Histogram of {feature}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

#   Heart Disease Prediction Page
elif selected == "Heart Disease Prediction":
    
    st.title("🩺 Heart Disease Risk Prediction")
    st.markdown("### Enter Your Health Details for a Risk Check")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("📅 Age", 20, 80, 50)
        oldpeak = st.number_input("📉 ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
        exang = st.selectbox("Exercise Induced Angina (Exang)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved (Thalach)", 70, 220, 150)
        ca = st.slider("Number of Major Vessels (CA)", 0, 4, 0)
        slope = st.selectbox("📈 Slope of ST Segment", [0, 1, 2])
        thal = st.selectbox("🧬 Thalassemia Type", [0, 1, 2, 3])
 
    col1, col2, col3 = st.columns([1, 3, 1])  

    with col2:
        predict = st.button("🔍 **Predict Risk**", use_container_width=True)

    st.divider()

    cp_encoded = [1 if cp == i else 0 for i in range(4)]
    thal_encoded = [1 if thal == i else 0 for i in range(4)]
    slope_encoded = [1 if slope == i else 0 for i in range(3)]

    feature_names = [
        "age", "sex", "thalach", "exang", "oldpeak", "ca",
        "cp_0", "cp_1", "cp_2", "cp_3",
        "thal_0", "thal_1", "thal_2", "thal_3",
        "slope_0", "slope_1", "slope_2"
    ]

    if predict:

        input_data = np.array([[age, sex, thalach, exang, oldpeak, ca] + cp_encoded + thal_encoded + slope_encoded])
        input_df = pd.DataFrame(input_data, columns=feature_names)
        model_path = os.path.join(os.path.dirname(__file__), "../data/model.pkl")

        st.subheader("🩺 Prediction Results")
        col1, col2 = st.columns(2)  

        model = joblib.load(model_path)
        ml_pred = model.predict(input_df)[0]
        risk_levels = {0: "✅ Low Risk", 1: "⚠️ High Risk"}

        expert_system = HeartDiseaseExpert()
        rule_pred = expert_system.predict(input_df.iloc[0].to_dict())
    
        with col1:
            st.success("📑 **Rule-Based Expert System**")
            st.metric(label="Risk Level", value=risk_levels.get(rule_pred))

        with col2:
            st.info("🤖 **Machine Learning Model**")
            st.metric(label="Risk Level", value=risk_levels.get(ml_pred))
        
    
