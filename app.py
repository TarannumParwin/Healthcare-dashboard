import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    /* Dashboard Title */
    .main-title {
        background-color:#2E86C1;
        padding:14px;
        border-radius:12px;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;
        margin-bottom:20px;
    }

    /* Card (metric) styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    div[data-testid="stMetric"] > label {
        color: #2E86C1;
        font-size: 18px;
        font-weight: bold;
    }
    div[data-testid="stMetric"] > div {
        color: #117A65;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="main-title">🏥 Healthcare Analytics & Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # Ensure file is in same folder
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ------------------ MODEL TRAINING ------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("🤖 Model Performance")
st.write(f"Logistic Regression Model Accuracy: **{acc*100:.2f}%**")

# ------------------ SIDEBAR USER INPUT ------------------
st.sidebar.header("🧑‍⚕️ Enter Patient Details")

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ------------------ PREDICTION ------------------
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("🔮 Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Diabetes ({prediction_proba[0][1]*100:.2f}%)")
else:
    st.success(f"✅ Low Risk of Diabetes ({prediction_proba[0][0]*100:.2f}%)")

# ------------------ DASHBOARD METRICS ------------------
st.subheader("📌 Key Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(df))
col2.metric("Diabetic Patients", df['Outcome'].sum())
col3.metric("Non-Diabetic Patients", len(df) - df['Outcome'].sum())

# ------------------ EDA ------------------
st.subheader("🔍 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Glucose Distribution")
    fig1 = px.histogram(df, x="Glucose", color="Outcome", nbins=30,
                        title="Glucose Distribution by Outcome",
                        color_discrete_sequence=["#2E86C1", "#E74C3C"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### Age vs BMI")
    fig2 = px.scatter(df, x="Age", y="BMI", color="Outcome",
                      title="Age vs BMI (Colored by Outcome)",
                      color_discrete_sequence=["#2E86C1", "#E74C3C"])
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ EXTRA VISUALS ------------------
st.subheader("📈 Additional Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Diabetes Distribution")
    fig3 = px.pie(df, names="Outcome", title="Diabetes vs Non-Diabetes",
                  color_discrete_sequence=["#2E86C1", "#E74C3C"])
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("### BMI vs Outcome (Boxplot)")
    fig4 = px.box(df, x="Outcome", y="BMI", color="Outcome",
                  title="BMI by Outcome",
                  color_discrete_sequence=["#2E86C1", "#E74C3C"])
    st.plotly_chart(fig4, use_container_width=True)

# ------------------ HEATMAP ------------------
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
st.pyplot(fig)
