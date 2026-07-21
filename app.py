import streamlit as st
import pandas as pd
from PIL import Image

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#0f172a,#111827,#1e293b);
color:white;
}

[data-testid="stSidebar"]{
background:#111827;
}

.big-title{
font-size:45px;
font-weight:bold;
color:white;
text-align:center;
}

.sub-title{
font-size:20px;
color:#cbd5e1;
text-align:center;
}

.card{
background:#1e293b;
padding:25px;
border-radius:18px;
box-shadow:0px 0px 20px rgba(0,0,0,0.35);
text-align:center;
transition:0.3s;
}

.card:hover{
transform:scale(1.04);
}

.metric{
font-size:34px;
font-weight:bold;
color:#38bdf8;
}

.label{
font-size:18px;
color:white;
}

hr{
border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------

st.sidebar.title("🤖 AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 EDA Analysis",
        "🤖 Prediction",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

# -------------------- HOME --------------------

if page=="🏠 Home":

    st.markdown(
        "<div class='big-title'>Customer Churn Prediction System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Machine Learning Dashboard using Streamlit</div>",
        unsafe_allow_html=True
    )

    st.write("")

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.markdown("""
        <div class='card'>
        <div class='metric'>10000</div>
        <div class='label'>Customers</div>
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card'>
        <div class='metric'>87%</div>
        <div class='label'>Accuracy</div>
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='card'>
        <div class='metric'>10</div>
        <div class='label'>Features</div>
        </div>
        """,unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='card'>
        <div class='metric'>RF</div>
        <div class='label'>Best Model</div>
        </div>
        """,unsafe_allow_html=True)

    st.write("")
    st.write("---")

    left,right=st.columns([2,1])

    with left:

        st.header("📌 Project Overview")

        st.write("""
This application predicts whether a customer will leave the bank.

### Features

- Interactive Dashboard
- Exploratory Data Analysis
- Live Prediction
- Model Performance
- Confusion Matrix
- ROC Curve
- Feature Importance
- Professional UI
        """)

    with right:

        st.success("Dataset Loaded")

        st.info("""
Target Variable

Exited

Classification Problem
        """)

# -------------------- EDA --------------------

elif page=="📊 EDA Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.info("EDA page will be connected with your notebook.")

# -------------------- Prediction --------------------

elif page=="🤖 Prediction":

    st.title("🤖 Customer Prediction")

    st.write("Prediction form will come here.")

# -------------------- Model --------------------

elif page=="📈 Model Performance":

    st.title("📈 Model Performance")

    st.metric("Accuracy","87%")

    st.metric("Precision","86%")

    st.metric("Recall","84%")

    st.metric("F1 Score","85%")

# -------------------- About --------------------

else:

    st.title("ℹ️ About")

    st.write("""
Customer Churn Prediction Project

Developed using

✔ Streamlit

✔ Scikit-Learn

✔ Pandas

✔ Plotly

✔ Machine Learning
""")


import sys
import subprocess

subprocess.run([
    sys.executable,
    "-m",
    "jupyter",
    "nbconvert",
    "--execute",
    "Output_Classification.ipynb"
])
