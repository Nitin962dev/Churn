import streamlit as st
import pandas as pd
import numpy as np

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


