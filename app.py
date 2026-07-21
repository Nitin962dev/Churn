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


# -------------------------------
# Load Model
# -------------------------------
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Customer Churn Prediction")
st.write("Predict whether a customer will leave the bank.")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Customer Details")

CreditScore = st.sidebar.number_input("Credit Score", 300, 900, 600)

Geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

Age = st.sidebar.number_input("Age", 18, 100, 35)

Tenure = st.sidebar.slider("Tenure", 0, 10, 5)

Balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)

NumOfProducts = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

HasCrCard = st.sidebar.selectbox(
    "Has Credit Card",
    [0,1]
)

IsActiveMember = st.sidebar.selectbox(
    "Is Active Member",
    [0,1]
)

EstimatedSalary = st.sidebar.number_input(
    "Estimated Salary",
    0.0,
    200000.0,
    50000.0
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    customer = pd.DataFrame({

        "CreditScore":[CreditScore],
        "Geography":[Geography],
        "Gender":[Gender],
        "Age":[Age],
        "Tenure":[Tenure],
        "Balance":[Balance],
        "NumOfProducts":[NumOfProducts],
        "HasCrCard":[HasCrCard],
        "IsActiveMember":[IsActiveMember],
        "EstimatedSalary":[EstimatedSalary]

    })

    prediction = model.predict(customer)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("❌ Customer is likely to Exit.")
    else:
        st.success("✅ Customer is likely to Stay.")
