import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ------------------------------
# Title
# ------------------------------
st.title("📊 Customer Churn Prediction System")

st.write(
    """
    This application predicts whether a customer is likely to leave the bank
    based on customer information.
    """
)

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.header("Enter Customer Details")

CreditScore = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

Geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

Age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

Tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

Balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

NumOfProducts = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

HasCrCard = st.sidebar.selectbox(
    "Has Credit Card",
    [0,1]
)

IsActiveMember = st.sidebar.selectbox(
    "Active Member",
    [0,1]
)

EstimatedSalary = st.sidebar.number_input(
    "Estimated Salary",
    value=50000.0
)

# ------------------------------
# Input DataFrame
# ------------------------------

input_data = pd.DataFrame({

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

st.subheader("Customer Information")

st.dataframe(input_data)


# ------------------------------
# Footer
# ------------------------------

st.markdown("---")

st.markdown(
"""
### About Project

**Machine Learning Model:** Classification

**Algorithms Used**
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Naive Bayes
- SVM

Developed using **Python**, **Scikit-Learn**, **Pandas**, and **Streamlit**.
"""
)
