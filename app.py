import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
try:
    model = joblib.load("churn_pipeline.pkl")
except FileNotFoundError:
    st.error("❌ pipeline.pkl not found.")
    st.info("Place pipeline.pkl in the same folder as app.py")
    st.stop()

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Customer Churn Prediction")

st.markdown(
"""
Predict whether a customer is likely to **Stay** or **Leave** the bank using Machine Learning.
"""
)

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("Customer Details")

CreditScore = st.sidebar.number_input(
    "Credit Score",300,900,600
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
    "Age",18,100,35
)

Tenure = st.sidebar.slider(
    "Tenure",0,10,5
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

# -------------------------------
# Create DataFrame
# -------------------------------

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

# -------------------------------
# Display Input
# -------------------------------

st.subheader("Customer Details")

st.dataframe(customer,use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Churn"):

    prediction = model.predict(customer)

    probability = model.predict_proba(customer)

    if prediction[0]==1:

        st.error("Customer is likely to CHURN ❌")

    else:

        st.success("Customer is likely to STAY ✅")

    st.subheader("Prediction Probability")

    col1,col2 = st.columns(2)

    with col1:
        st.metric(
            "Stay Probability",
            f"{probability[0][0]*100:.2f}%"
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability[0][1]*100:.2f}%"
        )

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.markdown("""
### Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Naive Bayes
- Support Vector Machine

**Developed using**

- Python
- Scikit-Learn
- Pandas
- Streamlit
""")
