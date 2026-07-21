import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------
# Load Models
# ----------------------------

models = {
    "Logistic Regression": joblib.load("churn_pipeline.pkl"),
    "Decision Tree": joblib.load("churn_pipeline_DT.pkl"),
    "KNN": joblib.load("churn_pipeline_KNN.pkl"),
    "Naive Bayes": joblib.load("churn_pipeline_NB.pkl"),
    "Random Forest": joblib.load("churn_pipeline_RF.pkl"),
    "SVM": joblib.load("churn_pipeline_SVC.pkl")
}

# ----------------------------
# Title
# ----------------------------

st.title("🏦 Customer Churn Prediction")

st.write(
"""
Predict whether a customer will leave the bank using Machine Learning.
"""
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("Customer Details")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

CreditScore = st.sidebar.number_input(
    "Credit Score",
    300,
    900,
    600
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

# ----------------------------
# Customer Data
# ----------------------------

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

st.subheader("Customer Information")

st.dataframe(customer, use_container_width=True)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    model = models[selected_model]

    prediction = model.predict(customer)

    st.subheader("Prediction")

    st.info(f"Selected Model : {selected_model}")

    if prediction[0] == 1:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(customer)

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

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown("""
### Models Included

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- Support Vector Machine

Developed using **Python**, **Scikit-Learn**, **Pandas**, and **Streamlit**.
""")
