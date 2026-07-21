import streamlit as st
import pandas as pd
import joblib
import os


st.write("Current directory:", os.getcwd())
st.write("Files in current directory:")
st.write(os.listdir())

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
# Prediction
# ------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("❌ Customer is likely to Churn.")
    else:
        st.success("✅ Customer is likely to Stay.")

    st.subheader("Prediction Probability")

    st.write(f"Stay Probability : **{probability[0][0]*100:.2f}%**")
    st.write(f"Churn Probability : **{probability[0][1]*100:.2f}%**")

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
