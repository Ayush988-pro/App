import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Bank Deposit Predictor", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("💰 Bank Deposit Prediction")

st.markdown("Fill customer details to predict subscription")

# ---------------- INPUTS ---------------- #

age = st.slider("Age", 18, 90, 30)

job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed",
    "services", "student", "technician", "unemployed"
])

marital = st.selectbox("Marital Status", ["single", "married", "divorced"])

education = st.selectbox("Education", ["primary", "secondary", "tertiary"])

default = st.selectbox("Has Credit Default?", ["no", "yes"])

balance = st.number_input("Account Balance", value=1000)

housing = st.selectbox("Housing Loan", ["no", "yes"])

loan = st.selectbox("Personal Loan", ["no", "yes"])

contact = st.selectbox("Contact Type", ["cellular", "telephone"])

month = st.selectbox("Last Contact Month", [
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
])

campaign = st.slider("Number of Contacts", 1, 20, 1)

pdays = st.number_input("Days Since Last Contact (-1 if none)", value=-1)

previous = st.number_input("Previous Contacts", value=0)

poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "other", "success"])

# ---------------- DATA PREP ---------------- #

input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# One-hot encoding
input_data = pd.get_dummies(input_data)

# Align with training columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Result:")

    if pred == 1:
        st.success("✅ Customer will SUBSCRIBE")
    else:
        st.error("❌ Customer will NOT SUBSCRIBE")

    st.write(f"📊 Probability of YES: **{prob:.2%}**")