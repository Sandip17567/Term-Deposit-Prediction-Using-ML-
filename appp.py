import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load
model = pickle.load(open("best_rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
onehot = pickle.load(open("one_hot_encoder.pkl", "rb"))
lower_bound, upper_bound = pickle.load(open("bound.pkl", "rb"))
st.title("Term Deposit Subscription Prediction")

# Inputs
age = st.slider("Age", 10, 100)
job = st.selectbox("Job", ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"])
marital = st.selectbox("Marital", ["divorced","married","single","unknown"])
education = st.selectbox("Education", ["primary","secondary","tertiary","unknown"])
default = st.selectbox("Default", ["no","yes","unknown"])
balance = st.number_input("Balance")
housing = st.selectbox("Housing", ["no","yes"])
loan = st.selectbox("Loan", ["no","yes"])
contact = st.selectbox("Contact", ["cellular","telephone","unknown"])
day=st.slider("Day", 1, 31)
month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
duration = st.number_input("Duration", min_value=0)
campaign = st.number_input("Campaign", min_value=1)
pdays = st.number_input("Pdays", min_value=-1)
previous = st.number_input("Previous", min_value=0)
poutcome = st.selectbox("Poutcome", ["failure","success","other","unknown"])

# Predict
if st.button("Predict"):

    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    input_df = pd.DataFrame([input_data])
    
    # -----------------------------
    # -----------------------------
    day_encoded = onehot.transform(input_df[['day']]).toarray()
    day_cols = onehot.get_feature_names_out(['day'])

    day_df = pd.DataFrame(day_encoded, columns=day_cols)

    input_df = input_df.drop('day', axis=1)

    input_df = pd.concat([input_df, day_df], axis=1)

    input_df['pdays'] = np.where(input_df['pdays'] >= 0, 1, 0)
    input_df['balance'] = np.log(input_df['balance'] + abs(input_df['balance'].min()) + 1)


    input_encoded = pd.get_dummies(input_df, drop_first=True)


    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    num_cols = input_encoded[['age','balance','duration','campaign','previous']]
    categori=[col for col in input_encoded.columns if col not in num_cols.columns]
    categorical_cols=input_encoded[categori]
    input= scaler.transform(num_cols)
    data=pd.concat([pd.DataFrame(input, columns=num_cols.columns,index=input_df.index),categorical_cols], axis=1)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Customer WILL subscribe")
    else:
        st.error("Customer will NOT subscribe")