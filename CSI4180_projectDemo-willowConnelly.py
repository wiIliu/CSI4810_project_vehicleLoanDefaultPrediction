from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('xgb.pkl')
test = pd.read_csv("slectedFeaturesTest.csv")

def prediction(uniqueid):
    return int(model.predict(StandardScaler().fit_transform([test.iloc[uniqueid,1:]]))[0])

st.title("Vehicle Loan Default Prediction - Willow Connelly")

st.write("This application predicts if a customer is likely to default on their car loan.")
st.write("0 indicates not likely, 1 indicates likely to default.")

uniqueid = st.slider("Select a unique ID from the test set", min_value=0, max_value=len(test) - 1, value=0)

if st.button("Predict"):
    pred = prediction(uniqueid)
    st.write(f"The prediction for ID {uniqueid} is:  **{'Likely to Default' if pred == 1 else 'Not Likely to Default'}**")
