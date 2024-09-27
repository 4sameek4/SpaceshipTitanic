import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("logistic_model.pkl")  
catboost_model = joblib.load("catboost_model.pkl")

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)  
    st.write("File uploaded successfully!")
else:
    df = pd.read_csv("train.csv") 

st.title("Spaceship Titanic - Predict Transported")

passenger_id = st.text_input("Enter PassengerId", "")

results = pd.DataFrame()

if st.button("Predict"):
    if passenger_id:
        
        passenger_data = df[df['PassengerId'] == passenger_id]

        if not passenger_data.empty:
    
            features = passenger_data.drop(columns=['Transported', 'PassengerId'])

            features_transformed = pipeline.named_steps['preprocessor'].transform(features)

            logistic_pred = pipeline.named_steps['classifier'].predict(features_transformed)

            catboost_pred = catboost_model.predict(features)

            final_prediction = (catboost_pred + logistic_pred) / 2
            final_prediction = 1 if final_prediction >= 0.5 else 0

            transported = "Yes" if final_prediction == 1 else "No"
            st.write(f"Prediction: Was the passenger transported? **{transported}**")

            results = passenger_data.copy()
            results['Prediction'] = transported
        else:
            st.write("PassengerId not found.")

if not results.empty:
    csv = results.to_csv(index=False)
    st.download_button(label="Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
