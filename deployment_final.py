import streamlit as st
import pandas as pd
import joblib

# Load the models
logistic_model = joblib.load('logistic_model.pkl')
catboost_model = joblib.load('catboost_model.pkl')

# Load the original training data to access PassengerId and categorical values
original_data = pd.read_csv('train.csv')

# Streamlit app layout
st.title('Titanic Model Prediction')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Select model
model_option = st.selectbox('Choose the model to use', ('Logistic Regression', 'CatBoost'))

# Input PassengerId or upload file
passenger_id = st.text_input("Enter PassengerId for prediction (or upload a CSV file)")


if st.button('Predict'):
    if uploaded_file is not None:
        
        input_data = pd.read_csv(uploaded_file)
    elif passenger_id:
        # Filter data by PassengerId
        input_data = original_data[original_data['PassengerId'].astype(str) == passenger_id]
        if input_data.empty:
            st.error("PassengerId not found in the dataset.")
            st.stop()
    else:
        st.error("Please upload a CSV file or provide a PassengerId.")
        st.stop()

    # Prepare input features
    input_features = input_data.drop(columns=['PassengerId', 'Name', 'Transported'], errors='ignore')

    categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
    for col in categorical_cols:
        input_features[col] = input_features[col].fillna('Unknown').astype(str)  # Fill missing values and convert to str

    # Different preprocessing for Logistic Regression and CatBoost
    if model_option == 'Logistic Regression':
        # For Logistic Regression: One-hot encode categorical columns
        input_features = pd.get_dummies(input_features, columns=categorical_cols, drop_first=True)

        # Ensure input features have the same columns as in training
        expected_cols = logistic_model.feature_names_in_  
        input_features = input_features.reindex(columns=expected_cols, fill_value=0) 

        # Predict using Logistic Regression
        predictions = logistic_model.predict(input_features)

    else:  # CatBoost
        # For CatBoost: Convert categorical columns to string
        required_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Cabin']

        # Ensure categorical features are cast to string, even for numerical values
        for col in required_columns:
            if col in ['Age']:  
                input_features[col] = input_features[col].fillna(-1) 
            else:
                input_features[col] = input_features[col].astype(str)

        # Predict using CatBoost
        predictions = catboost_model.predict(input_features)

    # Display predictions
    input_data['Prediction'] = predictions
    st.write(input_data)

    # Option to download predictions as CSV
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
