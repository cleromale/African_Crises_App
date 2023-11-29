# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:43:06 2023

@author: Windows
"""

# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and handling
from joblib import load  # For loading the pre-trained machine learning model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # For data preprocessing

# Load the pre-trained machine learning model
model = load('africa_crises_model2.joblib')

# Function to preprocess categorical columns
def preprocess_categorical(df):
    lb = LabelEncoder()

    # Encode 'country' and 'banking_crisis' columns to numerical values
    df['country'] = lb.fit_transform(df['country'])
    df['banking_crisis'] = lb.fit_transform(df['banking_crisis'])
    return df

# Function to preprocess numerical columns
def preprocess_numerical(df):
    # Scale numerical columns to a specific range
    scaler = MinMaxScaler()
    numerical_cols = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default',
                      'inflation_annual_cpi', 'independence', 'currency_crises', 'inflation_crises']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

# Function to preprocess input data
def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_categorical(input_df)
    input_df = preprocess_numerical(input_df)
    return input_df

# Main function to create the web app interface
def main():
    st.title('Africa Crises Predictor App')  # Title of the web app
    st.write('This app is built to determine if there is going to be any systemic crises depending on some available features. Please feel free to experiment with the input features below.')

    input_data = {}  # Dictionary to store user input data
    col1, col2 = st.columns(2)  # Split the interface into two columns

    with col1:
        # Collect user inputs for country and some financial indicators
        input_data['country'] = st.selectbox('Country', ['Egypt', 'South Africa', 'Algeria', 'Zimbabwe', 'Angola', 'Morocco', 'Zambia', 'Mauritius', 'Kenya', 'Tunisia', 'Nigeria',
                                                         'Central African Republic', 'Ivory Coast'])
        input_data['exch_usd'] = st.number_input('Exchange against USD', step=1)
        input_data['domestic_debt_in_default'] = st.number_input('Any Domestic Debt? If yes 1, if No, 0', min_value=0, max_value=1)
        input_data['sovereign_external_debt_default'] = st.number_input('Any Sovereign External Debt? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['inflation_annual_cpi'] = st.number_input('Inflation Annual CPI', step=1)

    with col2:
        # Collect user inputs for other indicators
        input_data['independence'] = st.number_input('Independence? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['currency_crises'] = st.number_input('Currency Crises? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['inflation_crises'] = st.number_input('Inflation Crises? if Yes 1', min_value=0, max_value=1)
        input_data['banking_crisis'] = st.selectbox('Banking Crisis?', ['crisis', 'no crisis'])

    input_df = pd.DataFrame([input_data])  # Convert collected data into a DataFrame
    st.write(input_df)  # Display the collected data on the app interface

    if st.button('Predict'):  # When the 'Predict' button is clicked
        final_df = preprocessor(input_df)  # Preprocess the collected data
        prediction = model.predict(final_df)[0]  # Use the model to predict the outcome
        
        # Display the prediction result
        if prediction == 1:
            st.write('There is a likelihood that there will be systemic crises.')
        else:
            st.write('There is a likelihood that there will not be any systemic crises')

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()
