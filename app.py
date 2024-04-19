import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import requests


with open('logistic_classifier.pkl', 'rb') as file:
    model_log = pickle.load(file)


def predict(input_data):
    prediction=model_log.predict(input_data)
    print("Prediction:", prediction[0])
    return prediction[0]

def perform_logistic_regression_on_user_input():
    # Take input from the user for 6 features
    options_mapping = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
    industrial_risk_option = st.selectbox("Select value for feature Industrial Risk:", list(options_mapping.keys()))
    management_risk_option = st.selectbox("Select value for feature Management Risk:", list(options_mapping.keys()))
    financial_flexibility_option = st.selectbox("Select value for feature Financial Flexibility:", list(options_mapping.keys()))
    credibility_option = st.selectbox("Select value for feature Credibility:", list(options_mapping.keys()))
    competitiveness_option = st.selectbox("Select value for feature Competitiveness:", list(options_mapping.keys()))
    operating_risk_option = st.selectbox("Select value for feature Operating Risk:", list(options_mapping.keys()))


    industrial_risk = options_mapping.get(industrial_risk_option)
    management_risk = options_mapping.get(management_risk_option)
    financial_flexibility = options_mapping.get(financial_flexibility_option)
    credibility = options_mapping.get(credibility_option)
    competitiveness = options_mapping.get(competitiveness_option)
    operating_risk = options_mapping.get(operating_risk_option)


    input_data = np.array([[industrial_risk,management_risk,financial_flexibility,
       credibility, competitiveness, operating_risk]])

    return input_data

    
    

def main():
    st.title('Bankruptcy prevention')
    st.write('Select the values for 6 features:')
    
    input_features=perform_logistic_regression_on_user_input()
    if st.button('Predict'):
        prediction_data=predict(input_features)

        st.write("## Prediction:")
        if prediction_data == 1:
            st.write("The prediction is: non-bankruptcy")
        else:
            st.write("The prediction is: bankruptcy")


if __name__ == '__main__':
    main()