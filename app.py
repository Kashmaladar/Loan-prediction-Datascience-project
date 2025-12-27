import streamlit as st
import joblib
import numpy as np
import pandas as pd




from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert Dependents safely
        X['Dependents'] = pd.to_numeric(X['Dependents'], errors='coerce').fillna(0)

        # Total income
        X['total_applicant_income'] = (
            X['Applicant_Income'] + X['Coapplicant_Income']
        )

        # Log transforms (SAFE)
        X['Log_Loan_Amount'] = np.log1p(X['Loan_Amount'])
        X['Log_Total_Income'] = np.log1p(X['total_applicant_income'])

        # Loan burden (log ratio)
        X['Loan_Burden_Log'] = np.log1p(
            X['Loan_Amount'] / (X['total_applicant_income'] + 1)
        )

        # Income pressure
        X['Income_pressure'] = (
            X['Dependents'] / (X['total_applicant_income'] + 1)
        )

        # Income stability
        X['Income_Stability'] = (
            (X['Self_Employed'] == 'No').astype(int) +
            (X['Coapplicant_Income'] > 0).astype(int)
        )

        # Loan burden ratio
        X['Loan_Burden_Ratio'] = (
            X['Loan_Amount'] /
            (X['Applicant_Income'] + X['Coapplicant_Income'] + 1)
        )

        # Income per dependent
        X['Income_per_Dependent'] = (
            (X['Applicant_Income'] + X['Coapplicant_Income']) /
            (X['Dependents'] + 1)
        )

        # Applicant contribution
        X['Applicant_Contribution'] = (
            X['Applicant_Income'] /
            (X['Applicant_Income'] + X['Coapplicant_Income'] + 1)
        )

        # Interaction feature
        X['Dependents_Stability'] = (
            X['Dependents'] * X['Income_Stability']

        )
        X['Dependents'] = pd.to_numeric(X['Dependents'], errors='coerce').fillna(0)


        return X


#load my model
model = joblib.load("bestmodel33.pkl")



st.title("Loan Prediction App")
st.divider()
st.write("Enter the values and hit the prediction button for getting a prediction.")
st.divider()

Gender = st.selectbox("Enter the Gender:",["Male","Female"])

Married = st.selectbox("Enter Martial Status :",["Yes","No"])

Dependents = st.selectbox("Enter Dependents :",["0","1","2","3+"])

Education = st.selectbox("Enter Education :",["Graduate","Not Graduate"])

Self_Employed = st.selectbox("Enter Employement Status :",["Yes","No"])
                            
Applicant_Income = st.number_input("Enter Applicant_Income",min_value=0,max_value=8_100_000,value=0,step=10_000)

Coapplicant_Income =  st.number_input("Enter Coapplicant_Income",min_value=0,max_value=4_200_000,value=0,step=10_000)

Loan_Amount    =  st.number_input("Enter Loan_Amount",min_value=0,max_value=70_000_000,value=0,step=10_000)

Term = st.number_input("Enter Term",min_value=0,max_value=460,value=0)
Credit_History  =  st.number_input("Enter Credit_History",max_value=1,value=1,step=1)
Area  = st.selectbox("Enter Area :",["Semiurban","Urban","Rural"])           

            



#creating predict button
prdictbutton = st.button("Predict")
if prdictbutton:
   #feature matrix x
   # Create a single-row DataFrame with the exact column names your pipeline expects
    input_df = pd.DataFrame({
        "Gender": [Gender],
        "Married": [Married],
        "Dependents": [Dependents],
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "Applicant_Income": [Applicant_Income],
        "Coapplicant_Income": [Coapplicant_Income],
        "Loan_Amount": [Loan_Amount],
        "Term": [Term],
        "Credit_History": [Credit_History],
        "Area": [Area]
        # add more columns if your model expects them
    })
    pred = model.predict(input_df)


    predicted = "Congratulations! Your loan is likely to be approved." if pred[0]==1 else "Sorry, your loan is unlikely to be approved."

    st.write(f"Predicted {predicted}")

else:
    st.write("Pls Enter the values and use predict button")


