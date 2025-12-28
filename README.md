Loan Prediction Project
Project Overview

This project predicts whether a loan applicant will be approved or rejected based on personal and financial features.
Because the dataset is highly imbalanced, the focus is on improving prediction quality for the rejected-loan class (minority class) instead of only maximizing accuracy.

Target Variable

‘Y’ → Approved (1)

‘N’ → Rejected (0)

Dataset Features
Demographics & Personal Info

Gender

Married

Dependents

Education

Financial Info

Applicant Income

Co-applicant Income

Loan Amount

Loan Term

Credit History

Derived / Engineered Features

Total household income

Loan burden ratio

Income per dependent

Income stability indicators

Education/area risk flags

Data Preparation

handled missing values

encoded categorical variables

feature engineering performed

non-informative features removed after evaluation

Handling Class Imbalance

Logistic Regression & KNN → Random Oversampling

Gradient Boosting & XGBoost → SMOTE

Optuna used for hyperparameter tuning of XGBoost

Models & Performance (Key Results)

Focus metric: minority class (Rejected loans = 0)

Model	Precision (Class 0)	Recall (Class 0)	AUC	Accuracy
Logistic Regression	0.52	0.63	0.709	0.73
K-Nearest Neighbors	0.45	0.56	~0.71	0.69
Gradient Boosting	0.43	0.62	~0.71	0.72
XGBoost	0.48	0.67	0.710	0.74
Optuna-Tuned XGBoost	0.89	0.40	0.723	0.78
Interpretation

tuned XGBoost achieves very high precision for rejected loans

but recall drops — it misses some risky applicants

logistic regression and untuned XGBoost are more balanced

oversampling techniques consistently improved minority-class metrics

Business Meaning

goal = avoid approving risky loans → tuned XGBoost

goal = identify most risky loans overall → logistic regression / baseline XGBoost

How to Run
git clone <https://github.com/Kashmaladar/Loan-prediction-Datascience-project.git>
cd <LOAN-PREDICTION-DATASCIENCE-PROJECT>
pip install -r requirements.txt
python app.py