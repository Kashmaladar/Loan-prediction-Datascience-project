Loan Prediction Project
Project Overview

This project predicts whether a loan applicant will be approved or rejected based on personal and financial features. The focus is on improving predictions for the minority class (rejected loans) in a highly imbalanced dataset.

Target Variable:

Status

Y → Approved (1)

N → Rejected (0)

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

Engineered Features

Total applicant income

Loan burden ratio

Income per dependent

Area and education risk

Income stability

Data Preparation
Handling Class Imbalance

Logistic Regression & KNN → Random Oversampling

Gradient Boosting & XGBoost → SMOTE

Feature Engineering & Selection

Created domain-driven features

Removed zero-importance features after model evaluation

Models & Performance

All metrics are reported on the test set.

🔹 Class label meaning

0 = Rejected loans (minority class of interest)

1 = Approved loans

📊 Model comparison
Model	Precision (Class 0)	Recall (Class 0)	F1 (Class 0)	Accuracy	AUC
Logistic Regression	0.52	0.63	0.57	0.73	0.709
K-Nearest Neighbors	0.45	0.56	0.50	0.69	~0.71
Gradient Boosting	0.43	0.62	0.51	0.72	~0.71
XGBoost	0.48	0.67	0.56	0.74	0.710
Optimized XGBoost (Optuna)	0.89	0.40	0.56	0.78	0.723
Interpretation

Optimized XGBoost reaches the highest AUC

Optimized XGBoost heavily prioritizes:

very high recall for approved loans (class 1)

very high precision for rejected loans (class 0)

Base XGBoost provides more balanced performance

Oversampling + SMOTE improves minority-class detection

Key Takeaways

Handling class imbalance is essential

Ensemble models outperform simple ones

Hyperparameter tuning matters — Optuna improved AUC

There is a trade-off:

catching more rejected loans

vs avoiding false alarms

How to Run
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt