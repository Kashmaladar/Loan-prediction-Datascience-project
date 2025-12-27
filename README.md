# Loan Prediction Project

## Project Overview
This project predicts whether a loan applicant will be **approved** or **rejected** based on personal and financial features.  
The model focuses on improving predictions for the **minority class (rejected loans)** in a highly imbalanced dataset.

**Target Variable:**  
- `Status`:  
  - `'Y'` → Approved (converted to `1`)  
  - `'N'` → Not Approved / Rejected (converted to `0`)  

---

## Dataset Features
- **Demographics & Personal Info:**  
  - Gender  
  - Married  
  - Dependents  
  - Education  

- **Financial Info:**  
  - Applicant Income  
  - Coapplicant Income  
  - Loan Amount  
  - Loan Term  
  - Credit History  

- **Derived/Engineered Features:**  
  - Total applicant income  
  - Loan burden ratio  
  - Area and education risk  
  - Income stability  
  - Income per dependent  

---

## Data Preparation
- **Handling Imbalance:**  
  - Logistic Regression & KNN: Random Oversampling  
  - XGBoost: SMOTE (Synthetic Minority Oversampling Technique)  

- **Feature Engineering & Selection:**  
  - Created domain-specific features.  
  - Removed non-informative features (zero importance) after model evaluation.  

---

## Models & Performance

| Model                  | Precision (Rejected Loans) | Recall (Rejected Loans) | F1-score (Rejected Loans) | AUC Score |
|------------------------|--------------------------|------------------------|--------------------------|-----------|
| Logistic Regression     | 50%                       | 55%                    | 53%                       | ~68%      |
| K-Nearest Neighbors     | 52%                       | 50%                    | 51%                       | 67%       |
| XGBoost (Optimized)     | 62%                       | 43%                    | 51%                       | ~68%      |

**Notes:**  
- Focused on minority class (rejected loans) rather than overall accuracy.  
- ROC curves were plotted for performance evaluation.  
- Feature importance helped remove irrelevant features for a cleaner model.  

---

## Key Takeaways
- Oversampling techniques significantly improve minority class prediction.  
- XGBoost with optimized hyperparameters slightly outperforms simpler models.  
- Feature importance helps improve interpretability without hurting performance.  
- The model is ready to deploy via a **web application** for real-time loan prediction.

---

## How to Run
1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-folder>
