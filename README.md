# Customer Churn Prediction Pipeline

## Project Overview
An end-to-end Machine Learning project that predicts customer churn for a subscription-based business.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Dataset
IBM Telco Customer Churn Dataset containing customer information, subscription details and churn status.

## Features
- Data cleaning and preprocessing
- Missing value handling
- One-hot encoding
- Feature scaling
- Feature engineering
- Logistic Regression
- Random Forest
- Decision Tree
- Hyperparameter tuning using GridSearchCV
- Model evaluation using ROC-AUC, Precision, Recall and PR-AUC
- REST API using FastAPI

## Engineered Features
1. Tenure Group
2. Average Monthly Spend
3. Total Services

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Start the API:

python -m uvicorn api:app --reload

Open API documentation:

http://127.0.0.1:8000/docs

## Prediction
The API accepts customer information and returns:
- Churn probability
- Churn prediction
- Risk level

## Conclusion
This project demonstrates a complete supervised machine learning pipeline for customer churn prediction and supports data-driven customer retention decisions.
