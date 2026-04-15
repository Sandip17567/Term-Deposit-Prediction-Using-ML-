# Term Deposit Subscription Prediction

## Overview
This project focuses on predicting whether a bank customer will subscribe to a term deposit using machine learning techniques. The analysis is based on a real-world marketing dataset and aims to improve targeted marketing strategies by identifying potential customers.

## Objectives
- Analyze customer data and marketing campaign information  
- Perform data preprocessing and cleaning  
- Detect and handle outliers  
- Handle class imbalance using SMOTE  
- Apply machine learning models for classification  
- Evaluate model performance using appropriate metrics  
- Identify the best model for prediction  

## Dataset
- Source: Kaggle (Bank Marketing Dataset)  
- Records: 45,000+ observations  
- Features include:
  - Demographics (age, job, education, marital status)  
  - Financial details (balance, loan, housing)  
  - Campaign information (contact type, duration, previous outcome)  
- Target Variable: Subscription to term deposit (Yes/No)  

## Data Preprocessing
- Checked for missing and duplicate values  
- Handled outliers using IQR and Z-score methods  
- Applied transformations to reduce skewness  
- Encoded categorical variables using One-Hot Encoding  
- Discretized important features (e.g., pdays)  
- Performed feature scaling using standardization  

## Handling Imbalanced Data
- Dataset had class imbalance (majority vs minority class)  
- Applied SMOTE to generate synthetic samples  
- Improved model performance for minority class prediction  

## Models Used
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  

## Model Evaluation
Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

## Results
- Random Forest achieved the best overall performance  
- XGBoost also showed strong predictive capability  
- Decision Tree and SVM provided stable and interpretable results  

## Key Insights
- Call duration is one of the most important factors influencing subscription  
- Previous campaign outcomes significantly impact customer decisions  
- Data preprocessing and balancing greatly improve model accuracy  
- Ensemble models perform better than individual models  
## Term Deposit Prediction App

This project includes a machine learning model + Streamlit app for real-time prediction.
# How to run
 streamlit run appp.py
# Demo
![App Screenshot][https://drive.google.com/file/d/116cAteY-Z7SLeBplOw986DXlTJTSbtZQ/view?usp=drive_link][https://drive.google.com/file/d/1TAz1UuswKMKLPZ-lGd9rxcQ5e1CC9lH9/view?usp=drive_link]

## Conclusion
This project demonstrates how machine learning can be applied to marketing data to predict customer behavior. Proper preprocessing, handling of imbalanced data, and model selection are critical for achieving high prediction accuracy. The results can help banks optimize marketing strategies, reduce costs, and improve customer targeting.
