Personalized Health Risk Prediction Platform

ABOUT THE PROJECT
Personalized Health Risk Prediction Platform is a Machine Learning powered web platform that predicts health risks before symptoms appear. It helps individuals adopt preventive habits and reduce the likelihood of serious diseases by analyzing their lifestyle and health data.

PROBLEM STATEMENT
Many individuals lack tools to understand potential health risks before symptoms appear. Lifestyle factors such as diet, sleep, physical activity, alcohol consumption, and stress significantly affect long-term health but most people cannot interpret their own health data effectively.

OUR SOLUTION
Health Risk Prediction Platform collects user health and lifestyle data and uses a Random Forest Classifier to predict the likelihood of 5 major diseases simultaneously. The system provides personalized insights, visual risk scores, and preventive recommendations.

FEATURES
5 Disease Predictions: Heart Disease, Diabetes, Stroke, High Blood Pressure, High Cholesterol
Visual Gauge Charts: Risk score shown as speedometer for each disease
Symptom Checker: Select multiple symptoms and see which diseases they match
Personalized Tips: Health recommendations based on your specific inputs
Lifestyle Summary: Overview of your lifestyle risk factors
Risk Factor Bar Chart: Visual summary of your active risk factors

TECH STACK
Frontend and UI    : Streamlit
ML Model           : Scikit-learn (Random Forest Classifier)
Data Processing    : Pandas, NumPy
Visualization      : Plotly
Model Saving       : Joblib
Language           : Python 3.9+

DATASET
Name     : Heart Disease Health Indicators BRFSS 2015
Source   : CDC (Centers for Disease Control and Prevention), USA
Records  : 250,000+ real health survey responses
Features : 22 health and lifestyle indicators
Link     :https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

AI MODEL DETAILS
Algorithm        : Random Forest Classifier
Trees per model  : 100 decision trees
Training split   : 80% training, 20% testing
Total models     : 5 separate models (one per disease)
Input features   : BMI, Smoking, Alcohol, Physical Activity, Stroke History,
Difficulty Walking, Sex, Age, Education, High BP,
High Cholesterol, Fruits, Vegetables
DISEASES PREDICTED
Heart Disease       : High BP, High Cholesterol, Smoking, Age
Diabetes            : BMI, Physical Inactivity, Age, Diet
Stroke              : High BP, Smoking, Alcohol, Age
High Blood Pressure : BMI, Smoking, Alcohol, Physical Inactivity
High Cholesterol    : High BP, BMI, Smoking, Diet

PROJECT STRUCTURE 
health-risk-prediction/
app.py              - Main Streamlit web app
model.py            - Trains and saves all 5 ML models
preprocess.py       - Loads and cleans the dataset
recommend.py        - Health tips per disease
symptoms.py         - Symptom matching logic
data/
heart_disease_health_indicators_BRFSS2015.csv

MODEL FILES
due to large file size,models are stored in google drive.
Download models here:https://drive.google.com/drive/folders/19LIxM6AH5r0eKcH6ZOJXVyoZvyQFrB0G?usp=sharing
After downloading,place the .pkl files in the project folder.

Try the application here:https://health-risk-prediction-ygbuyatsvhvzpn9uhntjc8.streamlit.app/
