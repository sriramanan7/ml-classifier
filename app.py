import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Diabetes Prediction Application", layout="wide")

st.title("Classificaton Model Deployment: Diabetes Prediction")
st.write("This application demonstrates various classification models on the Diabetes Health Indicators Dataset.")

# Sidebar
st.sidebar.header("User Input Features")

# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())
    
    # Preprocessing
    if 'Diabetes_binary' in data.columns:
        X = data.drop('Diabetes_binary', axis=1)
        y = data['Diabetes_binary']
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Model Selection
        model_name = st.sidebar.selectbox(
            "Select Classifier",
            ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
        )
        
        # Load model mapping
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "KNN": "knn.pkl",
            "Naive Bayes": "naive_bayes.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "xgboost.pkl"
        }
        
        import joblib
        import os

        if st.sidebar.button("Predict & Evaluate"):
            # Simple way to check and load model
            model_file = model_files[model_name]
            model_path = os.path.join("model", model_file)
            
            if os.path.exists(model_path):
                # Load the saved model
                model = joblib.load(model_path)
                
                # Make predictions
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                st.write(f"## Model: {model_name}")
                st.write(f"### Accuracy: {acc:.4f}")
                
                # Show Metrics
                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))
                
                # Show Confusion Matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
                
            else:
                st.error(f"File '{model_file}' not found in 'model' folder.")
                st.write("Please run the notebook to generate model files first.")
                
    else:
        st.error("Dataset must contain 'Diabetes_binary' column for this demo.")
        
else:
    st.info("Awaiting for CSV file to be uploaded. Please upload the test dataset.")
