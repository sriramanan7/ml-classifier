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
import joblib
import os

st.set_page_config(page_title="Diabetes Prediction Application", layout="wide")

st.title("Diabetes Prediction Application")
st.write("This application demonstrates various classification models on the Diabetes Health Indicators Dataset.")

st.sidebar.title("Options")

#Data source selection
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV", "Use Demo Data"])

data = None

if data_source == "Upload CSV":
    st.sidebar.info("Upload the diabetes dataset (CSV)")
    uploaded_file = st.sidebar.file_uploader("Choose file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    st.sidebar.info("Using demo dataset")
    github_url = "https://raw.githubusercontent.com/sriramanan7/ml-classifier/main/test_data_small.csv"
    
    try:
        data = pd.read_csv(github_url)
        st.sidebar.info("Loaded data from GitHub")
    except Exception as e:
        st.error("Could not load data from GitHub. Please upload a CSV.")
        st.error(f"GitHub Error: {e}")

if data is not None:
    st.write("### Dataset Preview:")
    st.dataframe(data.head())
    
    # Preprocessing
    if 'Diabetes_binary' in data.columns:
        X = data.drop('Diabetes_binary', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = data['Diabetes_binary']
        #The full uploaded dataset is used for testing directly as model files are saved
        X_test = pd.DataFrame(X_scaled, columns=X.columns)
        y_test = y
        st.write("### Model Evaluation:")
        
        # Model Selection
        model_name = st.selectbox(
            "Select Classifier Model",
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
        


        if st.button(f"Predict and Evaluate"):
            model_file = model_files[model_name]
            model_path = os.path.join("model", model_file)
            
            if os.path.exists(model_path):
                # Load the saved model
                try:
                    model = joblib.load(model_path)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Results for **{model_name}**")
                    st.metric("Accuracy: ", f"{acc:.4f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                    
                    with col2:
                        st.write("#### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error running model: {e}")
            else:
                st.error(f"Model file '{model_file}' not found in 'model' folder.")
                
    else:
        st.error("Dataset must contain 'Diabetes_binary' column for this demo.")
        
else:
    st.info("Please select a data source (Upload or Demo) to proceed.")





