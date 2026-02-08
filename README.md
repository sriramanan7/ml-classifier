# Classification Models - Diabetes prediction - Diabetes Health Indicators dataset

## 1. Problem Statement
To build and evaluate multiple machine learning classification models to predict whether a person has diabetes/pre-diabetes or not based on various health indicators.

## 2. Dataset Description
The dataset used is the **Diabetes Health Indicators Dataset** (50-50 split, binary class variant) from Kaggle.
- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Rows**: 70062
- **Columns**: 22
- **Target**: `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes/Pre-diabetes)
- **Features**: 21 features like BMI, HighBP, HighChol, Smoker, Age, PhysActivity etc.

## 3. Models Used & Comparison
Implemented 6 different classification models. Below is the performance comparison on the test dataset (large). The values below are from a sample run and might vary slightly based on the random seed/split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.7458 | 0.8232 | 0.7371 | 0.7639 | 0.7503 | 0.4919 |
| Decision Tree | 0.6516 | 0.6518 | 0.6556 | 0.6387 | 0.6470 | 0.3033 |
| KNN | 0.7129 | 0.7712 | 0.7024 | 0.7387 | 0.7201 | 0.4264 |
| Naive Bayes | 0.7153 | 0.7837 | 0.7206 | 0.7032 | 0.7118 | 0.4308 |
| Random Forest | 0.7487 | 0.8269 | 0.7287 | 0.7926 | 0.7593 | 0.4994 |
| XGBoost | 0.7483 | 0.8247 | 0.7291 | 0.7901 | 0.7584 | 0.4983 |


## 4. Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Performed very well (approx 75% accuracy), indicating a linear relationship works well for this data. Shows that there is not much non-linearity present in the dataset. |
| **Decision Tree** | Lowest performance (approx 65% accuracy), likely due to overfitting on the training data. Decision trees are usually prone to overfitting. |
| **KNN** | Moderate performance (71% accuracy), but can be computationally expensive and sensitive to the scale of data. |
| **Naive Bayes** | Good performance and fast to train, but assumes independence between features which might not be accurate as there are correlated features. |
| **Random Forest (Ensemble)** | Improved significantly over Decision Tree (approx 74% accuracy) by reducing variance, showing the power of bagging. |
| **XGBoost (Ensemble)** |  Best model for this dataset, alongside Logistic Regression, effectively capturing complex patterns through boosting. Accuracy is around 74% which is on par with logistic regression. But Recall is better (79%), and recall is the metric to look out for in medical field. |
