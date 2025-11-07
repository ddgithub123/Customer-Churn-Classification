# 📊 Customer Churn Classification – Model Development, Validation & Deployment  

This project focuses on predicting **customer churn** for a telecom company using machine learning models. It involves **data preprocessing, exploratory analysis, model development, evaluation, and deployment**, aligned with the requirements of **21AIC401T – Inferential Statistics and Predictive Analytics** case study.

---

## 📁 Dataset  
**Source:** [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Records:** 7,043  
**Features:** 21 (Demographic, Service, and Billing Attributes)  
**Target Variable:** `Churn` (Yes = Customer churned, No = Retained)

---

## ⚙️ Workflow Overview  

### 1. Data Preparation  
- Loaded and cleaned dataset (handled missing values, duplicates, and type conversions).  
- Encoded categorical features using LabelEncoder and OneHotEncoder.  
- Scaled numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`).  
- Conducted EDA with visualizations (correlation map, churn distribution, service type analysis).  

### 2. Model Development  
Implemented and compared multiple classification models:

| Model | Accuracy | ROC-AUC | Observation |
|-------|-----------|----------|-------------|
| CHAID-like Decision Tree | 0.784 | 0.808 | Good interpretability. |
| Logistic Regression | 0.795 | 0.839 | Stable linear baseline. |
| Random Forest (Tuned) | 0.796 | 0.837 | Reliable ensemble. |
| Gradient Boosting | 0.797 | 0.837 | Consistent performance. |
| XGBoost | 0.793 | 0.833 | Handles imbalance well. |
| LightGBM | 0.785 | 0.823 | Fast and efficient. |
| SVM | 0.796 | 0.801 | Performs well on scaled data. |
| KNN | 0.769 | 0.795 | Sensitive to scaling. |
| Naive Bayes | 0.656 | 0.814 | Weak alone but diverse. |
| **Stacking + SMOTE** | **0.843** | **0.924** | ✅ Best model with balanced precision and recall. |

---

## 📈 Evaluation Metrics
- **Accuracy**, **ROC-AUC**, **Precision**, **Recall**, **F1-Score**  
- **Confusion Matrix**, **Lift Chart**, and **Cumulative Gains Chart** for performance visualization  
- **Cross-validation (5-Fold)** and **SMOTE** used for model validation and balancing

---

## 🚀 Deployment
The final **Stacking Classifier** model was serialized using **Joblib** (`model.pkl`).  
It can be deployed via an API or integrated into a CRM dashboard for real-time churn prediction.

├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── Customer Churn CLassification.ipynb
├── visuals/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gains_chart.png
├── README.md


