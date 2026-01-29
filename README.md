# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques.  
Due to the highly imbalanced nature of fraud detection data, evaluation emphasizes **Precision, Recall, and F1-Score** rather than accuracy.

We compare a **baseline Logistic Regression model** with a **Random Forest model**, analyze feature importance, and save the best-performing model for reuse.

---

## 📂 Dataset
**Dataset Name:** Credit Card Fraud Detection  
**Source:** Kaggle  
**Download Link:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

---

## 🛠 Tools & Libraries Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Joblib  

---

## 🎯 Project Objectives
- Analyze fraud vs non-fraud transaction imbalance  
- Perform stratified train-test split  
- Train a baseline Logistic Regression model  
- Train and evaluate a Random Forest model  
- Compare model performance using Precision, Recall, and F1-Score  
- Plot feature importance to identify key fraud indicators  
- Save the best model for reuse  

---

## ⚙️ Workflow Steps
1. Load dataset and analyze class imbalance  
2. Preprocess and scale features  
3. Perform stratified train-test split  
4. Train Logistic Regression baseline model  
5. Train Random Forest classifier  
6. Evaluate models using classification metrics  
7. Plot confusion matrix and feature importance  
8. Save trained model using Joblib  

---

## 📊 Model Evaluation Metrics
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 🧠 Key Results Summary
| Model | Precision | Recall | F1-Score |
|------|----------|--------|---------|
| Logistic Regression | Baseline | Moderate | Moderate |
| Random Forest | High | High | Best |

> ✅ Random Forest performed best for fraud detection.

---

## 📈 Feature Importance
Random Forest identifies top fraud-related features based on transaction patterns and anomalies.

---

## 💾 Saved Model
The best trained model is saved as:

**best_fraud_detection_model.pkl**

---

## ✅ Conclusion
The Random Forest model performed best in detecting fraudulent transactions, achieving higher Precision, Recall, and F1-Score than Logistic Regression. This project demonstrates effective fraud detection on imbalanced financial data using machine learning.

