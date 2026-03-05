Customer Churn Prediction using Logistic Regression
Project Overview

Customer churn is one of the most critical problems for subscription-based businesses such as telecom companies, SaaS platforms, and banks.

Acquiring a new customer is estimated to cost 5–7× more than retaining an existing one, making churn prediction a valuable machine learning application.

This project builds a Logistic Regression classifier to predict whether a telecom customer will leave the service based on historical customer data.

The implementation focuses on building a clean end-to-end machine learning pipeline, covering preprocessing, feature encoding, model training, and evaluation.

Dataset

Dataset used: Telco Customer Churn

This dataset contains telecom customer information such as:

Feature	Description
Tenure	Number of months customer stayed
MonthlyCharges	Monthly subscription cost
TotalCharges	Total charges incurred
Contract	Contract type
InternetService	Type of internet service
PaymentMethod	Payment method

Target variable:

Churn
0 → Customer stays
1 → Customer leaves

Dataset size:

7043 rows
21 original features

Dataset Link:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Machine Learning Pipeline

The project follows a standard ML pipeline used in real-world ML systems.

Raw Dataset
      ↓
Data Cleaning
      ↓
Feature Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Logistic Regression Model
      ↓
Model Evaluation
Data Preprocessing

Key preprocessing steps performed:

Data Cleaning

Removed irrelevant column customerID

Converted TotalCharges column to numeric

Handled missing values using median imputation

Feature Encoding

Categorical features were converted into numerical representations using One-Hot Encoding.

Example transformation:

Contract Type

Month-to-month
One year
Two year

Becomes:

Contract_OneYear
Contract_TwoYear

The drop_first=True option is used to avoid the dummy variable trap.

Train-Test Split

Dataset split:

80% training data
20% testing data

Stratified sampling ensures the churn distribution remains consistent across both sets.

Feature Scaling

Logistic Regression performs better when features are normalized.

Standardization was applied using:

StandardScaler

This scales features to:

mean = 0
std = 1
Logistic Regression — Mathematical Intuition

Logistic Regression models the probability that a given input belongs to a particular class.

The model computes a linear combination of input features:

𝑧
=
𝑤
𝑇
𝑥
+
𝑏
z=w
T
x+b

Where:

𝑥
x = feature vector

𝑤
w = learned weights

𝑏
b = bias

This value is passed through the sigmoid activation function:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​


Which maps values to the range:

0 → 1

This output represents the probability of churn.

Prediction rule:

p > 0.5 → Customer likely to churn
p ≤ 0.5 → Customer likely to stay
Model Evaluation Metrics

Since churn datasets are often imbalanced, multiple evaluation metrics are used.

Metric	Purpose
Accuracy	Overall correctness
Precision	Correct churn predictions
Recall	Ability to detect churn customers
F1 Score	Balance between precision and recall
ROC-AUC	Model discrimination capability

ROC-AUC is particularly important for churn prediction.

Results

Example evaluation metrics:

Accuracy: ~0.80
Precision: ~0.65
Recall: ~0.55
F1 Score: ~0.60
ROC-AUC: ~0.84

Results may vary depending on preprocessing and random seed.

Time Complexity

Training complexity for Logistic Regression:

O(n × d)

Where:

n = number of samples
d = number of features

Prediction complexity:

O(d)

per sample.

Space Complexity

Model stores a weight for each feature:

O(d)

Additional memory required for dataset storage.

Key Learnings

Importance of proper categorical feature encoding

Understanding assumptions of Logistic Regression

Handling imbalanced datasets

Importance of ROC-AUC for classification tasks like churn prediction

Building a clean machine learning pipeline
