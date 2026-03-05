# Customer Churn Prediction using Logistic Regression

## 📌 Project Overview

Customer churn is one of the most critical problems for subscription-based businesses such as telecom companies, SaaS platforms, and banks.

Acquiring a new customer is estimated to cost **5–7× more than retaining an existing one**, making churn prediction an important machine learning application.

In this project, we build a **Logistic Regression classifier** to predict whether a telecom customer will leave the service based on historical customer data.

The implementation focuses on building a **clean end-to-end machine learning pipeline**, including data preprocessing, feature encoding, model training, and evaluation.

---

## 📊 Dataset

**Dataset Used:** Telco Customer Churn

This dataset contains telecom customer information such as:

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| Tenure          | Number of months the customer has stayed |
| MonthlyCharges  | Monthly subscription cost                |
| TotalCharges    | Total amount charged                     |
| Contract        | Contract type                            |
| InternetService | Type of internet service                 |
| PaymentMethod   | Customer payment method                  |

**Target Variable**

| Value | Meaning         |
| ----- | --------------- |
| 0     | Customer stays  |
| 1     | Customer churns |

**Dataset Size**

* **7043 rows**
* **21 original features**

Dataset Link:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## ⚙️ Machine Learning Pipeline

The following ML pipeline was implemented:

```
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
```

---

## 🧹 Data Preprocessing

### 1. Data Cleaning

* Removed unnecessary column **customerID**
* Converted **TotalCharges** to numeric
* Handled missing values using **median imputation**

---

### 2. Feature Encoding

Categorical variables were converted into numerical features using **One-Hot Encoding**.

Example transformation:

| Contract       | Contract_OneYear | Contract_TwoYear |
| -------------- | ---------------- | ---------------- |
| Month-to-month | 0                | 0                |
| One year       | 1                | 0                |
| Two year       | 0                | 1                |

`drop_first=True` is used to avoid the **dummy variable trap**.

---

## 🔀 Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified sampling is used to preserve the original churn distribution.

---

## 📏 Feature Scaling

Logistic Regression performs better when input features are standardized.

Scaling was performed using:

**StandardScaler**

This transforms features so that:

* Mean = **0**
* Standard deviation = **1**

---

## 🧠 Logistic Regression — Mathematical Intuition

Logistic Regression models the probability that an input belongs to a particular class.

The model first computes a linear combination of input features:

[
z = w^T x + b
]

Where:

* **x** → feature vector
* **w** → model weights
* **b** → bias

This value is then passed through the **sigmoid function**:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

The sigmoid function converts the value into a probability between **0 and 1**.

Decision rule:

```
If probability > 0.5 → Churn
Else → No churn
```

---

## 📈 Model Evaluation Metrics

Because churn datasets are often **imbalanced**, multiple metrics are used.

| Metric    | Description                          |
| --------- | ------------------------------------ |
| Accuracy  | Overall correctness                  |
| Precision | Correct churn predictions            |
| Recall    | Ability to detect churn customers    |
| F1 Score  | Balance between precision and recall |
| ROC-AUC   | Model discrimination ability         |

ROC-AUC is particularly important for churn prediction.

---

## 📊 Example Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~0.80 |
| Precision | ~0.65 |
| Recall    | ~0.55 |
| F1 Score  | ~0.60 |
| ROC-AUC   | ~0.84 |

*(Values may vary depending on preprocessing and random seed.)*

---

## ⏱ Time Complexity

Training complexity for Logistic Regression:

```
O(n × d)
```

Where:

* **n** = number of samples
* **d** = number of features

Prediction complexity:

```
O(d)
```

per sample.

---

## 💾 Space Complexity

The model stores a weight for each feature:

```
O(d)
```

Additional memory is required to store the dataset.


## 📚 Key Learnings

* Importance of correct categorical feature encoding
* Understanding assumptions of Logistic Regression
* Handling imbalanced datasets in classification
* Importance of ROC-AUC for churn prediction
* Building a clean machine learning pipeline

---


## 🚀 Future Improvements

Possible extensions include:

* Hyperparameter tuning
* Feature importance analysis
* Handling class imbalance using **SMOTE**
* Trying ensemble models such as **Random Forest** or **XGBoost**





