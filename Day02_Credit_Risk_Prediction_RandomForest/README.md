# Credit Risk Prediction

---

## 📌 Project Overview

Credit risk assessment is a critical problem for financial institutions such as banks and lending platforms.

Incorrectly approving a risky borrower can lead to significant financial loss. Machine learning models can help predict whether a loan applicant is likely to default based on historical financial data.

In this project, we build a **Random Forest classifier** to predict whether a borrower represents **good credit risk** or **bad credit risk**.

The implementation focuses on building a clean end-to-end machine learning pipeline, including exploratory data analysis, preprocessing, baseline modeling, hyperparameter tuning, and evaluation.

---

## 📊 Dataset

**Dataset Used:** [German Credit Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

This dataset contains financial and demographic information about loan applicants.

| Feature | Description |
|---|---|
| Age | Age of the applicant |
| Credit Amount | Total loan amount |
| Duration | Loan duration in months |
| Checking Account | Status of checking account |
| Savings Account | Savings balance category |
| Employment Duration | Years of employment |
| Housing | Housing status |

**Target Variable**

| Value | Meaning |
|---|---|
| 0 | Good credit |
| 1 | Bad credit (High Risk) |

**Dataset Size**
- 1000 rows
- 20 features

---

## ⚙️ Machine Learning Pipeline

```
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature Encoding
     ↓
Train/Test Split
     ↓
Baseline Random Forest Model
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Feature Importance Analysis
```

---

## 🔎 Exploratory Data Analysis (EDA)

EDA was performed to understand dataset structure, feature distributions, and potential data issues.

Key checks included:
- Inspecting dataset structure using `df.info()`
- Examining statistical properties using `df.describe()`
- Checking missing values using `df.isnull().sum()`

**Example visualization:**

```python
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.show()
```

This histogram shows how applicant ages are distributed across the dataset.

---

## 🧹 Data Preprocessing

### 1. Data Cleaning

- Verified dataset structure
- Checked for missing values
- Ensured numerical columns were correctly formatted

The German Credit dataset does not contain significant missing values, so minimal cleaning was required.

### 2. Feature Encoding

Categorical features were converted into numerical format so that they could be used by the machine learning model.

Tree-based models like Random Forest can work directly with integer-encoded categories, so extensive one-hot encoding was not required.

---

## 🔀 Train-Test Split

The dataset is divided into:
- **80%** Training Data
- **20%** Testing Data

Stratified sampling is used to preserve the distribution of credit risk classes.

---

## 🌲 Random Forest — Baseline Model

A baseline Random Forest classifier was first trained before performing hyperparameter tuning.

```python
RandomForestClassifier(
    n_estimators = 200,
    random_state = 42
)
```

Random Forest is an ensemble learning algorithm that combines multiple decision trees trained on random subsets of the dataset. Final predictions are determined through **majority voting** across trees.

---

## 🔧 Hyperparameter Tuning

After establishing the baseline model, hyperparameter tuning was performed using **GridSearchCV**.

| Parameter | Description |
|---|---|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum depth of trees |
| `min_samples_split` | Minimum samples required for split |
| `min_samples_leaf` | Minimum samples per leaf |
| `max_features` | Features considered per split |

**Example parameter grid:**

```python
n_estimators = [100, 200, 300]
max_depth = [None, 10, 20]
min_samples_split = [2, 5]
max_features = ["sqrt", "log2"]
```

---

## 📈 Model Evaluation Metrics

Because credit risk datasets can be imbalanced, multiple metrics were used.

| Metric | Description |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correctly predicted risky borrowers |
| Recall | Ability to detect risky borrowers |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Model's ability to distinguish between classes |

> **ROC-AUC** is particularly useful for evaluating classification models on imbalanced datasets.

---

## 📊 Results

### Baseline Model

| Metric | Value |
|---|---|
| Accuracy | ~0.74 |
| Precision | ~0.69 |
| Recall | ~0.63 |
| F1 Score | ~0.66 |
| ROC-AUC | ~0.79 |

### Tuned Model

| Metric | Value |
|---|---|
| Accuracy | ~0.78 |
| Precision | ~0.72 |
| Recall | ~0.68 |
| F1 Score | ~0.70 |
| ROC-AUC | ~0.83 |

Hyperparameter tuning improved the model's ability to detect high-risk borrowers.

---

## 📊 Feature Importance

Random Forest provides built-in feature importance scores based on **impurity reduction**.

| Feature | Importance |
|---|---|
| Credit Amount | High |
| Duration | High |
| Age | Moderate |
| Checking Account | Moderate |

Feature importance helps identify which attributes most influence credit risk prediction.

---

## ⏱ Complexity Analysis

### Time Complexity

Training complexity for Random Forest:

```
O(T × N log N)
```

Where:
- **T** = number of trees
- **N** = number of samples

Each decision tree requires sorting operations during split evaluation, resulting in approximately `O(N log N)` complexity per tree.

### Space Complexity

Random Forest stores multiple decision trees:

```
O(T × D)
```

Where:
- **T** = number of trees
- **D** = depth of trees

Additional memory is required to store the dataset and trained trees.

---

## 📚 Key Learnings

- Random Forest reduces overfitting through **ensemble learning**
- Hyperparameter tuning improves model performance
- Feature importance provides **interpretability**
- ROC-AUC is critical for evaluating classification models with imbalanced datasets
- Building structured ML pipelines improves **model reproducibility**
