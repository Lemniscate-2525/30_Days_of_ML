
# Customer Churn Prediction 

## Project Overview

Customer churn is one of the most important problems faced by subscription-based businesses such as telecom companies, SaaS platforms, and financial services.

Acquiring a new customer can cost **5–7× more than retaining an existing one**, making churn prediction a valuable machine learning application.

In this project, we build a **Logistic Regression classifier** to predict whether a telecom customer will leave the service based on historical data.

The focus of this implementation is building a **clean end-to-end machine learning pipeline**, including:

- Exploratory Data Analysis
- Data preprocessing
- Feature encoding
- Model training
- Cross validation
- Model evaluation
- Visualization

---

# Dataset

**Dataset Used:** Telco Customer Churn

Dataset Link  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset characteristics:

- **7043 samples**
- **21 original features**
- Binary classification problem

Example features include:

| Feature | Description |
|------|------|
| Tenure | Number of months customer stayed |
| MonthlyCharges | Monthly subscription cost |
| TotalCharges | Total billing amount |
| Contract | Contract type |
| InternetService | Internet service type |

Target Variable:

| Value | Meaning |
|------|------|
| 0 | Customer stays |
| 1 | Customer churns |

---

# Machine Learning Pipeline

The following pipeline was implemented:

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
Feature Scaling
     ↓
Logistic Regression Model
     ↓
Cross Validation
     ↓
Model Evaluation
     ↓
Visualization
```

---

# Exploratory Data Analysis

EDA helps understand the structure and quality of the dataset before modeling.

The following checks were performed:

- dataset shape
- feature types
- missing values
- class distribution

Key observations:

- dataset contains both numerical and categorical features
- `TotalCharges` contains missing values
- target classes are moderately imbalanced

Understanding these properties helps guide preprocessing decisions.

---

# Data Preprocessing

### Removing Irrelevant Features

The column `customerID` is a unique identifier and does not contribute to prediction.

```
df.drop('customerID', axis=1)
```

---

### Handling Missing Values

`TotalCharges` is converted to numeric and missing values are replaced using **median imputation**.

Median is chosen because it is less sensitive to outliers compared to the mean.

---

### Encoding Target Variable

The target variable `Churn` is converted to numeric values.

| Original | Encoded |
|------|------|
| Yes | 1 |
| No | 0 |

---

### One-Hot Encoding

Categorical features are converted to numeric features using **One-Hot Encoding**.

Example:

| Contract | Contract_OneYear | Contract_TwoYear |
|------|------|------|
| Month-to-month | 0 | 0 |
| One year | 1 | 0 |
| Two year | 0 | 1 |

`drop_first=True` is used to avoid the **dummy variable trap**.

---

# Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

Stratified sampling is used to preserve the original class distribution.

---

# Feature Scaling

Logistic Regression performs better when input features are standardized.

Scaling is performed using **StandardScaler**, which transforms features so that:

- Mean = 0
- Standard Deviation = 1

This ensures all features contribute equally to model training.

---

# Mathematical Intuition

Logistic Regression models the probability that a sample belongs to a particular class.

![Sigmoid Function](sf.png)

First, the model computes a linear combination of the input features:

```
z = wᵀx + b
```

Where:

- **x** = feature vector  
- **w** = model weights  
- **b** = bias term  

This value is then passed through the **sigmoid function**:

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid function converts any real value into a probability between **0 and 1**.

```
0 ≤ P(y = 1) ≤ 1
```

Decision rule:

```
If P(y = 1) ≥ 0.5 → churn
Else → no churn
```

---

# Cross Validation

A single train-test split may produce unstable performance depending on how the dataset is divided.

To obtain a more reliable estimate, **K-Fold Cross Validation** is used.

In this project:

```
5-Fold Cross Validation
```

Process:

1. The training data is split into **5 folds**
2. The model trains on **4 folds**
3. The remaining fold is used for validation
4. This process repeats **5 times**
5. The final score is the **average performance across folds**

Cross validation helps:

- reduce variance in evaluation
- detect overfitting
- produce a more robust performance estimate

---

# Model Evaluation

The model is evaluated using multiple classification metrics.

| Metric | Description |
|------|------|
| Accuracy | Overall prediction correctness |
| Precision | Correct churn predictions |
| Recall | Ability to detect churn customers |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Model discrimination capability |

Using multiple metrics provides a more complete understanding of model performance.

---

# Confusion Matrix

![Confusion Matrix](cm1.png)

The confusion matrix shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

This helps identify the types of errors made by the classifier.

---

# ROC Curve

![ROC Curve](roc1.png)

The ROC curve shows the tradeoff between:

- True Positive Rate
- False Positive Rate

The **Area Under the Curve (ROC-AUC)** measures how well the model distinguishes between churn and non-churn customers.

---

# Feature Importance

Logistic Regression learns a **weight for each feature**.

These weights represent how strongly each feature influences the prediction.

Feature importance is derived from the model coefficients:

```
model.coef_
```

Large positive coefficients indicate features that increase churn probability.

Large negative coefficients indicate features that reduce churn probability.

![Feature Importance](fi1.png)

---

# Time Complexity

Training complexity for Logistic Regression is approximately:

```
O(n × d)
```

Where:

- \(n\) = number of samples  
- \(d\) = number of features  

During each optimization step, the model computes the dot product between feature vectors and model weights.

This requires processing all features for every sample.

---

# Prediction Complexity

Prediction requires computing:

\[
z = w^T x + b
\]

This is a dot product operation.

Therefore prediction complexity per sample is:

```
O(d)
```

---

# Space Complexity

The model stores a weight for each feature.

Space complexity:

```
O(d)
```

Additional memory is required to store the dataset during training.

---

# Key Learnings

- Importance of feature encoding for categorical variables
- Understanding the sigmoid function in logistic regression
- Importance of feature scaling
- Using cross validation for robust model evaluation
- Interpreting model coefficients as feature importance

---

# Why Logistic Regression?

Logistic Regression is a strong baseline model for structured tabular datasets like customer churn.

It is especially useful when:

- The relationship between features and target is approximately linear
- Interpretability of model predictions is important
- Fast training and real-time inference are required

In production churn systems, logistic regression is often deployed as a **first benchmark model** before testing more complex algorithms.

---

# Engineering Tradeoffs

| Aspect | Logistic Regression |
|------|------|
| Training Speed | Very Fast |
| Inference Latency | Extremely Low |
| Interpretability | High |
| Non-linear Modeling | Limited |
| Feature Scaling Requirement | Mandatory |

Because it learns a **linear decision boundary**, performance may degrade when churn behaviour depends on complex feature interactions.

---

# Failure Case Analysis

After evaluating predictions, misclassified samples were inspected.

Observed limitations:

- Customers with similar tenure and billing patterns but different churn outcomes were difficult to separate linearly  
- Logistic Regression struggles when churn depends on **non-linear relationships between categorical service features**
- Model performance is sensitive to feature scaling and encoding quality  

This motivates testing tree-based and ensemble models in later experiments.

---

# Training Time Scaling Experiment

To understand model scalability, training time was measured by increasing dataset size fractions.

Observation:

- Training time increases approximately linearly with dataset size  
- This empirically validates theoretical training complexity: O(n x d)

This property makes Logistic Regression highly suitable for large-scale churn prediction pipelines.

---

# Inference Constraints

Prediction requires computing a **Dot product between feature vector and learned weights.**

Implications:

- Extremely low inference latency.  
- Suitable for real-time prediction APIs.  
- Can handle very high request throughput.  

This is a major advantage compared to kernel methods or deep models.

---

# When NOT to Use Logistic Regression

- When strong non-linear feature interactions exist.  
- When decision boundary is highly complex.  
- When dataset contains hierarchical or sequential patterns.  
- When feature engineering is limited.  

In such cases, tree ensembles or neural networks may perform better.

---

# Future Improvements : 

- Regularization tuning (L1 vs L2)
- Interaction feature engineering  
- Probability calibration  
- Ensemble comparison (Random Forest / Gradient Boosting / XGBoost)
- Drift monitoring for changing customer behaviour  

---

# Model Updates

### v1 — Initial Implementation  
Baseline logistic regression with preprocessing, cross-validation and evaluation.

### v2 — Engineering Upgrade (12-3-2026)  
- Added training time measurement.  
- Added inference latency analysis.  
- Added failure case auditing.  
- Added dataset scaling experiment.  
- Added engineering tradeoff discussion.  

---

