Day 2 — Credit Risk Prediction using Random Forest
📌 Project Overview

In this project, we build a Random Forest classifier to predict credit risk using the German Credit Dataset.

The objective is to determine whether a loan applicant is high risk (likely to default) or low risk based on financial and demographic attributes.

This project is part of the 30 Days of Machine Learning challenge, where each day focuses on implementing and understanding a machine learning model while building a complete ML pipeline.

📊 Problem Statement

Financial institutions must assess the creditworthiness of loan applicants to minimize default risk.

Incorrectly approving a risky borrower can lead to significant financial loss.

Using historical credit data, we train a machine learning model that predicts whether a borrower is:

0 → Good credit
1 → Bad credit (high risk)
📂 Dataset

Dataset: German Credit Dataset
Source: UCI Machine Learning Repository

Attribute	Value
Total samples	1000
Total features	20
Target variable	Risk

Example features:

Feature	Description
Age	Age of applicant
Credit Amount	Loan amount
Duration	Loan duration
Savings Account	Savings status
Checking Account	Checking account status
Employment Duration	Length of employment
⚙️ Machine Learning Pipeline

The full workflow implemented in this project:

Raw Dataset
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Encoding
      ↓
Train-Test Split
      ↓
Baseline Random Forest Model
      ↓
Hyperparameter Tuning
      ↓
Evaluation Metrics
      ↓
Feature Importance Analysis
🔎 Exploratory Data Analysis (EDA)

Before training the model, we explored the dataset to understand its structure and distributions.

Key inspection steps:

df.info()
df.describe()
df.isnull().sum()

These checks help identify:

Data types

Missing values

Summary statistics

Feature distributions

Example Distribution Plot
import matplotlib.pyplot as plt

plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.show()

X-axis: Age values
Y-axis: Number of individuals in each age range

🧹 Data Preprocessing
Handling Missing Values

Missing values were handled using domain-aware strategies.

Feature	Strategy
Age	Median
Credit Amount	Mean
Categorical variables	Mode or mapping

These approaches preserve dataset distribution while removing null values.

Feature Encoding

Categorical features were converted into numeric format.

Tree-based models like Random Forest can work with integer encoded categories, so one-hot encoding was optional.

🔀 Train-Test Split

The dataset was split into training and testing sets:

80% → Training Data
20% → Testing Data

Using stratified sampling to preserve class distribution.

train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
🌲 Baseline Model — Random Forest

We first trained a baseline Random Forest model without hyperparameter tuning.

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

Random Forest is an ensemble algorithm combining multiple decision trees trained on random subsets of data.

The final prediction is determined by majority voting.

🧠 Mathematical Intuition

Random Forest prediction:

ŷ = mode(T₁(x), T₂(x), ..., Tₙ(x))

Where:

Tᵢ = individual decision tree

x = feature vector

Randomization occurs through:

Bootstrap sampling of data

Random feature selection at splits

This reduces variance and overfitting.

⏱ Time Complexity

Training complexity of Random Forest:

O(T × N log N)

Where:

T = number of trees
N = number of samples

Explanation:

Each tree requires sorting to find optimal splits.

Sorting operations cost roughly O(N log N).

With T trees, complexity scales linearly.

🔧 Hyperparameter Tuning

To improve model performance, we used GridSearchCV.

Parameters tuned:

Parameter	Description
n_estimators	Number of trees
max_depth	Maximum tree depth
min_samples_split	Minimum samples required for split
min_samples_leaf	Minimum samples per leaf
max_features	Number of features considered per split

Example search space:

n_estimators = [100, 200, 300]
max_depth = [None, 10, 20]
min_samples_split = [2, 5]
max_features = ["sqrt", "log2"]

Cross-validation ensures the selected parameters generalize well.

📈 Evaluation Metrics

Since credit datasets are often imbalanced, multiple metrics were used.

Metric	Purpose
Accuracy	Overall correctness
Precision	Correctly predicted risky borrowers
Recall	Ability to detect risky borrowers
F1 Score	Balance between precision and recall
ROC-AUC	Overall ranking performance
📉 ROC-AUC Explained

ROC = Receiver Operating Characteristic

ROC curve plots:

True Positive Rate vs False Positive Rate

AUC measures the area under the ROC curve, representing how well the model separates classes.

AUC Score	Interpretation
0.5	Random guessing
0.7	Acceptable
0.8	Good
0.9	Excellent
📊 Baseline Model Results
Metric	Score
Accuracy	0.74
Precision	0.69
Recall	0.63
F1 Score	0.66
ROC-AUC	0.79
🚀 Tuned Model Results

After hyperparameter tuning:

Metric	Score
Accuracy	0.78
Precision	0.72
Recall	0.68
F1 Score	0.70
ROC-AUC	0.83

The tuned model demonstrates improved performance in identifying high-risk borrowers.

📊 Feature Importance

Random Forest calculates feature importance based on impurity reduction across trees.

Example output:

Feature	Importance
Credit Amount	0.19
Duration	0.16
Age	0.13
Checking Account	0.12

Feature importance highlights variables most influential in predicting credit risk.

📚 Key Learnings

Random Forest reduces overfitting through bagging.

Hyperparameter tuning improves predictive performance.

Feature importance provides interpretability.

ROC-AUC is crucial for evaluating imbalanced classification problems.
