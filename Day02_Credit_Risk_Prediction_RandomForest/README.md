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

Statistical properties

Feature distributions

Example Distribution Plot
import matplotlib.pyplot as plt

plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.show()

X-axis: Age values
Y-axis: Frequency (number of samples)

🧹 Data Preprocessing
Handling Missing Values

Missing values were handled using domain-aware strategies.

Feature	Strategy
Age	Median
Credit Amount	Mean
Categorical variables	Mode or category mapping

These approaches preserve dataset distribution while removing null values.

Feature Encoding

Categorical features were converted into numeric format.

Tree-based models like Random Forest can work with integer-encoded categories, so extensive scaling or normalization was not required.

🔀 Train-Test Split

The dataset was split into training and testing sets:

80% → Training Data

20% → Testing Data

Using stratified sampling:

train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
🌲 Baseline Model — Random Forest

We first trained a baseline Random Forest model without hyperparameter tuning.

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

Random Forest is an ensemble algorithm that combines multiple decision trees trained on random subsets of data.

Final predictions are determined by majority voting.

🧠 Mathematical Intuition

Random Forest prediction:

𝑦
^
=
𝑚
𝑜
𝑑
𝑒
(
𝑇
1
(
𝑥
)
,
𝑇
2
(
𝑥
)
,
.
.
.
,
𝑇
𝑛
(
𝑥
)
)
y
^
	​

=mode(T
1
	​

(x),T
2
	​

(x),...,T
n
	​

(x))

Where:

𝑇
𝑖
T
i
	​

 = individual decision tree

𝑥
x = feature vector

Randomization occurs through:

Bootstrap sampling of data

Random feature selection during splits

This reduces variance and overfitting.

⏱ Time Complexity

Training complexity:

𝑂
(
𝑇
⋅
𝑁
log
⁡
𝑁
)
O(T⋅NlogN)

Where:

T = number of trees

N = number of samples

Explanation:

Each decision tree evaluates splits that require sorting.

Sorting operations cost roughly O(N log N).

Random Forest trains T trees, making total complexity T × N log N.

🔧 Hyperparameter Tuning

To improve performance, we used GridSearchCV.

Parameters tuned:

Parameter	Description
n_estimators	Number of trees
max_depth	Maximum tree depth
min_samples_split	Minimum samples required for split
min_samples_leaf	Minimum samples in leaf node
max_features	Number of features considered per split

Example search space:

n_estimators = [100, 200, 300]
max_depth = [None, 10, 20]
min_samples_split = [2, 5]
max_features = ["sqrt", "log2"]
📈 Evaluation Metrics

Because credit risk datasets are often imbalanced, multiple metrics were used.

Metric	Purpose
Accuracy	Overall correctness
Precision	Correctly predicted risky borrowers
Recall	Ability to detect risky borrowers
F1 Score	Balance between precision and recall
ROC-AUC	Overall ranking performance
📉 ROC-AUC Explained

ROC = Receiver Operating Characteristic

ROC curve plots:

True Positive Rate

False Positive Rate

AUC represents the area under the ROC curve, measuring the model’s ability to distinguish between classes.

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

The tuned model shows improved ability to identify high-risk borrowers.

📊 Feature Importance

Random Forest calculates feature importance based on impurity reduction across all trees.

Example output:

Feature	Importance
Credit Amount	0.19
Duration	0.16
Age	0.13
Checking Account	0.12

These features contribute the most to predicting credit risk.

📚 Key Learnings

Random Forest reduces overfitting through bagging.

Hyperparameter tuning improves predictive performance.

Feature importance provides model interpretability.

ROC-AUC is essential for evaluating imbalanced classification problems.
