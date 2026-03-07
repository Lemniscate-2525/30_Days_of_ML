DAY 2 — CREDIT RISK PREDICTION USING RANDOM FOREST
PROJECT OVERVIEW

In this project we build a Random Forest classifier to predict credit risk using the German Credit Dataset.

The objective is to determine whether a loan applicant is high risk (likely to default) or low risk based on financial and demographic attributes.

This project is part of the 30 Days of Machine Learning Challenge, where each day focuses on implementing and understanding a machine learning model while building a complete ML pipeline.

PROBLEM STATEMENT

Financial institutions must assess the creditworthiness of loan applicants to minimize default risk.

Incorrectly approving a risky borrower can lead to financial losses.

Using historical credit data, we train a machine learning model that predicts whether a borrower is:

Label	Meaning
0	Good credit
1	Bad credit (High risk)
DATASET

Dataset used: German Credit Dataset

Source: UCI Machine Learning Repository

Attribute	Value
Total samples	1000
Total features	20
Target variable	Risk

Example dataset features:

Feature	Description
Age	Age of the applicant
Credit Amount	Total loan amount
Duration	Loan duration in months
Savings Account	Savings account category
Checking Account	Checking account status
Employment Duration	Years of employment
MACHINE LEARNING PIPELINE

The project follows a standard machine learning workflow:

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
Model Evaluation
↓
Feature Importance Analysis

EXPLORATORY DATA ANALYSIS (EDA)

Before training the model we analyze the dataset structure.

Dataset Inspection
df.info()
df.describe()
df.isnull().sum()

These commands reveal:

data types

missing values

statistical properties

feature distributions

Age Distribution Visualization
import matplotlib.pyplot as plt

plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.show()

Explanation:

Axis	Meaning
X-axis	Age values
Y-axis	Frequency (number of people in each age group)
DATA PREPROCESSING
Handling Missing Values

Missing values were handled using domain-aware strategies.

Feature	Method
Age	Median
Credit Amount	Mean
Categorical variables	Mode

This preserves distribution while removing null values.

Feature Encoding

Categorical variables were converted into numerical representations.

Random Forest can work with integer encoded categorical values, so extensive scaling was not required.

TRAIN TEST SPLIT

Dataset split into training and testing data.

Dataset	Percentage
Training data	80%
Testing data	20%

Implementation:

train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

Stratified sampling ensures the class distribution remains consistent.

BASELINE MODEL — RANDOM FOREST

We first trained a baseline Random Forest classifier.

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

Random Forest is an ensemble learning algorithm that combines multiple decision trees trained on random subsets of the data.

Final predictions are determined by majority voting across trees.

MATHEMATICAL INTUITION

Random Forest prediction can be expressed as:

ŷ = mode(T₁(x), T₂(x), … , Tₙ(x))

Where:

Symbol	Meaning
Tᵢ	Individual decision tree
x	Input feature vector

Randomization occurs through:

bootstrap sampling

random feature selection

This reduces variance and overfitting.

TIME COMPLEXITY

Training complexity of Random Forest:

O(T × N log N)

Variable	Meaning
T	Number of trees
N	Number of samples

Each decision tree requires sorting operations to evaluate splits, costing approximately O(N log N).

HYPERPARAMETER TUNING

Hyperparameters were optimized using GridSearchCV.

Parameters tuned:

Parameter	Description
n_estimators	Number of trees
max_depth	Maximum tree depth
min_samples_split	Minimum samples required to split
min_samples_leaf	Minimum samples in a leaf node
max_features	Features considered for splits

Example search space:

n_estimators = [100,200,300]
max_depth = [None,10,20]
min_samples_split = [2,5]
max_features = ["sqrt","log2"]
MODEL EVALUATION METRICS

Since credit datasets are often imbalanced, multiple evaluation metrics were used.

Metric	Purpose
Accuracy	Overall prediction correctness
Precision	Correctly predicted risky borrowers
Recall	Ability to detect risky borrowers
F1 Score	Balance between precision and recall
ROC-AUC	Ranking ability of the classifier
ROC-AUC EXPLAINED

ROC stands for Receiver Operating Characteristic.

The ROC curve plots:

True Positive Rate

False Positive Rate

AUC represents the Area Under the Curve.

AUC Score	Interpretation
0.5	Random guessing
0.7	Acceptable
0.8	Good
0.9	Excellent
BASELINE MODEL RESULTS
Metric	Score
Accuracy	0.74
Precision	0.69
Recall	0.63
F1 Score	0.66
ROC-AUC	0.79
TUNED MODEL RESULTS

After hyperparameter tuning:

Metric	Score
Accuracy	0.78
Precision	0.72
Recall	0.68
F1 Score	0.70
ROC-AUC	0.83

The tuned model demonstrates improved ability to detect high-risk borrowers.

FEATURE IMPORTANCE

Random Forest calculates feature importance using impurity reduction across trees.

Example output:

Feature	Importance
Credit Amount	0.19
Duration	0.16
Age	0.13
Checking Account	0.12

These features contribute most strongly to credit risk prediction.

KEY LEARNINGS

Random Forest reduces overfitting through bagging

Hyperparameter tuning improves predictive performance

Feature importance improves model interpretability

ROC-AUC is critical for evaluating imbalanced classification tasks
