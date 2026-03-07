import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Load Dataset
path = "/content/german_credit_data.csv"
df = pd.read_csv(path)

#Data Preprocessing and EDA : 

#Renaming German columns to English
df.columns = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "guarantor",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "risk"
]

#df.head()

#EDA
#df.shape

#df.info()
#df.describe()

#Checking for Missing Values
#df.isnull().sum()

#Checking Data Distribution:
#df["risk"].value_counts()

#df["age"].describe()

#plt.hist(df["age"], bins=20)
#plt.title("Age Distribution")
#plt.show()

#Correlation Heatmap
#df.corr()
#sns.heatmap(df.corr(), cmap = "coolwarm")

#Train-Test Split

X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#rf = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 42)
#rf.fit(X_train, y_train)

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[None,10,20],
    "min_samples_split":[2,5],
    "min_samples_leaf":[1,2],
    "max_features":["sqrt","log2"]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print(grid.best_params_)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:,-1]

#Feature Importance

#importance = best_rf.feature_importances_
#feat_imp = pd.DataFrame({"feature" : X.columns, "importance" : importance}).sort_values("importance", ascending = False)
#print(feat_imp.head(10))
#feat_imp.head(10).plot(x = "feature", y = "importance", kind = "barh", figsize = (8, 8))

#plt.title("Top Feature Importances")
#plt.show()

#Metrics and Evaluation

print("Accuracy:" , accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
