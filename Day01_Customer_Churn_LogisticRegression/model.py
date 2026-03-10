#Imports : 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Loading Datset : 
path = "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(path)

# EDA : 

# print("Shape : ", df.shape)

# df.describe()
# df.info()

# df.isnull().sum() -> ret sum of MV 

# print("Class Distribution : ", df["Churn"].value_counts())

# Data Preprocessing : 
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0}) # Label Encoding

df = pd.get_dummies(df, drop_first = True)

X = df.drop('Churn', axis = 1) # Independent Feature
y = df['Churn'] # Dependent Feature

# Train_Test_Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 42)

# Scaling :
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training :
clf = LogisticRegression(max_iter = 1000, solver="lbfgs")
clf.fit(X_train, y_train)

# Cross Validation :
cv_scores = cross_val_score(clf, X_train, y_train, cv = 5)
print(cv_scores)
print(cv_scores.mean())

# Predictions : 
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

# Training Metrics : 
accu = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ra = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accu:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC_AUC: {ra:.4f}")
print("\n")
print("Confusion Matrix:\n", cm)

# Visualization : 
plt.figure(figsize = (6,6))
sns.heatmap(cm, annot = True, fmt = "d" , cmap = "Blues")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label = "ROC Curve")
plt.plot([0,1],[0,1],'--')

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")

plt.legend()
plt.show()

#Feature Importance :
importance = clf.coef_[0]

feat_imp = pd.DataFrame({"Feature": X.columns, "importance": importance}).sort_values("importance", ascending = False)

#print("\nTop Important Features:")
#print(feat_imp.head(10))

feat_imp.head(10).plot(x="Feature", y="importance", kind="barh", figsize=(6,6))

plt.title("Top Feature Importances")
plt.show()
