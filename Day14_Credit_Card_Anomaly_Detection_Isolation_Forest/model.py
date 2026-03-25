import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Dataset : 
fp = "/content/creditcard.csv"
df = pd.read_csv(fp)

# EDA : 

#df.shape
#df.describe()

#anomaly_ratio = df['Class'].mean()*100
#print("Actual Fraud Vals Ratio : ",  anomaly_ratio)

#sns.scatterplot(data = df, x = 'Time', y = 'Amount', hue = 'Class', palette = ['blue', 'red'], alpha = 0.5)
#plt.title("Time vs Amount : ")
#plt.figure(figsize = (8, 8))
#plt.show()

#sns.kdeplot(data = df[df["Amount"] < 2000], x = 'Amount', hue = 'Class', palette = ['blue', 'red'], fill = True)
#plt.title("Amount Distribution : ")
#plt.show()

# Data Preprocessing : 
df.drop(['Time'], axis = 1, inplace = True) # Time inc monotonically, adds noise.

scaler  = RobustScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis = 1)
y = df['Class']

# Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

# Base Model Training : 
det_base = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', random_state = 42)

# Training Time and Inference Latency : 
start_time = time.time()
det_base.fit(X_train)
base_train_time = time.time() - start_time

start_inf = time.time()
pred_base = np.where(det_base.predict(X_test) == -1,1,0)
base_inf_lat = (time.time() - start_inf) / len(X_test)

# Base Model Metrics : 
scores_base = -det_base.decision_function(X_test)
base_prec, base_rec, base_thresh = precision_recall_curve(y_test, scores_base)
base_pr_auc = auc(base_rec, base_prec)
base_precision = precision_score(y_test, pred_base)
base_recall = recall_score(y_test, pred_base)
base_f1 = f1_score(y_test, pred_base)

# Tuned Model Training : 
known_contamination = y_train.mean()
det_tuned = IsolationForest(n_estimators = 300, max_samples = 256, contamination = known_contamination, random_state = 42)

# Training Time and Inference Latency : 
start_time = time.time()
det_tuned.fit(X_train)
tuned_train_time = time.time() - start_time

start_inf = time.time()
pred_tuned = np.where(det_tuned.predict(X_test) == -1,1,0)
tuned_inf_lat = (time.time() - start_time)/len(X_test)

# Tuned Model Metrics : 
scores_tuned = -det_tuned.decision_function(X_test)
tuned_prec, tuned_rec, tuned_thresh = precision_recall_curve(y_test, scores_tuned)
tuned_pr_auc = auc(tuned_rec, tuned_prec)
tuned_precision = precision_score(y_test, pred_tuned)
tuned_recall = recall_score(y_test, pred_tuned)
tuned_f1 = f1_score(y_test, pred_tuned)

# Comparison Table : 
res = pd.DataFrame({"Model":["Base", "Tuned"], "PR_AUC" : [base_pr_auc, tuned_pr_auc], 
"Precision": [base_precision, tuned_precision], "Recall": [base_recall, tuned_recall], "F1 Score": [base_f1, tuned_f1],
"Training Time(s)" : [base_train_time, tuned_train_time], "Inference Latency(s)" : [base_inf_lat, tuned_inf_lat]})

print(res.to_markdown(index=False))

# PR Curve : 
plt.figure(figsize = (8, 8))
plt.title("Precision-Recall Curve : Base vs Tuned Isolation Forest")
plt.plot(base_rec, base_prec, label = f'Base Model (AUC = {base_pr_auc:.3f})', color = 'blue')
plt.plot(tuned_rec, tuned_prec, label = f'Tuned Model (AUC = {tuned_pr_auc:.3f})', color = 'red', linestyle = '--')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()
