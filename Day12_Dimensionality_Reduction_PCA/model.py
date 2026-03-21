import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.datasets import load_breast_cancer

# Dataset : 
data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)

# EDA : 
plt.figure(figsize = (8, 8))
sns.heatmap(df.corr(), cmap = "coolwarm")
plt.title("Correlation Heatmap : ")
plt.show()


# Scaling : 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train Test Split : 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size = 0.2,
    random_state = 42
)

# Logistic Regression(Baseline) : 
start = time.time()

lr = LogisticRegression(max_iter = 5000)
lr.fit(X_train, y_train)

train_time_b = time.time() - start

start = time.time()
pred_b = lr.predict(X_test)
latency_b = time.time() - start

acc_b = accuracy_score(y_test, pred_b)

prec_b = precision_score(y_test, pred_b)
rec_b = recall_score(y_test, pred_b)
f1_b = f1_score(y_test, pred_b)

prob_b = lr.predict_proba(X_test)[:,1]
auc_b = roc_auc_score(y_test, prob_b)

# PCA Visualization : 
pca = PCA()
pca.fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.title("Explained Variance Curve : ")

plt.show()

# Principal Componenets(Features) capturing 95% Variance : 
pca = PCA(n_components = 0.95)

X_pca = pca.fit_transform(X_scaled)

print("Original Dim : ", X.shape[1])
print("Reduced Dim : ", X_pca.shape[1])

# Train Test Split : 
X_train_p, X_test_p, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)

# Model after PCA : 
start = time.time()

lr_p = LogisticRegression(max_iter = 5000)
lr_p.fit(X_train_p, y_train)

train_time_a = time.time() - start

start = time.time()
pred_a = lr_p.predict(X_test_p)
latency_a = time.time() - start

acc_a = accuracy_score(y_test, pred_a)
prec_a = precision_score(y_test, pred_a)
rec_a = recall_score(y_test, pred_a)
f1_a = f1_score(y_test, pred_a)

prob_a = lr_p.predict_proba(X_test_p)[:,1]
auc_a = roc_auc_score(y_test, prob_a)

# Comparison Table : 
res = pd.DataFrame({
    "Model": ["Before PCA", "After PCA"],
    "Accuracy": [acc_b, acc_a],
    "Precision": [prec_b, prec_a],
    "Recall": [rec_b, rec_a],
    "F1 Score": [f1_b, f1_a],
    "ROC AUC": [auc_b, auc_a],
    "Training Time": [train_time_b, train_time_a],
    "Inference Latency": [latency_b, latency_a]
})

print(res) 

# 2D Visualization using PCA : 
pca_2 = PCA(n_components = 2)
X_2 = pca_2.fit_transform(X_scaled)

plt.scatter(X_2[:,0], X_2[:,1], c=y, cmap="coolwarm")
plt.title("2D PCA Projection")
plt.show()

