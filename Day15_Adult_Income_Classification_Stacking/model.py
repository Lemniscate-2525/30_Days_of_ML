import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time as time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

fp = "/content/adult.csv"
df = pd.read_csv(fp)

# EDA : 
#df.shape

#df.describe()

#print(df.isnull().sum())  # 0 MV.

#sns.histplot(df["income"])
#plt.show()

#num_cols = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

#plt.figure(figsize = (6,6))
#sns.heatmap(df[num_cols].astype(float).corr(), annot = True, cmap = "coolwarm")

#plt.title("Feature Correlation Matrix : ")
#plt.show()

# Data Preprocessing : 
for col in df.columns:
  df[col] = df[col].astype(str).str.strip() # Removes Whitespaces.

df = df.replace("?", np.nan)   
df = df.dropna()

df["income"] = df["income"].map({"<=50K" : 0, ">50K" : 1, "<=50K.":0, ">50K.":1})  

num_cols = [
    "age",       # str vals to num in some cols.
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

for col in num_cols:
  df[col] = pd.to_numeric(df[col])

df = pd.get_dummies(df, drop_first = True) # Categorical Feature Encoding.

# Indep/Dep Data : 
X = df.drop("income", axis = 1)
y = df["income"]

# Scaling : 
scaler =  StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

# Helper fn common for all 4 Base learners : 
def evaluate(model, name):

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test)
    latency = (time.time() - start)/X_test.shape[0]

    prob = model.predict_proba(X_test)[:,1]

    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, prob)

    return {
        "Model": name,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
        "Train_Time": train_time,
        "Latency": latency,
        "Probs": prob
    }

# Log-Reg Model(B1) : 
lr = LogisticRegression(max_iter = 1000)

# KNN Model(B4) : 
knn = KNeighborsClassifier(n_neighbors = 15)

# RF Model(B2) : 
rf = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 42)

# XGB Model(B3) :
xgb = XGBClassifier(n_estimators = 200, max_depth = 5, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, eval_metric = "logloss")

res = []

res.append(evaluate(lr, "LogReg"))
res.append(evaluate(knn, "KNN"))
res.append(evaluate(rf, "RF"))
res.append(evaluate(xgb, "XGB"))

# Tuned Baseline XGBoost : 
param_grid = {"n_estimators" : [200,400], "max_depth" : [4,6], "learning_rate" : [0.03,0.07]}

grid = GridSearchCV(XGBClassifier(eval_metric = "logloss"), param_grid, scoring = "roc_auc", cv = 3, n_jobs = -1)

start = time.time()
grid.fit(X_train, y_train)
tuned_time = time.time() - start

best_xgb = grid.best_estimator_

start = time.time()
tuned_preds = best_xgb.predict(X_test)
tuned_latency = (time.time() - start)/X_test.shape[0]

tuned_prob = best_xgb.predict_proba(X_test)[: ,1]

res.append({
    "Model" : "Tuned_XGB",
    "Precision" : precision_score(y_test,tuned_preds),
    "Recall" : recall_score(y_test,tuned_preds),
    "F1" : f1_score(y_test,tuned_preds),
    "ROC_AUC" : (y_test,tuned_prob),
    "Train_Time" : tuned_time,
    "Latency" : tuned_latency,
    "Probs" : tuned_prob
})

# OOF Stacking : 
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

meta_train = np.zeros((X_train.shape[0], 4))
meta_test = np.zeros((X_test.shape[0], 4))

models = [lr, knn, rf, xgb]

for i, model in enumerate(models):

    test_fold_preds = []

    for train_idx, val_idx in kf.split(X_train):

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)

        meta_train[val_idx,i] = model.predict_proba(X_val)[: ,1]
        test_fold_preds.append(model.predict_proba(X_test)[: ,1])

    meta_test[: ,i] = np.mean(test_fold_preds, axis = 0)

# Meta Learner : 
meta_model = LogisticRegression()

start = time.time()
meta_model.fit(meta_train, y_train)
stack_time = time.time() - start

start = time.time()
stack_preds = meta_model.predict(meta_test)
stack_latency = (time.time() - start)/X_test.shape[0]

stack_prob = meta_model.predict_proba(meta_test)[: ,1]

res.append({
    "Model": "Stacking",
    "Precision": precision_score(y_test,stack_preds),
    "Recall": recall_score(y_test,stack_preds),
    "F1": f1_score(y_test,stack_preds),
    "ROC_AUC": roc_auc_score(y_test,stack_prob),
    "Train_Time": stack_time,
    "Latency": stack_latency,
    "Probs": stack_prob
})

# Comparison :
df_res = pd.DataFrame(res).drop(columns = "Probs")
print(df_res)

# ROC Curve : 
for r in res:
    fpr,tpr,_ = roc_curve(y_test, r["Probs"])
    plt.plot(fpr, tpr, label = r["Model"])

plt.legend()
plt.title("ROC Curve : ")
plt.show()

# PR Curve : 
for r in res:
    prec,rec,_ = precision_recall_curve(y_test, r["Probs"])
    plt.plot(rec,prec,label=r["Model"])

plt.legend()
plt.title("Precision Recall Curve")
plt.show()
