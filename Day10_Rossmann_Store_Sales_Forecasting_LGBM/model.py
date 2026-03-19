# Imports : 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time as time
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline

# Dataset  : 
fp = "/content/train.csv"
store_fp = "/content/store.csv"

df = pd.read_csv(fp, low_memory = False)
store = pd.read_csv(store_fp, low_memory = False)

df = df.merge(store, on = "Store", how = "left")  

# EDA : 
#df.head()
#df.shape

#df.describe()
#X.describe()

#df.isnull().sum().sort_values(ascending = False) # -> NO Missing Values -> :)

#sns.histplot(df["Sales"], bins = 60)
#plt.title("Sales Distribution :")
#plt.show()

#plt.figure(figsize = (8, 8))
#sns.heatmap(X_new.corr(), cmap = "coolwarm")
#plt.title("Correlation HeatMap : ")
#plt.show()

df = df[df["Open"] == 1]  # Open stores as 1, closed as 0 so no learning from them.

# Feature Engineering : 
df["Date"] = pd.to_datetime(df["Date"])   # Extracting useful features from Date col and then dropping it.

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

df.drop("Date", axis = 1, inplace = True)

df["Sales"] = np.log1p(df["Sales"]) # Log Transform(Fix skew in Sales Data).

# Data Preprocessing :
df["StateHoliday"] = df["StateHoliday"].astype(str)
df["PromoInterval"] = df["PromoInterval"].astype(str)  

cat_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"] 

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes  # Label Encoding for Categorical Data.

df.fillna(0, inplace = True) # Fill any NA vals.

# Independent/Dependent Variable : 
X = df.drop("Sales", axis = 1)
y = df["Sales"]

# Train Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Base Model : 
start = time.time()

lgb_base = LGBMRegressor (
    objective = "regression",
    n_estimators = 300,
    learning_rate = 0.05,
    num_leaves = 31,
    max_depth = -1,
    random_state = 42,
    n_jobs = -1
)

lgb_base.fit(X_train, y_train)

base_train_time = time.time() - start

# Pred Time and Inf Latency for Base Model : 
start = time.time()
base_pred = lgb_base.predict(X_test)
base_inf_time = (time.time() - start) / X_test.shape[0] 

# Reverse Log Transform : 
base_pred_real = np.expm1(base_pred)
y_test_real = np.expm1(y_test)

base_rmse = np.sqrt(mean_squared_error(y_test_real, base_pred_real))
base_r2 = r2_score(y_test_real, base_pred_real)

# Hyperparameter Tuning : 
param_grid = {"n_estimators":[300, 600], "learning_rate":[0.03, 0.05, 0.1], "num_leaves":[31, 63], "max_depth":[-1, 10]}

grid = GridSearchCV(LGBMRegressor(objective = "regression", random_state = 42, n_jobs = -1), param_grid, cv = 5, scoring = "neg_mean_squared_error", n_jobs = -1)

# Pred Time and Inf Latency for Tuned Model : 
start = time.time()  
grid.fit(X_train, y_train)
tuned_train_time = time.time() - start

best_model = grid.best_estimator_

start = time.time()
tuned_pred = best_model.predict(X_test)
tuned_inf_time = (time.time() - start) / X_test.shape[0]

tuned_pred_real = np.expm1(tuned_pred)

tuned_rmse = np.sqrt(mean_squared_error(y_test_real, tuned_pred_real))
tuned_r2 = r2_score(y_test_real, tuned_pred_real)

# Metrics Comparison : 
results = pd.DataFrame({
    "Model":["Base LGBM","Tuned LGBM"],
    "RMSE":[base_rmse, tuned_rmse],
    "R2":[base_r2, tuned_r2],
    "Training Time":[base_train_time, tuned_train_time],
    "Inference Latency":[base_inf_time, tuned_inf_time]
})

print(results)

print("\nBest Parameters: ", grid.best_params_)

# Residual Error Visualisation : 

residuals = y_test_real - tuned_pred_real

sns.scatterplot(x = tuned_pred_real, y = residuals)
plt.axhline(0, color = "red")
plt.title("Residual Plot")
plt.show()

# Error Distribution : 
sns.histplot(residuals, kde = True)
plt.title("Error Distribution")
plt.show()

# Feature Importance Bar Graph : 
imp = pd.Series(best_model.feature_importances_, index = X.columns)
imp.sort_values().plot(kind = "barh")
plt.title("Feature Importance")
plt.show()
