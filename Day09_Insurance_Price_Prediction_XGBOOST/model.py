# Imports : 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time as time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Dataset : 
fp = "/content/insurance.csv"
df = pd.read_csv(fp)

df_encoded = pd.get_dummies(df, drop_first = True)

X = df_encoded.drop('expenses', axis = 1)
y = df_encoded['expenses']


# EDA : 

#print(df.shape)  # 1338*7. 
#df.describe()    # Statistical Values.

#sns.pairplot(df)
#plt.show()

# Heatplot for Correlation b/w Features : 
#plt.figure(figsize = (8, 8))
#sns.heatmap(df_encoded.corr(), annot = True, cmap = "coolwarm")
#plt.title("Correlation Heatmap : ")
#plt.show()

#sns.histplot(df['expenses'] , kde = True)
#plt.title("Expense Dist : ")
#plt.show()

# Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Baseline Model, Pred Time & Inf Latency : 
start = time.time()

xgb_base = XGBRegressor(
    objective = "reg:squarederror",
    n_estimators = 300,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 1.0,
    colsample_bytree = 1.0,
    reg_lambda = 1,
    gamma = 0,
    random_state = 42,
    n_jobs = -1
)

xgb_base.fit(X_train, y_train)

base_train_time = time.time() - start

start = time.time()
base_pred = xgb_base.predict(X_test)
base_inf_time = (time.time() - start) / X_test.shape[0]

base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
base_r2 = r2_score(y_test, base_pred)


# Hyperparameter Tuning : 
param_grid = {
    "n_estimators" : [200,400],
    "learning_rate" : [0.03,0.05,0.1],
    "max_depth" : [3,4,5],
    "subsample" : [0.7,0.8],
    "colsample_bytree" : [0.7,0.8],
    "reg_lambda" : [1,5],
    "gamma" : [0,0.1]
}

grid = GridSearchCV(XGBRegressor(objective = "reg:squarederror", random_state = 42, n_jobs = -1), param_grid, cv = 3, scoring = "neg_mean_squared_error", n_jobs = -1)

# Tuned Model Inference Latency and Pred Time : 
start = time.time()
grid.fit(X_train, y_train)
tuned_train_time = time.time() - start

best_model = grid.best_estimator_

start = time.time()
tuned_pred = best_model.predict(X_test)
tuned_inf_time = (time.time() - start) / X_test.shape[0]

tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_r2 = r2_score(y_test, tuned_pred)

# Comparison of Results : 
results = pd.DataFrame({
    "Model" : ["Baseline XGB","Tuned XGB"],
    "RMSE" : [base_rmse, tuned_rmse],
    "R2" : [base_r2, tuned_r2],
    "Training Time" : [base_train_time, tuned_train_time],
    "Inference Latency" : [base_inf_time, tuned_inf_time]
})

print(results)
print("\nBest Parameters:", grid.best_params_)

# Residual Error Plot for Tuned Model :
residuals = y_test - tuned_pred

sns.scatterplot(x = tuned_pred, y = residuals)
plt.axhline(0, color = "red")
plt.title("Residual Plot (Tuned) : ")
plt.show()

# Error Distribution : 
sns.histplot(residuals, kde = True)
plt.title("Prediction Error Distribution")
plt.show()

# Bar Plot for Feature Importance : 
imp = pd.Series(best_model.feature_importances_, index = X.columns)
imp.sort_values().plot(kind = "barh")
plt.title("Feature Importance (Tuned XGBoost)")
plt.show()

# Boosting Curve Visualisation(Error vs Iterations):
best_model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose = False)
evals = best_model.evals_result()

plt.plot(evals["validation_0"]["rmse"], label = "Train")
plt.plot(evals["validation_1"]["rmse"], label = "Test")

plt.title("Boosting Error Visualization : ")
plt.xlabel("Iterations")
plt.ylabel("RMSE")

plt.legend()
plt.show()

