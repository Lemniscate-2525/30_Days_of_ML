#Imports :
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Dataset :
fp = "/content/insurance.csv"
df = pd.read_csv(fp)

#Encoding Categorical Features : 
df_encoded = pd.get_dummies(df, drop_first = True)
X = df_encoded.drop("expenses", axis = 1)
y = df_encoded["expenses"]

#EDA : 
#print(df.shape)  # 1338*7. 
#df.describe()    # Statistical Values.

#sns.pairplot(df)
#plt.show()

# Heatplot for Correlation b/w Features : 
plt.figure(figsize = (8,8))
sns.heatmap(df_encoded.corr(), annot = True, cmap = "coolwarm") 
plt.title("Correlation HeatMap : ")
plt.show()

#Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Baseline Model :
start = time.time()

baseline = GradientBoostingRegressor(n_estimators = 200, learning_rate = 0.05, max_depth = 3, random_state = 42)
baseline.fit(X_train, y_train)
baseline_train_time = time.time() - start

start = time.time()
baseline_pred = baseline.predict(X_test)
baseline_inf_time = (time.time() - start) / X_test.shape[0]

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

# HyperParameter Tuning(Cross Validation) : 
param_grid = {'n_estimators' : [100, 200, 400], 'learning_rate' : [0.01, 0.05, 0.1], 'max_depth' : [2, 3, 4]}

gbr = GridSearchCV(GradientBoostingRegressor(), param_grid, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
gbr.fit(X_train, y_train)
best_model = gbr.best_estimator_

# Inference Latency and Training Time : 
start = time.time()
gbr.fit(X_train, y_train)
tuned_train_time = time.time() - start
best_model = gbr.best_estimator_

start = time.time()
tuned_pred = best_model.predict(X_test)
tuned_inf_time = (time.time() - start) / X_test.shape[0]

# Metrics(Baseline & Tuned) :
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_r2 = r2_score(y_test, tuned_pred)

results = pd.DataFrame({"Model" : ["Baseline GB","Tuned GB"], "RMSE":[baseline_rmse, tuned_rmse], "R2":[baseline_r2, tuned_r2], "Training Time":[baseline_train_time, tuned_train_time], "Inference Latency":[baseline_inf_time, tuned_inf_time]
})

print(results)

#Residual Plot : 
residuals = y_test - tuned_pred

sns.scatterplot(x = tuned_pred, y = residuals)
plt.axhline(0, color = "red")
plt.title("Residual Plot (Tuned Model)")
plt.show()

#Feature Importance : 
imp = pd.Series(best_model.feature_importances_, index = X.columns)
imp.sort_values().plot(kind = "barh")
plt.title("Feature Importance")
plt.show()

# Feature Importance : 
imp = pd.Series(best_model.feature_importances_, index = X.columns)
imp.sort_values().plot(kind = "barh")
plt.title("Feature Importance (Tuned) : ")
plt.show()

# Boosting Curve(MSE vs Iterations Visualisation):
errors = []

for y_pred in best_model.staged_predict(X_test):
    errors.append(mean_squared_error(y_test, y_pred))

plt.plot(errors)
plt.title("Boosting Stage Error (Tuned) : ")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

print("\nBest Parameters Found:", gbr.best_params_)

