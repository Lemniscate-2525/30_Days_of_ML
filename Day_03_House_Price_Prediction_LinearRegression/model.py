import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

#Dataset

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns = housing.feature_names)
df["Price"] = housing.target

#EDA :

#df.head()
#df.shape
#df.describe()

#df.info() -> No NULL Values.

#Feature Corerelation : 
#df.corr()
#sns.heatmap(df.corr(), cmap = 'coolwarm')

#Data Preprocessing :
#df.isnull().sum() -> Confirms No NULL Values.


X = df.drop("Price", axis = 1)
y = df["Price"]


#Train-Test Split :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Scaling : 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

#Model Training :
model = LinearRegression()
model.fit(X_train, y_train)

#Model Predictions : 
y_pred = model.predict(X_test)


#Metrics :
mse = mean_squared_error(y_test, y_pred)
print("rmse:", mse ** 0.5)
print("r2:",  r2_score(y_test, y_pred))


#Visualization of Results : 
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted")

plt.show()
