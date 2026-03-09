import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris

#Dataset : 
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names

#Converting Dict->df : 
df = pd.DataFrame(X, columns = feature_names)
df["target"] = y


#EDA : 
#print(X.shape)
#print(y.shape)

#df.head()

#df.describe()
#df.info()

#df.isnull().sum() # -> No Missing Vals
#print(df["target"].value_counts()) # -> Class Dist

#sns.pairplot(df, hue="target")
#plt.show()

#Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Data Preprocessing : 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Cross Validation : 
k_vals = range(1, 21)
scores = []

for k in k_vals:
  model = KNeighborsClassifier(n_neighbors = k)
  cv_scores = cross_val_score(model, X, y, cv = 5)
  scores.append(cv_scores.mean())

# K vs Scores : 

plt.plot(k_vals, scores)

plt.xlabel("Value of K")
plt.ylabel("CV Accuracy")
plt.title("K vs Accuracy")

plt.show()

#Model Training : 
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)

#Evaluation : 
y_pred = model.predict(X_test)

#Metrics : 
print("Accuracy : ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Visualization :

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = "d")
plt.title("Confusion Matrix : ")
plt.show()
