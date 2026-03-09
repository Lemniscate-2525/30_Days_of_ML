import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

#Dataset : 
path = "/content/Titanic-Dataset.csv"
df = pd.read_csv(path)

#EDA:
#df.info() # -> No NULL Values ?.
#df.shape  # -> rows*columns. 

#df.describe()


#Data Preprocessing : 
#df.isnull().sum() -> Age, Cabin & Embarked have missing Values.

#MV-> Age with Median;

# Cabin had MV, but that is because higher class passengers have their Cabin codes, 
# but some lower class passengers don't; also 687 vals are missing in the cabin col;
# so by instinct we could just drop it completely or; we could extract something  
# meaningful out of it. We can choose Deck no as we see upon careful obs
# that deck no is the first letter in the cabin code which can be used as a meaningful feature.
# So, we extract deck from it and then drop the col.

#Embarked -> Mode.

df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")

df = pd.get_dummies(df, columns=["Deck"], drop_first=True)

df = df.drop(["Name","Ticket","Cabin","PassengerId"], axis=1)

#df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

#Encoding for Categorical Columns : 
df["Sex"] = df["Sex"].map({"male" : 0, "female" : 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Features : (Independent & Dependent) : 
X = df.drop("Survived", axis = 1) 
y = df["Survived"]

# Train/Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Model : 
clf = DecisionTreeClassifier(max_depth = 4, random_state = 42)
clf.fit(X_train, y_train)

# Cross Validation
scores = cross_val_score(clf, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
print("Std Dev:", scores.std())

# Predictions : 
y_pred = clf.predict(X_test)

# Metrics :

print("Accuracy : ", accuracy_score(y_test, y_pred))

print("Classification Report : ")
print(classification_report(y_test, y_pred))

print("Confusion Matrix : ")
print(confusion_matrix(y_test, y_pred))


# Visualization : 
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = "d")
plt.show()

# Tree : 
plt.figure(figsize = (10, 10))
plot_tree(clf, feature_names = X.columns , class_names = ["Died", "Survived"], filled = True)
plt.show()

# Feature_Importance : 
importance = clf.feature_importances_

feat_imp = pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values("importance", ascending = False)
#print(feat_imp.head(10))

feat_imp.head(10).plot(x = "feature", y = "importance", kind = "barh", figsize = (6, 6))

plt.title("Feature Importances : ")
plt.show()






