import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

df = pd.DataFrame(X, columns = feature_names)
df["target"] = y


# EDA :

#df.head()
#df.shape # -> 569*31

#df.describe()
#df.info()

#df.isnull().sum() # -> No MV.

#df['mean area'].describe() # -> Spec Feature Stats.
#df["target"].value_counts() # 357/212


#sns.heatmap(df.corr(), cmap = "coolwarm") # -> Correlation Heatmap b/w independent features.
#plt.title("Correlation Heatmap : ")
#plt.show()


# Train-Test Split : 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

# Feature Scaling : 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model :
clf = SVC(kernel = "linear", probability = True)
clf.fit(X_train, y_train)

# Hyperparameter Tuning : 
param_grid = { "C" : [0.1,1,10],
"kernel" : ["linear", "rbf"],
"gamma" : ["scale", "auto"]
}

grid = GridSearchCV(SVC(probability = True), param_grid, cv = 5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

scores = cross_val_score(best_model, X, y, cv = 5)
#print(scores.mean())

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = "d")
plt.title("Confusion Matrix : ")
#plt.show()

# ROC/AUC Curve : 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"SVM (AUC = {auc:.3f})")
plt.plot([0,1],[0,1],'--', label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

#plt.show()

#print("Accuracy : " , accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#clf.decision_function(X)


# SVM Margin Visualization : 
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

# Train SVM on reduced data
model_vis = SVC(kernel = 'linear')
model_vis.fit(X_pca, y)

# Plot decision boundary
plt.figure(figsize=(7,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap='coolwarm',
    edgecolors='k'
)

# Margin Visualization : 

step = 0.5

x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step),
    np.arange(y_min, y_max, step)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model_vis.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize = (7,6))

plt.contourf(xx, yy, Z, alpha = 0.3, cmap = "coolwarm")

plt.scatter(X_pca[:,0], X_pca[:,1], c = y, cmap = "coolwarm", edgecolors = 'k')

plt.scatter(
    model_vis.support_vectors_[:,0],
    model_vis.support_vectors_[:,1],
    s = 120,
    facecolors = 'none',
    edgecolors = 'black'
)

plt.title("SVM Decision Boundary")

plt.show()
