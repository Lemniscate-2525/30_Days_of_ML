import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans

from sklearn.model_selection import PredefinedSplit

from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

fp = "/content/Mall_Customers.csv"
df = pd.read_csv(fp)

# EDA : 

#df.shape
#df.describe()

#sns.scatterplot(x = df["Annual Income (k$)"], y = df["Spending Score (1-100)"])
#plt.title("Customer Dist : ")
#plt.show()

#sns.histplot(df["Annual Income (k$)"], kde = True)
#plt.title("Income Dist : ")
#plt.show()

#sns.histplot(df["Spending Score (1-100)"], kde = True)
#plt.title("Spending Score Dist : ")
#plt.show()

# Feature Selection : 
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scaling : # Vital as this model uses dist measures : 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SS for GSCV : 
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

# Values of k : 
k_range_vals = range(2, 11)

inertia_list = []
sil_list = []
db_list = []
ch_list = []

for k in k_range_vals:

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    inertia_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_scaled, labels))
    db_list.append(davies_bouldin_score(X_scaled, labels))
    ch_list.append(calinski_harabasz_score(X_scaled, labels))

# Selecting Values of K : (6 methods) : 

# 1. Empirical(Formula) : 
n = X.shape[0]
k_e = int(np.sqrt(n/2))
print("Empirical K Val : ", k_e)

# 2. Silhouette Score : 
k_s = k_range_vals[np.argmax(sil_list)]

# 3. Davies Bouldin : 
k_db = k_range_vals[np.argmin(db_list)]

# 4. CH Score : 
k_ch = k_range_vals[np.argmax(ch_list)]

# 5. Elbow Method : 
i_arr = np.array(inertia_list)
second_diff = np.diff(i_arr, 2)
k_elbow = k_range_vals[np.argmin(second_diff) + 1]

# 6. GridSearchCV : 
scores = {}
for k in k_range_vals:
    km_temp = KMeans(n_clusters = k, n_init = 20, random_state = 42)
    score = silhouette_scorer(km_temp, X_scaled)
    scores[k] = score

k_grid = max(scores, key = scores.get)

k_grid = max(scores, key = scores.get)
grid = type('obj', (object,) , {'best_params_': {'n_clusters': k_grid}})()

print("K_empirical : ", k_e)
print("K_silhouette : ", k_s)
print("K_DB : ", k_db)
print("K_CH : ", k_ch)
print("K_elbow : ", k_elbow)
print("k_GSCV : ", k_grid)

method_k = {
    "Empirical" : k_e,
    "Silhouette" : k_s,
    "Davies_Bouldin" : k_db,
    "Calinski_Harabasz" : k_ch,
    "Elbow" : k_elbow,
    "GridSearchCV" : k_grid
}

# Model Training, Inference Latency, Training Time : 
res = []
for meth, k in method_k.items():
    start = time.time()

    cls = KMeans(n_clusters = k, random_state = 42, n_init = 20)
    labels = cls.fit_predict(X_scaled)

    train_time = time.time() - start

    start = time.time()
    cls.predict(X_scaled[:1])
    inf_latency = time.time() - start

    inertia = cls.inertia_
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    # Intra Cluster Dist :
    intra = 0
    for i in range(k):
        pts = X_scaled[labels == i]
        centroid = cls.cluster_centers_[i]
        intra += np.mean(np.linalg.norm(pts - centroid, axis = 1))
    intra = intra / k

    # Inter Cluster Dist :
    centroids = cls.cluster_centers_
    inter = 0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            inter += np.linalg.norm(centroids[i] - centroids[j])
            count += 1
    inter = inter / count

    res.append([
        meth, k, inertia, sil, db, ch,
        intra, inter, train_time, inf_latency
    ])

    # Visualization per method :
    plt.figure(figsize = (6, 5))
    sns.scatterplot(
        x = X_scaled[:, 0],
        y = X_scaled[:, 1],
        hue = labels,
        palette = "tab10")

    plt.scatter(
        cls.cluster_centers_[:, 0],
        cls.cluster_centers_[:, 1],
        c = "black",
        s = 200,
        marker = "X")

    plt.title(f"Clusters using {meth} K={k}")
    plt.savefig(f"clusters_{meth.lower()}.png", bbox_inches = "tight")
    plt.show()

# Comparison Table : 

results = pd.DataFrame(
    res,
    columns = [
        "Method",
        "K",
        "Inertia",
        "Silhouette",
        "Davies_Bouldin",
        "Calinski_Harabasz",
        "Avg_Intra_Dist",
        "Avg_Inter_Dist",
        "Training_Time",
        "Inference_Latency"
    ]
)

print(results)

# Metrics vs K Visualisation : 
plt.plot(k_range_vals, inertia_list)
plt.title("Elbow Curve : ")
plt.show()

plt.plot(k_range_vals, sil_list)
plt.title("Silhouette Curve : ")
plt.show()

plt.plot(k_range_vals, db_list)
plt.title("Davies Bouldin Curve : ")
plt.show()

plt.plot(k_range_vals, ch_list)
plt.title("CH Score Curve : ")
plt.show()

