import tensorflow as tf

import time as time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

from tensorflow.keras.datasets import mnist

# Dataset : 
(X_train, _), (X_test, _) = mnist.load_data()

# EDA : 
plt.imshow(X_train[0], cmap = "gray")
plt.title("Sample Digit : ")
plt.show()

# Data Preprocessing :
X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Autoencoder Architechture : 
input_dim = 784
latent_dim = 32

input_layer = layers.Input(shape = (input_dim,))

# Encoder : 
encoder = layers.Dense(256, activation = "relu")(input_layer)
encoder = layers.Dense(64, activation = "relu")(encoder)
latent  = layers.Dense(latent_dim)(encoder)

# Decoder : 
decoder = layers.Dense(64, activation = "relu")(latent)
decoder = layers.Dense(256, activation = "relu")(decoder)
output_layer = layers.Dense(784, activation = "sigmoid")(decoder)

autoencoder = models.Model(inputs = input_layer, outputs = output_layer)

autoencoder.compile(optimizer = "adam", loss = "mse")

# Model Training : 
start_train = time.time()

autoencoder.fit(X_train, X_train, epochs = 10, batch_size = 256, shuffle = True, validation_data = (X_test, X_test), verbose = 1)

# Loss Curve Visualization : 
history = autoencoder.fit(X_train, X_train, epochs = 10, batch_size = 256, shuffle = True, validation_data = (X_test, X_test), verbose = 1)

plt.figure(figsize = (6, 6))
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"], label = "Val Loss")

plt.title("Autoencoder Training Loss Curve : ")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()

plt.show()

train_time = time.time() - start_train


# Reconstruction : 
start_inf = time.time()
recon = autoencoder.predict(X_test)
inf_lat = (time.time() - start_inf) / len(X_test)

ae_mse = mean_squared_error(X_test, recon)

# Comparison with PCA : 
pca = PCA(n_components = 32)

start = time.time()
pca.fit(X_train)
pca_train_time = time.time() - start

start = time.time()
pca_recon = pca.inverse_transform(pca.transform(X_test))
pca_inf_lat = (time.time() - start) / len(X_test)

pca_mse = mean_squared_error(X_test, pca_recon)

# Comparison : 
res = pd.DataFrame({
"Model": ["Autoencoder", "PCA"],
"Reconstruction_MSE ": [ae_mse, pca_mse],
"Training_Time ": [train_time, pca_train_time],
"Inference_Latency ": [inf_lat, pca_inf_lat]
})

print(res)

# Visualization : 
plt.figure(figsize = (8, 8))

for i in range(5):
  plt.subplot(2,5,i+1)
  plt.imshow(X_test[i].reshape(28, 28), cmap = "gray")
  plt.axis("off")


  plt.subplot(2,5, i+6)
  plt.imshow(recon[i].reshape(28, 28), cmap = "gray")
  plt.axis("off")

plt.suptitle("Original vs Autoencoder Reconstruction : ")
plt.show()
