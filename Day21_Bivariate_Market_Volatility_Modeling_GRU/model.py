import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import time
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Engine on: {device}")

# 2D CSV Data -> 3D Tensor : 
class TimeSeriesData(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.seq_len, :]  # Price, Volume.
        y = self.data[idx + self.seq_len, 0] 
        return X, y

# Data Preprocessing : 
df = pd.read_csv('all_stocks_5yr.csv')
df = df[df['Name'] == 'AAPL'].sort_values('date').reset_index(drop = True) # Isolating a single asset(Apple) for cont modelling.

# EDA : 
#print(f"AAPL Total Trading Days: {df.shape[0]}")
#plt.figure(figsize = (8, 8))
#plt.plot(df['close'].values, color = 'cyan')
#plt.title("AAPL Closing Price : ")
#plt.show()

raw_data = df[['close', 'volume']].values

# Scaling : 
scaler = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scaler.fit_transform(raw_data)

# Train/Test Split : 
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Hyperparameters : 
seq_len = 20 # Trading Month(20 days)
batch_size = 64
features = 2 # Input(Price, Volume)

train_dataset = TimeSeriesData(train_data, seq_len)
test_dataset = TimeSeriesData(test_data, seq_len)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# GRU Model : 
class GRU(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 2):
    super(GRU, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True, dropout = 0.1)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
    out, h_n = self.gru(x, h0)

    final_mem = out[:, -1, :]
    return self.fc(final_mem)

model = GRU(input_dim = features, hidden_dim = 64, output_dim = 1, num_layers = 2).to(device)
crit = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#BPTT : 
e = 30
start_time = time.time()
train_losses, val_losses = [], []

for epoch in range(e):
  model.train()
  running_loss = 0.0

  for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze() 
        loss = crit(y_pred, y_batch)
        
        loss.backward() # Backward Flow of Gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0) 
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        
  epoch_train_loss = running_loss / len(train_dataset)
  train_losses.append(epoch_train_loss)

# Model Validation & Loss Calculation : 
  model.eval()
  running_val_loss = 0.0

  with torch.no_grad():
    for X_batch, y_batch in test_loader:

      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      y_pred = model(X_batch).squeeze()

      loss = crit(y_pred, y_batch)
      running_val_loss += loss.item() * X_batch.size(0)
            
    epoch_val_loss = running_val_loss / len(test_dataset)
    val_losses.append(epoch_val_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:02d}/{e} | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f}")

print(f"\nTraining Complete in {time.time() - start_time:.2f} seconds.")

# Inference Latency : 
model.eval()
dummy_sequence = torch.randn(1, seq_len, features).to(device)

start_inf = time.time()

with torch.no_grad():
    _ = model(dummy_sequence)
inf_latency = time.time() - start_inf

print(f"Inference Latency : {inf_latency:.5f} seconds")

# Loss Visualization : 
plt.style.use('dark_background')
plt.figure(figsize = (8, 8))

plt.plot(train_losses, label = 'Train MSE', color = 'cyan')
plt.plot(val_losses, label = 'Val MSE', color = 'magenta')

plt.title('GRU Error Plot : ')

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

plt.legend()
plt.grid(True, alpha = 0.2)

plt.show()


