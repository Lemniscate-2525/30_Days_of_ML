import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
import time

from sklearn.preprocessing import MinMaxScaler

# Hardware : 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

# Sliding Window Dataset Generator :
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, horizon = 1):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len + self.horizon - 1]
        return X, y

# OOM-Safe SCADA Data Loading : 
print("Loading real SCADA Telemetry...")
file_path = 'B1_CL4_20.csv'

target_columns = ['GenTorqSP', 'DCC', 'DCV'] 

chunk_size = 100_000
chunks = []

try:
    for chunk in pd.read_csv(file_path, usecols = target_columns, chunksize = chunk_size):
        downsampled_chunk = chunk.iloc[::20, :]
        chunks.append(downsampled_chunk)

    df = pd.concat(chunks, ignore_index = True)

    df.ffill(inplace = True)    # MV Handling.
    df.dropna(inplace = True)

    raw_data = df.values
    print(f"Data compressed and loaded into RAM safely. Final shape: {raw_data.shape}")

except ValueError as e:
    print(f"Column Name Error. Read the metadata file and update 'target_columns'.\nSystem Error: {e}")
    print("Available columns are: ", pd.read_csv(file_path, nrows = 0).columns.tolist())

# EDA : 
print("Data Shape:")
print(f"Total Time Steps: {df.shape[0]}")
print(f"Features tracked: {df.shape[1]}")

print("\nMissing Values:")
print(df.isnull().sum()) # MV check.

print("\nStatistical Summary:")
print(df.describe()) # Stats.

# Raw Vibration vs Time : 
plt.figure(figsize = (8, 8))
plt.plot(df['GenTorqSP'].values[:5000], color = 'cyan', linewidth = 0.5)
plt.title("High-Frequency Generator Torque")
plt.show()

# Data Preprocessing : 
scaler = MinMaxScaler(feature_range = (-1, 1))
scaled_data = scaler.fit_transform(raw_data)

# Train/Test Split : 
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Hyperparameters : 
seq_len = 50
batch_size = 256
features = 3

train_dataset = TimeSeriesDataset(train_data, seq_len)
test_dataset = TimeSeriesDataset(test_data, seq_len)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# RNN Model : 
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True, nonlinearity = 'tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, h_n = self.rnn(x, h0)
  
        final_memory = out[:, -1, :]
        
        pred = self.fc(final_memory)
        return pred

model = VanillaRNN(input_dim = features, hidden_dim = 64, output_dim = features).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# BPTT: 
epochs = 10
start_time = time.time()
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        loss.backward() # BPTT
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)  # Grad Clipping to prevent exploding grad.
        
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    
    # Predictions, Loss and Validation Time :  
    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            running_test_loss += loss.item() * X_batch.size(0)
            
    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)
    
    print(f"Epoch {epoch+1}/{epochs} | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_test_loss:.5f}")

train_time = time.time() - start_time
print(f"\nTraining Complete in {train_time:.2f} seconds.")

# Inference Latency: 
model.eval()
dummy_sequence = torch.randn(1, seq_len, features).to(device)

start_inf = time.time()
with torch.no_grad():
    _ = model(dummy_sequence)
inf_latency = time.time() - start_inf

print(f"Per-sample Real-Time Prediction Latency: {inf_latency:.5f} seconds")

# Loss Curve Visualization: 
plt.style.use('dark_background')
plt.figure(figsize = (8, 8))

plt.plot(train_losses, label = 'Train Loss', color = 'cyan')
plt.plot(test_losses, label = 'Val Loss', color = 'magenta')

plt.title('RNN Training (BPTT) : ')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()
