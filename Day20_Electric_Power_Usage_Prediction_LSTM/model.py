import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import time
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Hardware : 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Engine on: {device}")

class TimeSeriesData(Dataset):
  def __init__ (self, data, seq_len, horizon = 1):
    self.data = torch.FloatTensor(data)
    self.seq_len = seq_len
    self.horizon = horizon


  def __len__ (self):
    return len(self.data) - self.seq_len - self.horizon + 1

  def __getitem__ (self, i):
    X = self.data[i : i + self.seq_len]
    y = self.data[i + self.seq_len + self.horizon - 1]
    return X, y

# Dataset : 
fp = "household_power_consumption.txt"
df = pd.read_csv(fp, sep = ";", parse_dates = {'Datetime' : ['Date', 'Time']}, infer_datetime_format = True, low_memory = False,
na_values = ['?'], dayfirst = True)

# Chronological Sorting : 
df = df.sort_values('Datetime').reset_index(drop = True)
df.set_index('Datetime', inplace = True)

# EDA : 
#print(f"Total 1-Minute Records : {df.shape[0]}")
#print(f"Missing Values (Sensor Drops): \n{df['Global_active_power'].isnull().sum()}")

df['Global_active_power'] = df['Global_active_power'].ffill() # Missing Value Handling 
df_hourly = df[['Global_active_power']].resample('h').mean()
df_hourly.dropna(inplace = True)

raw_data = df_hourly.values
#print(f"Compressed to Hourly Records : {raw_data.shape[0]}")

# Visualization : 
#plt.figure(figsize = (8, 8))
#plt.plot(raw_data[:1000], color = 'cyan', linewidth = 0.8)
#plt.title("Hourly Global Active Power : ")
#plt.ylabel("Kilowatts")

#plt.show()

# Data Preprocessing : 
scaler = MinMaxScaler(feature_range = (-1, 1)) # Scaling. 
scaled_data = scaler.fit_transform(raw_data)

# Train/Test Split : 
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Hyperparameters : 
seq_len = 24 
batch_size = 256
features = 1

train_dataset = TimeSeriesData(train_data, seq_len)
test_dataset = TimeSeriesData(test_data, seq_len)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# LSTM Model : 
class PowerLSTM(nn.Module):
  def __init__ (self, input_dim, hidden_dim, output_dim, num_layers = 2):
    super(PowerLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, dropout = 0.2)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

    out, (h_n, c_n) = self.lstm(x, (h0, c0))
    final_mem = out[:, -1, :]
    return self.fc(final_mem)

# Instantiate Model : 
model = PowerLSTM(input_dim = features, hidden_dim = 64, output_dim = features, num_layers = 2).to(device)
crit = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#BPTT : 
e = 10
start_time = time.time()
train_losses, val_losses = [], []
val_maes = []

for epoch in range(e):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = crit(y_pred, y_batch)
        
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        
    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = crit(y_pred, y_batch)
            running_val_loss += loss.item() * X_batch.size(0)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    epoch_val_loss = running_val_loss / len(test_dataset)
    val_losses.append(epoch_val_loss)
    
    # Calculate MAE in real-world scale (Inversely transforming the -1 to 1 outputs)
    preds_unscaled = scaler.inverse_transform(np.vstack(all_preds))
    targets_unscaled = scaler.inverse_transform(np.vstack(all_targets))
    epoch_mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    val_maes.append(epoch_mae)
    
    print(f"Epoch {epoch+1:02d}/{e} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f} | Val MAE: {epoch_mae:.4f} kW")

print(f"\nTraining Complete in {time.time() - start_time:.2f} seconds.")

# Inference Latency : 
model.eval()
dummy_sequence = torch.randn(1, seq_len, features).to(device)
start_inf = time.time()

with torch.no_grad():
    _ = model(dummy_sequence)
inf_latency = time.time() - start_inf
print(f"Inference Latency (Single Sequence): {inf_latency:.5f} seconds")

# Loss Curve Visualization : 
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.plot(train_losses, label = 'Train MSE', color = 'cyan')
ax1.plot(val_losses, label = 'Val MSE', color = 'magenta')
ax1.set_title('BPTT Loss Dynamics : ')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Mean Squared Error')

ax1.legend()
ax1.grid(True, alpha = 0.2)

ax2.plot(val_maes, label = 'Val MAE (kW)', color = 'yellow')

ax2.set_title('Absolute Error in Real-World Scale :  ')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Kilowatts (kW)')

ax2.legend()
ax2.grid(True, alpha = 0.2)

plt.show()
