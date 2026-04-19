import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.config import seq_len, vocab_size, batch_size

def generate_tick_data(num_samples = 5000):
  
    X = np.random.randint(0, vocab_size, (num_samples, seq_len))
    y = np.zeros(num_samples)
    
    crash_indices = np.random.choice(num_samples, size = int(num_samples * 0.1), replace = False)
    for idx in crash_indices:
        X[idx, 50:70] = np.random.randint(90, 100, 20) 
        X[idx, 70:90] = np.random.randint(10, 15, 20)
        y[idx] = 1
        
    return torch.tensor(X, dtype = torch.long), torch.tensor(y, dtype = torch.float32)

class TickDataset(Dataset):
  
    def __init__(self, X, y):
        self.X = X
        self.y = y
      
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_dataloaders():
  
    X_data, y_data = generate_tick_data()
    split = int(len(X_data) * 0.8)
    
    train_loader = DataLoader(TickDataset(X_data[:split], y_data[:split]), batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(TickDataset(X_data[split:], y_data[split:]), batch_size = batch_size)
  
    return train_loader, test_loader
