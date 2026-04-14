import time

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# Setup and Config : 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Deploying Spectral ViT Engine on: {device}")

batch_size = 64
epochs = 8
lr = 5e-4

img_size = 64     
patch_size = 8    
d_model = 128

num_heads = 4
num_layers = 3
num_classes = 2

# Spectral Dataset Generation (FFT) : 
def generate_spectral_dataset(num_samples = 1500):
    spectra, labels = [], []
    
    for _ in range(num_samples):

        x = np.linspace(-5, 5, img_size)
        y = np.linspace(-5, 5, img_size)

        X, Y = np.meshgrid(x, y)
        base_img = np.exp(-(X**2 + Y**2) / 10) + np.random.normal(0, 0.1, (img_size, img_size))
        
        is_fake = np.random.rand() > 0.5

        if is_fake:
            grid = np.sin(X * 15) * np.sin(Y * 15) * 0.5
            img = base_img + grid
            labels.append(1)

        else:
            img = base_img
            labels.append(0)
            
        fft_complex = np.fft.fft2(img)
        fft_shifted = np.fft.fftshift(fft_complex)

        magnitude = np.log(np.abs(fft_shifted) + 1e-8)
        
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        spectra.append(magnitude)
        
    return np.array(spectra, dtype = np.float32), np.array(labels, dtype = np.int64)

X_data, y_data = generate_spectral_dataset()

# EDA : 
real_idx = np.where(y_data == 0)[0][0]
fake_idx = np.where(y_data == 1)[0][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
ax1.imshow(X_data[real_idx], cmap = 'magma')

ax1.set_title('Real Image FFT : ')
ax1.axis('off')

ax2.imshow(X_data[fake_idx], cmap = 'magma')

ax2.set_title('Deepfake FFT : ')
ax2.axis('off')

plt.tight_layout()
plt.show()

class SpectralDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1) 
        self.y = torch.tensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

split = int(len(X_data) * 0.8)

train_loader = DataLoader(SpectralDataset(X_data[:split], y_data[:split]), batch_size = batch_size, shuffle = True)
test_loader = DataLoader(SpectralDataset(X_data[split:], y_data[split:]), batch_size = batch_size)

# ViT Architecture : 
class SpectralViT(nn.Module):

    def __init__(self, img_dim, patch_dim, hidden_dim, heads, layers, classes):
        super().__init__()

        self.patch_dim = patch_dim
        self.num_patches = (img_dim // patch_dim) ** 2
        
        self.patch_embed = nn.Linear(patch_dim * patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = heads, dim_feedforward = hidden_dim*4, batch_first = True, dropout = 0.1)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    def forward(self, x):
        B = x.shape[0]
        
        x = F.unfold(x, kernel_size = self.patch_dim, stride = self.patch_dim).transpose(1, 2)
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# Model Training and Loss Calculation : 
model = SpectralViT(img_size, patch_size, d_model, num_heads, num_layers, num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
criterion = nn.CrossEntropyLoss()

epoch_losses, epoch_accs = [], []
total_start_time = time.time()

for e in range(epochs):

    model.train()
    total_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    
    for X_batch, y_batch in train_loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim = 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    avg_loss = total_loss / len(train_loader)
    acc = (correct / total) * 100
    epoch_losses.append(avg_loss)
    epoch_accs.append(acc)
    
    print(f"Epoch {e+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time() - epoch_start:.2f}s")

print("="*50)

print(f"[PROFILER] Total Training Time: {(time.time() - total_start_time):.2f} Seconds")
print("="*50)

# Visualizations : 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
ax1.plot(range(1, epochs + 1), epoch_losses, marker = 'o', color = 'crimson')

ax1.set_title('ViT Cross Entropy Loss : ')

ax1.set_xlabel('Epoch')
ax1.grid(True, linestyle = '--', alpha = 0.6)

ax2.plot(range(1, epochs + 1), epoch_accs, marker = 'o', color = 'dodgerblue')

ax2.set_title('Artifact Detection Accuracy : ')

ax2.set_xlabel('Epoch')
ax2.grid(True, linestyle = '--', alpha = 0.6)

plt.tight_layout()
plt.show()

# Evaluation Metrics & Inference Latency : 
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim = 1)[:, 1].cpu().numpy() 
        preds = logits.argmax(dim = 1).cpu().numpy()
        
        all_probs.extend(probs)
        all_preds.extend(preds)

        all_labels.extend(y_batch.numpy())

f1 = f1_score(all_labels, all_preds)
auroc = roc_auc_score(all_labels, all_probs)

print(f"[METRICS] F1-Score: {f1:.4f}")
print(f"[METRICS] AUROC:    {auroc:.4f}")

print("\n" + classification_report(all_labels, all_preds, target_names=["Real (0)", "Deepfake (1)"]))

def scan_for_artifacts(img_tensor):

    start_inf = time.perf_counter()

    with torch.no_grad():

        logits = model(img_tensor.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim = 1)[0]
        pred = logits.argmax(1).item()
      
    latency = (time.perf_counter() - start_inf) * 1000
    
    status = "🚨 DEEPFAKE DETECTED" if pred == 1 else "✅ Authentic Image"
    print(f"Result: {status} | Confidence: {prob[pred].item()*100:.1f}% | Latency: {latency:.2f} ms")

sample_real, _ = test_loader.dataset[0]
sample_fake, _ = test_loader.dataset[-1]

scan_for_artifacts(sample_real)
scan_for_artifacts(sample_fake)
