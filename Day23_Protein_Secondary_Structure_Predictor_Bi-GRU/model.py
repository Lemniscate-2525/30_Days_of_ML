import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import time

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Amino Acid List: 
Amino_Acids_List = list("ACDEFGHIJKLMNPQRSTUVWY")
aa_to_ind = {"<PAD>": 0, "<UNK>": 1}
aa_to_ind.update({aa: i+2 for i , aa in enumerate(Amino_Acids_List)})

# Protein Structure Encoding : 
ind_to_struc = {0 : "H", 1: "E", 2: "C"}

# Hyperparameters : 
Pad_Label = -100
embed_dim = 64
hidden_dim = 128
batch_size = 32
epochs = 15
lr = 0.002

# Gen Synthetic Dataset : 
def generate_synthetic_proteins(num_samples = 2000):
  data = []
  for _ in range(num_samples):
    len = np.random.randint(50, 400)
    seq = [np.random.choice(Amino_Acids_List) for _ in range(len)]

    labels = []
    curr_state = np.random.choice([0, 1, 2])
    for _ in range(len):
      if np.random.rand() > 0.85:
        curr_state = np.random.choice([0, 1, 2])
      labels.append(curr_state)

    data.append({"seq" : seq, "labels" : labels})
  return data

train_data = generate_synthetic_proteins(3000)
val_data = generate_synthetic_proteins(500)

# EDA : 
seq_lengths = [len(item['seq']) for item in train_data]
print(f"Total Sequences : {len(train_data)}")
print(f"Max Length      : {max(seq_lengths)}")
print(f"Avg Length      : {sum(seq_lengths)/len(seq_lengths):.2f}\n")

all_labels = [label for item in train_data for label in item['labels']]
label_counts = Counter(all_labels)
print("Structural Class Distribution (H/E/C):")
for lbl_idx, count in label_counts.most_common():
    print(f"{ind_to_struc[lbl_idx]:>8} : {count}")

labels_plot = [ind_to_struc[idx] for idx in label_counts.keys()]
counts_plot = list(label_counts.values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

ax1.hist(seq_lengths, bins = 50, color = 'steelblue', edgecolor = 'black')
ax1.set_title('Protein Lengths : ')

ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Frequency')

ax2.bar(labels_plot, counts_plot, color = 'darkorange', edgecolor = 'black')

ax2.set_title('Class Distribution : ')

ax2.set_xlabel('Structure')
ax2.set_ylabel('Count')

plt.show()

# Dataset : 
class ProteinData(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        seq = [aa_to_ind.get(aa, 1) for aa in self.data[i]["seq"]]
        return torch.tensor(seq, dtype = torch.long), torch.tensor(self.data[i]["labels"], dtype = torch.long)

def protein_collate(batch):
    xx, yy = zip(*batch)
    max_len = max([len(x) for x in xx])
    
    x_padded = torch.full((len(xx), max_len), 0, dtype = torch.long)
    y_padded = torch.full((len(yy), max_len), Pad_Label, dtype = torch.long)
    mask = torch.zeros((len(xx), max_len), dtype = torch.bool)
    
    for i, (x, y) in enumerate(zip(xx, yy)):
        x_padded[i, :len(x)] = x
        y_padded[i, :len(y)] = y
        mask[i, : len(x)] = 1 
    return x_padded, y_padded, mask

train_loader = DataLoader(ProteinData(train_data), batch_size = batch_size, shuffle = True, collate_fn = protein_collate)
val_loader = DataLoader(ProteinData(val_data), batch_size = batch_size, shuffle = False, collate_fn = protein_collate)

# Bi-GRU Model : 
class BiGRU(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(len(aa_to_ind), embed_dim, padding_idx = 0)
    self.gru = nn.GRU(embed_dim, hidden_dim, num_layers = 2, bidirectional = True, batch_first = True, dropout = 0.3)
    self.fc = nn.Linear(hidden_dim*2 , 3)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x):
    embedded = self.dropout(self.embedding(x))
    gru_out, _ = self.gru(embedded)
    logits = self.fc(self.dropout(gru_out))
    return logits

# Optimizer and Loss fn. 
model = BiGRU().to(Device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

loss_fn = nn.CrossEntropyLoss(ignore_index = Pad_Label) # ignore_index handles Pad_Label internally, thus no MCE.

# Q3 Accuracy Metric : 
def q3_acc(logits, targets, mask):
    preds = torch.argmax(logits, dim = -1)
    correct = ((preds == targets) * mask).sum().item()
    total = mask.sum().item()
    return correct / total

scaler = torch.amp.GradScaler('cuda') if Device.type == 'cuda' else None

train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []
total_start_time = time.time()

for e in range(epochs):
    model.train()
    total_loss, total_acc = 0, 0
    start = time.time()
    
    for x, y, mask in train_loader:
        x, y, mask = x.to(Device), y.to(Device), mask.to(Device)
        optimizer.zero_grad()
        
        if scaler:

            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(x).view(-1, 3), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:

            loss = loss_fn(model(x).view(-1, 3), y.view(-1))
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        total_acc += q3_acc(model(x), y, mask)
        
    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)

    train_loss_hist.append(avg_train_loss)
    train_acc_hist.append(avg_train_acc)

    # Validation, Loss per Epoch  : 
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(Device), y.to(Device), mask.to(Device)

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    v_loss = loss_fn(logits.view(-1, 3), y.view(-1))
            else:
                logits = model(x)
                v_loss = loss_fn(logits.view(-1, 3), y.view(-1))
                
            val_loss += v_loss.item()
            val_acc += q3_acc(logits, y, mask)
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    val_loss_hist.append(avg_val_loss)
    val_acc_hist.append(avg_val_acc)
        
    print(f"Epoch {e+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} (Q3: {avg_train_acc:.2f}) | Val Loss: {avg_val_loss:.4f} (Q3: {avg_val_acc:.2f}) | Time: {time.time()-start:.2f}s")

print(f"\nTotal Training Time: {(time.time() - total_start_time)} s")

# Visualization(Loss and Accuracy Curves) : 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.plot(train_loss_hist, label = 'Train Loss')
ax1.plot(val_loss_hist, label = 'Val Loss')

ax1.set_title("Cross Entropy Loss")
ax1.legend()

ax2.plot(train_acc_hist, label = 'Train Q3 Acc', color = 'green')
ax2.plot(val_acc_hist, label = 'Val Q3 Acc', color = 'orange')

ax2.set_title("Q3 Structural Accuracy")
ax2.legend()

plt.show()

# Predictions : 
model.eval()
sample_prot_eg = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSA"
x_infer = torch.tensor([[aa_to_ind.get(aa, 1) for aa in sample_prot_eg]], dtype = torch.long).to(Device)

with torch.no_grad():
    preds = torch.argmax(model(x_infer), dim = -1).squeeze().tolist()

pred_struc = "".join([ind_to_struc[p] for p in preds])

print(f"Amino Acid Sequence :\n{sample_prot_eg}")
print(f"Predicted Structure :\n{pred_struc}")

# Inference Latency : 
start_time = time.perf_counter()
with torch.no_grad():
    preds = torch.argmax(model(x_infer), dim=-1).squeeze().tolist()
latency = (time.perf_counter() - start_time) * 1000 

predicted_structure = "".join([IDX_TO_STRUC[p] for p in preds])
print(f"Inference Latency   : {latency:.2f} ms")
