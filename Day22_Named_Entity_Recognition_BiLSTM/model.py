import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from collections import Counter
from datasets import load_dataset

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import time
import os

# Data Ingestion : 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("lhoestq/conll2003")
train_data = dataset['train']
val_data = dataset['validation']

# Data Preprocessing : 
tag_pad_ind = 9
ind_to_tag = {0: '0', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

# EDA : 
seq_len = [len(item['tokens']) for item in train_data]

#print("Total Seq :", len(train_data))
#print("Max Len :", max(seq_len))
#print("Avg Len :", sum(seq_len)/len(seq_len))

# Class Imbalance : 
all_tags = [tag for item  in train_data for tag in item['ner_tags']]
tag_cnts = Counter(all_tags)
#print("Class Imbalance : ")

#for tag_idx, count in tag_cnts.most_common():
#    print(f"{ind_to_tag[tag_idx]:>8} : {count}")

labels = [ind_to_tag[idx] for idx in tag_cnts.keys()]
counts = list(tag_cnts.values())

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    
#ax1.hist(seq_len, bins = 50, color = 'steelblue', edgecolor = 'black')
#ax1.set_title('Sequence Length Distribution : ')
#ax1.set_xlabel('Number of Tokens')
#ax1.set_ylabel('Frequency')

#ax2.bar(labels, counts, color = 'darkorange', edgecolor = 'black')
#ax2.set_title('NER Class Distribution : ')
#ax2.set_xlabel('BIO Tags')
#ax2.set_ylabel('Token Count')
#ax2.set_yscale('log') # Log due to O Class Dominance.
#plt.xticks(rotation = 45)

#plt.tight_layout()
#plt.show()

# Vocabulary and Dataset : 
word2ind = {"<PAD": 0, "<UNK>": 1}
ind = 2
counter = Counter([word.lower() for seq in train_data['tokens'] for word in seq])
for word, _ in counter.most_common(24998):
  word2ind[word] = ind
  ind += 1

class CoNLLData(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self): return len(self.data)

  def __getitem__(self, i):
    tokens = [w.lower() for w in self.data[i]['tokens']]
    token_ids = [word2ind.get(w, 1) for w in tokens]

    return torch.tensor(token_ids, dtype = torch.long), torch.tensor(self.data[i]['ner_tags'], dtype = torch.long)

def pad_collate(batch):
    xx, yy = zip(*batch)
    max_len = max([len(x) for x in xx])

    x_padded = torch.full((len(xx), max_len), 0, dtype = torch.long)
    y_padded = torch.full((len(yy), max_len), tag_pad_ind, dtype = torch.long)
    mask = torch.zeros((len(xx), max_len), dtype = torch.bool)

    for i, (x, y) in enumerate(zip(xx, yy)):
        x_padded[i, :len(x)] = x
        y_padded[i, :len(y)] = y
        mask[i, :len(x)] = 1 

    return x_padded, y_padded, mask

train_loader = DataLoader(CoNLLData(train_data), batch_size = 64, shuffle = True, collate_fn = pad_collate)
val_loader = DataLoader(CoNLLData(val_data), batch_size = 64, shuffle = False, collate_fn = pad_collate)

# BiLSTM Model : 
class BiLSTM_NER(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(word2ind), 100, padding_idx = 0)
        self.lstm = nn.LSTM(100, 256, num_layers = 2, bidirectional = True, batch_first = True, dropout = 0.3)
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        return self.fc(self.dropout(self.lstm(self.dropout(self.embedding(x)))[0]))

model = BiLSTM_NER().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def masked_loss(logits, targets, mask):
    loss = nn.CrossEntropyLoss(reduction = 'none')(logits.view(-1, 10), targets.view(-1))
    mask = mask.view(-1).float()
    return (loss * mask).sum() / mask.sum()

# Model Training and Loss Calculation : 
scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

train_loss_history = []
val_loss_history = []

for e in range(10) :
    model.train()
    total_train_loss = 0
    start = time.time()
    
    for x, y, mask in train_loader :
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        
        if scaler :

            with torch.amp.autocast('cuda'):
                loss = masked_loss(model(x), y, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else :
            loss = masked_loss(model(x), y, mask)
            loss.backward()
            optimizer.step()
            
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # Validation Phase and Loss/Epoch Calculation : 
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = masked_loss(model(x), y, mask)
            else:
                loss = masked_loss(model(x), y, mask)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
        
    print(f"Epoch {e + 1 : 02d}/10 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {time.time() - start:.2f}s")

# Inference Latency : 
model.eval()
text = "Apple CEO Tim Cook visited London yesterday for a conference."
tokens = [w.lower() for w in text.split()]
x = torch.tensor([[word2ind.get(w, 1) for w in tokens]], dtype = torch.long).to(DEVICE)

# Loss Curve : 
plt.figure(figsize = (8, 8))
plt.plot(range(1, 11), train_loss_history, label = 'Train Loss')
plt.plot(range(1, 11), val_loss_history, label = 'Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Masked Cross-Entropy Loss')

plt.title('BiLSTM Loss : ')

plt.legend()
plt.grid(True)
plt.show() 

