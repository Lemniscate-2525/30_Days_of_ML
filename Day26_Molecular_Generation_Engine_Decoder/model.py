import math

import time

import os

import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import urllib.request

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Device Setup and VRAM Config : 
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_ind, sos_ind, eos_ind, unk_ind = 0,1,2,3

max_seq_len = 60

batch_size = 128
epochs = 15
lr = 3e-4

d_model = 128
num_heads = 8
num_layers = 4
ffnn_dim = 512

# Data Extraction : 
def fetch_and_prep(num_samples = 25000):

  url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
  f_p = "qm9.csv"
    
  if not os.path.exists(f_p):
    urllib.request.urlretrieve(url, f_p)

  df = pd.read_csv(f_p)
  raw_smiles = df['smiles'].dropna().tolist()
    
  valid_smiles = [s for s in raw_smiles if len(s) < max_seq_len - 2]
  np.random.shuffle(valid_smiles)
  dataset = valid_smiles[:num_samples]

  # Vocabulary Building : 
  unique_chars = set()
  for s in dataset :
      unique_chars.update(list(s))
        
  chars = sorted(list(unique_chars))
  char_to_idx = {"<PAD>": pad_ind, "<SOS>": sos_ind, "<EOS>": eos_ind, "<UNK>": unk_ind}

  char_to_idx.update({c : i + 4 for i, c in enumerate(chars)})
  idx_to_char = {i: c for c, i in char_to_idx.items()}

  return dataset, char_to_idx, idx_to_char

train_raw, char_to_ind, ind_to_char = fetch_and_prep(25000)
vocab_size = len(char_to_ind)

# EDA : 
seq_lens = [len(s) for s in train_raw]

all_chars = "".join(train_raw)
char_counts = Counter(all_chars)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
ax1.hist(seq_lens, bins = 20, color = 'mediumseagreen', edgecolor = 'black')
ax1.set_title('Sequence Length Distribution : ')

ax1.set_xlabel('Length (Characters)')
ax1.set_ylabel('Frequency')

labels, values = zip(*char_counts.most_common(15))
ax2.bar(labels, values, color = 'coral', edgecolor = 'black')
ax2.set_title('Top 15 Character Frequencies : ')

ax2.set_xlabel('SMILES Token')

plt.tight_layout()
plt.show()

# DataLoader : 
class SmilesDataset(Dataset):

    def __init__(self, data, char_to_idx): 
        self.data = data
        self.char_to_idx = char_to_idx

    def __len__(self) : return len(self.data)

    def encode(self, text) : return [self.char_to_idx.get(c, unk_ind) for c in text]

    def __getitem__(self, i) :
        encoded = self.encode(self.data[i])
        src = [sos_ind] + encoded
        trg = encoded + [eos_ind]

        return torch.tensor(src, dtype = torch.long), torch.tensor(trg, dtype = torch.long)

def collate_fn(batch):
    src, trg = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src, padding_value = pad_ind, batch_first = True)
    trg_pad = nn.utils.rnn.pad_sequence(trg, padding_value = pad_ind, batch_first = True)

    return src_pad, trg_pad

train_loader = DataLoader(SmilesDataset(train_raw, char_to_ind), batch_size = batch_size, shuffle = True, collate_fn = collate_fn)

# Decoder Architecture : 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = max_seq_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model) 
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask = None):

        batch_size = x.size(0)
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)
        
        Q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim = -1)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(context)

class GPTBlock(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffnn_dim), nn.GELU(), nn.Linear(ffnn_dim, d_model))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask = mask)
        x = x + self.ffn(self.ln2(x))
        return x

class ChemGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_ind)
        self.pos_encoder = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([GPTBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size, bias = False)
        self.fc_out.weight = self.embedding.weight

    def create_causal_mask(self, size, device):
        mask = torch.tril(torch.ones((size, size), device = device)).bool()
        return mask.unsqueeze(0).unsqueeze(0) 

    def forward(self, x):
        mask = self.create_causal_mask(x.size(1), x.device)
        x = self.pos_encoder(self.embedding(x))

        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)

        return self.fc_out(x)

# Training Engine & Metrics : 
model = ChemGPT(vocab_size).to(Device)

optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index = pad_ind)

epoch_losses = []
epoch_accs = []

total_start_time = time.time()

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    start = time.time()
    
    for src, trg in train_loader:
        src, trg = src.to(Device), trg.to(Device)
        
        optimizer.zero_grad()
        output = model(src)
        
        loss = criterion(output.reshape(-1, output.shape[-1]), trg.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        total_loss += loss.item()
        
        preds = output.argmax(dim = -1)
        mask = (trg != pad_ind)

        correct_tokens += (preds == trg).masked_fill(~mask, False).sum().item()
        total_tokens += mask.sum().item()
        
    avg_loss = total_loss / len(train_loader)
    acc = (correct_tokens / total_tokens) * 100

    epoch_losses.append(avg_loss)
    epoch_accs.append(acc)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time() - start:.2f}s")

total_time = (time.time() - total_start_time) / 60

print("="*50)
print(f"Total Training Latency: {total_time:.2f} Minutes")
print("="*50)

# Loss and Accuracy Visualization : 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.plot(range(1, epochs+1), epoch_losses, marker = 'o', color = 'crimson')
ax1.set_title('Loss vs. Epoch : ')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross Entropy Loss')

ax1.grid(True, linestyle = '--', alpha = 0.6)

ax2.plot(range(1, epochs+1), epoch_accs, marker = 'o', color = 'dodgerblue')
ax2.set_title('Accuracy vs. Epoch : ')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

ax2.grid(True, linestyle = '--', alpha = 0.6)

plt.tight_layout()
plt.show()

# Inference : 
model.eval()

def generate_molecules(model, num_mols = 3, temperature = 1.0):

    generated = []
    latencies = []
    
    with torch.no_grad():

        for _ in range(num_mols):
            start_inf = time.perf_counter()
            input_ids = torch.tensor([[sos_ind]], dtype = torch.long).to(Device)
            
            for _ in range(max_seq_len):
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature

                probs = F.softmax(next_token_logits, dim = -1)
                next_token = torch.multinomial(probs, num_samples = 1).item()
                
                if next_token == eos_ind: break
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype = torch.long).to(Device)], dim = 1)
                
            latencies.append((time.perf_counter() - start_inf) * 1000)
            tokens = input_ids[0].cpu().tolist()

            mol_str = "".join([ind_to_char.get(idx, "?") for idx in tokens[1:]])
            generated.append(mol_str)
            
    avg_latency = sum(latencies) / len(latencies)
    return generated, avg_latency

print("\nHigh Probability / Safe Structures : ")
mols, lat = generate_molecules(model, temperature = 0.5)
for m in mols: print(f"-> {m}")

print(f"Avg Inference Latency: {lat:.2f} ms/mol")

print("\nHigh Variance / Novel Hallucinations :")
mols, lat = generate_molecules(model, temperature = 1.2)
for m in mols: print(f"-> {m}")

print(f"Avg Inference Latency: {lat:.2f} ms/mol")

