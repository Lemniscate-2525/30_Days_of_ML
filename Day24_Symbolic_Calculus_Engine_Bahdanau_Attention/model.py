import sympy as sp

import time

import random

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

# Device Configuration  :
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocab Config : 
chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^()., ")
char_to_ind = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
char_to_ind.update({c : i + 4 for i, c in enumerate(chars)})
ind_to_char = {i: c for c, i in char_to_ind.items()}

pad_ind, sos_ind, eos_ind = 0, 1, 2

# Hyperparameters : 
embed_dim = 64
hidden_dim = 128
Batch_Size = 128
epochs = 10
lr = 0.002
teacher_forcing_ratio = 0.5

def gen_calc_data(num_samples = 10000):
  data = []
  x = sp.Symbol('x')

# Possible Diff Math fns :
  fns = [lambda: sp.sin(random.randint(1, 3) * x),
         lambda: sp.cos(random.randint(1, 3) * x),
         lambda: sp.tan(random.randint(1, 3) * x),
         lambda: sp.exp(random.randint(1, 3) * x),
         lambda: sp.log(random.randint(1, 3) * x + 1),
         lambda: x**random.randint(1, 5),
         lambda: random.randint(2, 9) * x
  ]

  attempts = 0
  while len(data) < num_samples and attempts < num_samples * 5:
    attempts += 1
    f1, f2 = random.choice(fns)(), random.choice(fns)()

    struc = random.choice([f1*f2, f1+f2, f1.subs(x, f2)])

    try: 
      derivative = sp.diff(struc, x)
      src_str = str(struc).replace(" ", "")
      trg_str = str(derivative).replace(" ", "")

      if 3 < len(src_str) < 45 and 3 < len(trg_str) < 65:
        data.append((src_str, trg_str))
    except Exception:
      continue
            
  return data

raw_train_data = gen_calc_data(12000)
raw_val_data = gen_calc_data(1000)

# EDA : 
src_lens = [len(s) for s, t in raw_train_data]
trg_lens = [len(t) for s, t in raw_val_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.hist(src_lens, bins = 20, color = 'steelblue', edgecolor = 'black')
ax1.set_title('Source f(x) Char Len : ')

ax2.hist(trg_lens, bins = 20, color = 'darkorange', edgecolor = 'black')
ax2.set_title("Target f'(x) Char Len: ")


plt.show() 

# Creating Synthetic Dataset : 
class CalculusData(Dataset):

    def __init__(self, data): 
      self.data = data

    def __len__(self):
      return len(self.data)

    def encode(self, text):
       return [char_to_ind.get(c, char_to_ind["<UNK>"]) for c in text]

    def __getitem__(self, i):
        src = self.encode(self.data[i][0])
        trg = [sos_ind] + self.encode(self.data[i][1]) + [eos_ind]
        return torch.tensor(src, dtype = torch.long), torch.tensor(trg, dtype = torch.long)

def collate_fn(batch):
    src, trg = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src, padding_value = pad_ind, batch_first=True)
    trg_pad = torch.nn.utils.rnn.pad_sequence(trg, padding_value = pad_ind, batch_first=True)
    return src_pad, trg_pad

train_loader = DataLoader(CalculusData(raw_train_data), batch_size = Batch_Size, shuffle = True, collate_fn = collate_fn)
val_loader = DataLoader(CalculusData(raw_val_data), batch_size = Batch_Size, shuffle = False, collate_fn = collate_fn)

# Encoder, Decoder and Bahdanau Attention : 

class Encoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding = nn.Embedding(len(char_to_ind), embed_dim, padding_idx = pad_ind)
    self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional = True, batch_first = True)
    self.fc = nn.Linear(hidden_dim*2, hidden_dim)

  def forward(self, src):
    embedded = self.embedding(src)
    outputs, hidden = self.gru(embedded)
    hidden = torch.tanh(self.fc(torch.cat((hidden[-2, : , : ], hidden[-1, :, :]), dim = 1)))
    return outputs, hidden

class BahdanauAttention(nn.Module):

  def __init__(self):
    super().__init__()
    self.w = nn.Linear(hidden_dim * 2, hidden_dim) 
    self.u = nn.Linear(hidden_dim, hidden_dim)     
    self.v = nn.Linear(hidden_dim, 1)

  def forward(self, hidden, encoder_outputs):
    seq_len = encoder_outputs.shape[1]
    hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
    energy = torch.tanh(self.w(encoder_outputs) + self.u(hidden_expanded))
    attention_scores = self.v(energy).squeeze(2) 
    return F.softmax(attention_scores, dim = 1)

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention = BahdanauAttention()
        self.embedding = nn.Embedding(len(char_to_ind), embed_dim, padding_idx = pad_ind)
        self.gru = nn.GRU(embed_dim + (hidden_dim * 2), hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim + (hidden_dim * 2) + hidden_dim, len(char_to_ind))

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1) 

        embedded = self.embedding(input) 
        a = self.attention(hidden, encoder_outputs).unsqueeze(1) 

        context = torch.bmm(a, encoder_outputs) 

        rnn_input = torch.cat((embedded, context), dim = 2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))

        pred = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim = 1))
        return pred, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
  
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = len(char_to_ind)
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(Device)
        encoder_outputs, hidden = self.encoder(src)
        
        input_token = trg[:, 0] 
        
        for t in range(1, trg_len):
            pred, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = pred
            top1 = pred.argmax(1)
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
            
        return outputs

def calc_exact_match(preds, trgs):

    pred_indices = preds.argmax(dim = -1)
    batch_size = trgs.shape[0]
    exact_matches = 0
    
    for i in range(batch_size):
        p_seq = pred_indices[i].tolist()
        t_seq = trgs[i].tolist()
        
        if eos_ind in p_seq: p_seq = p_seq[:p_seq.index(eos_ind)]
        if eos_ind in t_seq: t_seq = t_seq[:t_seq.index(eos_ind)]
            
        if p_seq == t_seq:
            exact_matches += 1
            
    return exact_matches / batch_size

# Model, Training Time, Loss per Epoch Calculation : 
model = Seq2Seq().to(Device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index = pad_ind)
scaler = torch.amp.GradScaler('cuda') if Device.type == 'cuda' else None

total_start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss, total_ema = 0, 0
    start = time.time()
    
    for src, trg in train_loader:
        src, trg = src.to(Device), trg.to(Device)
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio)
        
        # Loss Computation : 
        output_flat = output[:, 1:].reshape(-1, output.shape[-1])
        trg_flat = trg[:, 1:].reshape(-1)
        loss = criterion(output_flat, trg_flat)
        
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
        total_ema += calc_exact_match(output[:, 1:], trg[:, 1:])
        
    avg_loss = total_loss / len(train_loader)
    avg_ema = (total_ema / len(train_loader)) * 100

    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Exact Match: {avg_ema:.2f}% | Time: {time.time() - start:.2f}s")

print(f"\nTotal Training Time: {(time.time() - total_start_time) / 60:.2f} minutes")

# Inference, Example Computation, Visualization : 
model.eval()

def diff_and_plot(input_str):

    input_str = input_str.replace(" ", "")
    src = torch.tensor([[char_to_ind.get(c, char_to_ind["<UNK>"]) for c in input_str]], dtype = torch.long).to(Device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        input_token = torch.tensor([sos_ind], dtype = torch.long).to(Device)
        
        decoded_chars = []
        attentions = []
        
        for _ in range(70): 
            pred, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)
            attentions.append(attn_weights.squeeze().cpu().numpy())
            
            top1 = pred.argmax(1).item()
            if top1 == eos_ind:
                break
            decoded_chars.append(ind_to_char[top1])
            input_token = torch.tensor([top1], dtype = torch.long).to(Device)

    latency = (time.perf_counter() - start_time) * 1000
    translated_str = "".join(decoded_chars)
    
    x = sp.Symbol('x')
    try:
        true_derivative = str(sp.diff(sp.sympify(input_str), x)).replace(" ", "")
    except:
        true_derivative = "Domain Error in SymPy"

    print(f"\nInput f(x) : {input_str}")
    print(f"True f'(x) : {true_derivative}")
    print(f"Predicted f'(x) : {translated_str}")
    print("-" * 40)
    print(f"Inference Latency: {latency:.2f} ms\n")
    
    attention_matrix = np.array(attentions)
    fig, ax = plt.subplots(figsize = (8, 8))
    cax = ax.matshow(attention_matrix, cmap = 'viridis')
    
    ax.set_xticks(range(len(input_str)))
    ax.set_xticklabels(list(input_str))

    ax.set_yticks(range(len(translated_str)))
    ax.set_yticklabels(list(translated_str))
    
    plt.title(f"Attention Heatmap : {input_str}", pad = 20)
    plt.colorbar(cax)

    plt.xlabel("Source Equation (Encoder)")
    plt.ylabel("Generated Derivative (Decoder)")

    plt.show()

diff_and_plot("x**2*exp(x)")



