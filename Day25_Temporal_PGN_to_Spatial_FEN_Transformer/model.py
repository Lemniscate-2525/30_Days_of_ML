from re import A
import time

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import chess
import chess.pgn

import random

# Device Config & VRAM Optimization : 
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chars = list("1234567890.abcdefgh PNRBQKpnrqk/xO-+=#")
char_to_ind = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

char_to_ind.update({c: i+4 for i, c in enumerate(chars)})
ind_to_char = {i: c for c, i in char_to_ind.items()}

pad_ind, SOS_IDX, EOS_IDX = 0, 1, 2

d_model = 64
num_heads= 8
FFNN_dim = 256
num_layers = 2
max_seq_len = 150 

Batch_size = 32     # VRAM Protected
epochs = 15
lr = 1e-4           # Stabilized for Attention Dynamics

# Constrained Real-World Dataset : 
def generate_chess_dataset(num_samples = 10000, max_moves = 20):
    dataset = []
    attempts = 0

    while len(dataset) < num_samples and attempts < num_samples * 3:

        attempts += 1
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        
        stop_ply = random.randint(1, max_moves * 2) 
        
        for _ in range(stop_ply):

            legal_moves = list(board.legal_moves)
            if not legal_moves: break 

            move = random.choice(legal_moves[:5]) 
            board.push(move)
            node = node.add_variation(move)
            
        exporter = chess.pgn.StringExporter(headers = False, variations = False, comments = False)
        pgn_string = game.accept(exporter).strip()
        fen_string = board.fen().split(' ')[0] 
        
        if len(pgn_string) < max_seq_len - 2 and len(fen_string) < max_seq_len - 2:
            dataset.append((pgn_string, fen_string))
            
    return dataset

# Train/Val Data : 
train_raw = generate_chess_dataset(8000)
val_raw = generate_chess_dataset(500)

class ChessData(Dataset):

    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)

    def encode(self, text): return [char_to_ind.get(c, char_to_ind["<UNK>"]) for c in text]

    def __getitem__(self, i):
        src = self.encode(self.data[i][0])
        trg = [SOS_IDX] + self.encode(self.data[i][1]) + [EOS_IDX]
        return torch.tensor(src, dtype = torch.long), torch.tensor(trg, dtype = torch.long)

def collate_fn(batch):

    src, trg = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src, padding_value = pad_ind, batch_first = True)
    trg_pad = nn.utils.rnn.pad_sequence(trg, padding_value = pad_ind, batch_first = True)
    return src_pad, trg_pad

train_loader = DataLoader(ChessData(train_raw), batch_size = Batch_size, shuffle = True, collate_fn = collate_fn)

# EDA : 
pgn_lens = [len(p) for p, f in train_raw]
fen_lens = [len(f) for p, f in train_raw]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8,8))

# PGN Dist : 
ax1.hist(pgn_lens, bins = 20, color = 'steelblue', edgecolor = 'black')
ax1.set_title('PGN Sequence Lengths : ')
ax1.set_xlabel('Characters')
ax1.set_ylabel('Frequency')

# FEN Dist : 
ax2.hist(fen_lens, bins = 20, color = 'darkorange', edgecolor = 'black')
ax2.set_title('FEN Sequence Lengths : ')
ax2.set_xlabel('Characters')

plt.tight_layout()
plt.show()

# Model Architecture : 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = max_seq_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim = -1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(context), attention_weights

class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, FFNN_dim), nn.ReLU(), nn.Linear(FFNN_dim, d_model))
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, trg_mask, src_mask):
        # 1. Masked Self-Attention : 
        self_attn_out, _ = self.self_attn(x, x, x, mask = trg_mask)
        x = self.norm1(x + self_attn_out)
        
        # 2. Cross-Attention : 
        cross_attn_out, attn_weights = self.cross_attn(q = x, k = enc_out, v = enc_out, mask = src_mask)
        x = self.norm2(x + cross_attn_out) 
        
        # 3. FFNN : 
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x, attn_weights

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(char_to_ind), d_model, padding_idx = pad_ind)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, len(char_to_ind))

    def create_pad_mask(self, matrix):
        return (matrix != pad_ind).unsqueeze(1).unsqueeze(2) # [B, 1, 1, SeqLen]

    def create_causal_mask(self, size, device):
        mask = torch.tril(torch.ones((size, size), device = device)).bool()
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, SeqLen, SeqLen]

    def forward(self, src, trg):
        # Masks
        src_mask = self.create_pad_mask(src)
        trg_pad_mask = self.create_pad_mask(trg)
        trg_causal_mask = self.create_causal_mask(trg.size(1), trg.device)
        trg_mask = trg_pad_mask & trg_causal_mask # Combine padding and look-ahead
        
        # Encoding : 
        enc_out = self.pos_encoder(self.embedding(src))
        dec_out = self.pos_encoder(self.embedding(trg))
        
        for layer in self.decoder_layers:
            dec_out, _ = layer(dec_out, enc_out, trg_mask, src_mask)
            
        return self.fc_out(dec_out)

# Model Training, Loss/Epoch : 
model = Transformer().to(Device)

optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index = pad_ind)

total_start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    start = time.time()

    for src, trg in train_loader:
        src, trg = src.to(Device), trg.to(Device)
        
        trg_input = trg[:, :-1]
        trg_target = trg[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, trg_input)
        
        loss = criterion(output.reshape(-1, output.shape[-1]), trg_target.reshape(-1))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Time: {time.time() - start:.2f}s")

# Output : 
total_time = (time.time() - total_start_time) / 60
print("="*50)
print(f"Total Training Latency: {total_time:.2f} Minutes")
print("="*50)

# Averaged-Head Inference & Visualization : 
model.eval()

def generate_fen_and_plot(pgn_str):
    src = torch.tensor([[char_to_ind.get(c, char_to_ind["<UNK>"]) for c in pgn_str]], dtype = torch.long).to(Device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        src_mask = model.create_pad_mask(src)
        enc_out = model.pos_encoder(model.embedding(src))
        
        trg_indices = [SOS_IDX]
        attention_matrices = []

        for _ in range(80): 
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(Device)
            trg_mask = model.create_causal_mask(trg_tensor.size(1), Device)
            
            dec_out = model.pos_encoder(model.embedding(trg_tensor))
            
            for layer in model.decoder_layers:
                dec_out, attn_weights = layer(dec_out, enc_out, trg_mask, src_mask)
            
            # Aggregating Attention across all 8 heads : 
            attn_avg = attn_weights.mean(dim = 1) 
            attention_matrices.append(attn_avg[0, -1, :].cpu().numpy())
            
            pred = model.fc_out(dec_out[:, -1, :])
            next_token = pred.argmax(1).item()
            
            if next_token == EOS_IDX: break
            trg_indices.append(next_token)

    latency = (time.perf_counter() - start_time) * 1000
    fen_output = "".join([ind_to_char[idx] for idx in trg_indices[1:]])
    
    print(f"\nInput PGN : {pgn_str}")
    print(f"Output FEN  : {fen_output}")
    print(f"Latency     : {latency:.2f} ms")
    
    attn_matrix = np.array(attention_matrices)
    fig, ax = plt.subplots(figsize = (8, 8))
    cax = ax.matshow(attn_matrix, cmap = 'viridis')

    ax.set_xticks(range(len(pgn_str)))
    ax.set_xticklabels(list(pgn_str))
    ax.set_yticks(range(len(fen_output)))
    ax.set_yticklabels([c if i % 5 == 0 else "" for i, c in enumerate(list(fen_output))])

    plt.title(f"Averaged Multi-Head Attention Map\n{pgn_str}", pad = 20)
    plt.colorbar(cax)
    plt.xlabel("Encoder")
    plt.ylabel("Decoder")

    plt.show()

# Sample Inference : 
sample_pgn = "1. e4 e5 2. Nf3 Nc6"
generate_fen_and_plot(sample_pgn)
