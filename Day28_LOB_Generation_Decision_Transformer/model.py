import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ==========================================
# 1. Global Setup & Config
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Deploying LOB-GPT on: {device}")

batch_size = 64
epochs = 15
lr = 1e-3
context_len = 20  

d_model = 128
num_heads = 4
num_layers = 3

state_dim = 4    
act_dim = 3      

# ==========================================
# 2. Synthetic Market Microstructure Engine
# ==========================================
print("\n--- Phase 1: Generating & Profiling LOB Trajectories ---")

def generate_market_trajectories(num_episodes=2000, max_steps=50):
    trajectories = []
    
    for _ in range(num_episodes):
        states, actions, rewards = [], [], []
        mid_price = 100.0
        
        for t in range(max_steps):
            spread = np.clip(np.random.normal(0.05, 0.01), 0.01, 0.20)
            bid_vol = np.random.randint(10, 500)
            ask_vol = np.random.randint(10, 500)
            mid_price += np.random.normal(0, 0.1) 
            
            state = [spread, mid_price, bid_vol, ask_vol]
            
            if spread < 0.04 and bid_vol > ask_vol * 1.5:
                action = 1 
                reward = np.random.normal(0.5, 0.1) 
            elif spread < 0.04 and ask_vol > bid_vol * 1.5:
                action = 2 
                reward = np.random.normal(0.5, 0.1) 
            else:
                action = 0 
                reward = -0.01 
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
        rtg = np.zeros_like(rewards)
        discount = 1.0
        for i in reversed(range(len(rewards))):
            rtg[i] = rewards[i] + (rtg[i+1] if i+1 < len(rewards) else 0) * discount
            
        trajectories.append({
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int64),
            'rtg': np.array(rtg, dtype=np.float32)
        })
    return trajectories

trajectories = generate_market_trajectories()

# --- EDA Telemetry ---
print("[SYSTEM] Rendering EDA distributions...")
all_actions = np.concatenate([t['actions'] for t in trajectories])
all_rtgs = np.concatenate([t['rtg'] for t in trajectories])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

act_counts = Counter(all_actions)
ax1.bar(['Hold (0)', 'Buy (1)', 'Sell (2)'], [act_counts[0], act_counts[1], act_counts[2]], color=['gray', 'green', 'red'], edgecolor='black')
ax1.set_title('Distribution of Executed Actions')
ax1.set_ylabel('Frequency')

ax2.hist(all_rtgs, bins = 30, color = 'dodgerblue', edgecolor = 'black')
ax2.set_title('RTG Distribution : ')
ax2.set_xlabel('Cumulative Future Reward')

plt.tight_layout()
plt.show()

# Normalize States : 
all_states = np.concatenate([t['states'] for t in trajectories])
state_mean, state_std = all_states.mean(0), all_states.std(0) + 1e-6

for t in trajectories:
    t['states'] = (t['states'] - state_mean) / state_std

class DecisionTransformerDataset(Dataset):
    def __init__(self, trajectories, k_len=context_len):
        self.trajectories = trajectories
        self.k_len = k_len

    def __len__(self): return len(self.trajectories) * 10
    
    def __getitem__(self, idx):
        traj = self.trajectories[np.random.randint(len(self.trajectories))]
        seq_len = len(traj['states'])
        
        start_idx = np.random.randint(0, seq_len - self.k_len)
        end_idx = start_idx + self.k_len
        
        s = torch.tensor(traj['states'][start_idx:end_idx])
        a = torch.tensor(traj['actions'][start_idx:end_idx])
        r = torch.tensor(traj['rtg'][start_idx:end_idx]).unsqueeze(-1)
        
        return s, a, r

train_loader = DataLoader(DecisionTransformerDataset(trajectories), batch_size = batch_size, shuffle = True)

# DT Architecture : 
class LOBGPT(nn.Module):
    def __init__(self, state_dim_in, act_dim_in, hidden_dim, heads, layers, max_timestep = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embed_state = nn.Linear(state_dim_in, hidden_dim)
        self.embed_action = nn.Embedding(act_dim_in, hidden_dim)
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_dim, nhead = heads, dim_feedforward = hidden_dim*4, 
            batch_first = True, dropout = 0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = layers)
        
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim_in)
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        B, T, _ = states.shape
        
        state_emb = self.embed_state(states) + self.embed_timestep(timesteps)
        action_emb = self.embed_action(actions) + self.embed_timestep(timesteps)
        rtg_emb = self.embed_rtg(returns_to_go) + self.embed_timestep(timesteps)
        
        stacked_inputs = torch.stack((rtg_emb, state_emb, action_emb), dim = 1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(B, 3*T, self.hidden_dim)
        
        mask = nn.Transformer.generate_square_subsequent_mask(3*T).to(device)
        x = self.transformer(stacked_inputs, mask = mask, is_causal = True)
        
        x = x.reshape(B, T, 3, self.hidden_dim)
        state_preds = x[:, :, 1] 
        
        return self.predict_action(state_preds)

# Model Training andOptimization : 
model = LOBGPT(state_dim, act_dim, d_model, num_heads, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
criterion = nn.CrossEntropyLoss()

epoch_losses = []
epoch_accs = []
total_start_time = time.time()

for epoch in range(epochs):
    
    model.train()
    total_loss, correct, total_preds = 0, 0, 0
    epoch_start = time.time()
    
    for s, a, r in train_loader:
        
        s, a, r = s.to(device), a.to(device), r.to(device)
        timesteps = torch.arange(context_len, device = device).unsqueeze(0).repeat(s.size(0), 1)
        
        optimizer.zero_grad()
        
        a_input = torch.cat([torch.zeros((a.size(0), 1), dtype = torch.long, device = device), a[:, :-1]], dim = 1)
        action_logits = model(s, a_input, r, timesteps)
        
        logits_flat = action_logits.reshape(-1, act_dim)
        target_flat = a.reshape(-1)
        
        loss = criterion(logits_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits_flat.argmax(dim = -1)
        correct += (preds == target_flat).sum().item()
        total_preds += target_flat.size(0)
        
    avg_loss = total_loss / len(train_loader)
    acc = (correct / total_preds) * 100
    epoch_losses.append(avg_loss)
    epoch_accs.append(acc)
    
    print(f"Epoch {e + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time() - epoch_start:.2f}s")

total_training_time = (time.time() - total_start_time) / 60
print("="*50)
print(f"[PROFILER] Total Training Time: {total_training_time:.2f} Minutes")
print("="*50)

# Visualizations : 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
ax1.plot(range(1, epochs + 1), epoch_losses, marker = 'o', color = 'crimson')
ax1.set_title('DT Loss : ')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross Entropy Loss')

ax1.grid(True, linestyle='--', alpha = 0.6)

ax2.plot(range(1, epochs+1), epoch_accs, marker = 'o', color = 'dodgerblue')

ax2.set_title('Target Action Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

ax2.grid(True, linestyle = '--', alpha = 0.6)

plt.tight_layout()
plt.show()


# Inference on Sample : 
model.eval()

def simulate_trade_trajectory(target_return):
    print(f"\nHallucinating execution for desired PnL: + {target_return:.2f}")
    
    s = torch.tensor([[[0.02, 100.5, 300, 100]]], dtype = torch.float32, device = device)
    a = torch.zeros((1, 1), dtype = torch.long, device = device) 
    r = torch.tensor([[[target_return]]], dtype = torch.float32, device = device)

    timesteps = torch.zeros((1, 1), dtype = torch.long, device = device)
    
    action_map = {0: "HOLD", 1: "BUY ", 2: "SELL"}
    latencies = []
    
    with torch.no_grad():
        for t in range(5): 

            start_inf = time.perf_counter()
            
            logits = model(s, a, r, timesteps)
            next_action = logits[0, -1].argmax().item()
            
            latency = (time.perf_counter() - start_inf) * 1000
            latencies.append(latency)

            print(f"Tick {t+1} | Market State Evaluated -> Order Sent: {action_map[next_action]} | Latency: {latency:.2f} ms")
            
            # Market Reaction to Trade : 
            next_state = torch.tensor([[[0.02, 100.5, 300, 100]]], dtype = torch.float32, device = device)
            next_rtg = r[0, -1, 0].item() - 0.1 
            
            # Synchronized Update (All tensors to t+1) : 
            s = torch.cat([s, next_state], dim = 1)[:, -context_len:]
            a = torch.cat([a, torch.tensor([[next_action]], device = device)], dim = 1)[:, -context_len:]
            r = torch.cat([r, torch.tensor([[[next_rtg]]], device = device)], dim = 1)[:, -context_len:]
            timesteps = torch.cat([timesteps, torch.tensor([[t+1]], device = device)], dim = 1)[:, -context_len:]
            
    avg_latency = sum(latencies) / len(latencies)
    print(f"[PROFILER] Average Execution Latency per Tick: {avg_latency:.2f} ms")

simulate_trade_trajectory(target_return=5.0)  
simulate_trade_trajectory(target_return=0.1)
