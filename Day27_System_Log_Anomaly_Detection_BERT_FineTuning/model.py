import math
import os
import time
import requests

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import DistilBertModel, DistilBertTokenizer

from io import BytesIO

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Device Setup & Config : 
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
epochs = 4
lr = 2e-3  

lora_rank = 8
lora_alpha = 16
max_seq_len = 64

# EC2 Dataset(HDFS LogPAI) : 
def fetch_real_hdfs_logs() :

    url = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
    response = requests.get(url)
    
    logs = response.text.strip().split('\n')
    texts, labels = [], []
    
    for log in logs:

        cleaned_log = " ".join(log.split()[5:]) 

        if "Exception" in cleaned_log or "not found" in cleaned_log or "Timeout" in cleaned_log:
            labels.append(1)

        else :
            labels.append(0)
        texts.append(cleaned_log)
        
    return texts, labels

texts, labels = fetch_real_hdfs_logs()

# EDA : 
seq_lens = [len(t.split()) for t in texts]
class_counts = Counter(labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.hist(seq_lens, bins = 20, color = 'steelblue', edgecolor = 'black')
ax1.set_title("Word Count Distribution : ")

ax1.set_xlabel('Word Count')
ax1.set_ylabel('Frequency')

ax2.bar(['Normal (0)', 'Anomaly (1)'], [class_counts[0], class_counts[1]], color = ['mediumseagreen', 'crimson'], edgecolor = 'black')
ax2.set_title("Class Distribution : ")
ax2.set_ylabel('Number of Logs')

plt.tight_layout()
plt.show()

# Data Splits : 
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size = 0.2, random_state = 42, stratify = labels
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class LogDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len = max_seq_len):
        self.encodings = tokenizer(texts, truncation = True, padding = 'max_length', max_length = max_len, return_tensors = 'pt')
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

train_loader = DataLoader(LogDataset(train_texts, train_labels, tokenizer), batch_size = batch_size, shuffle = True)
test_loader = DataLoader(LogDataset(test_texts, test_labels, tokenizer), batch_size = batch_size)

# Frozen LoRA Model(No Grad Updates) : 
class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, r = lora_rank, alpha = lora_alpha):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.randn(in_features, r) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features)) 
        self.scaling = alpha / r

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

# clf head : 
class BertLogAnomalyDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = LoRALinear(in_features = 768, out_features = 2, r = lora_rank)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# Training & Optimization : 
model = BertLogAnomalyDetector().to(Device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Update Ratio         : {(trainable_params / total_params) * 100:.4f}%\n")

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
criterion = nn.CrossEntropyLoss()

epoch_losses = []
epoch_accs = []
total_start_time = time.time()

for e in range(epochs):

    model.train()
    total_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    
    for batch, labels in train_loader:

        input_ids = batch['input_ids'].to(Device)
        attention_mask = batch['attention_mask'].to(Device)
        labels = labels.to(Device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim = 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = total_loss / len(train_loader)
    acc = (correct / total) * 100
    epoch_losses.append(avg_loss)
    epoch_accs.append(acc)
    
    print(f"Epoch {e+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")

total_training_time = (time.time() - total_start_time) / 60
print("="*50)

print(f"Total Training Latency: {total_training_time:.2f} Minutes")
print("="*50)

# Loss and Accuracy Visualizations : 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))

ax1.plot(range(1, epochs+1), epoch_losses, marker = 'o', color = 'crimson')
ax1.set_title("LoRA Fine-Tuning Loss : ")

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross Entropy Loss')

ax1.grid(True, linestyle = '--', alpha = 0.6)

ax2.plot(range(1, epochs+1), epoch_accs, marker = 'o', color = 'dodgerblue')
ax2.set_title("Classification Accuracy : ")

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

ax2.grid(True, linestyle = '--', alpha = 0.6)

plt.tight_layout()
plt.show()

# Model Evaluation & Inference Latency : 
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():

    for batch, labels in test_loader:
        input_ids = batch['input_ids'].to(Device)
        attention_mask = batch['attention_mask'].to(Device)
        
        outputs = model(input_ids, attention_mask)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, labels = [0, 1], target_names = ["Normal (0)", "Anomaly (1)"], zero_division = 0))

def check_log_with_latency(log_text) :

    inputs = tokenizer(log_text, return_tensors = "pt", truncation = True, padding = 'max_length', max_length = max_seq_len).to(Device)
    
    start_inf = time.perf_counter()

    with torch.no_grad():

        logits = model(inputs['input_ids'], inputs['attention_mask'])
        prob = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(1).item()

    latency = (time.perf_counter() - start_inf) * 1000
        
    status = "🚨 CRITICAL ANOMALY" if pred == 1 else "✅ Normal"
    print(f"Log: {log_text}")
    print(f"Result: {status} | Conf: {prob[pred].item()*100:.1f}% | Latency: {latency:.2f} ms\n")

print("--- Real-Time System Monitoring Simulation ---")

check_log_with_latency("Receiving block blk_-1608999687919862906 src: /10.250.19.102:50010 dest: /10.250.19.102:50010")
check_log_with_latency("PacketResponder 1 for block blk_-1608999687919862906 terminating")
check_log_with_latency("Block blk_3587508140051953248 is not found on DataNode 10.251.43.115")

