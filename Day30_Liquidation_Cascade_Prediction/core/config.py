import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
import math

from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

# Device Setup & Configuration :  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)
model_save_path = os.path.join(weights_dir, "llama_l3_engine.pth")

# Hyperparameters : 
batch_size = 32
epochs = 5
lr = 3e-4

# Architecture : 
seq_len = 128     
d_model = 256
num_heads = 8
num_layers = 4
vocab_size = 100  
