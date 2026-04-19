import sys
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# Appendingcore to path so that we can import our engine : 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import device, vocab_size, d_model, num_layers, num_heads, seq_len, model_save_path
from core.model import LlamaModel

# 1. Initializing Server : 
app = FastAPI(title = "Llama-3 HFT Engine API")

# 2. Loading Model into VRAM on startup : 
model = LlamaModel(vocab_size, d_model, num_layers, num_heads, seq_len).to(device)

if os.path.exists(model_save_path):
  
    model.load_state_dict(torch.load(model_save_path, map_location = device))
    model.eval()
  
else:
    print("[WARNING] Weights not found. Run scripts/train.py first.")

# 3. Defining Input/Output formats(enforces the JSON structure) : 
class TickSequence(BaseModel):
    ticks: List[int] = Field(..., min_length = seq_len, max_length = seq_len)

class PredictionResponse(BaseModel):
  
    probability: float
    cascade_imminent: bool
    latency_ms: float

# 4. The Execution Endpoint : 
@app.post("/predict", response_model = PredictionResponse)
async def predict_cascade(data: TickSequence):
    import time
    start = time.perf_counter()

    try:
        # Converting incoming JSON list to PyTorch Tensor -> Shape: (1, 128)
        input_tensor = torch.tensor([data.ticks], dtype=torch.long, device = device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()

        latency = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            probability = prob,
            cascade_imminent = prob > 0.85,
            latency_ms = latency
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
