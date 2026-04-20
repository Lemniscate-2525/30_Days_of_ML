from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import time
import logging

from core.config import vocab_size, d_model, num_layers, num_heads, seq_len, device, model_save_path
from core.model import LlamaModel

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ascension Engine API")

# --- Boot Sequence ---
try:
    logger.info(f"[SYSTEM] Compiling Llama-3 Microstructure Engine on {device}...")
    # Initialize with the correct arguments from config
    model = LlamaModel(vocab_size, d_model, num_layers, num_heads, seq_len).to(device)
    
    # Load the weights you downloaded from Colab
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    logger.info("[SYSTEM] Engine Armed and Locked.")
except Exception as e:
    logger.error(f"[FATAL ERROR] Engine Failed to Boot: {e}")
    model = None

class TickPayload(BaseModel):
    ticks: list[int]

@app.post("/predict")
def predict_cascade(payload: TickPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Engine offline due to boot failure.")
    
    if len(payload.ticks) != seq_len:
        raise HTTPException(status_code=400, detail=f"Expected {seq_len} ticks, got {len(payload.ticks)}")

    try:
        start_time = time.time()
        
        # Forward Pass
        with torch.no_grad():
            x_tensor = torch.tensor([payload.ticks], dtype=torch.long).to(device)
            logits = model(x_tensor)
            prob = torch.sigmoid(logits).item()
        
        latency = (time.time() - start_time) * 1000

        return {
            "probability": prob,
            "cascade_imminent": bool(prob > 0.85),
            "latency_ms": latency
        }
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
