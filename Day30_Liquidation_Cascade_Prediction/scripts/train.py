import time
import torch
import torch.nn as nn
import sys
import os

# Adding parent directory to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *
from core.data import get_dataloaders
from core.model import LlamaModel


def main():
    print(f"[SYSTEM] Initializing Training on {device}")
    train_loader, _ = get_dataloaders()

    # Reproducible initialization; data generation/splitting has its own RNG.
    torch.manual_seed(30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(30)

    model = LlamaModel(
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    for e in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {e + 1:02d}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Time: {time.time() - start:.2f}s"
        )

    # Save only weights learned from the training partition.
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[SYSTEM] Saved model weights to {model_save_path}")


if __name__ == "__main__":
    main()
