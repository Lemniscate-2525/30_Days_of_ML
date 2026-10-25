import time
import torch
import sys
import os
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *
from core.data import get_dataloaders
from core.model import LlamaModel


def main():
    _, test_loader = get_dataloaders()

    if not os.path.exists(model_save_path):
        raise FileNotFoundError(
            f"Trained weights not found at '{model_save_path}'. Run "
            "'python scripts/train.py' first."
        )

    model = LlamaModel(
        vocab_size, d_model, num_layers, num_heads, seq_len
    ).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Warm up the model before timing. Synchronize CUDA so asynchronous kernels
    # are included in the measured duration when evaluation runs on a GPU.
    warmup_X, _ = next(iter(test_loader))
    with torch.no_grad():
        model(warmup_X.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    all_preds, all_labels = [], []
    start_inf = time.perf_counter()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = torch.sigmoid(model(X_batch.to(device))).cpu().numpy()
            all_preds.extend(probs.reshape(-1).tolist())
            all_labels.extend(y_batch.numpy().reshape(-1).tolist())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_inf
    # This is amortized time per sequence for batched inference, not single-item latency.
    inference_ms_per_sequence = elapsed / len(test_loader.dataset) * 1000

    if len(set(all_labels)) < 2:
        raise ValueError(
            "The held-out set contains only one class; AUROC cannot be computed."
        )

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    average_precision = average_precision_score(all_labels, all_preds)

    print("=" * 56)
    print(f"Holdout samples: {len(test_loader.dataset)}")
    print(f"Holdout positive rate: {sum(all_labels) / len(all_labels):.3f}")
    print(f"AUROC: {roc_auc_score(all_labels, all_preds):.4f}")
    print(f"PR-AUC (trapezoidal): {pr_auc:.4f}")
    print(f"Average Precision (AP): {average_precision:.4f}")
    print(
        "Amortized batch inference time: "
        f"{inference_ms_per_sequence:.2f} ms per sequence"
    )
    print("=" * 56)


if __name__ == "__main__":
    main()
