import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.config import seq_len, vocab_size, batch_size

# Keep the generated data and split identical across independent train/evaluate runs.
DATA_SEED = 30
TRAIN_FRACTION = 0.8


def generate_tick_data(num_samples=5000, seed=DATA_SEED):
    """Generate a reproducible synthetic dataset.

    Labels identify samples with an injected cascade signature. The generator uses
    a local RNG so it does not perturb model initialization or other randomness.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=np.int64)
    y = np.zeros(num_samples, dtype=np.float32)

    crash_indices = rng.choice(
        num_samples, size=int(num_samples * 0.1), replace=False
    )
    for idx in crash_indices:
        X[idx, 50:70] = rng.integers(90, 100, size=20)
        X[idx, 70:90] = rng.integers(10, 15, size=20)
        y[idx] = 1.0

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


class TickDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders():
    """Return the same deterministic 80/20 train and held-out test sets.

    The split is created once from a fixed dataset seed and uses a separate
    permutation seed. Evaluation can therefore reproduce the exact holdout
    without accidentally evaluating on a freshly generated dataset.
    """
    X_data, y_data = generate_tick_data()
    split = int(len(X_data) * TRAIN_FRACTION)

    split_rng = np.random.default_rng(DATA_SEED + 1)
    indices = split_rng.permutation(len(X_data))
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_dataset = TickDataset(X_data[train_indices], y_data[train_indices])
    test_dataset = TickDataset(X_data[test_indices], y_data[test_indices])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(DATA_SEED + 2)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
