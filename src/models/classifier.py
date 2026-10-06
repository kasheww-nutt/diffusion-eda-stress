"""The FROZEN stress classifier used for every training-set variant (A-F).

Same architecture + same hyperparameters for every variant, so differences in
test F1 come only from the training DATA, not from the model.

Architecture: 3 x [Conv1d(k=7) -> BatchNorm -> ReLU -> MaxPool(2)]
              (channels 32/64/128) -> global average pool -> Dropout -> Linear(2)
Loss: class-weighted cross-entropy (weights from the training set it is given),
so the minority class (stress) is not ignored.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.utils import set_seed


class StressCNN(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3, in_ch=1):
        super().__init__()
        layers, c = [], in_ch
        for out in (32, 64, 128):
            layers += [nn.Conv1d(c, out, 7, padding=3), nn.BatchNorm1d(out), nn.ReLU(), nn.MaxPool1d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, n_classes))

    def embed(self, x):
        return self.features(x).mean(-1)          # global average pool -> [B, 128]

    def forward(self, x):
        return self.head(self.embed(x))


def train_classifier(X, y, seed, cfg, device, n_classes=2, verbose=False):
    """Train a fresh StressCNN on (X, y). X: [n, L] float, y: [n] int."""
    set_seed(seed)
    model = StressCNN(n_classes, cfg["dropout"]).to(device)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y, dtype=torch.long)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(seed)
    n, bs = len(yt), cfg["batch_size"]
    model.train()
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  ep {ep + 1}: loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32).unsqueeze(1).to(device)
        out.append(torch.softmax(model(xb), -1).cpu())
    return torch.cat(out).numpy()


def evaluate(model, X, y, device):
    pred = predict_proba(model, X, device).argmax(1)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
    }
