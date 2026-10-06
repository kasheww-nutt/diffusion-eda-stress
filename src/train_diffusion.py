"""Train the class-conditional diffusion model on ONE fold's training subjects.

- Input data: fold.X_train / fold.y_train only (asserted: no test subject).
- EMA (exponential moving average) weights are used for sampling: they are a
  smoothed average of the training weights and give cleaner samples.
- Milestone checkpoints (fp16, EMA weights) are saved at 25/50/100% of epochs
  for the utility-vs-memorization-over-epochs curve.
"""
import copy

import numpy as np
import torch

from src.data.loso import assert_no_leakage
from src.models.diffusion import Diffusion
from src.models.unet1d import UNet1D
from src.utils import CKPT_DIR, log, set_seed


def build(cfg, device):
    net = UNet1D(tuple(cfg["channels"]), cfg["emb_dim"]).to(device)
    return net, Diffusion(net, cfg["T"], cfg["p_uncond"], device=device)


@torch.no_grad()
def ema_update(ema, net, decay):
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), net.buffers()):
        be.copy_(b)


def ckpt_path(tag, milestone):
    return CKPT_DIR / tag / f"ema_{int(milestone * 100):03d}.pt"


def train_diffusion(X, y, s_ids, test_subject, cfg, device, tag, seed=0, epochs=None):
    """Return (ema_model, diffusion, loss_history). Saves milestone checkpoints under results/ckpt/<tag>/."""
    assert_no_leakage(s_ids, test_subject, "diffusion training set")
    set_seed(seed)
    epochs = epochs or cfg["epochs"]
    net, diff = build(cfg, device)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    n, bs = len(yt), cfg["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    milestones = {max(1, int(round(m * epochs))): m for m in cfg["milestones"]}
    hist = []
    (CKPT_DIR / tag).mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = diff.loss(Xt[idx], yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema_update(ema, net, cfg["ema"])
            tot += loss.item() * len(idx)
        hist.append(tot / n)
        if ep % 50 == 0 or ep == 1:
            log(f"  diffusion ep {ep}/{epochs} loss {hist[-1]:.4f}")
        if ep in milestones:
            sd = {k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()}
            torch.save(sd, ckpt_path(tag, milestones[ep]))
    diff.model = ema
    return ema, diff, hist


def load_ema(cfg, device, tag, milestone=1.0):
    net, diff = build(cfg, device)
    sd = torch.load(ckpt_path(tag, milestone), map_location=device)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    net.eval()
    diff.model = net
    return net, diff
