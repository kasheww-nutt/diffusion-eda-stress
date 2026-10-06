"""1D U-Net denoiser eps_theta(x_t, t, y).

Input: noisy window x_t [B, 1, 240], timestep t [B], class label y [B]
       (0 non-stress, 1 stress, 2 = "null" label used for classifier-free guidance).
Output: predicted noise eps [B, 1, 240].

Structure
- Conditioning vector c = MLP(sinusoidal(t)) + Embedding(y). It is added
  (as a per-channel bias) inside every ResBlock, so every layer knows how noisy
  the input is and which class to generate.
- Encoder: 240 -> 120 -> 60 samples, channels 64 -> 128 -> 256.
- Bottleneck with self-attention at length 60 (cheap; lets the model relate
  distant parts of the 60 s window, e.g. several SCRs).
- Decoder mirrors the encoder and concatenates skip connections (the "U").
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(c))[:, :, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)            # [B, L, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class UNet1D(nn.Module):
    def __init__(self, channels=(64, 128, 256), emb_dim=128, n_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(), nn.Linear(emb_dim * 2, emb_dim))
        self.y_emb = nn.Embedding(n_classes + 1, emb_dim)   # last index = null label
        self.inp = nn.Conv1d(1, channels[0], 3, padding=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips_ch = []
        c = channels[0]
        for i, ch in enumerate(channels):
            self.down.append(nn.ModuleList([ResBlock(c, ch, emb_dim), ResBlock(ch, ch, emb_dim)]))
            c = ch
            skips_ch.append(ch)
            last = i == len(channels) - 1
            self.downsample.append(nn.Identity() if last else nn.Conv1d(ch, ch, 4, stride=2, padding=1))

        self.mid1 = ResBlock(c, c, emb_dim)
        self.mid_attn = Attention(c)
        self.mid2 = ResBlock(c, c, emb_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            skip = skips_ch[i]
            self.up.append(nn.ModuleList([ResBlock(c + skip, ch, emb_dim), ResBlock(ch, ch, emb_dim)]))
            c = ch
            self.upsample.append(nn.Identity() if i == 0 else nn.ConvTranspose1d(ch, channels[i - 1], 4, stride=2, padding=1))
            if i != 0:
                c = channels[i - 1]

        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 1, 3, padding=1))

    def forward(self, x, t, y):
        c = self.t_mlp(sinusoidal(t, self.emb_dim)) + self.y_emb(y)
        h = self.inp(x)
        skips = []
        for (r1, r2), ds in zip(self.down, self.downsample):
            h = r2(r1(h, c), c)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.mid_attn(self.mid1(h, c)), c)
        for (r1, r2), us in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], 1)
            h = r2(r1(h, c), c)
            h = us(h)
        return self.out(h)
