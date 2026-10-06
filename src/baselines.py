"""Classical baselines that compete with diffusion synthetic data.

All of them take ONLY the small-real training subset (never test data) and
fill each class up to the same target count K as the diffusion variant C, so
the comparison is size-matched.

E: SMOTE - new window = a + u * (b - a), with b one of the k nearest
   same-class neighbours of a (u ~ U(0,1)). Interpolates in 240-D space.
F: Time-series augmentation of random real windows:
   - jitter: + Gaussian noise
   - scaling: multiply by ~N(1, sigma)
   - time-warp: smooth random speed-up/slow-down of the time axis
   - EDA-specific: tonic drift (slow random baseline change) and
     SCR insertion (Bateman-shaped phasic bumps: fast rise, slow decay).
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.interpolate import CubicSpline


def smote_fill(X, y, K, seed):
    counts = np.bincount(y, minlength=2)
    k = int(max(1, min(5, counts.min() - 1)))
    sm = SMOTE(sampling_strategy={c: int(max(K, counts[c])) for c in (0, 1)}, k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


# ---------- augmentation primitives (x: [L]) ----------
def jitter(x, rng, sigma):
    return x + rng.normal(0, sigma, x.shape)


def scaling(x, rng, sigma):
    return x * rng.normal(1, sigma)


def time_warp(x, rng, sigma, knots):
    L = len(x)
    pos = np.linspace(0, L - 1, knots + 2)
    speed = CubicSpline(pos, rng.normal(1, sigma, knots + 2))(np.arange(L))
    t = np.cumsum(np.clip(speed, 0.1, None))
    t = (t - t[0]) / (t[-1] - t[0]) * (L - 1)
    return np.interp(np.arange(L), t, x)


def tonic_drift(x, rng, sigma):
    L = len(x)
    pos = np.linspace(0, L - 1, 4)
    return x + CubicSpline(pos, rng.normal(0, sigma, 4))(np.arange(L))


def bateman(L, fs, onset, tau_rise=0.75, tau_decay=2.0):
    t = (np.arange(L) - onset) / fs
    s = np.where(t > 0, np.exp(-t / tau_decay) - np.exp(-t / tau_rise), 0.0)
    return s / (s.max() + 1e-8)


def scr_insert(x, rng, rate_per_min, amp, fs=4):
    L = len(x)
    n = rng.poisson(rate_per_min * L / fs / 60)
    out = x.copy()
    for _ in range(n):
        out += amp * rng.uniform(0.3, 1.0) * bateman(L, fs, rng.integers(0, L - fs * 4))
    return out


def augment_one(x, rng, cfg):
    x = jitter(x, rng, cfg["jitter_sigma"])
    x = scaling(x, rng, cfg["scale_sigma"])
    x = time_warp(x, rng, cfg["warp_sigma"], cfg["warp_knots"])
    if rng.random() < 0.5:
        x = tonic_drift(x, rng, cfg["drift_sigma"])
    if rng.random() < 0.5:
        x = scr_insert(x, rng, cfg["scr_rate_per_min"], cfg["scr_amp"])
    return x


def augment_fill(X, y, K, seed, cfg):
    rng = np.random.default_rng(seed)
    Xo, yo = [X], [y]
    for c in (0, 1):
        idx = np.where(y == c)[0]
        need = K - len(idx)
        if need <= 0 or len(idx) == 0:
            continue
        src = rng.choice(idx, need, replace=True)
        Xo.append(np.stack([augment_one(X[i].astype(np.float64), rng, cfg) for i in src]).astype(np.float32))
        yo.append(np.full(need, c))
    return np.concatenate(Xo), np.concatenate(yo)


def synth_fill(X, y, K, Xs, ys, seed):
    """Variant C: add synthetic windows (from the fold's diffusion pool) up to K per class."""
    rng = np.random.default_rng(seed)
    Xo, yo = [X], [y]
    for c in (0, 1):
        need = K - int((y == c).sum())
        if need <= 0:
            continue
        pool = np.where(ys == c)[0]
        pick = rng.choice(pool, need, replace=need > len(pool))
        Xo.append(Xs[pick])
        yo.append(ys[pick])
    return np.concatenate(Xo), np.concatenate(yo)
