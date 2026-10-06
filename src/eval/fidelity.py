"""Test 0 - Fidelity: does synthetic EDA look/behave like real EDA?

Each metric compares synthetic windows with the fold's REAL TRAINING windows,
per class. The held-out subject's real windows give an "unseen real person"
reference: a good generator should be about as close to train-real as a new
real person is.

- moments: per-window mean / std / skewness / kurtosis distributions
- PSD: Welch power spectrum (fs = 4 Hz), averaged, log-distance
- histogram / KDE of sample values (saved for plots)
- MMD: kernel two-sample distance (RBF, median-heuristic bandwidth); 0 = same distribution
- DTW: shape distance allowing small time shifts (Sakoe-Chiba band)
- discriminative score: |accuracy - 0.5| of a CNN trained to tell real from synthetic
  (0 = indistinguishable, 0.5 = trivially separable)
- EDA physiology: tonic level (SCL), SCR rate and amplitude via neurokit2
"""
import warnings

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import kurtosis, ks_2samp, skew

from src.models.classifier import evaluate, train_classifier

warnings.filterwarnings("ignore", module="neurokit2")


def moments(X):
    return {"mean": X.mean(1), "std": X.std(1), "skew": skew(X, 1), "kurt": kurtosis(X, 1)}


def moment_summary(Xr, Xs):
    mr, ms = moments(Xr), moments(Xs)
    return {k: {"real_mean": float(np.mean(mr[k])), "synth_mean": float(np.mean(ms[k])),
                "real_std": float(np.std(mr[k])), "synth_std": float(np.std(ms[k])),
                "ks": float(ks_2samp(mr[k], ms[k]).statistic)} for k in mr}


def psd(X, fs=4):
    f, p = welch(X, fs=fs, nperseg=120, axis=1)
    return f, p.mean(0)


def psd_distance(Xr, Xs, fs=4):
    f, pr = psd(Xr, fs)
    _, ps = psd(Xs, fs)
    return float(np.sqrt(np.mean((np.log10(pr + 1e-10) - np.log10(ps + 1e-10)) ** 2))), f, pr, ps


def mmd_rbf(A, B, max_n=600, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(max_n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(max_n, len(B)), replace=False)]
    a = torch.tensor(A, device=device, dtype=torch.float64)
    b = torch.tensor(B, device=device, dtype=torch.float64)
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z) ** 2
    sigma2 = torch.median(d2[d2 > 0])                     # median heuristic
    k = torch.exp(-d2 / sigma2)
    n, m = len(a), len(b)
    kaa = (k[:n, :n].sum() - k[:n, :n].diag().sum()) / (n * (n - 1))
    kbb = (k[n:, n:].sum() - k[n:, n:].diag().sum()) / (m * (m - 1))
    kab = k[:n, n:].mean()
    return float(kaa + kbb - 2 * kab)


def dtw(a, b, band):
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        cost = np.abs(a[i - 1] - b[lo - 1:hi])
        for j in range(lo, hi + 1):
            D[i, j] = cost[j - lo] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, n] / n


def mean_dtw(A, B, pairs, band, seed=0):
    rng = np.random.default_rng(seed)
    ia, ib = rng.integers(0, len(A), pairs), rng.integers(0, len(B), pairs)
    return float(np.mean([dtw(A[i], B[j], band) for i, j in zip(ia, ib)]))


def discriminative_score(Xr, Xs, clf_cfg, device, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(Xr), len(Xs))
    Xr = Xr[rng.choice(len(Xr), n, replace=False)]
    Xs = Xs[rng.choice(len(Xs), n, replace=False)]
    X = np.concatenate([Xr, Xs])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    cfg = dict(clf_cfg, epochs=20)
    m = train_classifier(X[tr], y[tr], seed, cfg, device)
    acc = evaluate(m, X[te], y[te], device)["bal_acc"]
    return float(abs(acc - 0.5))


def eda_features(X, fs=4, max_n=300, seed=0):
    """Per-window SCL (tonic mean), SCR rate (/min) and mean SCR amplitude."""
    import neurokit2 as nk
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(max_n, len(X)), replace=False)]
    scl, rate, amp = [], [], []
    for x in X:
        try:
            ph = nk.eda_phasic(x.astype(np.float64), sampling_rate=fs, method="highpass")
            _, info = nk.eda_peaks(ph["EDA_Phasic"].values, sampling_rate=fs, amplitude_min=0.05)
            peaks = np.asarray(info["SCR_Peaks"])
            a = np.asarray(info["SCR_Amplitude"], dtype=float)
            scl.append(float(ph["EDA_Tonic"].mean()))
            rate.append(len(peaks) / (len(x) / fs / 60))
            amp.append(float(np.nanmean(a)) if len(a) else 0.0)
        except Exception:
            continue
    return {"scl": np.array(scl), "scr_rate": np.array(rate), "scr_amp": np.array(amp)}


def eda_feature_summary(Xr, Xs):
    fr, fs_ = eda_features(Xr), eda_features(Xs)
    return {k: {"real_mean": float(np.mean(fr[k])), "synth_mean": float(np.mean(fs_[k])),
                "real_std": float(np.std(fr[k])), "synth_std": float(np.std(fs_[k])),
                "ks": float(ks_2samp(fr[k], fs_[k]).statistic)} for k in fr}


def run_fidelity(fold, Xs, ys, cfg, device):
    """Return (metrics dict, arrays dict for plotting)."""
    out, arrays = {}, {}
    mcfg = cfg["memorization"]
    for c, name in ((0, "nonstress"), (1, "stress")):
        Xr, Xsy, Xte = fold.X_train[fold.y_train == c], Xs[ys == c], fold.X_test[fold.y_test == c]
        d_psd, f, pr, ps = psd_distance(Xr, Xsy)
        res = {
            "moments": moment_summary(Xr, Xsy),
            "psd_logrmse": d_psd,
            "psd_logrmse_testreal": psd_distance(Xr, Xte)[0] if len(Xte) else None,
            "mmd": mmd_rbf(Xr, Xsy, device=device),
            "mmd_testreal": mmd_rbf(Xr, Xte, device=device) if len(Xte) > 2 else None,
            "dtw_synth_real": mean_dtw(Xsy, Xr, mcfg["dtw_pairs"], mcfg["dtw_band"]),
            "dtw_real_real": mean_dtw(Xr, Xr, mcfg["dtw_pairs"], mcfg["dtw_band"], seed=1),
            "eda": eda_feature_summary(Xr, Xsy),
        }
        out[name] = res
        arrays[f"psd_f"] = f
        arrays[f"psd_real_{name}"] = pr
        arrays[f"psd_synth_{name}"] = ps
        arrays[f"hist_real_{name}"] = np.histogram(Xr, bins=60, range=(-4, 6), density=True)[0]
        arrays[f"hist_synth_{name}"] = np.histogram(Xsy, bins=60, range=(-4, 6), density=True)[0]
    out["discriminative_score"] = discriminative_score(fold.X_train, Xs, cfg["classifier"], device)
    return out, arrays
