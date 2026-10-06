"""All report figures -> results/figures/*.png  (run after src.stats).

Usage: python -m src.plots
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocess import load_subject
from src.utils import FIG_DIR, FOLDS_DIR, TABLE_DIR, load_config, load_json

CLS = {0: "non-stress", 1: "stress"}
VAR_COL = {"A": "#444444", "B": "#8c564b", "C": "#1f77b4", "D": "#d62728", "E": "#2ca02c", "F": "#ff7f0e"}


def done_folds(name):
    return sorted([d.name for d in FOLDS_DIR.iterdir() if (d / name).exists()], key=lambda s: int(s[1:]))


def samples(subject="S2"):
    z = np.load(FOLDS_DIR / subject / "samples.npz")
    cfg = load_config()
    t = np.arange(240) / 4
    fig, ax = plt.subplots(2, 4, figsize=(14, 5), sharey=True)
    for r, c in enumerate((0, 1)):
        Xr = np.concatenate([load_subject(s)[0][load_subject(s)[1] == c][::7][:1]
                             for s in cfg["subjects"] if s != subject])[:2]
        Xs = z["X"][z["y"] == c][:2]
        for j in range(2):
            ax[r, j].plot(t, Xr[j], color="k")
            ax[r, j].set_title(f"real {CLS[c]}")
            ax[r, j + 2].plot(t, Xs[j], color=VAR_COL["C"])
            ax[r, j + 2].set_title(f"synthetic {CLS[c]}")
    for a in ax[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle(f"Real vs diffusion-generated EDA windows (fold {subject}, z-scored)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "samples.png", dpi=150)
    plt.close(fig)


def psd_hist():
    folds = done_folds("fidelity_arrays.npz")
    arr = [np.load(FOLDS_DIR / s / "fidelity_arrays.npz") for s in folds]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    bins = np.linspace(-4, 6, 61)
    mid = (bins[:-1] + bins[1:]) / 2
    for j, name in enumerate(("nonstress", "stress")):
        f = arr[0]["psd_f"]
        ax[j].semilogy(f, np.mean([a[f"psd_real_{name}"] for a in arr], 0), "k", label="real")
        ax[j].semilogy(f, np.mean([a[f"psd_synth_{name}"] for a in arr], 0), color=VAR_COL["C"], label="synthetic")
        ax[j].set_title(f"PSD {name}")
        ax[j].set_xlabel("Hz")
        ax[j + 2].plot(mid, np.mean([a[f"hist_real_{name}"] for a in arr], 0), "k", label="real")
        ax[j + 2].plot(mid, np.mean([a[f"hist_synth_{name}"] for a in arr], 0), color=VAR_COL["C"], label="synthetic")
        ax[j + 2].set_title(f"value density {name}")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "psd_density.png", dpi=150)
    plt.close(fig)


def utility():
    s = pd.read_csv(TABLE_DIR / "utility_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, (m, sd, lab) in enumerate((("f1_mean", "f1_std", "macro-F1"), ("bacc_mean", "bacc_std", "balanced accuracy"))):
        for v, name in (("C", "C small-real + diffusion"), ("D", "D small-real only"),
                        ("E", "E small-real + SMOTE"), ("F", "F small-real + augmentation")):
            d = s[s.variant == v].sort_values("frac")
            ax[j].errorbar(d.frac * 100, d[m], yerr=d[sd] / np.sqrt(d.n), marker="o", capsize=3,
                           color=VAR_COL[v], label=name)
        for v, ls in (("A", "--"), ("B", ":")):
            val = s[s.variant == v][m].values[0]
            ax[j].axhline(val, ls=ls, color=VAR_COL[v], label=f"{v} ({'100% real' if v == 'A' else '100% synthetic'})")
        ax[j].set_xscale("log")
        ax[j].set_xticks([5, 10, 25, 50, 100], ["5", "10", "25", "50", "100"])
        ax[j].set_xlabel("% of real training data")
        ax[j].set_ylabel(lab + " (mean ± s.e. over subjects)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "utility_vs_fraction.png", dpi=150)
    plt.close(fig)


def dcr():
    folds = done_folds("dcr_arrays.npz")
    a = [np.load(FOLDS_DIR / s / "dcr_arrays.npz") for s in folds]
    ds, dr = np.concatenate([x["dcr_synth"] for x in a]), np.concatenate([x["dcr_ref"] for x in a])
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.percentile(np.r_[ds, dr], 99), 60)
    ax.hist(dr, bins, density=True, alpha=.6, color="k", label="unseen-subject real -> train")
    ax.hist(ds, bins, density=True, alpha=.6, color=VAR_COL["C"], label="synthetic -> train")
    ax.set_xlabel("distance to closest training record (DCR)")
    ax.legend()
    ax.set_title(f"DCR, pooled over {len(folds)} folds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dcr_hist.png", dpi=150)
    plt.close(fig)


def curves():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, (name, lab) in enumerate((("curve_epochs", "epoch fraction"), ("curve_guidance", "guidance w"),
                                     ("curve_subjects", "# diffusion training subjects"))):
        rows = []
        for s in done_folds(f"{name}.json"):
            for k, v in load_json(FOLDS_DIR / s / f"{name}.json").items():
                rows.append({"setting": float(k), **{m: v[m] for m in ("util_C10_f1", "dcr_ratio")}})
        if not rows:
            continue
        d = pd.DataFrame(rows).groupby("setting").mean().reset_index()
        ax[j].plot(d.dcr_ratio, d.util_C10_f1, "o-", color=VAR_COL["C"])
        for _, r in d.iterrows():
            ax[j].annotate(f"{r.setting:g}", (r.dcr_ratio, r.util_C10_f1), textcoords="offset points", xytext=(5, 5))
        ax[j].set_xlabel("DCR ratio (synthetic / unseen real); lower = more memorization")
        ax[j].set_ylabel("macro-F1, C @ 10% real")
        ax[j].set_title(f"utility vs memorization: {lab}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tradeoff_curves.png", dpi=150)
    plt.close(fig)


def replacement():
    r = pd.read_csv(TABLE_DIR / "replacement_summary.csv")
    a = pd.read_csv(TABLE_DIR / "utility_summary.csv")
    base = a[a.variant == "A"].f1_mean.values[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.35
    for k, (mode, col) in enumerate((("random", "#1f77b4"), ("targeted", "#d62728"))):
        d = r[r["mode"] == mode].sort_values("rate")
        ax.bar(np.arange(len(d)) + k * w, d.f1_mean, w, yerr=d.f1_std, capsize=3, color=col, label=mode)
    ax.axhline(base, ls="--", color="k", label="100% real (A)")
    ax.set_xticks(np.arange(len(d)) + w / 2, [f"{x:.0%}" for x in d.rate])
    ax.set_xlabel("share of real windows replaced by synthetic")
    ax.set_ylabel("macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "replacement.png", dpi=150)
    plt.close(fig)


def loss(subject="S2"):
    p = FOLDS_DIR / subject / "diffusion_loss.json"
    if not p.exists():
        return
    h = load_json(p)["loss"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(h) + 1), h)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ε-MSE")
    ax.set_title(f"diffusion training loss ({subject})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "diffusion_loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (samples, psd_hist, utility, dcr, curves, replacement, loss):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, repr(e))
