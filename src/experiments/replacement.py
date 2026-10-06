"""Novelty extras: replacement experiments and a property check on mixed sets.

Replacement: start from 100% real training data and swap some real windows
for synthetic windows of the SAME label, keeping the set size fixed.
  random   : the swapped windows are drawn from all real windows
  targeted : the same NUMBER of swapped windows, but only stress windows
             (capped at the number of stress windows)
If utility barely drops, synthetic windows are good substitutes for real ones.

Property check on mixed real+synthetic sets (variant C):
  - class balance and synthetic share of the set
  - per-class moment drift of the mixed set vs pure real
  - label consistency: the 100%-real classifier (variant A) predicts the label
    the diffusion model was conditioned on? (conditional fidelity)
"""
import numpy as np

from src.baselines import synth_fill
from src.data.loso import subsample
from src.eval.fidelity import moments
from src.eval.utility import target_K
from src.models.classifier import evaluate, predict_proba, train_classifier


def replace_set(fold, Xs, ys, rate, mode, seed):
    rng = np.random.default_rng(seed)
    X, y = fold.X_train.copy(), fold.y_train.copy()
    n_rep = int(round(rate * len(y)))
    if mode == "random":
        idx = rng.choice(len(y), n_rep, replace=False)
    else:
        stress = np.where(y == 1)[0]
        idx = rng.choice(stress, min(n_rep, len(stress)), replace=False)
    for c in (0, 1):
        tgt = idx[y[idx] == c]
        pool = np.where(ys == c)[0]
        X[tgt] = Xs[rng.choice(pool, len(tgt), replace=len(tgt) > len(pool))]
    return X, y, len(idx)


def run_replacement(fold, Xs, ys, cfg, device):
    recs = []
    for rate in cfg["replacement"]["rates"]:
        for mode in ("random", "targeted"):
            for seed in cfg["classifier"]["seeds"]:
                X, y, n = replace_set(fold, Xs, ys, rate, mode, seed)
                m = train_classifier(X, y, seed, cfg["classifier"], device)
                r = evaluate(m, fold.X_test, fold.y_test, device)
                r.update(rate=rate, mode=mode, seed=seed, n_replaced=n)
                recs.append(r)
    return recs


def property_check(fold, Xs, ys, model_A, cfg, device):
    p = predict_proba(model_A, Xs, device).argmax(1)
    out = {"label_consistency": {str(c): float((p[ys == c] == c).mean()) for c in (0, 1)}}
    K = target_K(fold.y_train)
    for frac in (0.10, 0.50):
        Xr, yr, _ = subsample(fold, frac, seed=0)
        Xm, ym = synth_fill(Xr, yr, K, Xs, ys, seed=0)
        drift = {}
        for c in (0, 1):
            mr, mm = moments(fold.X_train[fold.y_train == c]), moments(Xm[ym == c])
            drift[str(c)] = {k: float(abs(np.mean(mm[k]) - np.mean(mr[k]))) for k in mr}
        out[f"mixed_{int(frac * 100)}"] = {
            "n": int(len(ym)), "stress_share": float(ym.mean()),
            "synthetic_share": float(1 - len(yr) / len(ym)),
            "moment_abs_drift_vs_real": drift,
        }
    return out
