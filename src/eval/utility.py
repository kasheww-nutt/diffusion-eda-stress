"""Test 1 - Utility: does synthetic data help a stress classifier on an UNSEEN subject?

Training-set variants (same frozen classifier for all):
  A  100% real
  B  100% synthetic (K windows per class)
  C  small real (f%) + synthetic, filled to K per class
  D  small real only
  E  small real + SMOTE to K per class
  F  small real + jitter/scale/time-warp/EDA augmentation to K per class
K = majority-class count of the full real training set (so C/E/F are size-matched).
Evaluated on the held-out subject: macro-F1 and balanced accuracy.
"""
import numpy as np

from src.baselines import augment_fill, smote_fill, synth_fill
from src.data.loso import assert_no_leakage, subsample
from src.models.classifier import evaluate, train_classifier
from src.utils import log


def target_K(y):
    return int(np.bincount(y, minlength=2).max())


def build_variant(variant, fold, frac, seed, Xs, ys, K, cfg):
    """Return the (X, y) training set for one variant/fraction/seed."""
    if variant == "B":
        rng = np.random.default_rng(seed)
        pick = np.concatenate([rng.choice(np.where(ys == c)[0], K, replace=(ys == c).sum() < K) for c in (0, 1)])
        return Xs[pick], ys[pick]
    Xr, yr, sr = subsample(fold, frac, seed=1000 * seed + int(frac * 100))
    assert_no_leakage(sr, fold.test_subject, f"variant {variant} real subset")
    if variant in ("A", "D"):
        return Xr, yr
    if variant == "C":
        return synth_fill(Xr, yr, K, Xs, ys, seed)
    if variant == "E":
        return smote_fill(Xr, yr, K, seed)
    if variant == "F":
        return augment_fill(Xr, yr, K, seed, cfg["augment"])
    raise ValueError(variant)


def run_one(variant, fold, frac, seed, Xs, ys, K, cfg, device, return_model=False):
    X, y = build_variant(variant, fold, frac, seed, Xs, ys, K, cfg)
    m = train_classifier(X, y, seed, cfg["classifier"], device)
    r = evaluate(m, fold.X_test, fold.y_test, device)
    r.update(variant=variant, frac=frac, seed=seed, n_train=len(y), n_stress=int(y.sum()))
    return (r, m) if return_model else r


def run_utility(fold, Xs, ys, cfg, device):
    """All variants x fractions x seeds for one fold. Returns (records, A-model seed 0)."""
    K = target_K(fold.y_train)
    seeds = cfg["classifier"]["seeds"]
    recs, model_A = [], None
    for seed in seeds:
        r, m = run_one("A", fold, 1.0, seed, Xs, ys, K, cfg, device, return_model=True)
        recs.append(r)
        if seed == seeds[0]:
            model_A = m
        recs.append(run_one("B", fold, 1.0, seed, Xs, ys, K, cfg, device))
        for frac in cfg["utility"]["fractions"]:
            for v in ("C", "D", "E", "F"):
                if v == "D" and frac >= 1.0:   # D at 100% is exactly A: reuse, don't retrain
                    a = next(x for x in recs if x["variant"] == "A" and x["seed"] == seed)
                    recs.append({**a, "variant": "D"})
                    continue
                recs.append(run_one(v, fold, frac, seed, Xs, ys, K, cfg, device))
        log(f"  utility seed {seed} done")
    return recs, model_A
