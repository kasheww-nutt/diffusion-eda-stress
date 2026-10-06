"""Novelty extra: utility-vs-memorization trade-off curves.

Each point on a curve = (utility, memorization) for one generator setting:
  utility      = macro-F1 of variant C at 10% real (mean over classifier seeds)
  memorization = DCR ratio (synthetic DCR / unseen-subject DCR; < 1 = closer
                 to training data than a new person is) and membership AUC.
Settings varied:
  - training epochs   (milestone checkpoints 25/50/100%)
  - guidance scale w  (sampling only; same model)
  - # training subjects for the diffusion model (4 / 8 / 14), with the number
    of gradient steps held constant so only data size changes.
"""
import dataclasses

import numpy as np

from src.data.loso import assert_no_leakage
from src.eval.memorization import run_memorization
from src.eval.utility import run_one, target_K
from src.train_diffusion import load_ema, train_diffusion
from src.utils import log


def generate_pool(diff, K, cfg, guidance=None, seed=0):
    d = cfg["diffusion"]
    g = d["guidance"] if guidance is None else guidance
    return diff.generate([K, K], guidance=g, steps=d["ddim_steps"], seed=seed)


def tradeoff_point(fold, Xs, ys, cfg, device, mem_fold=None):
    K = target_K(fold.y_train)
    f1 = [run_one("C", fold, 0.10, s, Xs, ys, K, cfg, device)["macro_f1"] for s in cfg["classifier"]["seeds"]]
    mem, _ = run_memorization(mem_fold or fold, Xs, ys, cfg, device, full=False)
    return {"util_C10_f1": float(np.mean(f1)), "dcr_ratio": mem["dcr"]["dcr_ratio"],
            "frac_below_p5": mem["dcr"]["frac_synth_below_ref_p5"], "mia_auc": mem["mia_auc"]}


def epoch_curve(fold, cfg, device, tag, final_point):
    out = {}
    for m in cfg["diffusion"]["milestones"]:
        if m == 1.0:
            out["1.0"] = final_point
            continue
        _, diff = load_ema(cfg["diffusion"], device, tag, m)
        Xs, ys = generate_pool(diff, target_K(fold.y_train), cfg)
        out[str(m)] = tradeoff_point(fold, Xs, ys, cfg, device)
        log(f"  epoch-curve {m}: {out[str(m)]}")
    return out


def guidance_curve(fold, diff, cfg, device, final_point):
    out = {}
    for w in cfg["curves"]["guidance"]:
        if w == cfg["diffusion"]["guidance"]:
            out[str(w)] = final_point
            continue
        Xs, ys = generate_pool(diff, target_K(fold.y_train), cfg, guidance=w)
        out[str(w)] = tradeoff_point(fold, Xs, ys, cfg, device)
        log(f"  guidance-curve w={w}: {out[str(w)]}")
    return out


def subject_curve(fold, cfg, device, full_point):
    """Retrain diffusion on n of the 14 training subjects (fixed #gradient steps)."""
    out = {}
    subs = fold.train_subjects
    rng = np.random.default_rng(0)
    order = list(rng.permutation(subs))
    for n in cfg["curves"]["n_train_subjects"]:
        if n >= len(subs):
            out[str(n)] = full_point
            continue
        chosen = order[:n]
        mask = np.isin(fold.s_train, chosen)
        assert_no_leakage(fold.s_train[mask], fold.test_subject, "subject-curve diffusion set")
        epochs = int(round(cfg["diffusion"]["epochs"] * len(fold.y_train) / mask.sum()))
        _, diff, _ = train_diffusion(fold.X_train[mask], fold.y_train[mask], fold.s_train[mask],
                                     fold.test_subject, cfg["diffusion"], device,
                                     tag=f"{fold.test_subject}_n{n}", epochs=epochs)
        Xs, ys = generate_pool(diff, target_K(fold.y_train), cfg)
        # memorization is measured against the windows this model was trained on
        mem_fold = dataclasses.replace(fold, X_train=fold.X_train[mask], y_train=fold.y_train[mask],
                                       s_train=fold.s_train[mask])
        out[str(n)] = {**tradeoff_point(fold, Xs, ys, cfg, device, mem_fold=mem_fold),
                       "subjects": chosen, "epochs": epochs}
        log(f"  subject-curve n={n}: {out[str(n)]}")
    return out
