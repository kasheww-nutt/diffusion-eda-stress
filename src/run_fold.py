"""Run the full pipeline for ONE LOSO fold (held-out subject).

Order: data -> diffusion (train subjects only) -> synthetic pool ->
Test 0 fidelity -> Test 1 utility -> Test 2 memorization -> curves ->
replacement + property check. Each stage writes its own JSON under
results/folds/<subject>/ and is skipped if that file already exists (resumable).

Usage:  python -m src.run_fold --subject S2 [--quick]
--quick shrinks epochs/seeds for a smoke test (results go to results/quick/).
"""
import argparse

import numpy as np

from src.data.loso import assert_no_leakage, get_fold
from src.eval.fidelity import run_fidelity
from src.eval.memorization import run_memorization
from src.eval.utility import run_utility, target_K
from src.experiments.curves import epoch_curve, generate_pool, guidance_curve, subject_curve
from src.experiments.replacement import property_check, run_replacement
from src.models.classifier import train_classifier
from src.train_diffusion import ckpt_path, load_ema, train_diffusion
from src.utils import (RESULTS, get_device, load_config, load_json, log, save_json, set_seed, timer)
import src.utils as U


def run(subject, cfg, quick=False, do_subject_curve=None):
    device = get_device()
    log(f"=== fold {subject} on {device} ===")
    out = U.fold_dir(subject)
    times = load_json(out / "timings.json") if (out / "timings.json").exists() else {}
    set_seed(cfg["seed"])
    fold = get_fold(subject, cfg["subjects"])
    log(f"train windows {len(fold.y_train)} (stress {fold.y_train.sum()}), "
        f"test windows {len(fold.y_test)} (stress {fold.y_test.sum()})")
    tag = subject

    # ---------- diffusion ----------
    if ckpt_path(tag, 1.0).exists():
        _, diff = load_ema(cfg["diffusion"], device, tag)
        log("loaded existing diffusion checkpoint")
    else:
        with timer("diffusion_train", times):
            _, diff, hist = train_diffusion(fold.X_train, fold.y_train, fold.s_train, subject,
                                            cfg["diffusion"], device, tag, seed=cfg["seed"])
        save_json({"loss": hist}, out / "diffusion_loss.json")
    K = target_K(fold.y_train)
    with timer("sampling", times):
        Xs, ys = generate_pool(diff, K, cfg)
    # synthetic windows are produced by a model fit on training subjects only
    assert_no_leakage(fold.s_train, subject, "diffusion source data")
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.where(ys == c)[0], 64, replace=False) for c in (0, 1)])
    np.savez_compressed(out / "samples.npz", X=Xs[keep], y=ys[keep])
    save_json(times, out / "timings.json")

    def stage(name, fn):
        path = out / f"{name}.json"
        if path.exists():
            log(f"skip {name} (exists)")
            return load_json(path)
        with timer(name, times):
            res = fn()
        save_json(res, path)
        save_json(times, out / "timings.json")
        return res

    def fid():
        res, arrays = run_fidelity(fold, Xs, ys, cfg, device)
        np.savez_compressed(out / "fidelity_arrays.npz", **arrays)
        return res

    def mem():
        res, arrays = run_memorization(fold, Xs, ys, cfg, device, full=True)
        np.savez_compressed(out / "dcr_arrays.npz", **arrays)
        return res

    stage("fidelity", fid)
    util = stage("utility", lambda: run_utility(fold, Xs, ys, cfg, device)[0])
    stage("memorization", mem)

    def final_point():
        f1 = [r["macro_f1"] for r in util if r["variant"] == "C" and abs(r["frac"] - 0.10) < 1e-9]
        m = load_json(out / "memorization.json")
        return {"util_C10_f1": float(np.mean(f1)), "dcr_ratio": m["dcr"]["dcr_ratio"],
                "frac_below_p5": m["dcr"]["frac_synth_below_ref_p5"], "mia_auc": m["mia_auc"]}

    fp = final_point()
    stage("curve_epochs", lambda: epoch_curve(fold, cfg, device, tag, fp))
    stage("curve_guidance", lambda: guidance_curve(fold, diff, cfg, device, fp))
    stage("replacement", lambda: run_replacement(fold, Xs, ys, cfg, device))

    def prop():
        set_seed(0)
        mA = train_classifier(fold.X_train, fold.y_train, cfg["classifier"]["seeds"][0], cfg["classifier"], device)
        return property_check(fold, Xs, ys, mA, cfg, device)

    stage("property", prop)
    if do_subject_curve is None:
        do_subject_curve = subject in cfg["curves"]["subject_curve_folds"]
    if do_subject_curve:
        stage("curve_subjects", lambda: subject_curve(fold, cfg, device, fp))
    save_json({"done": True}, out / "done.json")
    log(f"=== fold {subject} done ===")


def quick_cfg(cfg):
    cfg["diffusion"]["epochs"] = 8
    cfg["classifier"]["epochs"] = 3
    cfg["classifier"]["seeds"] = [0]
    cfg["utility"]["fractions"] = [0.10, 1.0]
    cfg["curves"]["guidance"] = [0.0, 2.0]
    cfg["curves"]["n_train_subjects"] = [4, 14]
    cfg["replacement"]["rates"] = [0.25]
    cfg["memorization"]["dtw_pairs"] = 20
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--subject-curve", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.quick:
        cfg = quick_cfg(cfg)
        U.FOLDS_DIR = RESULTS / "quick" / "folds"
        U.CKPT_DIR = RESULTS / "quick" / "ckpt"
        import src.train_diffusion as td
        td.CKPT_DIR = U.CKPT_DIR
    run(a.subject, cfg, a.quick, True if a.subject_curve else None)
