"""Run every LOSO fold (resumable: finished folds are skipped).

Usage: python -m src.run_all [--subjects S2 S3 ...]
"""
import argparse

from src.run_fold import run
from src.utils import FOLDS_DIR, load_config, log

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=cfg["subjects"])
    a = ap.parse_args()
    for s in a.subjects:
        if (FOLDS_DIR / s / "done.json").exists():
            log(f"skip {s} (done)")
            continue
        run(s, load_config())
    log("all folds finished")
