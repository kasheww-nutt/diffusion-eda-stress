"""Download WESAD and keep ONLY wrist EDA + labels (storage-minimal).

Steps
1. Stream WESAD.zip (~2.25 GB) to data/raw with resume support.
2. For each subject, read S{k}/S{k}.pkl straight from the zip into memory
   (never extracting the ~15 GB of raw files to disk).
3. Keep signal['wrist']['EDA'] (4 Hz) and the 700 Hz label vector re-sampled
   to 4 Hz by picking label[i * 175]  (700 / 4 = 175 label samples per EDA sample).
4. Save data/processed/S{k}.npz (a few hundred KB each), verify, delete the zip.

Note: WESAD's official format is a Python pickle (encoding='latin1' because it
was written with Python 2). We only unpickle the official dataset file.
"""
import argparse
import io
import pickle
import zipfile

import numpy as np
import requests

from src.utils import DATA_PROC, DATA_RAW, load_config, log

URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
ZIP_PATH = DATA_RAW / "WESAD.zip"
EXPECTED_SIZE = 2249444501


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    have = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
    if have == EXPECTED_SIZE:
        log("zip already complete")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    log(f"downloading from byte {have}")
    with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored Range -> restart
            have = 0
        mode = "ab" if have else "wb"
        done = have
        with open(ZIP_PATH, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (100 << 20) < (1 << 20):
                    log(f"  {done / 1e9:.2f} / {EXPECTED_SIZE / 1e9:.2f} GB")
    assert ZIP_PATH.stat().st_size == EXPECTED_SIZE, "download size mismatch"


def extract(subjects, label_fs=700, fs=4):
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    step = label_fs // fs  # 175
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        for s in subjects:
            out = DATA_PROC / f"{s}.npz"
            if out.exists():
                continue
            member = next(n for n in names if n.endswith(f"{s}/{s}.pkl"))
            log(f"reading {member}")
            with z.open(member) as fh:
                d = pickle.load(io.BytesIO(fh.read()), encoding="latin1")
            eda = np.asarray(d["signal"]["wrist"]["EDA"], dtype=np.float32).ravel()
            lab700 = np.asarray(d["label"]).ravel()
            idx = np.arange(len(eda)) * step
            idx = idx[idx < len(lab700)]
            eda = eda[: len(idx)]
            labels = lab700[idx].astype(np.int8)
            np.savez_compressed(out, eda=eda, labels=labels)
            log(f"  {s}: {len(eda)} samples ({len(eda) / fs / 60:.1f} min), "
                f"labels {np.unique(labels).tolist()}")
            del d


def verify(subjects):
    for s in subjects:
        z = np.load(DATA_PROC / f"{s}.npz")
        assert len(z["eda"]) == len(z["labels"]) > 1000, s
        assert {1, 2, 3}.issubset(set(np.unique(z["labels"]).tolist())), s
    log("all subject files verified")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    download()
    extract(cfg["subjects"], cfg["data"]["label_fs"], cfg["data"]["fs"])
    verify(cfg["subjects"])
    if not args.keep_zip:
        ZIP_PATH.unlink()
        log("deleted WESAD.zip (only wrist EDA npz kept)")
