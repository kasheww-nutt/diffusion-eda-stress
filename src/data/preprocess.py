"""Signal preprocessing -> labelled 60 s windows per subject.

Pipeline (per subject, independently):
1. Low-pass Butterworth (1 Hz, order 4) with filtfilt -> removes high-frequency
   noise; zero phase so SCR peaks are not shifted in time.
2. Per-subject z-score over the WHOLE recording. This uses no labels and no
   other subject's data, so it cannot leak test information into training.
   (Viva point: it is "transductive" for the test subject - we use its own
   unlabeled signal level - a standard choice for removing inter-person
   skin-conductance offsets.)
3. Keep protocol labels 1/2/3, map stress(2)->1, baseline/amusement->0.
4. Slide 240-sample windows (stride 60 samples); keep only windows whose 240
   labels are all identical (pure windows, no transitions).
Overlap exists only inside one subject; splits are by subject, so no window
of the test subject overlaps any training window.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt

from src.utils import DATA_PROC, load_config


def lowpass(x, fs, cutoff, order):
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def make_windows(x, labels, win, stride, keep_labels, stress_label):
    X, y, start = [], [], []
    for s in range(0, len(x) - win + 1, stride):
        lab = labels[s:s + win]
        if lab[0] not in keep_labels or not np.all(lab == lab[0]):
            continue
        X.append(x[s:s + win])
        y.append(int(lab[0] == stress_label))
        start.append(s)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(start, np.int64)


@lru_cache(maxsize=None)
def load_subject(subject: str):
    """Return (X [n,240], y [n], start_idx [n]) for one subject."""
    cfg = load_config()["data"]
    z = np.load(DATA_PROC / f"{subject}.npz")
    x = lowpass(z["eda"].astype(np.float64), cfg["fs"], cfg["lowpass_hz"], cfg["lowpass_order"])
    x = zscore(x)
    win = cfg["window_s"] * cfg["fs"]
    stride = cfg["stride_s"] * cfg["fs"]
    return make_windows(x, z["labels"], win, stride, set(cfg["keep_labels"]), cfg["stress_label"])


if __name__ == "__main__":
    for s in load_config()["subjects"]:
        X, y, _ = load_subject(s)
        print(f"{s}: {len(y)} windows, stress={int(y.sum())}, non-stress={int((1 - y).sum())}")
