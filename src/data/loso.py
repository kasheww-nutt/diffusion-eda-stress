"""Leave-One-Subject-Out folds, real-data subsampling, and leakage asserts.

A Fold holds training windows from 14 subjects and test windows from 1 subject.
Each window carries its subject id so we can PROVE (assert) that the test
subject never enters diffusion / SMOTE / augmentation / classifier training.
"""
from dataclasses import dataclass

import numpy as np

from src.data.preprocess import load_subject
from src.utils import load_config


@dataclass
class Fold:
    test_subject: str
    X_train: np.ndarray
    y_train: np.ndarray
    s_train: np.ndarray      # subject id per training window
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def train_subjects(self):
        return sorted(set(self.s_train.tolist()))


def get_fold(test_subject, subjects=None, train_subjects=None):
    subjects = subjects or load_config()["subjects"]
    assert test_subject in subjects
    train_subjects = train_subjects or [s for s in subjects if s != test_subject]
    Xs, ys, ss = [], [], []
    for s in train_subjects:
        X, y, _ = load_subject(s)
        Xs.append(X)
        ys.append(y)
        ss.append(np.full(len(y), s))
    X_te, y_te, _ = load_subject(test_subject)
    fold = Fold(test_subject, np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), X_te, y_te)
    assert_no_leakage(fold.s_train, test_subject)
    return fold


def assert_no_leakage(subject_ids, test_subject, where="training data"):
    """Hard rule: the held-out subject must never appear in anything we fit."""
    assert test_subject not in set(np.asarray(subject_ids).tolist()), \
        f"LEAKAGE: test subject {test_subject} found in {where}"


def subsample(fold: Fold, frac: float, seed: int):
    """Stratified (subject x class) random fraction of the real training set.

    Keeps at least 1 window per non-empty (subject, class) cell so every
    training subject and both classes stay represented even at 5%.
    """
    if frac >= 1.0:
        return fold.X_train, fold.y_train, fold.s_train
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(fold.s_train):
        for c in (0, 1):
            idx = np.where((fold.s_train == s) & (fold.y_train == c))[0]
            if len(idx) == 0:
                continue
            n = max(1, int(round(frac * len(idx))))
            keep.append(rng.choice(idx, n, replace=False))
    keep = np.sort(np.concatenate(keep))
    return fold.X_train[keep], fold.y_train[keep], fold.s_train[keep]
