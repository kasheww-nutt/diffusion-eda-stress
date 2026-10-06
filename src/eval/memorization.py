"""Test 2 - Memorization audit (NOT a privacy guarantee).

Question: are synthetic windows near-copies of specific training windows?

- DCR (distance to closest record): Euclidean distance from each synthetic
  window to its nearest REAL TRAINING window.
  Baseline: DCR of the held-out subject's real windows to the same training set.
  That subject was never seen, so its DCR shows how close an UNSEEN real
  person naturally gets. If synthetic DCR << baseline DCR -> copying.
- NNDR: nearest / 2nd-nearest distance. Values near 0 mean "one training window
  is much closer than all others" (typical of a copy).
- Per-subject breakdown: which training subject owns each synthetic sample's
  nearest neighbour; compared with that subject's share of the training data
  (chi-square). Over-represented subjects = the model leans on them.
- Subject re-identification: a CNN learns to tell the 14 training subjects
  apart. If it is as confident on synthetic windows as on held-out windows of
  known subjects, synthetic windows carry subject identity.
- Distance-based membership inference: score = -(distance to nearest synthetic).
  Members = training windows, non-members = held-out subject. AUC 0.5 = no
  membership signal.
"""
import numpy as np
import torch
from scipy.stats import chisquare, mannwhitneyu
from sklearn.metrics import roc_auc_score

from src.models.classifier import predict_proba, train_classifier


def nn_dists(Q, R, device, k=2, bs=2048):
    """Return (dist [n,k], idx [n,k]) of the k nearest rows of R for each row of Q."""
    Rt = torch.tensor(R, device=device, dtype=torch.float32)
    ds, ix = [], []
    for i in range(0, len(Q), bs):
        q = torch.tensor(Q[i:i + bs], device=device, dtype=torch.float32)
        d = torch.cdist(q, Rt)
        v, j = d.topk(k, dim=1, largest=False)
        ds.append(v.cpu())
        ix.append(j.cpu())
    return torch.cat(ds).numpy(), torch.cat(ix).numpy()


def dcr_stats(Xs, X_train, X_ref, device):
    ds, idx = nn_dists(Xs, X_train, device)
    dr, _ = nn_dists(X_ref, X_train, device)
    dcr_s, dcr_r = ds[:, 0], dr[:, 0]
    nndr_s = ds[:, 0] / (ds[:, 1] + 1e-8)
    nndr_r = dr[:, 0] / (dr[:, 1] + 1e-8)
    p5 = np.percentile(dcr_r, 5)
    return {
        "dcr_synth_median": float(np.median(dcr_s)),
        "dcr_ref_median": float(np.median(dcr_r)),
        "dcr_ratio": float(np.median(dcr_s) / np.median(dcr_r)),
        "frac_synth_below_ref_p5": float(np.mean(dcr_s < p5)),
        "mwu_p_synth_closer": float(mannwhitneyu(dcr_s, dcr_r, alternative="less").pvalue),
        "nndr_synth_median": float(np.median(nndr_s)),
        "nndr_ref_median": float(np.median(nndr_r)),
    }, dcr_s, dcr_r, idx[:, 0]


def per_subject_breakdown(nn_idx, s_train):
    subs, counts = np.unique(s_train, return_counts=True)
    owner = s_train[nn_idx]
    obs = np.array([(owner == s).sum() for s in subs])
    exp = counts / counts.sum() * len(nn_idx)
    share = obs / obs.sum()
    return {
        "observed_share": {s: float(v) for s, v in zip(subs, share)},
        "expected_share": {s: float(v) for s, v in zip(subs, counts / counts.sum())},
        "chi2_p": float(chisquare(obs, exp).pvalue),
        "max_over_expected": float(np.max(share / (counts / counts.sum()))),
    }


def membership_auc(X_members, X_nonmembers, Xs, device):
    dm, _ = nn_dists(X_members, Xs, device, k=1)
    dn, _ = nn_dists(X_nonmembers, Xs, device, k=1)
    y = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
    score = -np.r_[dm[:, 0], dn[:, 0]]
    return float(roc_auc_score(y, score))


def reid_split(s_train, block=10, gap=3):
    """Within-subject time-block split for the re-ID classifier.

    Windows overlap (stride 15 s, length 60 s), so neighbours share data.
    Every 5th block of 10 consecutive windows is held out, and train windows
    within `gap` positions of a held-out block are dropped (no overlap leak).
    """
    tr, te = [], []
    for s in np.unique(s_train):
        idx = np.where(s_train == s)[0]          # already in time order
        pos = np.arange(len(idx))
        held = (pos // block) % 5 == 4
        near = np.zeros(len(idx), bool)
        for p in np.where(held)[0]:
            near[max(0, p - gap):p + gap + 1] = True
        te.append(idx[held])
        tr.append(idx[~near])
    return np.concatenate(tr), np.concatenate(te)


def subject_reid(fold, Xs, clf_cfg, device, seed=0):
    subs = sorted(set(fold.s_train.tolist()))
    lab = np.array([subs.index(s) for s in fold.s_train])
    tr, te = reid_split(fold.s_train)
    m = train_classifier(fold.X_train[tr], lab[tr], seed, clf_cfg, device, n_classes=len(subs))

    def conf(X):
        p = predict_proba(m, X, device)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        return float(p.max(1).mean()), float(ent.mean())

    p_te = predict_proba(m, fold.X_train[te], device)
    c_in, e_in = conf(fold.X_train[te])
    c_syn, e_syn = conf(Xs)
    c_new, e_new = conf(fold.X_test)
    return {
        "reid_acc_heldout_known": float((p_te.argmax(1) == lab[te]).mean()),
        "chance": 1 / len(subs),
        "maxprob_known_subject": c_in, "entropy_known_subject": e_in,
        "maxprob_synthetic": c_syn, "entropy_synthetic": e_syn,
        "maxprob_unseen_subject": c_new, "entropy_unseen_subject": e_new,
    }


def run_memorization(fold, Xs, ys, cfg, device, full=True):
    stats, dcr_s, dcr_r, nn_idx = dcr_stats(Xs, fold.X_train, fold.X_test, device)
    out = {"dcr": stats,
           "mia_auc": membership_auc(fold.X_train, fold.X_test, Xs, device)}
    if full:
        out["per_subject"] = per_subject_breakdown(nn_idx, fold.s_train)
        out["per_class_dcr_ratio"] = {
            str(c): dcr_stats(Xs[ys == c], fold.X_train, fold.X_test, device)[0]["dcr_ratio"] for c in (0, 1)}
        out["reid"] = subject_reid(fold, Xs, cfg["classifier"], device)
    return out, {"dcr_synth": dcr_s, "dcr_ref": dcr_r}
