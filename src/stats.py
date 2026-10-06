"""Aggregate all folds -> tables + significance tests.

Unit of analysis = held-out SUBJECT (classifier seeds are averaged first), so
n = number of folds. Paired Wilcoxon signed-rank tests compare two variants on
the same subjects; Holm correction controls the family-wise error over all
fraction x comparison tests.

Usage: python -m src.stats
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils import FOLDS_DIR, TABLE_DIR, load_config, load_json


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def folds_with(name):
    return sorted([d.name for d in FOLDS_DIR.iterdir() if (d / f"{name}.json").exists()],
                  key=lambda s: int(s[1:]))


def utility_tables():
    rows = []
    for s in folds_with("utility"):
        for r in load_json(FOLDS_DIR / s / "utility.json"):
            rows.append({**r, "subject": s})
    df = pd.DataFrame(rows)
    per_subj = df.groupby(["subject", "variant", "frac"], as_index=False)[["macro_f1", "bal_acc"]].mean()
    summ = per_subj.groupby(["variant", "frac"]).agg(
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        bacc_mean=("bal_acc", "mean"), bacc_std=("bal_acc", "std"), n=("subject", "nunique")).reset_index()

    tests = []
    piv = per_subj.pivot_table(index="subject", columns=["variant", "frac"], values="macro_f1")
    fracs = sorted(per_subj[per_subj.variant == "C"].frac.unique())
    for frac in fracs:
        for a, b in (("C", "D"), ("C", "E"), ("C", "F"), ("E", "D"), ("F", "D")):
            x, y = piv[(a, frac)], piv[(b, frac)]
            diff = x - y
            p = wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
            tests.append({"frac": frac, "comparison": f"{a} vs {b}", "mean_diff_f1": diff.mean(),
                          "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p})
    xa, xb = piv[("B", 1.0)], piv[("A", 1.0)]
    tests.append({"frac": 1.0, "comparison": "B vs A", "mean_diff_f1": (xa - xb).mean(),
                  "wins": int((xa > xb).sum()), "losses": int((xa < xb).sum()), "p": wilcoxon(xa, xb).pvalue})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p)
    return per_subj, summ, tests


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, (int, float)) and v is not None:
            out[key] = v
    return out


def json_table(name, keep=None):
    rows = []
    for s in folds_with(name):
        f = flat(load_json(FOLDS_DIR / s / f"{name}.json"))
        if keep:
            f = {k: v for k, v in f.items() if any(k.startswith(p) for p in keep)}
        rows.append({"subject": s, **f})
    return pd.DataFrame(rows)


def curve_table(name):
    rows = []
    for s in folds_with(name):
        for setting, v in load_json(FOLDS_DIR / s / f"{name}.json").items():
            rows.append({"subject": s, "setting": float(setting),
                         **{k: v[k] for k in ("util_C10_f1", "dcr_ratio", "frac_below_p5", "mia_auc")}})
    df = pd.DataFrame(rows)
    return df.groupby("setting").agg(["mean", "std"]).drop(columns="subject", errors="ignore") if len(df) else df


def replacement_table():
    rows = []
    for s in folds_with("replacement"):
        for r in load_json(FOLDS_DIR / s / "replacement.json"):
            rows.append({**r, "subject": s})
    df = pd.DataFrame(rows)
    per = df.groupby(["subject", "mode", "rate"], as_index=False)[["macro_f1", "bal_acc"]].mean()
    return per.groupby(["mode", "rate"]).agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                                             bacc_mean=("bal_acc", "mean")).reset_index()


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    per_subj, summ, tests = utility_tables()
    per_subj.to_csv(TABLE_DIR / "utility_per_subject.csv", index=False)
    summ.to_csv(TABLE_DIR / "utility_summary.csv", index=False)
    tests.to_csv(TABLE_DIR / "utility_wilcoxon.csv", index=False)

    fid = json_table("fidelity")
    mem = json_table("memorization", keep=["dcr.", "mia_auc", "per_subject.chi2_p", "per_subject.max_over",
                                           "per_class_dcr_ratio", "reid."])
    prop = json_table("property")
    for name, df in (("fidelity", fid), ("memorization", mem), ("property", prop)):
        df.to_csv(TABLE_DIR / f"{name}_per_subject.csv", index=False)
        df.drop(columns="subject").agg(["mean", "std"]).T.to_csv(TABLE_DIR / f"{name}_summary.csv")
    rep = replacement_table()
    rep.to_csv(TABLE_DIR / "replacement_summary.csv", index=False)
    curves = {n: curve_table(n) for n in ("curve_epochs", "curve_guidance", "curve_subjects")}
    for n, df in curves.items():
        df.to_csv(TABLE_DIR / f"{n}.csv")

    with open(TABLE_DIR / "summary.md", "w") as f:
        f.write(f"# Results summary ({summ.n.max()} folds)\n\n## Utility (macro-F1, mean ± std over subjects)\n\n")
        wide = summ.assign(cell=lambda d: d.f1_mean.map("{:.3f}".format) + " ± " + d.f1_std.map("{:.3f}".format))
        f.write(wide.pivot(index="variant", columns="frac", values="cell").to_markdown() + "\n\n")
        f.write("## Wilcoxon signed-rank (paired over subjects, Holm-corrected)\n\n")
        f.write(tests.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Fidelity (mean over folds)\n\n")
        f.write(fid.drop(columns="subject").agg(["mean", "std"]).T.to_markdown(floatfmt=".4f") + "\n\n")
        f.write("## Memorization (mean over folds)\n\n")
        f.write(mem.drop(columns="subject").agg(["mean", "std"]).T.to_markdown(floatfmt=".4f") + "\n\n")
        f.write("## Replacement\n\n" + rep.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Property check on mixed sets\n\n")
        f.write(prop.drop(columns="subject").agg(["mean", "std"]).T.to_markdown(floatfmt=".4f") + "\n\n")
        for n, df in curves.items():
            f.write(f"## {n}\n\n{df.to_markdown(floatfmt='.4f')}\n\n")
    print(open(TABLE_DIR / "summary.md").read())


if __name__ == "__main__":
    main()
