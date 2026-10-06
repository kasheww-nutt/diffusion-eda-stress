# Diffusion-Based Synthetic EDA Generation for Data-Efficient Stress Detection
### Evaluating fidelity, utility, and memorization (leakage-safe LOSO on WESAD)

This project trains a class-conditional 1D DDPM to generate wrist EDA windows (stress vs. non-stress) inside every leave-one-subject-out fold of WESAD. The synthetic data is then evaluated in three ways:
- **Fidelity:** does it look and behave like real EDA?
- **Utility:** does it help a frozen stress classifier on an *unseen* person, compared with SMOTE and classical augmentation?
- **Memorization:** are the samples near-copies of training windows?

Results: [`results/report.md`](results/report.md). Design rationale: [`docs/VIVA_NOTES.md`](docs/VIVA_NOTES.md).

## Leakage rules (enforced with asserts)
- The split is always at subject level. The held-out subject never touches diffusion, SMOTE, augmentation or classifier training.
- Everything is fit inside each fold, on training subjects only.
- The same frozen classifier and hyperparameters are used for every training-set variant.
- Seeds are fixed, and every fold's results are logged to `results/folds/<subject>/`.

## Setup
```
pip install torch numpy pandas scipy scikit-learn imbalanced-learn neurokit2 matplotlib pyyaml tabulate requests
```
Tested with Python 3.12, torch 2.5.1+cu121 and an RTX 4060 Laptop GPU.

## Reproduce
```
python -m src.data.download                # ~2.25 GB download -> 0.5 MB of wrist-EDA npz; zip deleted
python -m src.data.preprocess              # prints windows per subject
python -m src.run_fold --subject S2 --quick  # 3-min smoke test (results/quick/)
python -m src.run_fold --subject S2        # one full fold (~25 min on GPU)
python -m src.run_all                      # all 15 folds (resumable)
python -m src.stats                        # tables + Wilcoxon -> results/tables/
python -m src.plots                        # figures -> results/figures/
```
All hyperparameters are in `configs/default.yaml`.

## Layout
```
configs/        default.yaml (all hyperparameters)
src/data/       download, preprocess (filter, z-score, windows), loso (folds, subsampling, leakage asserts)
src/models/     classifier (frozen 1D CNN), unet1d, diffusion (DDPM + CFG + DDIM)
src/train_diffusion.py
src/baselines.py            SMOTE, jitter/scale/time-warp, tonic drift, SCR insertion
src/eval/                   fidelity, utility (variants A-F), memorization (DCR, NNDR, re-ID, MIA)
src/experiments/            trade-off curves, replacement, mixed-set property check
src/run_fold.py, run_all.py, stats.py, plots.py
notebooks/explore.ipynb     CUDA check, toy loop, raw signal plots
results/                    folds/ (per-fold JSON), tables/, figures/, report.md
```

## Training-set variants
| | Training set |
|---|---|
| A | 100% real |
| B | 100% synthetic |
| C | small real (5/10/25/50/100%) + diffusion synthetic, filled to K per class |
| D | small real only |
| E | small real + SMOTE to K per class |
| F | small real + jitter/scale/time-warp/EDA augmentation to K per class |

K = the majority-class count of the full real training set, so C, E and F are size-matched.

## Data
[WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) (Schmidt et al., ICMI 2018). The dataset is not redistributed here; download it with the script above.
