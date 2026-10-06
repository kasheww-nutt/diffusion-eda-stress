# Project: Diffusion-Based Synthetic EDA for Data-Efficient Stress Detection

Title: "Diffusion-Based Synthetic EDA Generation for Data-Efficient Stress Detection: Evaluating Fidelity, Utility, and Memorization"

## How to work with me (MOST IMPORTANT)
- I am a student learning deep learning. I must be able to explain every line in a viva.
- Do NOT write full files for me unless I explicitly say "write it".
- Default mode: explain the concept, give me a skeleton or hints, I write the code, you review it.
- When I'm stuck: smallest hint first, then a bigger hint, then the answer.
- After each phase, quiz me with 2-3 questions before moving on.
- Be blunt. Point out bugs, leakage, and weak reasoning directly. No flattery.
- Keep replies short, bullet points.

## Hard rules (never break)
- Subject-level split ALWAYS. The held-out subject never touches diffusion OR classifier training.
- Everything (normalization stats, diffusion model, SMOTE, classifier) is fit inside each LOSO fold on training subjects only.
- Same frozen classifier architecture for every training-set variant.
- Fixed random seeds. Log every fold's results.
- Run 1 fold before running all 15.

## Data
- WESAD, 15 subjects. Labels: 1 baseline, 2 stress, 3 amusement -> binary stress (1) vs non-stress (0).
- Single-channel EDA. Wrist E4 = 4 Hz, chest RespiBAN = 700 Hz (decision pending: start with wrist).
- Preprocess: low-pass filter, per-subject z-score, 60 s windows (240 samples at 4 Hz), non-overlapping or no overlap across splits.

## Method
- Class-conditional 1D DDPM (1D U-Net denoiser, timestep + label embedding, classifier-free guidance, DDIM sampling).
- Retrain diffusion inside every LOSO fold.

## Evaluation
- Test 0 Fidelity: mean/std/skew/kurtosis, PSD, histograms/KDE, MMD, DTW, discriminative score, tonic/phasic EDA features (SCR rate, amplitude, SCL) per class.
- Test 1 Utility: frozen classifier on
  - A 100% real, B 100% synthetic, C small-real + synthetic, D small-real only
  - E small-real + SMOTE, F small-real + jitter/scale/time-warp (+ EDA-specific augmentations)
  - Real-data fractions 5/10/25/50/100%. Key result: C vs D. Wilcoxon test across subjects. Metric: macro-F1 / balanced accuracy, mean ± std.
- Test 2 Memorization: distance to closest record (DCR) + NNDR vs cross-subject real baseline; per-subject breakdown; subject re-identification test; optional membership inference.
- Novelty extras: utility-vs-memorization curves (training epochs, # training subjects, guidance scale); random vs stress-targeted replacement of real windows; property check on mixed real+synthetic sets.

## Novelty claim (don't overclaim)
- NOT first diffusion for EDA/WESAD (Di Martino & Delmastro 2025 did it, window-level split, no privacy audit).
- NOT first LOSO synthetic WESAD (Lange et al. 2024 did it with GANs).
- OUR claim: first leakage-safe, LOSO-evaluated study of diffusion-generated EDA that jointly measures utility vs classical imbalance baselines AND memorization, including utility-memorization trade-off curves.
- Never say "privacy-preserving" - say "memorization audit".

## Phases (track progress here)
- [ ] 0 Setup: env, CUDA check, repo, toy training loop
- [ ] 1 Data: load, filter, label, normalize, window, LOSO split + leakage assert
- [ ] 2 Baseline 1D CNN classifier, variants A and D
- [ ] 3 Diffusion model on 1 fold, sample plots, time per fold
- [ ] 4 Tests 0/1/2 on 1 fold, then full LOSO
- [ ] 5 Baselines (SMOTE, augmentation), replacement experiments, trade-off curves
- [ ] 6 Full run, results table, figures, report, clean README

## Repo layout
data/  src/  notebooks/  results/  configs/

## Stack
Python, PyTorch (CUDA, RTX 4060), numpy, pandas, scipy, scikit-learn, imbalanced-learn, neurokit2, matplotlib, wandb (optional).
