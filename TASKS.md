# Project task list

## Phase 0: Setup
- [x] Venv check (torch 2.5.1 + CUDA, RTX 4060 working)
- [x] Install imbalanced-learn, neurokit2, tabulate into the same venv
- [x] Repo layout, `.gitignore`, git init, private GitHub repo + push
- [x] Toy GPU training loop (`notebooks/explore.ipynb`)

## Phase 1: Data
- [x] Download WESAD → keep only wrist EDA + labels (0.5 MB), delete zip
- [x] Low-pass filter, per-subject z-score, binary labels, 60 s windows (15 s stride)
- [x] LOSO folds + stratified subsampling + leakage asserts

## Phase 2: Classifier
- [x] Frozen 1D CNN, class-weighted CE, macro-F1 / balanced accuracy

## Phase 3: Diffusion
- [x] 1D U-Net + DDPM + classifier-free guidance + DDIM sampler + EMA
- [x] Bug found: ε-prediction blew up window levels → switched to v-prediction
- [x] Guidance default w=0 (chosen on S2 *training* subjects only)
- [x] Synthetic output gets the same 1 Hz low-pass as real data (PSD fix)

## Phase 4: Tests on 1 fold (S2)
- [x] Diffusion trained (7 min), samples look EDA-like
- [x] Test 0 Fidelity
- [x] Test 1 Utility (variants A–F × 5 fractions × 3 seeds)
- [x] Test 2 Memorization (DCR, NNDR, per-subject, re-ID, MIA)
- [ ] Curves (epochs ✓; guidance, # subjects) + replacement + property check
- [ ] Review S2 results (sanity: A > D at 5%, no leakage)

## Phase 4b: Full LOSO
- [ ] Run all 15 folds (`python -m src.run_all`, ~5–6 h) ← STOPPED by user, resumable

## Phase 5–6: Aggregate + write-up
- [ ] `src.stats`: tables, Wilcoxon C vs D/E/F (Holm)
- [ ] `src.plots`: samples, PSD, utility curves, DCR histogram, trade-off curves, replacement
- [x] `notebooks/project.ipynb`: whole project + live dashboard (§11)
- [ ] `results/report.md` (careful claims: "memorization audit")
- [ ] Finalize README + `docs/VIVA_NOTES.md`, tick phases in CLAUDE.md
- [ ] Commit + push
