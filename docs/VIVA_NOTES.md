# Viva notes: what each part does and why

Read the files in this order: `loso.py` → `preprocess.py` → `unet1d.py` → `diffusion.py` → `utility.py` → `memorization.py`.

---

## 1. Data (`src/data/`)

| Step | What | Why |
|---|---|---|
| Download | Stream each `S*.pkl` out of the zip and keep only wrist EDA + labels | The full extraction is about 15 GB. We need 0.5 MB. |
| Label alignment | `label[i*175]` | Labels are at 700 Hz and EDA at 4 Hz, and 700/4 = 175. |
| Low-pass 1 Hz, order 4, `filtfilt` | Removes high-frequency noise | EDA is slow (SCRs last seconds). `filtfilt` is zero-phase, so peaks don't shift. |
| Per-subject z-score | Each subject is normalized using only their own signal | People have very different skin-conductance baselines. Normalization uses no labels and no other subject. |
| Binary labels | 2 (stress) → 1; 1 (baseline) and 3 (amusement) → 0 | Standard WESAD binary task. Labels 0 and 4–7 (transient, meditation, ignore) are dropped. |
| 60 s windows, 15 s stride, pure label only | 240 samples. Windows that cross a label change are dropped | Overlap is only *within* a subject. Splits are by subject, so no test window overlaps a training window. |

Result: ~135 windows per subject (~30% stress), and ~1900 training windows per fold.

**Likely question:** "Isn't the per-subject z-score of the test subject leakage?"
No. It uses that subject's *own unlabeled* signal, never labels, and never other subjects. It is transductive calibration. The alternative, global training stats, would mix baseline offsets between people. Admit this openly as a design choice.

## 2. LOSO + leakage (`src/data/loso.py`)
- 15 folds. In each fold one subject is test and 14 are train.
- `assert_no_leakage` is called on:
  - the fold's training set
  - every real subset used by a variant
  - the diffusion training set
  - the subject-curve training subsets
- `subsample(frac)` is stratified by (subject × class) and keeps ≥1 window per cell. At 5% every subject and both classes are still present.
- The fold is the unit of analysis. Statistics run over 15 subjects, not over windows, because windows from one person are not independent.

## 3. Classifier (`src/models/classifier.py`)
- 3 × [Conv(k=7) → BN → ReLU → MaxPool], then global average pool, dropout and linear.
- Class-weighted cross-entropy, so the stress minority isn't ignored.
- **Frozen**: the same architecture and hyperparameters (30 epochs, Adam 1e-3, batch 64) for every variant. There is no early stopping on the test subject, and no tuning per fold.
- Macro-F1 averages F1 over both classes. Balanced accuracy averages recall over both classes. Both are robust to the 70/30 imbalance.

## 4. Diffusion model (`src/models/`)
**Forward process.** `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`. This is noise added in closed form. Cosine schedule, T = 1000.

**Training.** Pick a random t and add noise. The U-Net predicts **v = sqrt(ᾱ)·ε − sqrt(1−ᾱ)·x₀**, and the loss is MSE(v̂, v). Then x̂₀ = sqrt(ᾱ)·x_t − sqrt(1−ᾱ)·v̂.

**Why v and not ε (a real bug we found).**
- The first S2 model used ε-prediction. Its samples had window levels of ±4.7 SD, against ±0.7 for real windows, and 11% of samples hit the clip.
- Recovering x₀ from ε divides by sqrt(ᾱ_t). At the noisy end of the cosine schedule that is about 10⁻⁵, so tiny ε errors explode.
- The window mean (the tonic level) is the dimension that suffers most, and it is the main stress cue.
- v-prediction is well-posed at every noise level. With it, synthetic levels were −0.54 ± 0.56 against −0.50 ± 0.69 for real (non-stress), and 1.67 ± 0.76 against 1.59 ± 0.92 (stress).
- This choice was made on **S2's training subjects only**, comparing samples with training windows. The test subject was never looked at.

**Post-filter.**
- Synthetic windows go through the same 1 Hz low-pass filter as real windows.
- Without it, leftover sampler noise above 1 Hz put the synthetic power spectrum off by a log-RMSE of 3.2. The real-vs-unseen-real reference is 0.4. With the filter it is 0.3, and MMD is unchanged.
- This is the same preprocessing, not a trick. Every real window already went through it.

**Default guidance w = 0.** w = 1 and w = 2 squeezed the level spread to 0.2–0.3, a loss of diversity, and pushed means to extremes. The other w values are reported on the guidance curve.

**U-Net.**
- Encoder 240→120→60 samples (channels 64→128→256), self-attention at length 60, and a decoder with skip connections.
- Timestep (sinusoidal + MLP) and label embedding are added inside every ResBlock.
- 4.1 M parameters.

**Classifier-free guidance.**
- During training, 10% of labels are replaced by a "null" label, so one network learns both conditional and unconditional generation.
- At sampling: `ε = (1+w)·ε(x,t,y) − w·ε(x,t,∅)`. w = 0 is plain conditional sampling. Larger w gives more class-typical samples, with a possible cost in diversity or memorization. That cost is our guidance curve.

**DDIM sampling.** Deterministic (η = 0), 50 steps instead of 1000, so it is about 20× faster. At each step we predict x̂₀, clip it to [−8, 8], and step to the previous noise level.

**EMA weights.** An exponential moving average of the weights (decay 0.999) is used for sampling, because it gives smoother, better samples.

**Per fold.** The diffusion model is retrained on the 14 training subjects only.

## 5. Training-set variants (`src/eval/utility.py`, `src/baselines.py`)
- K = the majority-class count of the full real training set.
- C, E and F all fill each class up to K, so they have the same size and the same balance. Only the source of the extra windows differs:
  - C: diffusion
  - E: SMOTE (linear interpolation between same-class neighbours)
  - F: jitter, scaling, time-warp, tonic drift, and Bateman-shaped SCR insertion
- **The key comparison is C vs D**: does adding synthetic data beat small-real alone? C vs E/F asks whether it beats the cheap classical alternatives.
- The test is a paired Wilcoxon signed-rank over the 15 subjects, with Holm correction for the multiple tests.

## 6. Fidelity (`src/eval/fidelity.py`)
- Moments (KS statistic between real and synthetic per-window statistics), Welch PSD, value histograms, MMD (RBF, median bandwidth), DTW.
- Discriminative score: |accuracy − 0.5| of a CNN that tries to tell real from synthetic. 0 means indistinguishable.
- EDA physiology via neurokit2: tonic level (SCL), SCR rate and SCR amplitude.
- **Reference values:** the same metrics computed for the *unseen test subject's real data*. "Synthetic is as close to training data as a new real person is" is the bar.

## 7. Memorization audit (`src/eval/memorization.py`)
**DCR.** Distance from each synthetic window to its closest training window.
- Baseline: the DCR of the unseen subject's real windows.
- `dcr_ratio` = median DCR(synthetic) / median DCR(unseen real). A ratio well below 1 means copying.
- `frac_below_p5` is the fraction of synthetic windows closer than the 5th percentile of the baseline.

**NNDR.** d1/d2. Near 0 means one training window is uniquely close, which is what a copy looks like.

**Per-subject breakdown.** Which subject owns each synthetic window's nearest neighbour, compared with that subject's share of the data (χ²).

**Subject re-ID.**
- A 14-way subject classifier is trained with a time-block split, with a gap so overlapping windows can't leak.
- We compare its confidence on synthetic windows with its confidence on known-subject held-out windows and on the unseen subject.

**Membership inference.**
- Score = −distance to the nearest synthetic window.
- Members are training windows; non-members are the unseen subject's windows.
- AUC ≈ 0.5 means no membership signal.

**Wording.** This is a *memorization audit*, NOT a privacy guarantee (there is no differential privacy).

## 8. Novelty extras (`src/experiments/`)
**Trade-off curves.** Each point is (macro-F1 of C at 10% real, DCR ratio). The point moves as we vary:
- training epochs (25/50/100%)
- the guidance scale w
- the number of subjects the diffusion model is trained on (4/8/14, with gradient steps held fixed; run on 5 folds)

**Replacement.**
- Swap 10/25/50% of the real windows for synthetic ones with the same label.
- Random (from all windows) vs stress-targeted (the same number, stress windows only).

**Property check.**
- Class balance and synthetic share of the mixed sets.
- Moment drift of the mixed sets vs real.
- Label consistency: does a real-trained classifier agree with the label the diffusion model was conditioned on?

## 9. Novelty claim (say exactly this)
- Not the first diffusion model for EDA/WESAD (Di Martino & Delmastro 2025 used a window-level split).
- Not the first LOSO synthetic WESAD (Lange et al. 2024 used GANs).
- **Ours:** the first leakage-safe, LOSO-evaluated study of diffusion-generated EDA that jointly measures utility against classical imbalance baselines AND memorization, including utility–memorization trade-off curves.

---

## Quiz questions (one block per phase)
**Phase 1, Data**
1. Why must the split be by subject rather than by window? What would a window-level split inflate?
2. Why `filtfilt` instead of `lfilter`?
3. Overlapping windows: when are they fine, and when would they leak?

**Phase 2, Classifier**
1. Why freeze the classifier across variants?
2. Why macro-F1 and not accuracy on a 70/30 dataset?

**Phase 3, Diffusion**
1. Write the forward-noising equation. What does the network predict, and why v instead of ε here?
2. What does classifier-free guidance do, and how is the null label trained?
3. Why is DDIM faster than DDPM sampling, and what does η = 0 mean?

**Phase 4, Tests**
1. What does a DCR ratio of 0.6 mean? And 1.1?
2. Why use the unseen subject as the DCR baseline instead of other training windows?
3. Why Wilcoxon over subjects instead of a t-test over windows?

**Phase 5, Extras**
1. Why hold gradient steps fixed in the subject-count curve?
2. What would you conclude if stress-targeted replacement hurts more than random replacement?
