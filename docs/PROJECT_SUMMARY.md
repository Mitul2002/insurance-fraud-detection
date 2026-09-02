# Fraud Detection Project — Resume Summary

## Project Overview

End-to-end ML pipeline for health insurance claim fraud detection on a dataset of 474,619 claims (104 features) across two classification targets:
- **Target 1 — Investigation flag**: whether a claim should be routed for investigation (1:14 class imbalance)
- **Target 2 — Fraud flag**: whether a claim is confirmed fraud (1:35 class imbalance)

**Key design constraint:** Extreme temporal distribution shift — an adversarial classifier achieves AUC = 1.000 separating train from test, meaning every standard ML assumption about i.i.d. data is violated. The entire project is structured around solving this shift problem.

---

## Data & Setup

| Aspect | Detail |
|--------|--------|
| Dataset | 474,619 health insurance claims, 104 raw columns, anonymized (`CFM_anon_final.csv`) |
| Train split | Claims before 2025-07-01 → 415,751 rows (18 months) |
| Test split | Claims from 2025-07-01 onward → 58,868 rows (6 months, future period) |
| Metric | **PR-AUC** (Precision-Recall AUC / `average_precision_score`) — chosen because ROC-AUC is insensitive to class imbalance at 1:35 ratios |
| Shift severity | Adversarial validation RF AUC = **1.000** — train and test come from completely different distributions; no off-the-shelf DA technique resolves this |

---

## Feature Engineering (40+ features, 200 retained after adversarial drop)

### Entity Intelligence Features

Aggregations computed on training data only to prevent leakage, then merged onto all rows:

- **Hospital-level**: fraud rate, investigation rate, mean/std claim amount, unique doctor count, unique diagnosis count, mean length-of-stay (LOS), **Diagnosis HHI** (Herfindahl-Hirschman Index of disease concentration per hospital — a known fraud ring signal per OIG/RGA research)
- **Doctor-level**: fraud rate, claim count, hospital count, **diagnosis HHI**, **exclusivity score** (fraction of claims at one hospital — multi-hospital doctors are higher risk)
- **Employer-level** (group policy): fraud rate, claim count, unique hospital count, **hospital flocking ratio** (fraction of employer's claims at one hospital — group insurance fraud ring signal)
- **Disease-level**: fraud rate, expected LOS by diagnosis, LOS standard deviation (catches upcoding — the #1 OIG audit flag)
- **Pincode-level**: fraud rate, hospital count

### Velocity / Temporal Features
- Days since policy inception at time of claim (early-claim fraud pattern)
- Days-to-date x Hospital Cash add-on product (interaction between claim timing and add-on cover fraud)
- 30d/90d/180d velocity ratios on investigation flag

### Z-Score Deviation Features
- Claim amount Z-score vs hospital mean (outlier billing)
- LOS deviation vs disease-expected LOS (`short_stay_upcode` flag)

### Add-On Cover Features
- Number of add-on covers claimed (TTD, PTD, PPD, Hospital Cash, Ambulance, Critical Illness, etc.)
- Add-on total amount
- Add-on to base claim ratio (high ratio = fraud signal per RGA 2024)

### OOF Target Encoding
- 5-fold time-sorted out-of-fold encoding on 8 high-cardinality categorical features (hospital ID, doctor ID, disease category, etc.) — prevents target leakage while capturing base-rate signal

### Anomaly Features (external notebook)
- Denoising Autoencoder reconstruction error (`dae_recon_error`)
- Isolation Forest anomaly score (`isoforest_score`)

---

## Adversarial Validation & Feature Selection

- Trained a Random Forest to distinguish train rows from test rows
- Feature importances above threshold 0.008 indicate features that "explain the shift" and are dropped from modeling
- Result: 200 features retained (from ~320 initial) in `checkpoint.npz` (415,751 x 200)
- This is a principled anti-overfit technique — removes features that would give high CV scores but generalize poorly to the future period

---

## Graph Neural Network (Heterogeneous GNN)

### Graph Schema

A heterogeneous graph over the full 474,619-claim dataset:
- **7 node types**: Claim, Hospital, Doctor, Corporate Client, Disease Category, Pincode, Employer (group policy)
- **6 directed edge types**: Claim -> Hospital, Claim -> Doctor, Claim -> Client, Claim -> Disease, Claim -> Pincode, Claim -> Employer
- Made undirected via `ToUndirected` so entity nodes aggregate back into claim nodes (bidirectional message passing)
- ~550K nodes total, ~5.6M edges (12 edge types after making undirected)

### Node Feature Construction

Each node type has its own aggregated feature vector computed on training data:
- Hospital nodes: fraud rate, claim mean/std/count, uniq doctors, uniq diagnoses, mean LOS, HHI
- Doctor nodes: fraud rate, claim stats, hospital count, exclusivity score, diagnosis HHI
- Employer nodes: fraud rate, claim stats, hospital count, flocking ratio
- Claim nodes: 40+ numerical/categorical features (claim amount, age, room charges, add-ons, encoded categoricals)

### Architecture: ResHeteroSAGE v2

- **4-layer Heterogeneous GraphSAGE** with inductive design — handles test nodes not seen during training
- Hidden dim = 256, output dim = 128 (vs 128/64 in v1)
- **Residual connections** on layers 2 and 3 (prevents vanishing gradients in deep hetero graphs)
- **Per-node-type LayerNorm** after each layer (stabilizes heterogeneous feature scale differences)
- Dropout = 0.3, ELU activation
- Dual classification heads: one for fraud, one for investigation (multi-task)
- 200 epochs, focal loss (gamma=2) to focus on hard examples, OneCycleLR scheduler (10% warmup + cosine decay)
- Hardware: NVIDIA RTX 4070 12GB (~3-4 GB VRAM)

### GNN Integration Experiments (5 methods tested)

| Method | PR-AUC (Fraud) | vs Baseline (0.0478) |
|--------|---------------|----------------------|
| Standalone GNN (128-dim LGBM) | 0.0607 | +27% |
| MI Feature Selection (top-20 dims) | 0.0502 | +5% |
| PCA compression (5 components, 80% var) | 0.0499 | +4% |
| Interaction features (top-5 GNN x top-10 tabular) | 0.0494 | +3% |
| K-Means clustering (k=50 GNN space) | 0.0528 | +10% |
| **Late Fusion (50/50 prob blend)** | **0.0620** | **+30%** |

**Winner: Late Fusion** — keeping GNN as a completely separate model and blending probabilities (50/50) outperforms all feature concatenation approaches. Rationale: graph-structural signal and tabular signal are orthogonal enough that blending probabilities is more effective than forcing a single model to reconcile both.

---

## Distribution Shift Mitigation (10 systematic experiments)

### Experiment 1 — Quantile Normalization & CORAL Alignment
- Hypothesis: align train feature distribution to test distribution before training
- Result: Negligible (<0.5% gain). With adversarial AUC = 1.0, the shift is so severe that feature-level alignment doesn't help — the distributions are fundamentally different time periods, not just scaling differences

### Experiment 2 — Window Training Sweep (lean features)
- Hypothesis: training on only the most recent N months reduces the gap between training and test distribution
- Result: **Fraud peaks at 4-month window (+29.6% vs full training)**. Investigation is less sensitive (~7%). Key insight: the last few months of training data resemble the test period far more than 18-month-old data

### Experiment 3 — Window Training with Full Pipeline Features
- Finding: Window training **hurts Fraud RF** when combined with full-history entity features
- Root cause diagnosed: `hosp_fraud_rate`, `doc_fraud_rate`, etc. are computed over all 18 months of training data. When the model trains on only the last 3 months, these entity statistics describe patterns from 18 months ago — the features and training rows are temporally inconsistent. LGBM (more regularized) is more resilient to this conflict than RF

### Experiment 7 — Density Ratio Reweighting (DRW)
- Method: Train LGBM domain classifier (train=0, test=1). Use `p/(1-p)` as sample weights clipped at 99th percentile (max 9.8x), combined multiplicatively with existing time-decay weights. Mean `p_shift_tr` = 0.007 — most training samples look nothing like test
- Result: +1.6–6.3% gain on window models. Hurts full-train models (same entity feature conflict)

---

## Breakthrough: Rolling Entity Features

### Problem Diagnosed

Experiment 3 revealed a fundamental conflict: entity statistics (hospital fraud rates, doctor fraud rates) computed over full 18-month history lose temporal alignment with the future test period. April–June 2025 hospital fraud rates predict July–December 2025 far better than the average over January 2024–June 2025.

### Solution

**Recompute all entity-level aggregate features using only the last 3 months of training data, while keeping all 415K training rows intact.**

- Approach A (winner): Full 415K rows for training, but entity feature columns replaced with 3-month rolling statistics
- Approach B (loser): Restrict both training rows and entity stats to 3-month window — loses too much training data (34K vs 415K rows), hurts model quality

30 entity features survive adversarial drop. Replaced with rolling versions by recomputing hospital/doctor/employer aggregations on only rows where `data_created_at >= cutoff_date - 3 months`.

### Results (3-month rolling window vs 18-month static baseline)

| Target | Model | Baseline | Rolling Entity 3mo | Improvement |
|--------|-------|----------|--------------------|-------------|
| Investigation | RF | 0.1385 | 0.2466 | **+78%** |
| Investigation | LGBM | 0.0965 | 0.2548 | **+164%** |
| Fraud | RF | 0.0546 | 0.1442 | **+164%** |
| Fraud | LGBM | 0.0564 | 0.1449 | **+157%** |

Window size sensitivity: 3mo >> 6mo >> 12mo >> 18mo (full) — even 6-month entity stats are significantly worse than 3-month. The fraud ring patterns are highly non-stationary.

---

## LGBM Hyperparameter Tuning

Grid search over 54 combinations:
- `num_leaves` in {31, 63, 127}
- `min_child_samples` in {10, 30, 50}
- `reg_lambda` in {1, 5, 10}
- `colsample_bytree` in {0.6, 0.8}
- `n_estimators` = 800

Pattern: **all winning hyperparameters lean toward more regularization** — higher lambda, larger min_child_samples, smaller colsample. This makes sense: the model must generalize from an old distribution to a new one, so regularization acts as implicit domain adaptation.

| Target | Best params | PR-AUC | vs default |
|--------|-------------|--------|------------|
| Investigation (5mo window) | nl=63 mcs=50 rl=10 cbt=0.6 | 0.1134 | **+12.1%** |
| Fraud (3mo window) | nl=63 mcs=30 rl=5 cbt=0.6 | 0.0641 | **+8.5%** |

`num_leaves=127` gives essentially zero gain over 63 (tested at combo 37, INV=0.1135 vs 0.1134) — 63 leaves is the sweet spot for this dataset size and shift severity.

---

## What Was Ruled Out (and Why)

| Technique | Outcome | Root Cause |
|-----------|---------|------------|
| Platt scaling / isotonic calibration | No effect | PR-AUC is a ranking metric; monotonic transforms don't change rank order |
| Pseudo-labeling (high-conf test predictions as training data) | -8% to -17% | Adversarial AUC = 1.0 means test samples are from a different distribution; adding them as training confuses the model |
| 6 new engineered features (velocity ratio, inception bins, amount ratio, TTD x HospCash) | Marginal (+3.3% one config, hurts others) | Features don't address the temporal shift root cause |
| Stacking meta-learner (OOF LR on 4 base models) | Neutral/hurts INV, marginal +2% FRAUD | Best single models remain best; OOF CV leaks temporal info |
| Importance Weighting (IW) on top of time-decay | Hurts | Time-decay weights already applied; double-compounding creates instability |
| Focal loss with alpha=0.75 | All predictions -> class 0 | At 1:35 imbalance, correct alpha ~0.97, not 0.75; use scale_pos_weight instead |
| SMOTE | Not attempted (correctly) | Oversampling in feature space doesn't address temporal shift; scale_pos_weight is sufficient |
| Approach B (window training + rolling entity) | Worse than Approach A | 34K rows vs 415K — the rolling entity features already capture recency; restricting rows adds no benefit and loses model capacity |

---

## Final Results

| Target | Previous Best | Method | Final Best | Total Improvement |
|--------|--------------|--------|------------|-------------------|
| Investigation | Window RF 5mo (0.1759) | RE-LGBM 3mo rolling entity, tuned | **0.2724** | **+65% vs 0.1759** |
| Fraud | GNN Late Fusion (0.0673) | RE-LGBM 3mo rolling entity, tuned | **0.1574** | **+134% vs 0.0673** |

---

## Technology Stack

- **ML Models**: LightGBM, Random Forest (scikit-learn), CatBoost, XGBoost, Neural Networks (PyTorch), Graph Neural Networks (PyTorch Geometric)
- **GNN**: PyTorch Geometric — `HeteroConv`, `SAGEConv`, `ToUndirected`, `HeteroData`; custom `ResHeteroSAGE` architecture
- **Data Processing**: pandas, numpy, scikit-learn (`RobustScaler`, `LabelEncoder`, `average_precision_score`, `mutual_info_classif`)
- **Anomaly Detection**: PyTorch Denoising Autoencoder, scikit-learn Isolation Forest
- **Hardware**: NVIDIA RTX 4070 12GB (GNN training)
- **Infrastructure**: Jupyter notebooks + standalone .py experiment files; micromamba Python environment

---

## Resume-Ready Bullet Points

### Strong Impact Bullets

- Designed and implemented a health insurance claim fraud detection pipeline on 474K claims achieving **PR-AUC of 0.27 (investigation) and 0.16 (fraud)**, representing 65–134% improvement over baselines under severe temporal distribution shift (adversarial AUC = 1.0)
- Discovered and resolved a temporal misalignment between entity-level aggregate features and training window, implementing **rolling 3-month entity statistics** that improved PR-AUC by 78–164% across all models and targets
- Built a **4-layer residual Heterogeneous GraphSAGE (ResHeteroSAGE)** on a 550K-node, 5.6M-edge graph spanning claims, hospitals, doctors, employers, diseases, and pincodes; achieved standalone fraud PR-AUC of 0.061 and +30% via late-fusion probability blending
- Conducted **10 systematic ablation experiments** (quantile normalization, CORAL alignment, window sweeps, pseudo-labeling, density ratio reweighting, stacking, calibration, GNN integration) to isolate which techniques generalize under extreme covariate shift

### Technical Depth Bullets

- Applied **adversarial validation** (RF distinguishing train vs test rows) to identify and drop 120+ features that explain temporal shift, retaining 200 distribution-agnostic features
- Engineered **fraud-ring detection features** inspired by OIG audit guidelines and RGA 2024 actuarial survey: Herfindahl-Hirschman Index per hospital diagnosis, doctor exclusivity score, employer hospital-flocking ratio, LOS deviation from disease-expected
- Implemented **density ratio reweighting** using a domain classifier (`p_shift / 1-p_shift`), combined multiplicatively with time-decay sample weights, yielding +6% LGBM improvement on window models
- Performed LGBM hyperparameter grid search (54 combinations) across `num_leaves`, `min_child_samples`, `reg_lambda`, `colsample_bytree`; identified that heavier regularization across all axes is the dominant pattern for distribution-shifted fraud detection (+8.5–12.1%)
- Used **OOF 5-fold time-sorted target encoding** for high-cardinality categoricals to prevent temporal leakage while capturing base-rate signals
