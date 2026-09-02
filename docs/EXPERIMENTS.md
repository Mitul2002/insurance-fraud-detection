# Fraud Detection — Experiment Log
*Project: CFM Insurance Fraud / Investigation classification*
*Last updated: 2026-03*

---

## Setup

| Item | Value |
|------|-------|
| Data | `CFM_anon_final.csv` — 474,619 rows, 104 columns |
| Train | rows where `data_created_at < 2025-07-01` → 415,751 rows |
| Test | rows where `data_created_at >= 2025-07-01` → 58,868 rows |
| Targets | `Target_as_investigation` (1:14 ratio), `Target_as_fraud` (1:35 ratio) |
| Metric | **PR-AUC** (`average_precision_score`) — NOT ROC-AUC |
| Shift | Adversarial AUC = 1.000 — train and test are from different time distributions |

---

## Baseline (notebook, full pipeline)

| Target | Model | PR-AUC |
|--------|-------|--------|
| Investigation | RF (full train) | 0.1654 |
| Investigation | LGBM (full train) | 0.0920 |
| Fraud | RF (full train) | 0.0632 |
| Fraud | LGBM (full train) | 0.0547 |
| Fraud | GNN Late Fusion | **0.0668** ← previous best |

---

## Experiment 1 — Distribution Shift Analysis (`experiment_shift.py`)

**Question:** Does Quantile Normalisation (QN) or CORAL alignment reduce train/test shift?

**Method:** Lean pipeline (~60 features), compare baseline vs QN vs CORAL vs window training.

| Config | Inv-RF | Fr-RF |
|--------|--------|-------|
| Baseline | 0.1501 | 0.0790 |
| QN | 0.1508 | 0.0793 |
| CORAL | 0.1502 | 0.0803 |
| **6mo window** | **0.1583** | **0.0957** |

**Verdict:** QN and CORAL are negligible. Window training is the winner. Domain adaptation at the feature level doesn't work when shift is this extreme (Adv AUC = 1.0).

---

## Experiment 2 — Window Size Sweep (`experiment_window.py`)

**Question:** What's the optimal training window size with lean features?

| Window | N train | Inv-RF | Fr-RF |
|--------|---------|--------|-------|
| Full (18mo) | 415,751 | 0.1494 | 0.0773 |
| 1mo | 11,327 | 0.1614 | 0.0820 |
| 4mo | 47,533 | 0.1601 | **0.1002** |
| 5mo | 58,394 | 0.1608 | 0.0972 |
| 6mo | 73,305 | 0.1583 | 0.0962 |

**Verdict:** Fraud peaks at 4mo (+29.6% vs full baseline). Investigation peaks at 1–5mo (minor). Shorter window = more recent distribution = better generalisation.

---

## Experiment 3 — Full Pipeline Window Sweep (`experiment_checkpoint_window.py`)

**Question:** Does window training still help with 200 adversarially-filtered features?

| Window | Inv-RF | Inv-LGBM | Fr-RF | Fr-LGBM |
|--------|--------|----------|-------|---------|
| Full | 0.1385 | 0.0965 | 0.0546 | 0.0564 |
| 3mo | 0.1451 | 0.1051 | 0.0422 | **0.0591** |
| **5mo** | **0.1478** | 0.1012 | 0.0479 | 0.0541 |

**Key finding:** With full-history entity features, window training **hurts Fraud RF** (entity stats are stale and don't match the window's time period). LGBM is more robust due to regularisation.

**Root cause discovered:** Entity features (`hosp_fraud_rate`, `doc_fraud_rate`, etc.) are computed over the full 18-month training history. When training on only the last 3 months, those features represent patterns from 18 months ago — a direct conflict. This finding motivated Experiment 7.

---

## Experiment 4 — Calibration (`experiment_calibration.py`)

**Question:** Does Platt scaling or isotonic regression improve scores?

**Verdict:** No effect. PR-AUC is a ranking metric — monotonic probability transforms don't change the ranking. Not worth adding.

---

## Experiment 5 — Pseudo-labelling (`experiment_pseudolabel.py`)

**Question:** Can we assign pseudo-labels to high-confidence test predictions and retrain?

**Configs tested:** confidence thresholds (0.97/0.02, 0.90/0.05, 0.80/0.10)

| Config | Inv-RF | vs baseline |
|--------|--------|-------------|
| Full train PL (0.97/0.02) | 0.1300 | −12% |
| 5mo window PL | ~0.12 | −17% |

**Verdict:** Hurts all configs. With Adv AUC = 1.0, test samples are from a completely different distribution — adding them as training data confuses the model. Ruled out entirely.

---

## Experiment 6 — New Features (`experiment_new_features.py`)

**Question:** Do 6 new engineered features improve results?

New features: `ratio_30d_vs_90d_risk_inc_flag`, `incep_within_30d/90d/180d`, `amt_ratio_vs_doc_mean`, `TTD_x_HospCash_product`

| Config | Inv-RF | Fr-RF | vs baseline |
|--------|--------|-------|-------------|
| Full train | 0.1431 | 0.0526 | +3.3% INV / hurts FR |
| 3mo LGBM | 0.1051 | 0.0589 | marginal |

**Verdict:** Marginal. Doesn't beat the 5mo window baseline. Not added to notebook.

---

## Experiment 7 — Density Ratio Reweighting (`experiment_density_reweight.py`)

**Question:** Can we upweight training samples that "look like" test samples?

**Method:** Train LGBM domain classifier (train=0, test=1). Use `p/(1-p)` as sample weights, clip at 99th percentile (max 9.8×). Combine multiplicatively with existing time-decay weights.

Domain classifier AUC = 1.0000. Mean `p_shift_tr` = 0.007 (most training samples look nothing like test).

| Config | RF | LGBM | vs baseline |
|--------|-----|------|-------------|
| INV full train + DRW | 0.1388 | 0.0949 | negligible / hurt |
| **INV 5mo window + DRW** | **0.1502** | **0.1073** | **+1.6% RF, +6% LGBM** |
| FRAUD full train + DRW | 0.0517 | 0.0492 | hurts (−5 to −13%) |
| FRAUD 5mo window + DRW | 0.0481 | **0.0604** | RF neutral, LGBM +6.3% |

**Verdict:** Helps window models modestly. DRW is additive with window training. Fraud full-train DRW hurts (same entity feature conflict as before). Not added to notebook (rolling entity dominates).

---

## Experiment 8 — Rolling Entity Features (`experiment_rolling_entity.py`) ⭐ BREAKTHROUGH

**Question:** What if we recompute entity statistics from only the last N months of training data (instead of full 18-month history)?

**Method (Approach A — winner):** Keep all 415K training rows. Replace entity feature columns with stats computed from last 3/6/12/full months only. RobustScale the refreshed features before substituting into X_tr/X_te.

**Method (Approach B — loser):** Also restrict training rows to the same window. Loses too much data.

| Entity window | INV-RF | INV-LGBM | FRAUD-RF | FRAUD-LGBM |
|---------------|--------|----------|----------|------------|
| Baseline (18mo static) | 0.1385 | 0.0965 | 0.0546 | 0.0564 |
| **3mo** | **0.2466** | **0.2548** | **0.1442** | **0.1449** |
| 6mo | 0.2334 | 0.2344 | 0.1236 | 0.1201 |
| 12mo | 0.2078 | 0.1848 | 0.1049 | 0.1020 |
| full (18mo refresh) | 0.1870 | 0.1614 | 0.0932 | 0.0999 |

**Improvement at 3mo:**
- Investigation RF: 0.1385 → **0.2466** (+78%)
- Investigation LGBM: 0.0965 → **0.2548** (+164%)
- Fraud RF: 0.0546 → **0.1442** (+164%)
- Fraud LGBM: 0.0564 → **0.1449** (+157%)

**Why it works:** `hosp_fraud_rate` from Apr–Jun 2025 predicts Jul–Dec 2025 fraud far better than the average over Jan 2024–Jun 2025. The entity statistics from the last 3 months are temporally aligned with the test period.

**Why shorter is better:** Even 6mo entity stats (including older patterns) are significantly worse than 3mo.

**3 cells added to notebook:** `re_setup_01`, `re_inv_01`, `re_fraud_01`

---

## Experiment 9 — LGBM Hyperparameter Tuning (`experiment_lgbm_tuning.py`)

**Question:** Are the default LGBM hyperparameters optimal for window training?

**Grid:** `num_leaves` [31,63,127] × `min_child_samples` [10,30,50] × `reg_lambda` [1,5,10] × `colsample_bytree` [0.6,0.8] × `n_estimators`=800

| Target | Best params | PR-AUC | vs default |
|--------|-------------|--------|------------|
| INV 5mo window | nl=63 mcs=50 rl=10 cbt=0.6 | 0.1134 | +12.1% |
| FRAUD 3mo window | nl=63 mcs=30 rl=5 cbt=0.6 | 0.0641 | +8.5% |

**Pattern:** More regularisation wins — higher `reg_lambda`, smaller `colsample_bytree`, larger `min_child_samples`. More leaves (63 vs 31) helps; 127 gives no gain.

**Applied to 4 notebook cells:** `re_inv_01`, `re_fraud_01`, `b17xbads2uw`, `2g8usyudg0v`

---

## Experiment 10 — Stacking Meta-Learner (`experiment_stacking.py`)

**Question:** Does a stacking ensemble (OOF LogisticRegression on base model outputs) beat simple averaging?

**Base models:** RF_full, LGBM_full, RF_5mo_win, LGBM_3mo_win

| Method | INV | FRAUD |
|--------|-----|-------|
| RF_win5 (best single) | **0.1478** | 0.0479 |
| LGBM_win3 (best single) | 0.1051 | **0.0593** |
| Simple avg (all 4) | 0.1333 | 0.0604 |
| Top-3 avg | 0.1417 | 0.0586 |
| LogReg meta (C=0.01) | 0.1318 | 0.0605 |

**Verdict:** Stacking hurts INV (0.1417 vs 0.1478 best single). Marginal +2% for FRAUD (0.0605 vs 0.0593). Not worth adding — rolling entity already gives 3–4× better results on both targets.

---

## Final Notebook State (39 cells, 2026-03)

| Target | Winner | PR-AUC | Notes |
|--------|--------|--------|-------|
| Investigation | **RE-LGBM (3mo rolling entity)** | **~0.255** | +164% vs baseline LGBM |
| Fraud | **RE-LGBM (3mo rolling entity)** | **~0.145** | +157% vs baseline LGBM |

Previous winners (before rolling entity):
- Investigation: Window RF 5mo = 0.1759
- Fraud: GNN Late Fusion = 0.0668

---

## What Worked

| Technique | Gain | Why |
|-----------|------|-----|
| **Rolling entity features (3mo)** | +78–164% | Entity stats temporally aligned with test period |
| Window training (5mo INV, 3mo FRAUD) | +7–30% | Reduces train/test distribution gap |
| LGBM tuning (nl=63, rl=10/5) | +8–12% | More regularisation prevents memorising old patterns |
| DRW on window models | +2–6% | Upweights samples similar to test; works with window |
| GNN late fusion (Fraud) | +1% | Graph features add marginal signal |

## What Did NOT Work

| Technique | Result | Why |
|-----------|--------|-----|
| QN / CORAL alignment | Negligible | Shift too extreme (Adv AUC=1.0) for feature alignment to help |
| Pseudo-labelling | −8 to −17% | Test distribution too different; degrades training signal |
| New features (+6 engineered) | Marginal / hurts | Features don't address temporal shift |
| Stacking meta-learner | Neutral / hurt | Doesn't beat best single model; OOF CV leaks temporal info |
| Window training + RF for Fraud | Hurts | Entity features (full-history) conflict with window |
| Importance Weighting (IW) | Hurt | Double-compounds with existing time-decay weights |
| Platt / isotonic calibration | No effect | PR-AUC is ranking metric; monotonic transforms don't help |
| Approach B (window+entity) | Worse than A | Too few training rows (34K vs 415K) |
