# Fraud Detection — Master Plan
*Last updated: 2026-02*

## What the Research Told Us (Key Findings)

1. **Entity aggregation is the #1 signal** — UID construction + per-entity behavioral stats pushed one team from top 30% to top 2%
2. **GroupKFold by month beats everything** — StratifiedKFold leaks future info; TimeSeriesSplit wastes training data
3. **Simple equal-weight blend beats stacking** — 3-6 diverse models, average probabilities, done
4. **Compute frequency encodings on train+test combined** (no labels used) — better representations, standard practice
5. **Graph features** are underexplored in competitions but capture fraud rings — opportunity
6. **Adversarial validation** to detect + remove features that shift between train/test periods

---

## Current State

| Notebook | What It Does | Status |
|----------|-------------|--------|
| `scale pos fraud.ipynb` | Main pipeline: 40+ features, 6 models, PR-AUC, threshold opt | Ready to run |
| `fraud.ipynb` | Same without SMOTE, older structure | Deprecated |
| `fraud_optimized_complete.ipynb` | With SMOTE | Deprecated |
| `cfm_shift_v3.ipynb` | Reference: graph features, focal loss, OOF stacking (Colab) | Reference only |

**What `scale pos fraud.ipynb` is missing:**
- No entity-level behavioral aggregations (the #1 signal)
- No velocity/burst features
- No deviation-from-baseline per entity
- No GroupKFold — uses single train/test split
- No adversarial validation
- No GNN embeddings
- Frequency encoding only on train (should be train+test for representations)

---

## The Plan — In Order of Expected Impact

### Phase 1 — GNN Feature Extractor  `gnn_features.ipynb`
**Why first:** Graph embeddings are expensive to compute, independent of the main pipeline, and need to be pre-computed once. Everything else can reload them.

**What it builds:**
- Heterogeneous graph: Claim · Hospital · Doctor · Patient · Disease · Pincode nodes
- Edges: Claim→Hospital, Claim→Doctor, Claim→Patient, Claim→Disease, Claim→Pincode
- Model: 3-layer HeteroSAGE (GraphSAGE for heterogeneous graphs)
- Output: 64-dim embedding per claim → saved as `gnn_embeddings.parquet`

**Hardware:** RTX 4070 12GB (est. ~3GB VRAM usage), ~30 min training

**Key design:**
- Node features built from TRAIN statistics only (no leakage)
- Inductive (GraphSAGE) — handles new test nodes correctly
- Trained as a classification task (fraud label) then embeddings extracted from penultimate layer
- Saved embeddings loaded by `scale pos fraud.ipynb` as extra features

---

### Phase 2A — Entity Intelligence Features  (new cells in `scale pos fraud.ipynb`)
**Why high impact:** Published importance rankings show provider-level features dominate (60-70% of fraud is provider-side per SmartLight Analytics).

**Group 1 — Provider aggregations (train-only):**
- Per-hospital: fraud_rate, inv_rate, mean/std Claimed_Amt, unique_doctors, unique_diagnoses, unique_patients, unique_insured_cities (geo diversity), Type_of_Hospital distribution
- Per-doctor: fraud_rate, mean_claim, hospital_count (multi-hospital = flag), diagnosis Herfindahl index (specialty mismatch signal), exclusivity score (claims at top hospital / total claims)
- Per-disease: fraud_rate, mean_claim, expected_LOS (from train median hosp_days), unique_hospitals

**Group 2 — Top research-validated interaction features:**
- **Diagnosis entropy per hospital** — Herfindahl index of Disease_Category per HID_anon (low entropy = always same disease = fraud ring). Research: #1 discriminating signal
- **LOS vs expected LOS** — `hosp_days - expected_LOS_for_disease` (short LOS for high-severity DRG = upcoding). Research: primary OIG audit flag
- **Doctor specialty mismatch** — diseases treated vs expected for doctor's most common disease
- **Employer × Hospital flocking** — `claims(Policy_number, HID_anon) / total_claims(Policy_number)` — what % of a company's claims go to one hospital
- **Doctor exclusivity score** — `claims(doctor, hospital) / total_claims(doctor)`

**Group 3 — Velocity features (30d/90d windows, empirically validated):**
- Claims per hospital in last 7/30/60/90/180 days
- Claims per patient in last 30/90 days
- Velocity ratio: `count_30d / count_90d` (detects spikes)
- Policy inception signal: `days_from_policy_inception` at claim time, flag <30d, <60d, <90d

**Group 4 — Add-on cover features (top SHAP per Computational Economics 2025):**
- `num_addon_covers_claimed` = count of non-zero add-on columns (TTD, PTD, PPD, Death, Accidental_Hospitalisation, Hospital_Cash_Allowance, etc.)
- Binary interactions: `TTD_AND_HospitalCash`, `Hospitalization_AND_TTD_AND_HospitalCash`
- `has_TTD`, `has_HospitalCash` — individual flags (most fraud-prone per RGA survey)
- `addon_to_base_ratio` = total add-on amount / Claimed_Amt

**Group 5 — Deviation-from-baseline (Amex-style, shift-resistant):**
- `(Claimed_Amt - hospital_mean_claim) / hospital_std_claim` — z-score vs hospital baseline
- `(Total_Bill - diagnosis_mean_bill) / diagnosis_std_bill` — z-score vs disease baseline
- `(hosp_days - expected_LOS) / expected_LOS_std`

**Group 6 — Adversarial validation + PSI:**
- Train classifier to predict train vs test rows
- Compute PSI per feature between train and test periods
- Remove features with adversarial AUC > 0.75 OR PSI > 0.20

### Phase 2B — Distribution Shift Handling
- **Population Stability Index (PSI)** monitoring per feature (flag PSI > 0.2)
- **Rolling deviation features** — relative to rolling baseline (inherently shift-resistant)
- **Time-weighted training** — `sample_weight = exp(0.001 * days_from_train_start)` — recency emphasis

---

### Phase 3 — GroupKFold Cross-Validation  (replace current single split in `scale pos fraud.ipynb`)
**Why:** Research shows GroupKFold by month is more reliable than single split.

**What to change:**
- Extract month from `data_created_at`
- Use 6-fold GroupKFold (one month per fold) for OOF evaluation
- Final test = July–Dec 2025 (kept sacred, never used in CV)
- OOF predictions from CV become meta-features for ensemble
- Report both CV PR-AUC (monthly) AND test PR-AUC

---

### Phase 4 — Anomaly Scoring as a Feature  `anomaly_features.ipynb`
**Based on:** Porto Seguro 1st place (DAE), IEEE research (Isolation Forest → XGBoost)

**What to build:**
- Denoising Autoencoder (DAE) trained on TRAIN data only
- Reconstruction error per claim = unsupervised anomaly signal
- Isolation Forest score per claim
- Both scores added as features to main pipeline

---

### Phase 5 — Final Ensemble  (update `scale pos fraud.ipynb`)
**Based on:** Simple equal-weight blend of diverse models beats complex stacking

**What to change:**
- 5 base models: LightGBM (focal), LightGBM (scale_pos_weight), XGBoost, CatBoost, NN
- Add GNN embeddings as features to all models
- Blend: equal-weight probability average (not rank average — PR-AUC is probability-sensitive)
- Calibrate probabilities (Platt scaling) before blending

---

## File Map

```
fraud/
├── plan.md                        <- this file
├── gnn_features.ipynb             <- Phase 1: GNN, outputs gnn_embeddings.parquet
├── anomaly_features.ipynb         <- Phase 4: DAE + IsoForest, outputs anomaly_scores.parquet
├── scale pos fraud.ipynb          <- Main pipeline (updated with all phases)
├── gnn_embeddings.parquet         <- GNN embeddings (64 dims per claim)
├── anomaly_scores.parquet         <- Anomaly scores (2 features per claim)
└── CFM_anon_final.csv             <- Source data
```

---

## What We Are NOT Doing (and Why)

| Technique | Decision | Reason |
|-----------|----------|--------|
| SMOTE | Skip | Research confirms scale_pos_weight is sufficient |
| Complex stacking (LR meta-learner) | Skip | Equal-weight blend consistently wins |
| Pseudo-labeling | Skip | No evidence it helps with temporal shift; can hurt |
| Target encoding as primary | Skip | Frequency + aggregations dominate; target encoding secondary |
| TimeSeriesSplit | Skip | Worst performer in head-to-head comparison |
| Full GNN end-to-end | Use as feature extractor only | More reliable; embeddings fed to GBM |

---

## Expected Impact (conservative estimates)

| Phase | Expected PR-AUC Gain |
|-------|---------------------|
| Phase 1: GNN embeddings | +0.01–0.03 |
| Phase 2: Entity intelligence | +0.03–0.06 (largest gain) |
| Phase 3: GroupKFold | Better calibration, not raw score gain |
| Phase 4: Anomaly scoring | +0.01–0.02 |
| Phase 5: Final ensemble | +0.01–0.02 |

---

## Next Steps Right Now
1. `gnn_features.ipynb` — build and run to generate embeddings
2. Add Phase 2 entity features to `scale pos fraud.ipynb`
3. Run full pipeline, compare PR-AUC before/after
