# Insurance Fraud Detection Under Severe Temporal Shift

An end-to-end ML pipeline for health insurance claim fraud detection on 474,619 claims (104 raw features), built around one central obstacle: **the train and test periods are so different that a classifier can separate them with AUC = 1.000.** Almost every technique here exists to answer the question *"how do you build a model that still generalizes when the future doesn't look like the past?"*

## The problem

Two targets are predicted per claim:

| Target | Meaning | Class imbalance |
|---|---|---|
| `Target_as_investigation` | Should this claim be routed for investigation? | 1 : 14 |
| `Target_as_fraud` | Is this claim confirmed fraud? | 1 : 35 |

The split is temporal, not random — train is claims before `2025-07-01` (415,751 rows, 18 months), test is everything from `2025-07-01` onward (58,868 rows, 6 months). That's the only split that reflects how the model would actually be deployed, and it's what exposes the shift: an adversarial classifier trained to distinguish train rows from test rows achieves **AUC = 1.000**. No amount of feature-level domain adaptation (quantile normalization, CORAL) fixes that — the two periods are structurally different, not just differently scaled.

## Results

| Target | Metric | Best model | PR-AUC | vs. original baseline |
|---|---|---|---|---|
| Investigation | PR-AUC | Rolling-Entity LightGBM (3mo, tuned) | **0.2724** | +65% vs. windowed RF (0.1759) |
| Fraud | PR-AUC | Rolling-Entity LightGBM (3mo, tuned) | **0.1574** | +134% vs. GNN late-fusion (0.0673) |

PR-AUC (`average_precision_score`) is used throughout instead of ROC-AUC, since ROC-AUC is insensitive to imbalance this extreme (1:35).

## The core finding: rolling entity features

The single biggest lever in this project wasn't a model — it was diagnosing a subtle leakage-adjacent bug in feature *recency*.

Entity-level features (hospital fraud rate, doctor fraud rate, employer fraud rate, etc.) are the strongest fraud signal available, but they were originally computed once over the full 18-month training history. That's a problem: **fraud rings are non-stationary.** A hospital's fraud rate from April–June 2025 predicts July–December 2025 far better than its average rate since January 2024. Training on the full 18 months while using stale 18-month-average entity stats actively works against the model — the rows are recent, the features describing them are not.

The fix: keep all 415K training rows, but recompute every entity aggregate using only a **rolling 3-month window** ending at each cutoff. Effect:

| Target | Model | Static (18mo) entity stats | Rolling 3mo entity stats | Improvement |
|---|---|---|---|---|
| Investigation | RF | 0.1385 | 0.2466 | +78% |
| Investigation | LGBM | 0.0965 | 0.2548 | +164% |
| Fraud | RF | 0.0546 | 0.1442 | +164% |
| Fraud | LGBM | 0.0564 | 0.1449 | +157% |

Window size was swept (3 / 6 / 12 / 18 months) and 3 months won outright — even 6-month stats trail noticeably, which says something about how fast these fraud patterns actually move.

## Approach

- **Adversarial validation** — a Random Forest trained to separate train from test rows is used to prune ~120 features whose importance for that task exceeds a threshold. What's left (200 features) is the set least responsible for the train/test gap, which is a more principled filter than plain correlation or variance screening.
- **Entity intelligence features** — fraud rate, investigation rate, claim-amount and length-of-stay statistics per hospital, doctor, employer, disease, and pincode; plus fraud-ring signals from OIG audit guidance and RGA actuarial research: hospital diagnosis HHI (Herfindahl-Hirschman concentration index), doctor exclusivity score, employer hospital-flocking ratio, and LOS deviation from disease-expected (an upcoding signal).
- **OOF target encoding** — 5-fold time-sorted out-of-fold encoding for 8 high-cardinality categoricals, so base-rate signal is captured without leaking labels forward in time.
- **Anomaly features** — a denoising autoencoder's reconstruction error and an Isolation Forest score, trained separately and merged in as extra signal.
- **Heterogeneous GNN** — a 4-layer residual HeteroGraphSAGE (`ResHeteroSAGE`) over a graph of claims, hospitals, doctors, employers, diseases, and pincodes (~550K nodes, 5.6M edges after making edges undirected). Five integration strategies were tested; late-fusion probability blending (keep the GNN as a fully separate model, average its output with the tabular model's) beat every form of feature concatenation.
- **Density ratio reweighting** — a domain classifier's `p/(1-p)` output, clipped at the 99th percentile, used as a multiplicative sample weight alongside time-decay weighting.
- **Regularization-heavy LightGBM tuning** — a 54-combination grid search consistently favored more regularization (higher `reg_lambda`, larger `min_child_samples`, lower `colsample_bytree`) — the model has to generalize across a distribution gap, and regularization functions as implicit domain adaptation here.

Full technique-by-technique writeup: [docs/NOTEBOOK_EXPLAINED.md](docs/NOTEBOOK_EXPLAINED.md).

## What didn't work

Equally important — a systematic ablation log of techniques that were tried and ruled out, and why:

| Technique | Outcome | Root cause |
|---|---|---|
| Platt scaling / isotonic calibration | No effect | PR-AUC is rank-based; monotonic transforms don't change ranking |
| Pseudo-labeling on high-confidence test predictions | -8% to -17% | Adversarial AUC = 1.0 — test samples are from a genuinely different distribution, not just noisy |
| Quantile normalization / CORAL alignment | Negligible (<0.5%) | Shift is a structural period change, not a scaling artifact |
| Stacking (OOF logistic regression over 4 base models) | Neutral to negative | OOF CV itself leaks temporal information under this shift |
| Importance weighting stacked on time-decay | Hurts | Double-compounds with existing time-decay weights |
| Focal loss, α = 0.75 | Collapses to all-negative | At 1:35 imbalance the correct α is ≈0.97; `scale_pos_weight` used instead |
| SMOTE | Not used | Oversampling in feature space doesn't address a temporal distribution shift |

Full experiment log with every configuration tested: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Repo structure

```
notebooks/
  scale_pos_fraud.ipynb    Main pipeline — feature engineering, adversarial
                            validation, rolling-entity features, model
                            training/evaluation for both targets
  gnn_features.ipynb        Builds the heterogeneous graph and trains
                            ResHeteroSAGE -> claim embeddings
  anomaly_features.ipynb    Denoising autoencoder + Isolation Forest
                            -> anomaly scores

experiments/                 13 standalone ablation scripts (one per
                            hypothesis): shift-alignment (QN/CORAL),
                            window-size sweeps, rolling entity features,
                            density-ratio reweighting, LGBM hyperparameter
                            search, calibration, pseudo-labeling, stacking,
                            focal loss, adversarial feature iteration

docs/
  PROJECT_SUMMARY.md         Resume-style project overview
  NOTEBOOK_EXPLAINED.md      Every technique explained from first principles
  EXPERIMENTS.md             Full experiment log with all results
  PLAN.md                    Original research plan and prioritization
  research/                  Literature review notes that informed the
                            approach (Kaggle fraud-competition strategies,
                            OIG/RGA practitioner guidance, one-class MoE
                            architectures, auxiliary-target OOF techniques)

environment.yml              Conda environment (Python, LightGBM, XGBoost,
                            CatBoost, PyTorch, PyTorch Geometric)
```

## Reproducing this

```bash
conda env create -f environment.yml
conda activate set
jupyter lab
```

The raw dataset (`CFM_anon_final.csv`) and generated artifacts (`checkpoint.npz`, GNN embeddings, anomaly scores) are not included — they're large (300MB–900MB each) and derived from claims data that isn't public. Run `notebooks/scale_pos_fraud.ipynb` end-to-end to regenerate `checkpoint.npz`; the scripts in `experiments/` each load that checkpoint to run their specific ablation.

## Tech stack

**Models:** LightGBM, Random Forest (scikit-learn), CatBoost, XGBoost, PyTorch (NN), PyTorch Geometric (GNN)
**GNN:** `HeteroConv`, `SAGEConv`, `ToUndirected`, `HeteroData` — custom `ResHeteroSAGE` architecture
**Data:** pandas, numpy, scikit-learn (`RobustScaler`, `LabelEncoder`, `average_precision_score`, `mutual_info_classif`)
**Anomaly detection:** PyTorch denoising autoencoder, scikit-learn Isolation Forest
**Hardware:** NVIDIA RTX 4070 12GB (GNN training)
