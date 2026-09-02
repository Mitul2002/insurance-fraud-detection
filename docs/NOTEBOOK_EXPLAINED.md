# Fraud Detection Notebook — Complete Method Explanation
*Every technique, every decision, explained from first principles.*
*Updated to reflect `scale pos fraud copy.ipynb` (47 cells, 2026-03)*

---

## The Problem

We have insurance claims data. Two things we want to predict for each claim:
1. **Target_as_investigation** — will this claim be flagged for investigation?
2. **Target_as_fraud** — will this claim be confirmed as fraud?

**Why it's hard:**
- Only ~7% of claims get investigated (1 in 14)
- Only ~2% are confirmed fraud (1 in 50)
- Train data is Jan 2024 – Jun 2025. Test data is Jul – Dec 2025. They come from different time periods, so the statistical patterns shift.

---

## Step 1: Train/Test Split (Temporal Split)

**What:** We split data by date, not randomly.
- Train: all claims before `2025-07-01`
- Test: all claims from `2025-07-01` onwards

**Why not random split?**
In real deployment, you always predict the future from the past. If you split randomly, your model trains on July data and tests on March data — that never happens in production. A random split would give falsely optimistic scores.

**The consequence:** The train and test distributions are very different. We ran an "adversarial validation" to measure this — the model could tell train from test with AUC = 1.00. That's as different as it gets.

---

## Step 2: Removing Leaky Features

**What:** Before doing anything, we drop two categories of columns.

**What is a leaky feature?**
A feature that contains information about the outcome — information you wouldn't actually have at the time of making the prediction. In fraud detection: you're scoring a claim when it is FILED. Anything that is only known AFTER the claim is reviewed is leaky.

---

### Category A — Outcome columns (dropped from the entire dataset immediately)

These columns directly encode what happened after the claim was reviewed. Using them would be like giving the model the answer sheet.

| Column | Why It's Leaky |
|--------|---------------|
| `Investigation Outcome` | The result of the investigation — this IS what we're trying to predict |
| `Investigation Outcome1` | A second version of the same outcome label |
| `Old_Target` | A previous labelling of the target — encodes outcome knowledge |
| `Final Status` | Whether the claim was ultimately approved or rejected — decided POST review |
| `Investigated` | Binary flag: was this claim investigated? — again, this IS the target |
| `Assign Date` | The date an investigator was assigned — only exists AFTER the claim was flagged |
| `Claim Type` | Classification assigned during/after review process |
| `Approved_Amount_INR` | The amount the insurer approved paying — decided AFTER reviewing the claim. A fraudulent claim that got caught might have zero approved amount. A legitimate claim gets its full amount. Using this = the model learns the decision, not the signal. |

**The `Approved_Amount_INR` trap specifically:**
This column is the most tempting because it's numerical and looks like a claim feature. But it represents the insurer's POST-REVIEW decision. Any model trained with it will achieve near-perfect accuracy — and fail completely in deployment where the approved amount doesn't exist yet.

---

### Category B — ID and date columns (kept in `df_clean` for aggregations, but removed from the feature matrix `feat`)

These don't directly leak the outcome, but they must not go into the model as raw features for different reasons.

| Column | Why Removed from Feature Matrix |
|--------|--------------------------------|
| `Claim_No` | Unique claim identifier — no predictive signal, just an ID |
| `hospital_name_anon` | Raw anonymised hospital name string — already captured properly as numeric entity features (fraud rate, HHI, etc.) |
| `Hospital_HID` | Raw hospital ID number — same reason; the entity features encode this more meaningfully |
| `Policy_number` | Policy ID — used internally to compute employer-level aggregations, but the raw ID itself has no meaning |
| `Miscllenious_policy_ID` | Another ID column — no predictive value |
| `data_created_at` | Raw date string of when the claim was created — if we feed the date directly, the model learns "claims filed after Jul 2025 are test data" and overfits to calendar artifacts |
| `data_created_at_parsed` | Parsed datetime version of the same column — same issue |
| `Target_as_investigation` | The label itself — obviously cannot be a feature |
| `Target_as_fraud` | The label itself — obviously cannot be a feature |

**Why keep them in `df_clean` but not in `feat`?**
We still need `Policy_number` to compute employer-level fraud rates (which hospital/employer does this policy belong to?). We need `data_created_at_parsed` to compute velocity features (how many claims in the last 30 days?). We need `Hospital_HID` for entity aggregations. So they stay in the dataframe for computation — they just never enter the final model input matrix.

---

**Why does this matter?**
If you train with outcome columns, your model achieves 99% accuracy on training data and 5% on production data. This is the single most common mistake in fraud modelling. We verified our pipeline has zero leakage by checking that removing these columns drops CV accuracy from ~99% to ~87% (the remaining performance comes from genuine fraud signals, not cheating).

---

## Step 3: Entity Intelligence Features (Phase 2A)

**What:** For each claim, we look at the hospital, doctor, disease, and employer involved — and calculate statistics about their historical behaviour.

**Examples:**
- `hosp_fraud_rate` — what fraction of all past claims at this hospital were fraud?
- `doc_fraud_rate` — what fraction of all past claims by this doctor were fraud?
- `hosp_diag_hhi` — Herfindahl–Hirschman Index. Measures how concentrated a hospital's diagnoses are. A hospital that treats 500 different conditions has low HHI. A hospital that only ever bills for 2 diagnosis codes has suspiciously high HHI.
- `doc_exclusivity` — does this doctor only work with one hospital? High exclusivity is a red flag.
- `emp_fraud_rate` — is this employer's employees making lots of fraud claims?
- `hosp_repeat_patient_rate` — does this hospital keep seeing the same patients repeatedly?

**Why computed only on training data:**
If we compute `hosp_fraud_rate` using ALL data including the test set, then when we evaluate on the test set, the hospital's fraud rate already includes test labels — the model has seen the answer. So we only aggregate from the training rows.

**Critical rule:** Each training sample's entity feature is computed WITHOUT counting itself (this is OOF — see Step 6). This prevents "label leakage."

---

## Step 4: Graph + Geo + Hospital Deep Features (Phase 2A Extra)

**What:** 8 additional features that capture geographic fraud signals and hospital behavioural patterns. These come from building a simple network view of the data (which hospital is in which city, which patients visit which hospitals) rather than just entity statistics.

**The 8 features:**

| Feature | What it measures | Fraud signal |
|---------|-----------------|-------------|
| `city_mismatch` | Is insured's city ≠ hospital's city? | Patients don't normally travel far for routine treatment |
| `state_mismatch` | Is insured's state ≠ hospital's state? | Stronger geographic mismatch signal |
| `hosp_repeat_patient_rate` | % of claims at this hospital from patients who appear multiple times | Fraud rings bring the same people back repeatedly |
| `hosp_weekend_adm_rate` | % of admissions that are on Saturday/Sunday | Staged admissions are often scheduled on weekends to avoid scrutiny |
| `hosp_short_stay_rate` | % of stays with LOS ≤ 1 day | Day-care fraud: admit, file claim, discharge same day |
| `hosp_high_claim_rate` | % of claims above hospital's own 75th percentile | Outlier concentration — a hospital that keeps billing above its own norm |
| `pincode_claim_count` | Total claims filed from this pincode | High-volume pincodes = claim hotspots |
| `pincode_hosp_diversity` | Unique hospitals used per pincode | One pincode funnelling claims through many hospitals = suspicious network |

**All aggregations train-only:** Just like entity features, these statistics are computed from training rows only. The test set looks up the values for its hospitals/pincodes.

**Why these aren't in the original entity cell:**
They require different joins (patient × hospital for repeat rate, pincode × hospital for diversity). They're computed as a separate pass over the data to keep the code organised.

---

## Step 5: Velocity Features (Phase 2A Group 3)

**What:** How many claims has this entity made in the last 30 days? Last 90 days?

**Examples:**
- `hosp_claims_30d` — number of claims at this hospital in the past 30 days
- `doc_claims_90d` — number of claims by this doctor in the past 90 days

**Why float64 matters (not float32):**
We use rolling time windows. The rolling aggregation in pandas returns float64 naturally. If we accidentally cast to float32, counts like `3.0` become `3.0` fine, but very large numbers (like 100,000) get rounded due to float32's limited precision. The feature becomes noisy and drops in importance. We explicitly keep float64.

**Why it works:**
A legitimate doctor might see 5 patients per month. A fraud ring doctor might file 200 claims in 30 days. Velocity catches bursts.

---

## Step 6: OOF Target Encoding (Phase 2C)

**What:** We turn categorical entity IDs (hospital code, doctor ID, etc.) into numbers by replacing them with their fraud rate — but using Out-Of-Fold (OOF) to prevent leakage.

---

**The leakage problem WITHOUT OOF — concrete example:**

Hospital X has 4 claims in training:

| Claim | Fraud? |
|-------|--------|
| Claim 1 | Yes |
| Claim 2 | No |
| Claim 3 | Yes |
| Claim 4 | No |

Naive encoding: Hospital X fraud rate = 2/4 = **0.50**. Assign 0.50 to ALL 4 claims.

Now when the model trains on Claim 1, it sees `hosp_fraud_rate = 0.50` and label = `Fraud`. But that 0.50 was computed partly USING Claim 1's own label. The model has been subtly told "this claim contributed to the fraud rate." It memorises the training data instead of learning a pattern.

This inflates training accuracy and collapses on test data.

---

**OOF solution — concrete example:**

Hospital X has 5 claims. We split into 2 chronological folds:

| Claim | Fold | Fraud? |
|-------|------|--------|
| Claim 1 | Fold 1 | No |
| Claim 2 | Fold 1 | No |
| Claim 3 | Fold 1 | No |
| Claim 4 | Fold 1 | No |
| **Claim 5** | **Fold 2** | **Yes** ← the one fraud claim |

**Without OOF (naive):**
Rate = 1 fraud / 5 claims = **0.20** → assigned to ALL 5 claims including Claim 5.
The model trains on Claim 5 with `hosp_fraud_rate = 0.20` and label = Fraud.
That 0.20 was computed using Claim 5's own fraud label. The feature already "knows" about the fraud.

**With OOF:**
- Fold 1 claims (1–4) → look at Fold 2 only: 1 fraud / 1 claim = **1.00**
- Fold 2 claims (Claim 5) → look at Fold 1 only: 0 fraud / 4 claims = **0.00**

Claim 5 now gets `hosp_fraud_rate_oof = 0.00` with label = Fraud.

**What the model now has to learn:**
"This claim came from a hospital with zero prior fraud history — yet it IS fraud. There must be something in the other features (claim amount, diagnosis, doctor, velocity) that makes it suspicious."

That is genuine learning. Without OOF the model just memorises "hospital with rate=0.20 → fraud" which is a circular hint, not a pattern.

**Real scenario where it matters:**

Hospital Y has 10 claims: 9 are not fraud, 1 is fraud (Claim 7).
- Without OOF: Claim 7 gets `hosp_fraud_rate = 1/10 = 0.10`
- With OOF: Claim 7 gets `hosp_fraud_rate` computed from the other 9 claims only = `0/9 = 0.0`

Now when the model trains, Claim 7 has `hosp_fraud_rate = 0.0` (no fraud history at this hospital based on other claims) but label = Fraud. The model learns: "this was an unusual event at an otherwise clean hospital" — which is genuine signal, not a leak.

---

**In our notebook — 5 time-sorted folds (expanding window, past-only):**

```
Fold 1  [Jan–Apr 2024]   ← encoded using global prior only (no past data yet)
Fold 2  [May–Aug 2024]   ← encoded using Fold 1
Fold 3  [Sep–Dec 2024]   ← encoded using Folds 1+2
Fold 4  [Jan–Apr 2025]   ← encoded using Folds 1+2+3
Fold 5  [May–Jun 2025]   ← encoded using Folds 1+2+3+4
```

**The rule:** encoding data time < sample time. A claim is NEVER encoded using data from after its own date.

**Why expanding window, not leave-one-out?**
An alternative is: for each fold, use all OTHER folds (including future ones). This would give richer statistics — but it leaks the future. A Jan 2024 claim would be encoded using fraud rates computed from 2025 data. The model would learn patterns from future information that don't exist at prediction time. Expanding window is the only correct approach for temporal data.

**Why time-sorted folds (not random folds)?**
If we used random folds, Fold 1 might contain a mix of Jan 2024 and Jun 2025 claims. When computing Fold 1's entity rates from the other folds, we'd be using Jun 2025 data to encode a Jan 2024 claim — that's future data leaking into the past. Time-sorted folds keep the temporal integrity intact.

**Fold 1 has no prior data — what happens?**
Every entity in Fold 1 gets the global training mean (via additive smoothing fallback). That's the best estimate when you have no history. It's a conservative encoding — the model knows nothing special about that hospital yet.

**Additive smoothing:**
`rate = (fraud_count + k × global_mean) / (claim_count + k)` where `k = 10`

A hospital with only 1 claim gets pulled strongly towards the global average. A hospital with 500 claims is barely affected. This prevents rare entities from having extreme (noisy) rates.

**For the test set:**
We compute entity rates from ALL training data (all 5 folds combined). This is safe — test labels are never used in the calculation. Test claims simply look up: "what is Hospital X's fraud rate over the entire 18-month training period?"

**Result:** 8 OOF-encoded features — one each for hospital, doctor, disease code, employer (fraud + investigation targets each → 4 entities × 2 targets = 8 features).

---

## Step 7: Auxiliary Target OOF Meta-Features

**What:** Instead of only encoding the main fraud/investigation targets, we build 5 proxy sub-models that predict related billing outcomes — and use their OOF predictions as additional features.

**Why proxy targets?**
Fraud correlates with many billing anomalies that don't require the main fraud label to compute. For example: a claim where the amount exceeds the sum insured is suspicious regardless of whether it was ever flagged as fraud. By training separate models to predict these proxies, we extract billing-pattern signals that the main model might miss.

**The 3 proxy targets created:**

| Feature name | Proxy target | Why it signals fraud |
|-------------|-------------|---------------------|
| `oof_aux_claim_inflated` | Claimed_Amt > Total_Bill (binary) | Billing inflation — claiming more than the actual hospital bill |
| `oof_aux_si_exhaustion` | Claimed_Amt > Sum_Insured (binary) | Coverage abuse — claiming above the policy limit |
| `oof_aux_claim_to_si` | Claimed_Amt / Sum_Insured (continuous) | How aggressively is the policy being used? |

*(Two more proxies using Approved_Amount_INR are skipped — that column is dropped as leaky in Step 2.)*

**How it works:**
1. For each proxy target, train a LightGBM model via 5-fold **expanding-window OOF** on all entity + velocity + OOF-encoded features computed so far
2. The OOF predictions (training) and full-train predictions (test) become new columns in `feat`
3. These 3 meta-features are then available to all downstream models

**Why OOF for the meta-features too?**
Same reason as Step 6. If we trained a model on the full training data and then used its predictions as features, the model has already "seen" every training label. OOF ensures the meta-feature for each training claim is generated from a model that never saw that claim's label.

**Why LightGBM for the sub-models?**
Fast, handles mixed features and NaN natively, and 300 estimators is sufficient for a proxy target. No hypertuning needed — we just want a reasonable prediction, not the best possible.

---

## Step 8: GNN — Graph Neural Network (Late Fusion)

**What:** A Graph Neural Network that learns embeddings by treating the insurance network as a graph.

---

### Which GNN training type do we use?

| Training type | Prediction target | Used here? |
|---------------|-------------------|------------|
| **Node classification** | Label per node | **YES — our approach** |
| Link prediction | Edge (connection) exists? | No |
| Graph classification | Label for whole graph | No |
| Self-supervised | Embeddings without labels | No |
| Sampling training | Scalability for huge graphs | No |
| Temporal GNN | Dynamic/changing graphs | No |

We use **node classification**: each claim is a node, and we train the GNN to predict `y_fraud` and `y_inv` directly on claim nodes using a `train_mask`. The GNN learns node embeddings that are optimised specifically to discriminate fraud from non-fraud.

---

### The graph structure

**7 node types:**
- `claim` — each insurance claim (474K nodes, the primary node)
- `hospital` — hospital entity (HID_anon)
- `doctor` — treating doctor (Treating_Dr)
- `client` — corporate insurance client (CID_anon, 33 clients)
- `pincode` — geographic pincode
- `disease` — disease category
- `employer` — employer entity

**Edges (undirected, so messages flow both ways):**
- `claim → hospital` (claim was filed at this hospital)
- `claim → doctor` (claim was treated by this doctor)
- `claim → client` (claim belongs to this corporate client)
- `claim → pincode` (claim's pincode)
- `claim → disease` (claim's diagnosis category)
- `claim → employer` (claim's employer)

---

### Why a GNN at all? — Manual features vs learned embeddings

**You could ask:** can't we just compute the features the GNN uses directly, without training a model?

**For 1-hop features — yes, we already do:**

| Feature | How computed | Where |
|---------|-------------|-------|
| `hosp_fraud_rate` | `fraud_count(Hospital X) / total_claims(Hospital X)` | Phase 2A |
| `doc_fraud_rate` | `fraud_count(Doctor Y) / total_claims(Doctor Y)` | Phase 2A |

These are simple 1-hop aggregations. The GNN's first layer does essentially this too.

**For 2+ hop features — this is where manual computation breaks down:**

The GNN runs **4 layers** of message passing. Each layer pushes information one hop further through the graph:

```
Layer 1: claim learns from → its hospital, doctor, pincode, employer, disease
Layer 2: claim learns from → other claims at the same hospital
          = "what do co-patients at my hospital look like?"
Layer 3: claim learns from → other doctors at those co-patients' hospitals
          = "who else works at the hospitals my co-patients use?"
Layer 4: even deeper cross-entity patterns
```

After 4 layers, a claim's 128-dim embedding encodes its **entire local subgraph** — not just its hospital's fraud rate, but the fraud rates of every doctor at that hospital, the pincodes those doctors' other patients come from, the employers of those patients, and so on.

You could manually compute `hosp_avg_doc_fraud_rate` (average fraud rate of all doctors at hospital X). But with 7 node types and 4 hops, the number of possible cross-entity combinations explodes. You'd be guessing which combinations matter. The GNN finds which ones are discriminative automatically — via gradient descent on the actual fraud labels.

**The embeddings are LEARNED, not just computed:**
A manual feature `hosp_fraud_rate = 0.15` is a single number.
A GNN embedding for Hospital X is a **128-dimensional vector** where every dimension is a non-linear transformation of its full neighbourhood, jointly optimised to separate fraud from non-fraud. You cannot reverse-engineer that into named features — it is an implicit, learned representation.

---

### Architecture — ResHeteroSAGE

4-layer heterogeneous GraphSAGE with residual connections:
```
Input (per node type, different dims)
  → Layer 1: → 256 hidden  (LayerNorm + ELU + Dropout 0.3)
  → Layer 2: → 256 hidden  + residual connection
  → Layer 3: → 256 hidden  + residual connection
  → Layer 4: → 128 output  (the embedding)
```

- **Residual connections** (layers 2–3): prevents vanishing gradients in deeper layers — same idea as ResNet in computer vision
- **LayerNorm per node type**: different node types (claim vs hospital vs doctor) have different feature scales; normalising each type separately stabilises training
- **`aggr="mean"`**: each node averages its neighbours' messages — robust to variable-degree nodes
- **Trained with focal loss** (gamma=2): downweights easy negatives, focuses learning on hard fraud cases

---

### Why it's NOT in X_train (the main feature matrix)

GNN produces 128-dimensional embeddings per claim. When added directly to the 200-column feature matrix, adversarial validation flagged them as highly shift-prone — the GNN was trained on the full 18-month graph, but the test period (Jul–Dec 2025) has a different network structure (new hospitals, new fraud patterns). Adding them directly made models worse.

**Solution — Late Fusion:**
Train a SEPARATE LightGBM on only the GNN embeddings. Blend its output with the main model at the probability level:
`final_prob = 0.5 × main_prob + 0.5 × gnn_prob`

The GNN contributes signal without contaminating the main feature space.

**Why it barely helps (+1% PR-AUC):**
The same temporal shift that hurts all features hits the GNN even harder. The graph structure from 2024 is a poor proxy for the 2025 test graph. This is fundamentally a **static GNN trained on a dynamic fraud network** — a Temporal GNN (which tracks graph evolution over time) would be the right tool, but is significantly more complex to build.

**Why GNN embeddings are separate:**
The GNN is trained in `gnn_features_colab.ipynb` (on Google Colab for GPU) and saved to `gnn_embeddings.parquet`. We load those embeddings here. The GNN itself is too computationally expensive to retrain every notebook run.

---

## Step 9: Anomaly Scores (Phase 4)

**What:** Two unsupervised models that score how "abnormal" each claim is, without ever seeing the labels.

**Model 1 — Denoising Autoencoder (DAE):**
An autoencoder is trained to compress a claim's features and reconstruct them. Normal claims reconstruct well. Anomalous claims (unusual combinations of diagnosis, hospital, amount) reconstruct poorly — high reconstruction error = suspicious.

**Model 2 — Isolation Forest:**
Builds random decision trees that try to isolate (separate) each data point. Normal points take many splits to isolate (they live in dense regions). Anomalous points are isolated quickly (they live in sparse regions). Score = how quickly isolated = how anomalous.

**Why unsupervised?**
Labels are noisy and incomplete. Many actual frauds go uncaught. Unsupervised methods find claims that are statistically unusual regardless of whether they were caught. This captures fraud patterns the labels miss.

**Key rule:** Both models are trained ONLY on training data, then scored on both train and test. Never use test labels.

---

## Step 10: Text / Diagnosis Embedding Features

**What:** Convert free-text diagnosis and treatment descriptions into numeric features using TF-IDF and dimensionality reduction.

**Columns used:**
- `Final_Diagnosis` — free text (e.g. "OPD Dental", "Fever with cough and cold")
- `Treatment_Type` — procedure description (e.g. "OPD- Dental treatment", "Elective Surgery")
- `Code_TLD` — ICD-style code string (e.g. "(Z01)", "(A09)")

**Method:**
1. TF-IDF vectorisation: converts each text into a sparse bag-of-words vector, weighted by how distinctive each term is across all documents
2. TruncatedSVD: compresses the sparse TF-IDF matrix into 12 dense numeric components per column
3. Result: up to 36 new numeric columns added to `feat`

**Why this captures fraud:**
Fraudsters reuse diagnosis descriptions. A fraud ring billing for "OPD Dental" across hundreds of claims filed on the same day has a detectable textual pattern. A legitimate hospital shows varied, specific diagnoses. TF-IDF captures these vocabulary distributions.

**Train-only fitting:**
The TF-IDF vocabulary and SVD components are fitted on training text only, then applied to test. This prevents test vocabulary from influencing the encoding.

**Adversarial safety net:**
If any of these text features are too distribution-shifted between train and test (e.g. diagnosis codes used only in test period), the RF adversarial validation in Step 12 will drop them automatically.

---

## Step 11: engineer_features Function

**What:** Creates 40+ additional derived features from the raw columns.

**Examples of what's computed:**
- `claim_to_si_ratio` — claimed amount / sum insured. High ratio = suspicious
- `los_vs_expected` — actual length of stay vs. expected for the diagnosis
- `amt_per_day` — amount claimed per day of hospitalisation
- `has_hosp_cash` — binary flag: does this policy have hospital cash benefit?
- `addon_to_base` — ratio of add-on cover to base premium
- `z_score_amt_hosp` — how many standard deviations is this claim amount from the hospital's mean? (Computed from train stats only, never test stats)
- `high_si_low_premium` — flag for unusually high sum insured relative to premium paid

**Train-only statistics:**
Z-scores and ratios that compare to a hospital's or doctor's "normal" are computed from training data mean/std, then applied to test. If we computed test z-scores using test statistics, we'd be implicitly using future information.

---

## Step 12: prepare_features Function

**What:** Encodes categorical columns and handles missing values.

**Frequency encoding:**
For categorical columns like `hospital_city` or `diagnosis_code`, we replace each category with how often it appears in the data. "Mumbai" appears 50,000 times → encoded as 50000. This is better than one-hot encoding for high-cardinality columns because it doesn't create thousands of columns.

**Why combined train+test for frequency encoding:**
Frequency encoding doesn't use the label — it just counts occurrences. Using combined data gives better frequency estimates for rare categories. This is NOT leakage because no target information is used.

**Imputation:**
Missing values filled with -1 (a sentinel value that tree models handle naturally — they can split on "is this -1 or not").

---

## Step 13: Train/Test Arrays + RobustScaler + Sample Weights

**RobustScaler:**
Normalises features using median and interquartile range instead of mean and std. Why? Insurance claim amounts have extreme outliers (a ₹10L claim vs a ₹1Cr claim). Standard z-score normalisation would be dominated by those outliers. RobustScaler clips their influence.

**Why scale at all for tree models?**
Strictly, Random Forest and LightGBM don't need scaling — they split on thresholds, not distances. We scale for the Neural Network, which does need normalised inputs. We apply the same scaler to all models for consistency.

**Time-decay sample weights:**
`weight = exp(0.001 × days_since_start_of_training)`

Older claims (Jan 2024) get weight ~1.0. Recent claims (Jun 2025) get weight ~1.7. This tells the model: recent fraud patterns matter more than old ones. The test set is Jul–Dec 2025 — claims from Jun 2025 are more representative of that than claims from Jan 2024.

**Why exp(0.001) and not something stronger?**
0.001 per day is a gentle decay — it upweights recent claims by about 70% over 18 months. Too strong a decay (like 0.01/day) would effectively train on only the last few months, losing too much data.

---

## Step 14: RF Adversarial Validation (Feature Dropping)

**What:** A technique to find and remove features that are "too different" between train and test, to make the model more generalisable.

**How it works — step by step:**
1. Label all training rows as class `0` ("train") and all test rows as class `1` ("test")
2. Train a Random Forest to predict: is this row from train or test?
3. If the RF achieves AUC = 1.0 (perfect), the train and test distributions are completely different
4. Look at which features the RF used most to tell them apart — these are the most "shift-prone" features
5. Drop any feature with importance > 0.008 (our threshold)

**Why this works:**
If a feature looks very different in train vs test, the model will learn a pattern from that feature that doesn't hold in test. For example, if `claim_month` is very different (train = Jan–Jun, test = Jul–Dec), any pattern the model learns from `claim_month` won't generalise.

**Result in our notebook:**
- Before drop: ~225 features (including all new graph, aux OOF, text features)
- After drop: ~200 features (shift-prone ones removed automatically)
- Adversarial AUC = 0.9961 — even after dropping, train and test are extremely different. This is expected — 18 months of distribution shift is fundamental.

**Why RF specifically for adversarial validation?**
Random Forest is fast, doesn't need hypertuning, and gives reliable feature importance scores. We don't need a perfect adversarial classifier — we just need importance rankings.

**Safety net for new features:**
Any new feature added (graph features, text embeddings, aux OOF) that happens to be distribution-shifted will automatically be dropped here. We don't need to manually vet each new feature.

---

## Step 15: Evaluation Metric — PR-AUC (Not ROC-AUC)

**PR-AUC (Precision-Recall Area Under Curve):**
- Precision = of all claims we flagged, what fraction were actually fraud?
- Recall = of all actual fraud claims, what fraction did we catch?
- PR-AUC = area under the precision-recall curve as you vary the threshold
- Higher = better. Random classifier gets PR-AUC ≈ fraud rate (e.g., 0.02 for 2% fraud)

**Why not ROC-AUC?**
ROC-AUC measures how well you separate positives from negatives. For 98% negative class, a model that predicts "not fraud" for everyone gets ROC-AUC ~0.5. But it can still get a high ROC-AUC by correctly ranking the very obvious frauds. ROC-AUC inflates with class imbalance.

PR-AUC is harder to game — it directly measures whether your flagged cases are actually fraud (precision) while catching enough real fraud (recall).

**Threshold optimisation:**
After training, we find the threshold `t` such that:
- Recall ≥ 0.60 (we must catch at least 60% of actual fraud)
- Among valid thresholds, maximise F1

Why min_recall=0.6? Because missing fraud (false negative) is more costly than a false alarm (false positive). An investigator can dismiss a false alarm — but a missed fraud costs the company money.

---

## Step 16: Class Imbalance Handling

**The problem:** 93% of claims are NOT investigation (7% are). A model that predicts "not investigation" for everything would be 93% accurate — and completely useless.

**Our solution: scale_pos_weight**
`scale_pos_weight = count(negative) / count(positive) = ~13 for investigation, ~35 for fraud`

This tells LightGBM/XGBoost: "treat each positive (fraud) sample as if it appears 13× more than it does." Effectively multiplies the loss for misclassifying a fraud case.

**Why not SMOTE (Synthetic Minority Oversampling)?**
SMOTE creates synthetic fraud cases by interpolating between existing fraud cases. But our train/test shift is so extreme (train data looks completely different from test) that synthetic samples from the training distribution would make the problem worse — they'd reinforce the wrong patterns.

**Why not focal loss?**
Focal loss downweights easy negatives. With `alpha=0.75` and a 1:35 imbalance, the math works out such that ALL samples get downweighted to near zero — the model predicts nothing. You'd need `alpha≈0.97` to make it work, which we tested and found still inferior to scale_pos_weight.

---

## Step 17: Models Used

### Random Forest
- Builds 300–400 decision trees, each on a random subset of features and rows
- Final prediction = average of all trees
- **Why for fraud?** Bagging (random subsets) makes it more robust to distribution shift. Each tree sees a slightly different version of the data, so the ensemble is less sensitive to any one unusual pattern.

### LightGBM
- Gradient boosting: builds trees sequentially, each correcting the errors of the previous
- Very fast due to histogram-based splitting
- **Key params in our notebook:**
  - `num_leaves=63` — controls tree complexity. More leaves = more complex patterns.
  - `min_child_samples=50` — minimum samples in a leaf. Prevents memorising rare patterns.
  - `reg_lambda=10` — L2 regularisation. Penalises large weights. Prevents overfitting to training noise.
  - `colsample_bytree=0.6` — use only 60% of features per tree. Forces diversity.

### XGBoost
- Similar gradient boosting to LightGBM, slightly different algorithm
- `tree_method='hist'` for speed
- Generally comparable to LightGBM; included for ensemble diversity

### CatBoost
- Gradient boosting optimised for categorical features
- Has built-in ordered target encoding that prevents leakage internally
- **Known issue:** Cannot use in sklearn's VotingClassifier (clone bug). We manually average probabilities.

### Neural Network (PyTorch)
- 4-layer MLP: [n_features → 256 → 128 → 64 → 1]
- BatchNorm between layers, Dropout 0.3 for regularisation
- `BCEWithLogitsLoss` with `pos_weight` for imbalance
- **Why included?** Neural networks can learn non-linear interactions between features that tree models miss. But they're more sensitive to distribution shift, so they don't always win.

---

## Step 18: Window Training

**Problem:** Model trained on Jan 2024–Jun 2025 learns old patterns. By Jul 2025, fraud patterns have evolved.

**Solution:** Train only on the last N months of training data (the "window").

**For Investigation:**
- 5-month window (Feb–Jun 2025): RF gets 0.1478 vs 0.1385 full train (+6.7%)
- Why 5mo? 4mo has too little data; 6mo includes too much old data

**For Fraud:**
- 3-month window + LGBM: 0.0591 vs 0.0564 full train (+4.8%)
- RF with window HURTS fraud (see Rolling Entity section for why)

**Time-decay weights within window:**
Even within the 5-month window, we recompute time-decay weights so that claims from month 5 are weighted more than claims from month 1.

---

## Step 19: Ensemble

**What:** Take the top-3 models by PR-AUC, average their probability outputs.

`final_probability = (prob_model1 + prob_model2 + prob_model3) / 3`

**Why top-3, not all?**
Adding weak models dilutes the ensemble. The worst model drags down the average.

**Why probability averaging, not majority voting?**
Majority voting loses the calibration — a model that's 95% confident and one that's 51% confident both count as "1 vote." Probability averaging preserves the confidence information.

**Why not VotingClassifier from sklearn?**
CatBoost has a bug with sklearn's `clone()` function which VotingClassifier uses internally. It crashes. We manually average instead.

---

## Step 20: Rolling Entity Features (Breakthrough)

**The root cause problem:**
`hosp_fraud_rate` is computed over Jan 2024–Jun 2025 (18 months). But we're predicting Jul–Dec 2025. Hospital fraud patterns change — a hospital that was legitimate in 2024 might be running a fraud ring by mid-2025.

**The fix:**
Instead of using 18-month entity statistics, recompute them from only the last 3 months (Apr–Jun 2025). These are then used as features for ALL training rows (not just those 3 months).

**Why "all training rows" but "3-month statistics"?**
- Statistics: computed from Apr–Jun 2025 (most recent, most relevant to test)
- Training data: all 415K rows (we need enough data for the model to learn)

This is the key insight: we separate "what statistics to use as features" from "how much data to train on."

**Results:**
| | Old (18mo stats) | New (3mo stats) |
|--|--|--|
| Investigation RF | 0.1385 | 0.2466 (+78%) |
| Investigation LGBM | 0.0965 | 0.2724 (+182%) |
| Fraud RF | 0.0546 | 0.1442 (+164%) |
| Fraud LGBM | 0.0564 | 0.1574 (+179%) |

**Why 3 months, not 1 or 6?**
- 1 month: too little data, statistics are noisy
- 6 months: includes older patterns that are less relevant to the test period
- 3 months (Apr–Jun 2025): directly precedes the test period (Jul–Dec 2025), capturing current fraud behaviour

---

## Step 21: GroupKFold Temporal Stability (Phase 3)

**What:** Cross-validation on the training data using months as groups.

**How:**
- Split training months into 6 chronological folds
- Train on folds 1–5, test on fold 6. Repeat for each fold.
- Report PR-AUC mean ± standard deviation across folds

**Why GroupKFold (not regular KFold)?**
Regular KFold would mix January and June data randomly. That means a model trained on 80% of January data tests on the other 20% of January — easy! GroupKFold ensures you always predict future months from past months — the real problem.

**What we learned:**
- CV PR-AUC ≈ 0.87–0.93 (high — within training, model is great)
- Test PR-AUC ≈ 0.27 (real test — massive drop)
- Only 30% of CV performance survives to test

This "30% retention" confirms the distribution shift is the main bottleneck, not model capacity.

**Note:** This cell evaluates the standard `X_tr` feature matrix (not `X_tr_re`), so it measures the window models, not the rolling entity models. It is a diagnostic tool, not a final benchmark.

---

## Step 22: PR and ROC Curves

**PR Curve:**
X-axis = Recall (0 to 1), Y-axis = Precision (0 to 1)
As you lower your threshold (flag more claims), recall goes up but precision drops.
The curve shows the precision-recall trade-off at every possible threshold.
Area under this curve = PR-AUC.

**ROC Curve:**
X-axis = False Positive Rate, Y-axis = True Positive Rate
Area under = ROC-AUC. Less informative for imbalanced problems but shown for completeness.

---

## Step 23: Final Summary

**What the last cell does:**
1. Recomputes `best_inv` and `best_fraud` from the full results list (including rolling entity models)
2. Prints the winner, its PR-AUC, ROC-AUC
3. Prints classification report at the optimised threshold

**Why recompute?**
The rolling entity models (RE-LGBM) are trained AFTER the ensemble cell that originally set `best_inv`. Without recomputing, the final summary would still report the old winner (Window RF) instead of RE-LGBM.

---

## Summary Table: Every Technique and Why

| Technique | What It Does | Why We Use It |
|-----------|-------------|---------------|
| Temporal train/test split | Train < Jul 2025, Test >= Jul 2025 | Real deployment scenario; no future leakage |
| Remove leaky columns | Drop Approved_Amount, outcome columns | These don't exist at prediction time |
| Entity features (fraud rate, HHI) | Summarise hospital/doctor history | Network-level patterns catch organised fraud |
| Graph + Geo features | City/state mismatch, weekend admissions, short-stay, pincode hotspots | Network view of fraud; geographic and behavioural signals |
| Velocity features (30d/90d counts) | Count recent claims per entity | Burst patterns indicate fraud rings |
| OOF target encoding (expanding window) | Encode entity IDs using only past data | No future leakage; correct temporal structure |
| Auxiliary OOF meta-features | Sub-models predict billing proxies (claim inflation, SI exhaustion) | Billing anomaly signals without requiring main fraud labels |
| GNN embeddings (late fusion) | Graph relationships between entities | Propagates fraud signal through the network |
| Anomaly scores (DAE + IsoForest) | Flag statistically unusual claims | Catches fraud the labels miss |
| Text embeddings (TF-IDF + SVD) | Encode diagnosis/treatment text into 36 numeric features | Detects repeated diagnosis patterns used by fraud rings |
| engineer_features | 40+ derived ratios and z-scores | Domain knowledge encoded as numbers |
| RobustScaler | Normalise using median/IQR | Outlier-robust; needed for Neural Network |
| Time-decay weights | Upweight recent claims | Test period is recent; recent patterns matter more |
| RF Adversarial Validation | Drop features that differ most between train/test | Automatically removes shift-prone features including new ones |
| scale_pos_weight | Weight fraud cases 13–35x higher | Fixes class imbalance without synthetic data |
| Window training (3–5 months) | Train only on recent data | Recent patterns more predictive of future |
| Rolling entity features (3mo) | Refresh entity stats from last 3 months | Entity fraud rates change; fresh stats = better signal |
| Ensemble (top-3 avg) | Average probabilities of 3 best models | Reduces variance; no single model dominates |
| PR-AUC metric | Area under precision-recall curve | Correct metric for imbalanced fraud detection |
| Threshold optimisation (min_recall=0.6) | Find threshold that catches ≥60% fraud | Missing fraud is costlier than false alarms |
| GroupKFold CV | Time-sorted cross-validation | Diagnoses overfitting vs distribution shift |
| Save/load results | Persist results to disk | Avoid rerunning expensive training every session |

---

## The One-Line Answer to "Why Are Your Results Lower Than Their Notebook?"

Their notebook trains on data from May–June 2025 and then tests on that same data (their test set overlaps with their training set by 2 months). Our test is strictly future data (Jul 2025 onwards) that the model has never seen in any form. Our 0.27 is real. Their 0.41 is not.
