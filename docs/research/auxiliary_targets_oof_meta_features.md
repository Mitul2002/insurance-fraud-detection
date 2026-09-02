# Auxiliary targets and OOF meta-features for insurance fraud detection

**Health insurance fraud detection at 0.1–2% positive rates demands more than a single binary classifier.** The most effective pipelines combine weak label engineering, out-of-fold (OOF) meta-feature stacking, multi-task learning, pseudo-labeling, and temporal shift handling into one unified system. This guide provides the complete technical blueprint—architecture choices, loss functions, cross-validation strategies, and production code patterns—for building such a pipeline on group corporate hospitalization claims data with temporal train/test splits. Every technique discussed has been validated in either peer-reviewed research or top Kaggle competition solutions (IEEE-CIS Fraud Detection, AmEx Default Prediction), and the guide addresses the specific signals available: binary fraud labels, claim-level approval/rejection rates, ICD code patterns, and provider behavioral data.

---

## A. Turning approval and rejection rates into engineered weak labels

Insurer approval/rejection decisions are **noisy proxies** for fraud—not all rejected claims are fraudulent, and some approved claims are. The key is to treat these signals as weak labels and apply noise-aware learning methods.

### Converting insurer decisions to probabilistic fraud scores

The most principled approach models the relationship between observed labels (approved/rejected) and true fraud status via a **noise transition matrix** T, where T[i,j] = P(observed=i | true=j). For binary fraud detection, this 2×2 matrix has only two free parameters: P(labeled_fraud | true_legit) and P(labeled_legit | true_fraud). These can be estimated empirically from a small audit sample, or via the anchor point assumption (finding examples where the model is near-certain of the true label).

For direct conversion, a practical formula is:

```python
# Convert rejection decisions to soft fraud scores
fraud_soft_label = np.clip(
    rejection_probability * calibration_factor,  # calibration_factor from audit sample
    0.05, 0.95  # avoid hard 0/1 labels
)
```

**Confident Learning (cleanlab)** provides the most mature framework for this. It estimates the joint distribution between noisy and true labels Q(ỹ, y*) using OOF predicted probabilities, identifies mislabeled examples, and enables noise-robust training with any scikit-learn-compatible classifier:

```python
from cleanlab.classification import CleanLearning
from xgboost import XGBClassifier

cl = CleanLearning(XGBClassifier(scale_pos_weight=50), cv_n_folds=5)
# Find claims where rejection labels likely disagree with true fraud status
label_issues = cl.find_label_issues(X_train, y_noisy_rejection_labels)
# Train robust model that automatically handles label noise
cl.fit(X_train, y_noisy_rejection_labels)
```

If you already have OOF predictions from your own cross-validation pipeline, pass them directly:

```python
from cleanlab.filter import find_label_issues
ranked_issues = find_label_issues(
    labels=y_train, pred_probs=oof_pred_probs,
    return_indices_ranked_by="self_confidence"
)
```

**Label smoothing** offers a simpler alternative: replace hard labels {0, 1} with soft targets {ε/2, 1−ε/2} where ε ≈ 0.1. Research on tabular EHR data (PMC, 2024) confirms this helps but requires adaptation from image-domain methods. GBDTs show **natural robustness** to symmetric label noise in early training stages—early stopping is often sufficient.

**Cross-fit noise correction** (Forward T, Patrini et al. 2017) modifies the loss function as L_corrected = T⁻¹ · L_noisy, producing an unbiased estimator of the clean loss. The FasTEN method (ECCV 2022) efficiently estimates T using a two-head architecture with a label correction threshold safeguard.

### Provider-level rejection rate aggregations as auxiliary targets

Provider-level statistics create powerful auxiliary targets for claim-level models. The key aggregations:

```python
provider_features = claims.groupby('provider_id').agg(
    mean_rejection_rate=('rejected', 'mean'),
    rejection_rate_std=('rejected', 'std'),
    claim_volume=('claim_id', 'count'),
    mean_claim_amount=('amount', 'mean'),
    specialty_adjusted_rate=('rejection_residual', 'mean'),  # vs peer group
    icd_diversity=('icd_code', 'nunique'),
).reset_index()
```

Research from NBER (Shekhar et al.) represents hospitals through ICD code frequency distributions and uses ensemble subspace outlier detectors (SOD, Isolation Forest, LODA) on provider ICD profiles. Aggregated anomaly rankings via rank-based voting produce robust provider-level fraud scores. These become either **auxiliary regression targets** for multi-task learning or **input features** for the claim-level model.

### ICD code anomaly scoring as auxiliary targets

Three approaches ranked by effectiveness:

- **Autoencoder reconstruction error**: Train deep autoencoders on ICD+CPT code combinations; high reconstruction error flags anomalous code pairs. Filter rare CPT codes (<100 occurrences) to avoid flagging rarity as anomaly.
- **DRG upcoding detection**: Compare expected DRG assignment given ICD codes vs. actual DRG claimed. Excess cost relative to peer group (same specialty, region) indicates potential upcoding.
- **Subspace outlier detection**: Use Isolation Forest on ICD frequency vectors per provider, producing a continuous anomaly score that serves as a regression auxiliary target.

---

## B. OOF meta-feature construction done right

### Purged walk-forward CV for temporally ordered fraud data

Standard K-fold is **catastrophic** for temporal fraud data because it leaks future patterns into training. The correct approach uses purged walk-forward cross-validation with embargo periods.

**Purging** removes any training observation whose label horizon overlaps with the test period. **Embargoing** adds a temporal buffer after each test fold boundary to handle residual serial correlation. For health insurance claims, the embargo should match the maximum label delay period—typically **30–90 days** for fraud investigation outcomes.

```python
from sklearn.model_selection import TimeSeriesSplit

# Simple approach: sklearn with gap parameter
tscv = TimeSeriesSplit(n_splits=5, test_size=1000, gap=90)  # 90-day embargo

# Full implementation: PurgedGroupTimeSeriesSplit
cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=np.inf,
    group_gap=5  # number of time groups to purge
)
for train_idx, test_idx in cv.split(X, y, groups=month_groups):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
```

**Combinatorial Purged CV** (CPCV, from López de Prado) generates C(N,k) test combinations from N sequential groups, producing a distribution of performance estimates rather than a single point estimate. The `skfolio` library implements this:

```python
from skfolio.model_selection import CombinatorialPurgedCV
cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1, embargo_size=1)
```

### Generating OOF predictions from multiple auxiliary targets

The core pipeline generates OOF predictions from diverse models and targets, then uses these as meta-features:

```python
import lightgbm as lgb
import numpy as np

def generate_oof_predictions(X_train, y_train, X_test, params, cv_splitter):
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups)):
        dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
        
        model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=10000,
                          callbacks=[lgb.early_stopping(100)])
        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])
        test_preds += model.predict(X_test) / cv_splitter.n_splits
    
    return oof_preds, test_preds

# Generate OOF from multiple targets
oof_fraud, test_fraud = generate_oof_predictions(X, y_fraud, X_test, params, cv)
oof_approval, test_approval = generate_oof_predictions(X, y_approval_rate, X_test, params, cv)
oof_rejection, test_rejection = generate_oof_predictions(X, y_rejection_rate, X_test, params, cv)
oof_anomaly, test_anomaly = generate_oof_predictions(X, y_icd_anomaly, X_test, params, cv)

# Stack as meta-features
meta_train = np.column_stack([oof_fraud, oof_approval, oof_rejection, oof_anomaly])
meta_test = np.column_stack([test_fraud, test_approval, test_rejection, test_anomaly])

# Add derived meta-features (Deotte's approach)
meta_train_confidence = np.std(meta_train, axis=1)
meta_train_consensus = np.mean(meta_train, axis=1)
```

Chris Deotte's 1st-place Kaggle solution (April 2025) demonstrated four distinct prediction targets from the same data: **predict target directly**, **predict ratio** (target/key feature), **predict residual** from a linear baseline, and **predict missing features**. Each creates a separate OOF column that stacks well because it captures different aspects of the data.

### Leakage prevention checklist

- **Target encoding must be OOF**: For temporal data, use only prior data (expanding window), never future data
- **All OOF predictions use the same CV strategy**: Purged time-series CV with identical fold assignments
- **Never use future data for rolling aggregates**: All temporal features must look backward only
- **Adversarial validation**: Train a classifier to distinguish train vs. test—features with AUC > 0.7 are likely leaky
- **Time Consistency Feature Selection** (from IEEE-CIS 1st place): Train model on first month with one feature, validate on last month. Drop features where validation AUC < 0.5

### Stacking architectures that win competitions

The **IEEE-CIS Fraud Detection 1st place** (FraudSquad, AUC 0.9459) used 20 LightGBM + 5 CatBoost base models stacked with a LightGBM meta-learner, plus UID-based prediction averaging as postprocessing. The **AmEx Default 1st place** (jxzly) used a 7-stage sequential pipeline where Stage 3 LightGBM OOF predictions became features for Stage 5 main models, with final ensemble weights: `0.30 × lgb_manual + 0.35 × lgb_manual_series + 0.15 × nn_series + 0.10 × nn_all`.

A proven three-level stacking architecture:

```
Level 1: 20-75 diverse base models
├── LightGBM (varied hyperparameters, feature sets, random seeds)
├── XGBoost (different max_depth, learning_rate)
├── CatBoost (different border_count, l2_leaf_reg)
└── Neural nets (MLP, FT-Transformer)

Level 2: Meta-learners on Level 1 OOF predictions
├── LightGBM meta-model (forward feature selection of Level 1 outputs)
└── Logistic regression (for calibration)

Level 3: Weighted average of Level 2 outputs
└── Optimize weights in log-odds space using Optuna on OOF
```

**Target encoding with OOF smoothing** is essential for high-cardinality features like ICD codes and provider IDs:

```python
def target_encode_oof(df, col, target, n_splits=5, smoothing=10):
    global_mean = df[target].mean()
    oof_encoded = pd.Series(index=df.index, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df, df[target]):
        stats = df.iloc[train_idx].groupby(col)[target].agg(['mean', 'count'])
        smoother = 1 / (1 + np.exp(-(stats['count'] - 1) / smoothing))
        smooth_mean = global_mean * (1 - smoother) + stats['mean'] * smoother
        oof_encoded.iloc[val_idx] = df.iloc[val_idx][col].map(smooth_mean).fillna(global_mean)
    return oof_encoded
```

---

## C. Multi-task learning architectures for tabular fraud data

### Architecture choices: from shared-bottom to mixture-of-experts

The architecture spectrum for multi-task fraud detection with four heads (binary fraud, approval probability, ICD anomaly score, rejection rate):

**Shared-Bottom** is simplest: a shared MLP encoder feeds lightweight task-specific heads (1–2 FC layers each). It works when tasks are highly correlated but suffers from negative transfer when they aren't.

**MMoE (Multi-gate Mixture-of-Experts)** replaces the shared bottom with N expert sub-networks (e.g., 8 MLP experts). Each task has its own gating network that computes a softmax-weighted combination of expert outputs, allowing different tasks to leverage different experts. This mitigates negative transfer significantly.

**PLE (Progressive Layered Extraction)** extends MMoE by separating experts into shared and task-specific experts, stacking multiple layers for progressive refinement. PLE outperformed MMoE by ~0.3% AUC in Tencent production systems and directly addresses the "seesaw phenomenon" where improving one task hurts another.

For transformer-based approaches, **MultiTab-Net** (2025) is the first transformer-based MTL model specifically designed for tabular data. It uses multi-token mechanisms where each task gets its own [CLS]-like token, with multitask masked attention to inhibit task interference:

```python
class MultiTaskFTTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(num_features, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers)
        # Per-task CLS tokens (MultiTab-Net style)
        self.task_tokens = nn.ParameterDict({
            'fraud': nn.Parameter(torch.randn(1, 1, d_model)),
            'approval': nn.Parameter(torch.randn(1, 1, d_model)),
            'anomaly': nn.Parameter(torch.randn(1, 1, d_model)),
            'rejection': nn.Parameter(torch.randn(1, 1, d_model)),
        })
        self.heads = nn.ModuleDict({
            'fraud': nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)),
            'approval': nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)),
            'anomaly': nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)),
            'rejection': nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)),
        })

    def forward(self, x_num, x_cat):
        tokens = self.feature_tokenizer(x_num, x_cat)  # (B, N, d_model)
        # Prepend all task tokens
        task_toks = torch.cat([v.expand(tokens.size(0), -1, -1)
                               for v in self.task_tokens.values()], dim=1)
        tokens = torch.cat([task_toks, tokens], dim=1)
        encoded = self.transformer(tokens)
        outputs = {}
        for i, (name, head) in enumerate(self.heads.items()):
            outputs[name] = head(encoded[:, i, :])  # each task reads its CLS token
        return outputs
```

A practical MLP-based multi-task network for production use:

```python
class MultiTaskTabularNet(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, embed_dim=8, hidden=[256, 128]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        input_dim = n_numeric + len(cat_cardinalities) * embed_dim
        layers = []
        for h in hidden:
            layers.extend([nn.Linear(input_dim, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(0.3)])
            input_dim = h
        self.shared = nn.Sequential(*layers)
        self.fraud_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.approval_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.anomaly_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.rejection_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
```

### Loss weighting strategies for imbalanced multi-task setups

**Focal loss** (Lin et al., 2017) for the binary fraud head is essential at 0.1–2% fraud rates. Use **γ=1.5–2.0** and **α=0.75** (weighted toward fraud):

```python
def sigmoid_focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * targets + (1-alpha) * (1-targets)) * ((1 - p_t) ** gamma) * ce
```

**Uncertainty weighting** (Kendall et al., CVPR 2018) learns per-task homoscedastic uncertainty σ_k, automatically down-weighting noisier auxiliary tasks:

```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
```

**GradNorm** (Chen et al., ICML 2018) dynamically tunes gradient magnitudes to equalize task training rates. It has a single hyperparameter α=1.5 and applies only to the last shared layer. GradNorm addresses magnitude imbalance but not direction conflicts—for that, you need PCGrad or CAGrad.

The combined loss for the four-head architecture:

```python
# Combined training step
fraud_loss = sigmoid_focal_loss(outputs['fraud'], y_fraud, alpha=0.75, gamma=2.0)
approval_loss = F.mse_loss(torch.sigmoid(outputs['approval']), y_approval)
anomaly_loss = F.mse_loss(outputs['anomaly'], y_anomaly)
rejection_loss = F.mse_loss(torch.sigmoid(outputs['rejection']), y_rejection)

total_loss = uncertainty_weighter(fraud_loss, approval_loss, anomaly_loss, rejection_loss)
```

### Gradient conflict resolution: PCGrad vs. CAGrad

**PCGrad** (Yu et al., NeurIPS 2020) detects conflicting gradient directions between tasks and projects conflicting gradients onto the normal plane of each other:

```python
def pcgrad_update(task_grads):
    pc_grads = [g.clone() for g in task_grads]
    for i in range(len(task_grads)):
        for j in random.sample(range(len(task_grads)), len(task_grads)):
            if i == j: continue
            cos_sim = torch.dot(pc_grads[i], task_grads[j])
            if cos_sim < 0:  # conflict
                pc_grads[i] -= (cos_sim / (task_grads[j].norm()**2 + 1e-8)) * task_grads[j]
    return sum(pc_grads) / len(pc_grads)
```

**CAGrad** (Liu et al., NeurIPS 2021) improves on PCGrad with **provable convergence** guarantees. It finds the update vector maximizing the worst local improvement of any task within a neighborhood of the average gradient. CAGrad is recommended over PCGrad for production use. Official implementation: `github.com/Cranial-XIX/CAGrad`.

### Which auxiliary tasks help fraud detection vs. distract

Research findings on task selection:

- **Approval probability prediction** is the strongest auxiliary task—it's directly correlated with fraud patterns and uses readily available labels
- **Rejection rate prediction** provides complementary signal, especially at the provider level
- **ICD anomaly scoring** helps when fraud involves upcoding or unusual procedure combinations
- Tasks with Pearson correlation > 0.15 to the primary fraud target tend to help; below that, negative transfer risk increases

**ForkMerge** (NeurIPS 2023) found that 23 of 30 task combinations showed negative transfer with equal weighting. The solution: periodically fork the model, search for optimal task weights that minimize target validation error, then merge branches. Monitor per-task validation metrics continuously—if any auxiliary degrades main fraud AUC by > 1%, consider dropping it.

---

## D. Pseudo-labeling and semi-supervised learning for sparse fraud labels

### Self-training with calibrated confidence thresholds

The critical challenge: at 0.1–2% fraud rates, standard pseudo-labeling thresholds (τ=0.95) produce almost zero fraud-class pseudo-labels, creating severe majority-class bias. **Class-specific adaptive thresholds are essential.**

```python
def pseudo_label_with_adaptive_thresholds(model, X_labeled, y_labeled, X_unlabeled,
                                           tau_fraud=0.7, tau_legit=0.95, max_iters=5):
    X_train, y_train = X_labeled.copy(), y_labeled.copy()
    remaining = X_unlabeled.copy()
    
    for iteration in range(max_iters):
        model.fit(X_train, y_train)
        probs = model.predict_proba(remaining)[:, 1]  # P(fraud)
        
        # Class-specific thresholds
        fraud_mask = probs >= tau_fraud           # confident fraud predictions
        legit_mask = (1 - probs) >= tau_legit     # confident legitimate predictions
        
        X_pseudo_fraud = remaining[fraud_mask]
        X_pseudo_legit = remaining[legit_mask]
        
        X_train = np.vstack([X_train, X_pseudo_fraud, X_pseudo_legit])
        y_train = np.concatenate([y_train,
                                  np.ones(len(X_pseudo_fraud)),
                                  np.zeros(len(X_pseudo_legit))])
        remaining = remaining[~(fraud_mask | legit_mask)]
        
        print(f"Iter {iteration}: +{fraud_mask.sum()} fraud, +{legit_mask.sum()} legit pseudo-labels")
```

**FlexMatch** adjusts thresholds per-class based on learning status. **CReST** (CVPR 2021) progressively rebalances sampling to oversample minority pseudo-labels, improving FixMatch by up to 4.0% on imbalanced data. For tabular data specifically, **VIME** (NeurIPS 2020) provides the strongest SSL framework—it corrupts tabular features via masking, trains an encoder to recover values and mask vectors, then uses the encoder for semi-supervised consistency regularization.

### Snorkel-style programmatic labeling for health insurance fraud

Write labeling functions that encode domain expertise, then let Snorkel's generative model learn their accuracies without ground truth:

```python
from snorkel.labeling import labeling_function, LFApplier, LabelModel
FRAUD, LEGIT, ABSTAIN = 1, 0, -1

@labeling_function()
def lf_high_rejection_provider(claim):
    if provider_rejection_rate[claim.provider_id] > 0.30: return FRAUD
    return ABSTAIN

@labeling_function()
def lf_unusual_icd_pair(claim):
    if icd_pair_frequency[(claim.icd_primary, claim.icd_secondary)] < 0.001: return FRAUD
    return ABSTAIN

@labeling_function()
def lf_amount_outlier(claim):
    if claim.amount > peer_mean + 3 * peer_std: return FRAUD
    return ABSTAIN

@labeling_function()
def lf_verified_provider(claim):
    if claim.provider_id in verified_clean_providers: return LEGIT
    return ABSTAIN

# Apply LFs and train label model
lfs = [lf_high_rejection_provider, lf_unusual_icd_pair, lf_amount_outlier, lf_verified_provider]
applier = LFApplier(lfs=lfs)
L_train = applier.apply(df_unlabeled)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=500, lr=0.001)
probabilistic_labels = label_model.predict_proba(L_train)  # soft labels for training
```

**Critical for imbalanced data**: include labeling functions for the negative class (known legitimate providers, standard procedure combinations), otherwise the model achieves high precision but low recall.

### Calibration before pseudo-labeling

Modern neural networks are systematically overconfident, and imbalanced training biases probabilities toward the majority class. **Temperature scaling** is the minimum calibration step before pseudo-labeling:

```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def calibrate(self, val_logits, val_labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    def forward(self, logits):
        return logits / self.temperature
```

For imbalanced binary classification, **Platt scaling** (sigmoid calibration) with an additional intercept is preferable to temperature scaling alone—it can correct the systematic bias toward the majority class. The recommended pipeline: train model → calibrate with Platt scaling on held-out data → generate pseudo-labels using calibrated probabilities → apply class-specific thresholds (τ_fraud ≈ 0.7, τ_legit ≈ 0.95).

---

## E. Handling temporal shift across the train/test boundary

### Adversarial validation to quantify and exploit distributional shift

Adversarial validation trains a binary classifier to distinguish train from test samples. If the AUC significantly exceeds 0.5, distributional shift exists:

```python
from catboost import CatBoostClassifier

train_df['is_test'] = 0
test_df['is_test'] = 1
combined = pd.concat([train_df[features + ['is_test']], test_df[features + ['is_test']]])

adv_model = CatBoostClassifier(iterations=500, eval_metric='AUC', verbose=0)
adv_model.fit(combined[features], combined['is_test'])

# Feature importance reveals which features shift most over time
shifting_features = pd.DataFrame({
    'feature': features,
    'importance': adv_model.get_feature_importance()
}).sort_values('importance', ascending=False)

# Use adversarial scores to create a better validation set
train_df['p_test'] = adv_model.predict_proba(train_df[features])[:, 1]
# Top-N most test-like training samples become validation set
val_set = train_df.nlargest(5000, 'p_test')
```

The key insight from IEEE-CIS top solutions: temporal features like raw timestamps are the strongest train/test separators. Replacing absolute timestamps with relative features ("days since last claim," "hour of day") significantly reduces separability while preserving predictive signal.

### Domain adaptation methods that work for tabular fraud data

**Importance weighting** (covariate shift correction) reweights training samples by the density ratio w(x) = p_test(x) / p_train(x):

```python
from densratio import densratio
result = densratio(X_train, X_test)
weights = result.compute_density_ratio(X_train)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train, sample_weight=weights)
```

**CORAL** (CORrelation ALignment) aligns second-order statistics between source and target distributions—"frustratingly easy" to implement:

```python
def coral_transform(X_source, X_target):
    Cs = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 1e-5
    Ct = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 1e-5
    Cs_half_inv = np.linalg.cholesky(np.linalg.inv(Cs))
    Ct_half = np.linalg.cholesky(Ct)
    return X_source @ Cs_half_inv @ Ct_half
```

**AdapTable** (Kim et al., 2024) is a test-time adaptation framework specifically for tabular data, using a GNN-based shift-aware uncertainty calibrator that achieves **16–27% improvement** on shifted tabular datasets.

### Concept drift detection and the label delay problem

**ADWIN** (ADaptive WINdowing) is the best-performing drift detector for fraud detection (ROSFD framework, 2025). It maintains a variable-length window and detects when statistics between sub-windows diverge:

```python
from concept_drift.adwin import AdWin
adwin = AdWin(delta=0.002)
for prediction_error in streaming_predictions:
    if adwin.set_input(prediction_error):
        print("Drift detected — trigger retraining")
```

The **label delay problem** is acute in health insurance fraud: investigation outcomes arrive 1–6 months after prediction. Three strategies address this:

- **Post-claim signals as pseudo-labels** (Meta, KDD 2025): Use post-claim activity (follow-up appointments, amended claims, provider behavior changes) to generate pseudo-labels before true labels arrive
- **Survival analysis framing**: Model time-to-fraud-report as censored data, enabling use of partially labeled recent data
- **PRODEM** (ECML PKDD 2025): A meta-model trained via reverse distillation that predicts when the primary model will make errors—detecting degradation *before* ground truth labels arrive

### Temporal feature engineering that bridges gaps

Rolling aggregates across multiple windows are the backbone of fraud feature engineering:

```python
def compute_temporal_features(df, entity='provider_id', time='claim_date', amt='amount'):
    df = df.sort_values([entity, time])
    df['time_since_last'] = df.groupby(entity)[time].diff().dt.days
    
    for window in ['7D', '30D', '90D', '365D']:
        grp = df.set_index(time).groupby(entity)[amt]
        df[f'mean_amt_{window}'] = grp.rolling(window).mean().values
        df[f'std_amt_{window}'] = grp.rolling(window).std().values
        df[f'count_{window}'] = grp.rolling(window).count().values
    
    # Deviation from baseline (spike detection)
    df['amt_zscore_30d'] = (df[amt] - df['mean_amt_30D']) / (df['std_amt_30D'] + 1e-6)
    # Velocity features
    df['claim_velocity_7d'] = df['count_7D'] / 7
    # Periodic features
    df['day_of_week'] = df[time].dt.dayofweek
    df['is_month_end'] = df[time].dt.day > 25
    return df
```

Research by Bahnsen et al. (2016) showed periodic features (analyzing cyclic patterns in hour-of-day and day-of-week using von Mises distributions) provide a **13% average increase** in fraud detection savings. For new providers with limited history, use cohort-level statistics (population priors from same specialty/region) and gradually shift to entity-level as data accumulates.

---

## F. Kaggle competition strategies that transfer to production fraud systems

### IEEE-CIS Fraud Detection: the UID discovery that changed everything

The **1st place solution** (FraudSquad, AUC 0.9459) made one discovery worth more than all other techniques combined: constructing a **synthetic unique client ID** from `card1 + addr1 + D1` (where D1 = days since card first used, making `TransactionDay − D1` approximately constant per client). This unlocked entity-level aggregation features that dramatically improved performance.

Key patterns from top IEEE-CIS solutions:

- **Feature engineering** (~262 features): group aggregations by UID (mean, std, count of amounts), frequency encoding of categoricals, OOF target encoding, and temporal D-column normalization
- **V-feature handling**: Grouped 340 V-features into 15 groups by NaN patterns, applied PCA within groups
- **Validation**: GroupKFold with ordered months, plus Time Consistency feature selection
- **Ensemble**: 20 LightGBM + 5 CatBoost → LightGBM meta-learner → UID-based prediction averaging

### AmEx Default Prediction: OOF predictions as features in a staged pipeline

The **1st place solution** (jxzly, score 0.80977) used a 7-stage sequential pipeline where Stage 3 LightGBM OOF predictions became input features for Stage 5 main models. The feature engineering pattern for temporal customer data: compute **mean, std, min, max, last value, and first-to-last difference/ratio** across monthly statements for every feature. Chris Deotte's 15th place solution introduced **Transformer with LightGBM Knowledge Distillation**: train LightGBM, save OOF predictions, use them to pseudo-label the test set, then train a Transformer on both OOF and pseudo-labels.

### Transferable winning patterns

The most impactful techniques across Kaggle fraud competitions, ranked by typical lift:

- **Entity identification and group features**: Construct UIDs, compute entity-level aggregates (rejection rates, claim patterns, temporal behavior). This is consistently the #1 performance driver.
- **Multi-window temporal aggregates**: RFM features (recency, frequency, monetary) across 1h/24h/7d/30d/90d windows with z-score deviation features.
- **OOF stacking with diverse base models**: Use multiple algorithms (LightGBM, XGBoost, CatBoost, neural nets), multiple feature sets, and multiple random seeds. Forward feature selection of Level 1 models for Level 2 meta-learner.
- **Target encoding with OOF + Bayesian smoothing**: α = 5–10 for smoothing factor; for temporal data, use expanding-window encoding (only past data).
- **Postprocessing**: Entity-based prediction averaging (replace all predictions for a single entity with that entity's average), and log-odds space weight optimization for ensembles.

---

## The unified pipeline: putting it all together

The complete pipeline for health insurance hospitalization fraud detection integrates all six components into a coherent system:

```
Phase 1: Weak Label Engineering
├── Compute provider-level rejection rates, ICD anomaly scores
├── Create Snorkel labeling functions from domain rules
├── Generate probabilistic weak labels via Snorkel LabelModel
└── Run cleanlab on available labels to identify/correct noise

Phase 2: Temporal Feature Engineering
├── Multi-window rolling aggregates (7d/30d/90d/365d)
├── Entity-level behavioral features (provider, patient, facility)
├── ICD code embeddings and anomaly scores
└── Adversarial validation to identify and transform shifting features

Phase 3: OOF Meta-Feature Generation (Purged Walk-Forward CV)
├── LightGBM → fraud OOF predictions
├── LightGBM → approval rate OOF predictions
├── XGBoost → rejection rate OOF predictions
├── Neural net → ICD anomaly OOF predictions
└── Derived meta-features: confidence (std), consensus (mean)

Phase 4: Multi-Task Learning
├── FT-Transformer/MMoE shared encoder
├── Four heads: fraud (focal loss), approval, anomaly, rejection
├── Uncertainty weighting + CAGrad for gradient conflicts
└── Neural net OOF predictions join Phase 3 meta-features

Phase 5: Stacking Meta-Learner
├── All Phase 3+4 OOF predictions as meta-features
├── Optionally include original features (passthrough)
├── LightGBM meta-learner with forward feature selection
└── Calibrate with Platt scaling

Phase 6: Pseudo-Labeling & Temporal Adaptation
├── Calibrate final model predictions (temperature scaling)
├── Pseudo-label unlabeled/recent data with adaptive thresholds
├── Retrain with expanded labeled set
├── Deploy ADWIN drift monitoring
└── Sliding-window retraining when drift detected
```

This pipeline respects temporal ordering at every stage, addresses the extreme class imbalance through focal loss and adaptive thresholds, leverages all available signals (approval rates, ICD patterns, provider behavior) as both features and auxiliary targets, and handles the inevitable distributional shift between training and deployment periods. The entity-level feature engineering (provider and patient behavioral aggregates) will likely provide the largest single performance improvement, followed by the OOF stacking of diverse models and targets.