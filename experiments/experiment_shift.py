"""
experiment_shift.py
===================
Quick experiment script to test distribution-shift mitigation techniques.
Runs a lean pipeline (numeric + anomaly + GNN features, no complex engineering)
so each experiment completes in a few minutes.

Techniques tested:
  0. Baseline (existing time-decay weights, no extra shift correction)
  1. Quantile Normalization  — align X_tr feature distributions to X_te
  2. CORAL                   — align X_tr covariance matrix to X_te
  3. Time window training    — train only on last N months before split
  4. IW (Importance Weighting) standalone (not combined with class weights)

Usage:
  python experiment_shift.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
ANOMALY_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\anomaly_scores.parquet"
GNN_PATH     = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\gnn_embeddings.parquet"
SPLIT_DATE   = pd.Timestamp('2025-07-01')

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
df['data_created_at_parsed'] = pd.to_datetime(df['data_created_at'], format='ISO8601')

train_mask = df['data_created_at_parsed'] < SPLIT_DATE
test_mask  = ~train_mask
print(f"  {len(df):,} rows | Train: {train_mask.sum():,} | Test: {test_mask.sum():,} | {time.time()-t0:.1f}s")

# ── Target labels ──────────────────────────────────────────────────────────────
# Investigation target
y_inv = pd.to_numeric(df.get('Target_as_investigation', pd.Series(0, index=df.index)),
                       errors='coerce').fillna(0).astype(int)

# Fraud target
fraud_col = next((c for c in ['Fraud_Outcome', 'Target_as_fraud'] if c in df.columns), None)
if fraud_col:
    y_fraud = pd.to_numeric(df[fraud_col], errors='coerce').fillna(0).astype(int)
else:
    y_fraud = pd.Series(0, index=df.index)

print(f"  Inv  - train pos: {y_inv[train_mask].sum():,}  test pos: {y_inv[test_mask].sum():,}")
print(f"  Fraud- train pos: {y_fraud[train_mask].sum():,}  test pos: {y_fraud[test_mask].sum():,}")

# ── Lean feature set ───────────────────────────────────────────────────────────
# Keep: numeric columns from raw data (no complex engineering)
# Drop: leaky / post-adjudication / identifier / date columns
DROP_COLS = [
    'Claim_No', 'data_created_at', 'data_created_at_parsed',
    'Target_as_investigation', 'Target_as_fraud', 'Investigated', 'Old_Target',
    'Claims_Stage', 'Claim_Reserve', 'Approved_Amount_INR',
    'Buffer_Available', 'Buffer_Consumed', 'Balance_Buffer',
    'Investigation Outcome', 'Fraud_Outcome',
    'Assign Date', 'Case_Close_Date',
    'Company_Date_of_Joining', 'Date_of_Joining_the_Policy',
    'Risk_Inception_Date', 'Risk_Expiry_Date',
    'Expected_Date_of_Admission', 'Expected_Date_of_Discharge',
    'Actual_Date_of_Admission', 'Actual_Date_of_Discharge',
    'Document_Received_Date', 'Claim_Intimation_Date',
    'Medical_Management_Date', 'Last_Document_Received_Date',
    'HID_anon', 'CID_anon', 'Policy_number', 'DID_anon',
    'Final_Diagnosis', 'Claim_Type', 'Insurer',
]
df_feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
num_cols = df_feat.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
X_raw = df_feat[num_cols].copy()

# Impute with train median
tr_med = X_raw[train_mask].median()
X_raw  = X_raw.fillna(tr_med).fillna(0)

# Add anomaly scores
print("\nLoading anomaly scores...")
anom = pd.read_parquet(ANOMALY_PATH)
X_raw['dae_recon_error'] = np.log1p(anom['dae_recon_error'].values)
X_raw['isoforest_score'] = anom['isoforest_score'].values

# Add GNN embeddings (top-16 by variance on train, to keep feature count manageable)
print("Loading GNN embeddings (top-16 by train variance)...")
gnn = pd.read_parquet(GNN_PATH)
if 'gnn_emb_0' in gnn.columns:
    gnn_cols = [c for c in gnn.columns if c.startswith('gnn_emb_')]
    gnn_arr  = gnn[gnn_cols].values
    gnn_var  = gnn_arr[train_mask].var(axis=0)
    top16    = np.argsort(gnn_var)[::-1][:16]
    for k in top16:
        X_raw[f'gnn_{k}'] = gnn_arr[:, k]
    print(f"  Added {len(top16)} GNN dims")

print(f"\nFeature set: {X_raw.shape[1]} cols")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = RobustScaler()
X_tr_raw = scaler.fit_transform(X_raw[train_mask])
X_te_raw = scaler.transform(X_raw[test_mask])

# ── Sample weights (time-decay, same as notebook) ─────────────────────────────
_train_dates = df.loc[train_mask, 'data_created_at_parsed']
_days        = (_train_dates - _train_dates.min()).dt.days.values.astype(float)
time_decay_w = np.exp(0.001 * _days).astype(np.float32)
time_decay_w /= time_decay_w.mean()

# ── Labels for train/test ─────────────────────────────────────────────────────
y_tr_inv   = y_inv[train_mask].values
y_te_inv   = y_inv[test_mask].values
y_tr_fraud = y_fraud[train_mask].values
y_te_fraud = y_fraud[test_mask].values

spw_inv   = (y_tr_inv   == 0).sum() / max((y_tr_inv   == 1).sum(), 1)
spw_fraud = (y_tr_fraud == 0).sum() / max((y_tr_fraud == 1).sum(), 1)
print(f"\nspw_inv={spw_inv:.1f}  spw_fraud={spw_fraud:.1f}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_sample_weight(y, spw, base_w):
    """Combine class balance weight with time-decay weight."""
    class_w = np.where(y == 1, spw, 1.0).astype(np.float32)
    return class_w * base_w

def evaluate(name, y_true, y_prob):
    pr  = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    print(f"    {name:<35} PR-AUC={pr:.4f}  ROC={roc:.4f}")
    return pr

def run_models(X_tr, X_te, y_tr, y_te, spw, sw_train, label):
    """Run RF + LGBM and return best PR-AUC."""
    print(f"\n  [{label}] n_train={X_tr.shape[0]:,} | n_test={X_te.shape[0]:,} | feats={X_tr.shape[1]}")
    results = {}

    # RF
    t = time.time()
    rf = RFC(n_estimators=300, max_depth=None, min_samples_leaf=5,
             class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=sw_train)
    p = rf.predict_proba(X_te)[:, 1]
    results['RF'] = evaluate('RF', y_te, p)
    print(f"      ({time.time()-t:.1f}s)")

    # LGBM
    t = time.time()
    lgb = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                         scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr, y_tr, sample_weight=sw_train)
    p = lgb.predict_proba(X_te)[:, 1]
    results['LGBM'] = evaluate('LGBM', y_te, p)
    print(f"      ({time.time()-t:.1f}s)")

    best = max(results.values())
    print(f"    --> BEST: {best:.4f}")
    return results

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 0: Baseline
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 0: Baseline (time-decay weights, no shift correction)")
print("="*70)
sw_inv_base   = make_sample_weight(y_tr_inv,   spw_inv,   time_decay_w)
sw_fraud_base = make_sample_weight(y_tr_fraud, spw_fraud, time_decay_w)

print("Investigation:")
r0_inv   = run_models(X_tr_raw, X_te_raw, y_tr_inv,   y_te_inv,   spw_inv,   sw_inv_base,   "baseline")
print("Fraud:")
r0_fraud = run_models(X_tr_raw, X_te_raw, y_tr_fraud, y_te_fraud, spw_fraud, sw_fraud_base, "baseline")

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Quantile Normalization (fit on X_te, transform X_tr)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 1: Quantile Normalization (test-distribution alignment)")
print("="*70)
t = time.time()
qt = QuantileTransformer(n_quantiles=min(1000, len(X_te_raw)),
                         output_distribution='normal',
                         subsample=min(50000, len(X_te_raw)),
                         random_state=42)
qt.fit(X_te_raw)
X_tr_qn = qt.transform(X_tr_raw)
X_te_qn = qt.transform(X_te_raw)
print(f"  QN fit+transform done in {time.time()-t:.1f}s")

print("Investigation:")
r1_inv   = run_models(X_tr_qn, X_te_qn, y_tr_inv,   y_te_inv,   spw_inv,   sw_inv_base,   "QN")
print("Fraud:")
r1_fraud = run_models(X_tr_qn, X_te_qn, y_tr_fraud, y_te_fraud, spw_fraud, sw_fraud_base, "QN")

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: CORAL (covariance alignment, regularized via LedoitWolf)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 2: CORAL (second-order statistics alignment)")
print("="*70)
try:
    from sklearn.covariance import LedoitWolf
    from scipy.linalg import cholesky, solve_triangular, LinAlgError

    t = time.time()
    # Subsample for covariance estimation (full 400k rows is slow)
    n_coral = min(20000, len(X_tr_raw))
    idx_sub = np.random.choice(len(X_tr_raw), n_coral, replace=False)

    lw_tr = LedoitWolf().fit(X_tr_raw[idx_sub])
    lw_te = LedoitWolf().fit(X_te_raw)

    Cs = lw_tr.covariance_
    Ct = lw_te.covariance_

    # CORAL transform: X_aligned = (X_tr - mu_tr) @ Cs^(-1/2) @ Ct^(1/2) + mu_te
    mu_tr = X_tr_raw.mean(axis=0)
    mu_te = X_te_raw.mean(axis=0)

    L_s = cholesky(Cs + 1e-5 * np.eye(Cs.shape[0]), lower=True)  # Cs = L_s @ L_s.T
    L_t = cholesky(Ct + 1e-5 * np.eye(Ct.shape[0]), lower=True)  # Ct = L_t @ L_t.T

    # X_aligned = (X - mu_tr) @ inv(L_s).T @ L_t + mu_te
    #   inv(L_s) via solve_triangular: inv(L_s) @ X.T  =>  solve(L_s, X.T)
    X_tr_c = X_tr_raw - mu_tr
    X_te_c = X_te_raw - mu_tr  # subtract train mean (source mean)

    # solve triangular: L_s @ Y = X_tr_c.T  => Y = inv(L_s) @ X_tr_c.T
    X_tr_white = solve_triangular(L_s, X_tr_c.T, lower=True).T   # (n, d) whitened
    X_te_white = solve_triangular(L_s, X_te_c.T, lower=True).T

    X_tr_coral = X_tr_white @ L_t + mu_te
    X_te_coral = X_te_white @ L_t + mu_te

    # Clip extremes (whitening can produce large values for OOD samples)
    clip_val = np.percentile(np.abs(X_tr_coral), 99.5)
    X_tr_coral = np.clip(X_tr_coral, -clip_val, clip_val)
    X_te_coral = np.clip(X_te_coral, -clip_val, clip_val)

    print(f"  CORAL transform done in {time.time()-t:.1f}s")
    print(f"  X_tr_coral: mean={X_tr_coral.mean():.3f} std={X_tr_coral.std():.3f}")

    print("Investigation:")
    r2_inv   = run_models(X_tr_coral, X_te_coral, y_tr_inv,   y_te_inv,   spw_inv,   sw_inv_base,   "CORAL")
    print("Fraud:")
    r2_fraud = run_models(X_tr_coral, X_te_coral, y_tr_fraud, y_te_fraud, spw_fraud, sw_fraud_base, "CORAL")

except Exception as e:
    print(f"  CORAL failed: {e}")
    r2_inv   = {'RF': 0, 'LGBM': 0}
    r2_fraud = {'RF': 0, 'LGBM': 0}

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Time-window training (last N months only)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 3: Time-window training (last 6 months of training data)")
print("="*70)
WINDOW_MONTHS = 6
window_start = SPLIT_DATE - pd.DateOffset(months=WINDOW_MONTHS)
window_mask  = (df['data_created_at_parsed'] >= window_start) & train_mask
print(f"  Window: {window_start.date()} to {SPLIT_DATE.date()}")
print(f"  Window train size: {window_mask.sum():,} (vs {train_mask.sum():,} full train)")
print(f"  Inv pos in window: {y_inv[window_mask].sum():,}  Fraud pos: {y_fraud[window_mask].sum():,}")

# We need to scale with train-only stats but restrict to window
# Re-fit scaler on full train (not window) to keep same feature space
# but train models only on window rows
X_tr_win = X_tr_raw[window_mask[train_mask].values]
y_tr_win_inv   = y_tr_inv[window_mask[train_mask].values]
y_tr_win_fraud = y_tr_fraud[window_mask[train_mask].values]
# Time decay within window
_win_dates = df.loc[window_mask, 'data_created_at_parsed']
_win_days  = (_win_dates - _win_dates.min()).dt.days.values.astype(float)
_win_decay = np.exp(0.001 * _win_days).astype(np.float32)
_win_decay /= _win_decay.mean()
sw_win_inv   = make_sample_weight(y_tr_win_inv,   spw_inv,   _win_decay)
sw_win_fraud = make_sample_weight(y_tr_win_fraud, spw_fraud, _win_decay)

# Check if there are enough positives
if y_tr_win_inv.sum() > 10 and y_tr_win_fraud.sum() > 5:
    print("Investigation:")
    r3_inv   = run_models(X_tr_win, X_te_raw, y_tr_win_inv,   y_te_inv,   spw_inv,   sw_win_inv,   "6mo window")
    print("Fraud:")
    r3_fraud = run_models(X_tr_win, X_te_raw, y_tr_win_fraud, y_te_fraud, spw_fraud, sw_win_fraud, "6mo window")
else:
    print(f"  Skipping — too few positives in window (inv={y_tr_win_inv.sum()}, fraud={y_tr_win_fraud.sum()})")
    r3_inv   = {'RF': 0, 'LGBM': 0}
    r3_fraud = {'RF': 0, 'LGBM': 0}

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: QN + 6-month window (combination)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 4: QN + 6-month window (combination)")
print("="*70)
if y_tr_win_inv.sum() > 10:
    X_tr_win_qn = qt.transform(X_tr_win)   # QN fitted on X_te already
    print("Investigation:")
    r4_inv   = run_models(X_tr_win_qn, X_te_qn, y_tr_win_inv,   y_te_inv,   spw_inv,   sw_win_inv,   "QN+6mo")
    print("Fraud:")
    r4_fraud = run_models(X_tr_win_qn, X_te_qn, y_tr_win_fraud, y_te_fraud, spw_fraud, sw_win_fraud, "QN+6mo")
else:
    r4_inv = r4_fraud = {'RF': 0, 'LGBM': 0}

# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
experiments = [
    ("Baseline",          r0_inv, r0_fraud),
    ("QN",                r1_inv, r1_fraud),
    ("CORAL",             r2_inv, r2_fraud),
    ("6mo window",        r3_inv, r3_fraud),
    ("QN + 6mo window",   r4_inv, r4_fraud),
]
print(f"{'Experiment':<22} {'Inv-RF':>8} {'Inv-LGBM':>10} {'Fr-RF':>8} {'Fr-LGBM':>10}")
print("-"*60)
for name, ri, rf in experiments:
    print(f"  {name:<20} {ri.get('RF',0):8.4f} {ri.get('LGBM',0):10.4f}"
          f" {rf.get('RF',0):8.4f} {rf.get('LGBM',0):10.4f}")

print("\nDone.")
