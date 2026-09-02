"""
experiment_window.py
====================
Find optimal training window size for the temporal shift problem.
Tests: 1, 2, 3, 4, 5, 6, 9, 12, 18, 24 months before split.
Also tests blending full-data model + window model.

Usage: python experiment_window.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
ANOMALY_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\anomaly_scores.parquet"
GNN_PATH     = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\gnn_embeddings.parquet"
SPLIT_DATE   = pd.Timestamp('2025-07-01')

print("Loading data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
df['data_created_at_parsed'] = pd.to_datetime(df['data_created_at'], format='ISO8601')
train_mask = df['data_created_at_parsed'] < SPLIT_DATE
test_mask  = ~train_mask
print(f"  {len(df):,} rows | Train: {train_mask.sum():,} | Test: {test_mask.sum():,} | {time.time()-t0:.1f}s")

# ── Targets ────────────────────────────────────────────────────────────────────
y_inv = pd.to_numeric(df.get('Target_as_investigation', pd.Series(0, index=df.index)),
                       errors='coerce').fillna(0).astype(int)
fraud_col = next((c for c in ['Fraud_Outcome', 'Target_as_fraud'] if c in df.columns), None)
y_fraud = pd.to_numeric(df[fraud_col], errors='coerce').fillna(0).astype(int) if fraud_col else pd.Series(0, index=df.index)

# ── Feature set ────────────────────────────────────────────────────────────────
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
tr_med = X_raw[train_mask].median()
X_raw  = X_raw.fillna(tr_med).fillna(0)

anom = pd.read_parquet(ANOMALY_PATH)
X_raw['dae_recon_error'] = np.log1p(anom['dae_recon_error'].values)
X_raw['isoforest_score'] = anom['isoforest_score'].values

gnn = pd.read_parquet(GNN_PATH)
if 'gnn_emb_0' in gnn.columns:
    gnn_cols = [c for c in gnn.columns if c.startswith('gnn_emb_')]
    gnn_arr  = gnn[gnn_cols].values
    gnn_var  = gnn_arr[train_mask].var(axis=0)
    top16    = np.argsort(gnn_var)[::-1][:16]
    for k in top16:
        X_raw[f'gnn_{k}'] = gnn_arr[:, k]

# Scale with full train stats (kept constant across experiments for fair comparison)
scaler = RobustScaler()
X_tr_full = scaler.fit_transform(X_raw[train_mask])
X_te      = scaler.transform(X_raw[test_mask])

y_te_inv   = y_inv[test_mask].values
y_te_fraud = y_fraud[test_mask].values
y_tr_inv_full   = y_inv[train_mask].values
y_tr_fraud_full = y_fraud[train_mask].values

spw_inv   = (y_tr_inv_full   == 0).sum() / max((y_tr_inv_full   == 1).sum(), 1)
spw_fraud = (y_tr_fraud_full == 0).sum() / max((y_tr_fraud_full == 1).sum(), 1)

# Full train time-decay weights
_tdates = df.loc[train_mask, 'data_created_at_parsed']
_days   = (_tdates - _tdates.min()).dt.days.values.astype(float)
_tdecay = np.exp(0.001 * _days).astype(np.float32)
_tdecay /= _tdecay.mean()

def make_sw(y, spw, base_w):
    return (np.where(y == 1, spw, 1.0).astype(np.float32)) * base_w

def pr_auc(y_true, y_prob):
    return average_precision_score(y_true, y_prob)

def run_rf_lgbm(X_tr, X_te, y_tr, y_te, spw, sw, tag=""):
    """Returns dict with RF and LGBM PR-AUC."""
    # RF
    rf = RFC(n_estimators=200, max_depth=None, min_samples_leaf=5,
             class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    p_rf = rf.predict_proba(X_te)[:, 1]

    # LGBM
    lgb = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr, y_tr, sample_weight=sw)
    p_lgb = lgb.predict_proba(X_te)[:, 1]

    return {'RF': pr_auc(y_te, p_rf), 'LGBM': pr_auc(y_te, p_lgb),
            'p_rf': p_rf, 'p_lgb': p_lgb}

# ── Full baseline (all train data) ─────────────────────────────────────────────
print("\n--- Full baseline ---")
t = time.time()
sw_inv_full   = make_sw(y_tr_inv_full,   spw_inv,   _tdecay)
sw_fraud_full = make_sw(y_tr_fraud_full, spw_fraud, _tdecay)
r_full_inv   = run_rf_lgbm(X_tr_full, X_te, y_tr_inv_full,   y_te_inv,   spw_inv,   sw_inv_full)
r_full_fraud = run_rf_lgbm(X_tr_full, X_te, y_tr_fraud_full, y_te_fraud, spw_fraud, sw_fraud_full)
print(f"  Inv   RF={r_full_inv['RF']:.4f}  LGBM={r_full_inv['LGBM']:.4f}")
print(f"  Fraud RF={r_full_fraud['RF']:.4f}  LGBM={r_full_fraud['LGBM']:.4f}")
print(f"  ({time.time()-t:.1f}s)")

# ── Window sweep ───────────────────────────────────────────────────────────────
WINDOWS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24]
results_inv   = {}
results_fraud = {}

for w_months in WINDOWS:
    window_start = SPLIT_DATE - pd.DateOffset(months=w_months)
    wmask = (df['data_created_at_parsed'] >= window_start) & train_mask
    n_win = wmask.sum()
    n_inv_pos   = y_inv[wmask].sum()
    n_fraud_pos = y_fraud[wmask].sum()

    if n_inv_pos < 20 or n_fraud_pos < 5:
        print(f"\n  [{w_months}mo] skip (inv={n_inv_pos}, fraud={n_fraud_pos})")
        continue

    # Get window subset of X_tr_full (already scaled)
    win_in_train = wmask[train_mask].values
    X_tr_win = X_tr_full[win_in_train]
    y_tr_win_inv   = y_tr_inv_full[win_in_train]
    y_tr_win_fraud = y_tr_fraud_full[win_in_train]

    # Time decay within window
    wdates  = df.loc[wmask, 'data_created_at_parsed']
    wdays   = (wdates - wdates.min()).dt.days.values.astype(float)
    wdecay  = np.exp(0.001 * wdays).astype(np.float32)
    wdecay /= wdecay.mean()

    sw_inv_w   = make_sw(y_tr_win_inv,   spw_inv,   wdecay)
    sw_fraud_w = make_sw(y_tr_win_fraud, spw_fraud, wdecay)

    t = time.time()
    ri = run_rf_lgbm(X_tr_win, X_te, y_tr_win_inv,   y_te_inv,   spw_inv,   sw_inv_w)
    rf = run_rf_lgbm(X_tr_win, X_te, y_tr_win_fraud, y_te_fraud, spw_fraud, sw_fraud_w)
    elapsed = time.time()-t

    results_inv[w_months]   = ri
    results_fraud[w_months] = rf

    print(f"  [{w_months:2d}mo | n={n_win:>6,} | inv_pos={n_inv_pos:>4,} | fraud_pos={n_fraud_pos:>4,}]"
          f"  Inv RF={ri['RF']:.4f} LGBM={ri['LGBM']:.4f}"
          f"  Fraud RF={rf['RF']:.4f} LGBM={rf['LGBM']:.4f}  ({elapsed:.1f}s)")

# ── Blend: full-data model + best-window model ─────────────────────────────────
print("\n--- Blend: full-data + window model ---")
best_w = max(results_fraud, key=lambda w: results_fraud[w]['RF'])
print(f"Best window for Fraud: {best_w} months")

# Re-run best window to get probabilities
window_start = SPLIT_DATE - pd.DateOffset(months=best_w)
wmask = (df['data_created_at_parsed'] >= window_start) & train_mask
win_in_train = wmask[train_mask].values
X_tr_bw = X_tr_full[win_in_train]
y_tr_bw_fraud = y_tr_fraud_full[win_in_train]
wdates  = df.loc[wmask, 'data_created_at_parsed']
wdays   = (wdates - wdates.min()).dt.days.values.astype(float)
wdecay  = np.exp(0.001 * wdays).astype(np.float32)
wdecay /= wdecay.mean()
sw_bw   = make_sw(y_tr_bw_fraud, spw_fraud, wdecay)

r_bw = run_rf_lgbm(X_tr_bw, X_te, y_tr_bw_fraud, y_te_fraud, spw_fraud, sw_bw)

# Blend RF predictions
for alpha in [0.3, 0.5, 0.7]:
    p_blend = alpha * r_bw['p_rf'] + (1-alpha) * r_full_fraud['p_rf']
    blend_pr = pr_auc(y_te_fraud, p_blend)
    print(f"  Fraud RF blend: alpha={alpha} (window) => PR-AUC={blend_pr:.4f}")

# Best window for Inv
best_w_inv = max(results_inv, key=lambda w: results_inv[w]['RF'])
print(f"\nBest window for Inv: {best_w_inv} months")
window_start_inv = SPLIT_DATE - pd.DateOffset(months=best_w_inv)
wmask_inv = (df['data_created_at_parsed'] >= window_start_inv) & train_mask
win_in_train_inv = wmask_inv[train_mask].values
X_tr_bw_inv = X_tr_full[win_in_train_inv]
y_tr_bw_inv = y_tr_inv_full[win_in_train_inv]
wdates_inv  = df.loc[wmask_inv, 'data_created_at_parsed']
wdays_inv   = (wdates_inv - wdates_inv.min()).dt.days.values.astype(float)
wdecay_inv  = np.exp(0.001 * wdays_inv).astype(np.float32)
wdecay_inv /= wdecay_inv.mean()
sw_bw_inv   = make_sw(y_tr_bw_inv, spw_inv, wdecay_inv)

r_bw_inv = run_rf_lgbm(X_tr_bw_inv, X_te, y_tr_bw_inv, y_te_inv, spw_inv, sw_bw_inv)
for alpha in [0.3, 0.5, 0.7]:
    p_blend = alpha * r_bw_inv['p_rf'] + (1-alpha) * r_full_inv['p_rf']
    blend_pr = pr_auc(y_te_inv, p_blend)
    print(f"  Inv RF blend: alpha={alpha} (window) => PR-AUC={blend_pr:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("WINDOW SWEEP SUMMARY")
print("="*70)
print(f"{'Window':<10} {'N_train':>8} {'Inv-RF':>8} {'Inv-LGBM':>10} {'Fr-RF':>8} {'Fr-LGBM':>10}")
print("-"*55)
print(f"  {'Full':8} {train_mask.sum():>8,}"
      f" {r_full_inv['RF']:>8.4f} {r_full_inv['LGBM']:>10.4f}"
      f" {r_full_fraud['RF']:>8.4f} {r_full_fraud['LGBM']:>10.4f}")
for w_months in WINDOWS:
    if w_months not in results_inv:
        continue
    wmask = (df['data_created_at_parsed'] >= (SPLIT_DATE - pd.DateOffset(months=w_months))) & train_mask
    n_w = wmask.sum()
    ri = results_inv[w_months]
    rf = results_fraud[w_months]
    tag = " <--" if w_months == best_w or w_months == best_w_inv else ""
    print(f"  {w_months:2d}mo    {n_w:>8,}"
          f" {ri['RF']:>8.4f} {ri['LGBM']:>10.4f}"
          f" {rf['RF']:>8.4f} {rf['LGBM']:>10.4f}{tag}")

print("\nDone.")
