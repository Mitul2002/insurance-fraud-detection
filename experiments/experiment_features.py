"""
experiment_features.py
======================
Tests: 4-month window + enriched features (hospital HHI, fraud rates, OOF).
Compares different feature sets to understand what helps most.

Usage: python experiment_features.py
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

DATA_PATH    = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
ANOMALY_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\anomaly_scores.parquet"
GNN_PATH     = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\gnn_embeddings.parquet"
SPLIT_DATE   = pd.Timestamp('2025-07-01')

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
df['data_created_at_parsed'] = pd.to_datetime(df['data_created_at'], format='ISO8601')
train_mask = df['data_created_at_parsed'] < SPLIT_DATE
test_mask  = ~train_mask
print(f"  {len(df):,} rows  train={train_mask.sum():,}  test={test_mask.sum():,}  {time.time()-t0:.1f}s")

# ── Targets ────────────────────────────────────────────────────────────────────
y_inv = pd.to_numeric(df.get('Target_as_investigation', pd.Series(0, index=df.index)),
                       errors='coerce').fillna(0).astype(int)
fraud_col = next((c for c in ['Fraud_Outcome', 'Target_as_fraud'] if c in df.columns), None)
y_fraud = pd.to_numeric(df[fraud_col], errors='coerce').fillna(0).astype(int) if fraud_col else pd.Series(0, index=df.index)

# ── Base numeric features ──────────────────────────────────────────────────────
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
feat = df_feat[num_cols].copy()
tr_med = feat[train_mask].median()
feat = feat.fillna(tr_med).fillna(0)

# ── Anomaly scores ─────────────────────────────────────────────────────────────
anom = pd.read_parquet(ANOMALY_PATH)
feat['dae_recon_error'] = np.log1p(anom['dae_recon_error'].values)
feat['isoforest_score'] = anom['isoforest_score'].values

# ── GNN top-16 by train variance ───────────────────────────────────────────────
gnn = pd.read_parquet(GNN_PATH)
if 'gnn_emb_0' in gnn.columns:
    gnn_cols = [c for c in gnn.columns if c.startswith('gnn_emb_')]
    gnn_arr  = gnn[gnn_cols].values
    gnn_var  = gnn_arr[train_mask].var(axis=0)
    top16    = np.argsort(gnn_var)[::-1][:16]
    for k in top16:
        feat[f'gnn_{k}'] = gnn_arr[:, k]

print(f"Base features: {feat.shape[1]}")

# ── Entity intelligence features (train-only, no leakage) ─────────────────────
print("Computing entity intelligence features (train-only)...")
t = time.time()
train_df = df[train_mask].copy()
train_df['y_inv']   = y_inv[train_mask].values
train_df['y_fraud'] = y_fraud[train_mask].values

HID_COL = next((c for c in ['HID_anon', 'Hosp_ID', 'Hospital_ID'] if c in df.columns), None)
DID_COL = next((c for c in ['DID_anon', 'Doctor_ID'] if c in df.columns), None)
DIS_COL = next((c for c in ['Final_Diagnosis', 'ICD_Code', 'Diagnosis_Code'] if c in df.columns), None)
EMP_COL = next((c for c in ['CID_anon', 'Employer_ID', 'Company_ID'] if c in df.columns), None)
CLAIM_AMT = next((c for c in ['Claimed_Amt', 'Claim_Amount', 'Total_Claimed'] if c in df.columns), None)
LOS_COL   = next((c for c in ['hosp_days', 'LOS', 'Length_of_Stay'] if c in df.columns), None)

print(f"  HID={HID_COL}, DID={DID_COL}, DIS={DIS_COL}, EMP={EMP_COL}, AMT={CLAIM_AMT}, LOS={LOS_COL}")

def safe_agg(df_all, df_tr, id_col, metric_col, agg, new_name, fallback=0.0):
    """Compute train-only aggregation, merge to all rows."""
    if id_col is None or id_col not in df_tr.columns:
        return pd.Series(fallback, index=df_all.index)
    if metric_col not in df_tr.columns:
        return pd.Series(fallback, index=df_all.index)
    grp = df_tr.groupby(id_col)[metric_col].agg(agg)
    result = df_all[id_col].map(grp).fillna(fallback)
    return result

def herfindahl(series):
    """HHI: sum of squared shares."""
    vc = series.value_counts(normalize=True)
    return (vc ** 2).sum()

# Hospital features
if HID_COL:
    # Fraud/inv rates
    feat['hosp_fraud_rate'] = safe_agg(df, train_df, HID_COL, 'y_fraud', 'mean', 'hfr')
    feat['hosp_inv_rate']   = safe_agg(df, train_df, HID_COL, 'y_inv',   'mean', 'hir')
    feat['hosp_claim_count']= safe_agg(df, train_df, HID_COL, 'y_fraud', 'count', 'hcc')

    # Hospital HHI (diagnosis concentration — #1 signal)
    if DIS_COL and DIS_COL in train_df.columns:
        hhi = train_df.groupby(HID_COL)[DIS_COL].apply(herfindahl)
        feat['hosp_diag_hhi'] = df[HID_COL].map(hhi).fillna(0.0)

    # Hospital mean/std claim amount
    if CLAIM_AMT and CLAIM_AMT in train_df.columns:
        feat['hosp_mean_claim'] = safe_agg(df, train_df, HID_COL, CLAIM_AMT, 'mean', 'hmc')
        feat['hosp_std_claim']  = safe_agg(df, train_df, HID_COL, CLAIM_AMT, 'std',  'hsc', fallback=1.0)

    # Hospital mean LOS
    if LOS_COL and LOS_COL in train_df.columns:
        feat['hosp_mean_los'] = safe_agg(df, train_df, HID_COL, LOS_COL, 'mean', 'hml')

# Doctor features
if DID_COL:
    feat['doc_fraud_rate'] = safe_agg(df, train_df, DID_COL, 'y_fraud', 'mean', 'dfr')
    feat['doc_inv_rate']   = safe_agg(df, train_df, DID_COL, 'y_inv',   'mean', 'dir')
    feat['doc_claim_count']= safe_agg(df, train_df, DID_COL, 'y_fraud', 'count', 'dcc')
    if DIS_COL and DIS_COL in train_df.columns:
        doc_hhi = train_df.groupby(DID_COL)[DIS_COL].apply(herfindahl)
        feat['doc_diag_hhi'] = df[DID_COL].map(doc_hhi).fillna(0.0)

# Employer features
if EMP_COL:
    feat['emp_fraud_rate'] = safe_agg(df, train_df, EMP_COL, 'y_fraud', 'mean', 'efr')
    feat['emp_inv_rate']   = safe_agg(df, train_df, EMP_COL, 'y_inv',   'mean', 'eir')

# Z-score: claim vs hospital average
if HID_COL and CLAIM_AMT and CLAIM_AMT in df.columns and 'hosp_mean_claim' in feat.columns:
    hosp_std = feat['hosp_std_claim'].replace(0, np.nan).fillna(1.0)
    feat['claim_z_vs_hosp'] = (df[CLAIM_AMT].fillna(0) - feat['hosp_mean_claim']) / hosp_std

print(f"  Entity features added. Total: {feat.shape[1]}  {time.time()-t:.1f}s")

# ── OOF Target Encoding (5-fold time-sorted) ───────────────────────────────────
print("Computing OOF target encoding...")
t = time.time()

# Encode top high-cardinality categoricals with OOF mean target encoding
HIGH_CARD_COLS = [c for c in ['HID_anon', 'DID_anon', 'CID_anon'] if c in df.columns]

def oof_target_encode(df_full, train_mask, col, target_series, n_folds=5, global_fallback=None):
    """5-fold time-sorted OOF target encoding."""
    result = pd.Series(np.nan, index=df_full.index)
    if col not in df_full.columns:
        return result
    tr_idx = np.where(train_mask)[0]
    fold_size = len(tr_idx) // n_folds
    global_mean = target_series.iloc[tr_idx].mean() if global_fallback is None else global_fallback

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < n_folds - 1 else len(tr_idx)
        train_fold_idx = np.concatenate([tr_idx[:val_start], tr_idx[val_end:]])
        val_fold_idx   = tr_idx[val_start:val_end]

        grp = target_series.iloc[train_fold_idx].groupby(df_full[col].iloc[train_fold_idx]).mean()
        result.iloc[val_fold_idx] = df_full[col].iloc[val_fold_idx].map(grp).fillna(global_mean)

    # Test: use full train stats
    grp_all = target_series.iloc[tr_idx].groupby(df_full[col].iloc[tr_idx]).mean()
    te_idx = np.where(test_mask)[0]
    result.iloc[te_idx] = df_full[col].iloc[te_idx].map(grp_all).fillna(global_mean)
    return result

for col in HIGH_CARD_COLS[:2]:  # just top 2 to keep it fast
    feat[f'oof_{col}_inv']   = oof_target_encode(df, train_mask, col, y_inv)
    feat[f'oof_{col}_fraud'] = oof_target_encode(df, train_mask, col, y_fraud)

feat = feat.fillna(0)
print(f"  OOF target encoding done. Total: {feat.shape[1]}  {time.time()-t:.1f}s")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler  = RobustScaler()
X_tr_full = scaler.fit_transform(feat[train_mask])
X_te      = scaler.transform(feat[test_mask])

y_te_inv   = y_inv[test_mask].values
y_te_fraud = y_fraud[test_mask].values
y_tr_inv   = y_inv[train_mask].values
y_tr_fraud = y_fraud[train_mask].values

spw_inv   = (y_tr_inv   == 0).sum() / max((y_tr_inv   == 1).sum(), 1)
spw_fraud = (y_tr_fraud == 0).sum() / max((y_tr_fraud == 1).sum(), 1)
print(f"\nFeatures: {feat.shape[1]}  spw_inv={spw_inv:.1f}  spw_fraud={spw_fraud:.1f}")

# Time-decay weights (full)
_tdates = df.loc[train_mask, 'data_created_at_parsed']
_days   = (_tdates - _tdates.min()).dt.days.values.astype(float)
_tdecay = np.exp(0.001 * _days).astype(np.float32)
_tdecay /= _tdecay.mean()

def make_sw(y, spw, base):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base

def eval_all(label, X_tr, X_te, y_tr, y_te, spw, sw, n_est_rf=300, n_est_lgb=500):
    t = time.time()
    rf = RFC(n_estimators=n_est_rf, max_depth=None, min_samples_leaf=5,
             class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr, sample_weight=sw)
    p_rf = rf.predict_proba(X_te)[:, 1]
    pr_rf = average_precision_score(y_te, p_rf)

    lgb = LGBMClassifier(n_estimators=n_est_lgb, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr, y_tr, sample_weight=sw)
    p_lgb = lgb.predict_proba(X_te)[:, 1]
    pr_lgb = average_precision_score(y_te, p_lgb)

    print(f"  {label:<35} RF={pr_rf:.4f}  LGBM={pr_lgb:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf, pr_lgb, p_rf, p_lgb

# ── Baseline (full train) ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("Investigation — full train vs window")
print("="*65)
sw_inv_full   = make_sw(y_tr_inv,   spw_inv,   _tdecay)
_, _, p_rf_inv_full, _ = eval_all("Full train", X_tr_full, X_te, y_tr_inv, y_te_inv, spw_inv, sw_inv_full)

# Window versions (1, 3, 4, 5 months)
for w_mo in [1, 3, 4, 5, 6]:
    wstart = SPLIT_DATE - pd.DateOffset(months=w_mo)
    wmask  = (df['data_created_at_parsed'] >= wstart) & train_mask
    win_in_tr = wmask[train_mask].values
    X_tr_w = X_tr_full[win_in_tr]
    y_tr_w = y_tr_inv[win_in_tr]
    wdates  = df.loc[wmask, 'data_created_at_parsed']
    wdays   = (wdates - wdates.min()).dt.days.values.astype(float)
    wdecay  = np.exp(0.001 * wdays).astype(np.float32); wdecay /= wdecay.mean()
    sw_w    = make_sw(y_tr_w, spw_inv, wdecay)
    n_pos   = y_tr_w.sum()
    if n_pos < 15:
        print(f"  {w_mo}mo window: skip (only {n_pos} positives)")
        continue
    eval_all(f"{w_mo}mo window (n={wmask.sum():,}, pos={n_pos})", X_tr_w, X_te, y_tr_w, y_te_inv, spw_inv, sw_w)

print("\n" + "="*65)
print("Fraud — full train vs window")
print("="*65)
sw_fraud_full = make_sw(y_tr_fraud, spw_fraud, _tdecay)
_, _, p_rf_fr_full, _ = eval_all("Full train", X_tr_full, X_te, y_tr_fraud, y_te_fraud, spw_fraud, sw_fraud_full)

for w_mo in [2, 3, 4, 5, 6]:
    wstart = SPLIT_DATE - pd.DateOffset(months=w_mo)
    wmask  = (df['data_created_at_parsed'] >= wstart) & train_mask
    win_in_tr = wmask[train_mask].values
    X_tr_w = X_tr_full[win_in_tr]
    y_tr_w = y_tr_fraud[win_in_tr]
    wdates  = df.loc[wmask, 'data_created_at_parsed']
    wdays   = (wdates - wdates.min()).dt.days.values.astype(float)
    wdecay  = np.exp(0.001 * wdays).astype(np.float32); wdecay /= wdecay.mean()
    sw_w    = make_sw(y_tr_w, spw_fraud, wdecay)
    n_pos   = y_tr_w.sum()
    if n_pos < 10:
        print(f"  {w_mo}mo window: skip (only {n_pos} positives)")
        continue
    pr_rf, pr_lgb, p_rf_w, p_lgb_w = eval_all(
        f"{w_mo}mo window (n={wmask.sum():,}, pos={n_pos})", X_tr_w, X_te, y_tr_w, y_te_fraud, spw_fraud, sw_w)

    # Blend window RF + full RF
    for alpha in [0.5, 0.7, 0.9]:
        p_blend = alpha * p_rf_w + (1-alpha) * p_rf_fr_full
        blend_pr = average_precision_score(y_te_fraud, p_blend)
        print(f"      blend {alpha:.0%}window+{1-alpha:.0%}full RF => {blend_pr:.4f}")

print("\nDone.")
