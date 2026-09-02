"""
experiment_window_focal.py
==========================
Combines our rolling entity features (3mo) with Anshul's techniques:
  - 9-month training window  (quantile 0.7 cutoff, ~Sep 2024)
  - Focal loss LGBM          (alpha=0.75, gamma=1.0)
  - 15-seed averaging

Configurations tested (Investigation + Fraud):
  A  Baseline RE  — scale_pos_weight, full 415K rows        (our current best ~0.27)
  B  RE + 9mo     — scale_pos_weight, 9mo window
  C  RE + Focal   — focal loss,       full 415K, 15 seeds
  D  RE + 9mo + Focal — focal loss,   9mo window,  15 seeds  (main test)
  E  Anshul-like  — no rolling entity, 9mo window, focal, 15 seeds

Question: does combining our rolling entity breakthrough with Anshul's focal + 9mo beat both?
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import warnings, time
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
DATA_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
RANDOM_STATE = 42

# ── Load checkpoint ───────────────────────────────────────────────────────────
print("Loading checkpoint...")
d = np.load(CKPT_PATH, allow_pickle=True)
X_tr_ckpt      = d['X_tr'].copy()          # (415751, 238)
X_te_ckpt      = d['X_te'].copy()          # (58868,  238)
y_train_inv    = d['y_train_inv']
y_test_inv     = d['y_test_inv']
y_train_fraud  = d['y_train_fraud']
y_test_fraud   = d['y_test_fraud']
sample_weight  = d['sample_weight_tr']
train_dates    = pd.to_datetime(d['train_dates'])
feat_names     = list(d['feat_names'])
print(f"  X_tr: {X_tr_ckpt.shape}  X_te: {X_te_ckpt.shape}")
print(f"  Train dates: {train_dates.min().date()} -> {train_dates.max().date()}")

# ── Load raw data for 3mo entity recomputation ────────────────────────────────
print("\nLoading raw data for rolling entity features...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df['data_created_at_parsed'] = pd.to_datetime(df['data_created_at'], errors='coerce')
# NaT rows (1,005 unparseable dates) fall into train — fill with early date to match checkpoint
df['data_created_at_parsed'] = df['data_created_at_parsed'].fillna(pd.Timestamp('2020-01-01'))
train_mask = df['data_created_at_parsed'] < '2025-07-01'
test_mask  = df['data_created_at_parsed'] >= '2025-07-01'
print(f"  Train: {train_mask.sum():,}  Test: {test_mask.sum():,}")

# ── Rolling entity feature computation (3mo window) ──────────────────────────
print("\nComputing 3-month rolling entity features...")

ENTITY_WINDOW_MONTHS = 3
SMOOTH = 10    # same as main notebook

win_end   = df.loc[train_mask, 'data_created_at_parsed'].max()
win_start = win_end - pd.DateOffset(months=ENTITY_WINDOW_MONTHS)
win_mask  = (df['data_created_at_parsed'] >= win_start) & (df['data_created_at_parsed'] <= win_end)
df_win    = df[win_mask & train_mask].copy()
print(f"  3mo window: {win_start.date()} -> {win_end.date()}  ({win_mask.sum():,} rows)")

# Also parse admission dates (for LOS-based features)
for col in ['Actual_Date_of_Admission', 'Actual_Date_of_Discharge']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')
df['hosp_days_raw'] = (df['Actual_Date_of_Discharge'] - df['Actual_Date_of_Admission']).dt.days.clip(0, 365).fillna(0)

glob_inv_mean   = df.loc[train_mask, 'Target_as_investigation'].mean()
glob_fraud_mean = df.loc[train_mask, 'Target_as_fraud'].mean()

def smooth_rate(series_sum, series_count, global_mean):
    return (series_sum + SMOOTH * global_mean) / (series_count + SMOOTH)

def compute_entity_stats(group_col, df_ref, df_all, global_inv_mean, global_fraud_mean):
    """Compute entity stats from df_ref (window), map to df_all."""
    stats = {}
    grp = df_ref.groupby(group_col)
    n   = df_all.groupby(group_col)[group_col].transform('count').reindex(df_all.index).fillna(0)

    for tgt, gm, sfx in [
        ('Target_as_fraud',         global_fraud_mean, 'fraud_rate'),
        ('Target_as_investigation', global_inv_mean,   'inv_rate'),
    ]:
        agg = grp[tgt].agg(['sum', 'count']).reset_index()
        agg['rate'] = smooth_rate(agg['sum'], agg['count'], gm)
        mapped = df_all[[group_col]].merge(agg[[group_col, 'rate']], on=group_col, how='left')['rate'].fillna(gm).values
        stats[f'{sfx}'] = mapped

    # Additional numeric aggs
    if 'Claimed_Amt' in df_ref.columns:
        agg_claim = grp['Claimed_Amt'].agg(['mean', 'count']).reset_index()
        agg_claim.columns = [group_col, 'mean', 'count']
        stats['claim_mean'] = df_all[[group_col]].merge(agg_claim[[group_col, 'mean']], on=group_col, how='left')['mean'].fillna(0).values
        stats['claim_count'] = df_all[[group_col]].merge(agg_claim[[group_col, 'count']], on=group_col, how='left')['count'].fillna(0).values

    if 'hosp_days_raw' in df_ref.columns:
        agg_los = grp['hosp_days_raw'].mean().reset_index()
        agg_los.columns = [group_col, 'mean_LOS']
        stats['mean_LOS'] = df_all[[group_col]].merge(agg_los, on=group_col, how='left')['mean_LOS'].fillna(0).values

    return stats

# Hospital features
hosp_stats = compute_entity_stats('HID_anon', df_win, df, glob_inv_mean, glob_fraud_mean)
# Doctor features
doc_stats  = compute_entity_stats('Treating_Dr', df_win, df, glob_inv_mean, glob_fraud_mean)
# Disease features
dis_stats  = compute_entity_stats('Disease_Category', df_win, df, glob_inv_mean, glob_fraud_mean)
# Employer features (Policy_number as employer proxy)
emp_stats  = compute_entity_stats('Policy_number', df_win, df, glob_inv_mean, glob_fraud_mean)

# Map to train and test positions
entity_replacements = {
    'hosp_fraud_rate':   hosp_stats['fraud_rate'],
    'hosp_inv_rate':     hosp_stats['inv_rate'],
    'hosp_mean_LOS':     hosp_stats.get('mean_LOS', np.zeros(len(df))),
    'doc_fraud_rate':    doc_stats['fraud_rate'],
    'doc_inv_rate':      doc_stats['inv_rate'],
    'doc_claim_mean':    doc_stats.get('claim_mean', np.zeros(len(df))),
    'doc_claim_count':   doc_stats.get('claim_count', np.zeros(len(df))),
    'dis_fraud_rate':    dis_stats['fraud_rate'],
    'emp_fraud_rate':    emp_stats['fraud_rate'],
    'emp_inv_rate':      emp_stats['inv_rate'],
    'emp_claim_count':   emp_stats.get('claim_count', np.zeros(len(df))),
    'emp_claim_mean':    emp_stats.get('claim_mean', np.zeros(len(df))),
}

# Build X_tr_re and X_te_re by replacing entity columns
train_pos = np.where(train_mask.values)[0]
test_pos  = np.where(test_mask.values)[0]

X_tr_re = X_tr_ckpt.copy()
X_te_re = X_te_ckpt.copy()

replaced = []
for col_name, full_arr in entity_replacements.items():
    if col_name in feat_names:
        ci = feat_names.index(col_name)
        # RobustScale the replacement values using train window stats
        tr_vals = full_arr[train_pos].reshape(-1, 1)
        te_vals = full_arr[test_pos].reshape(-1, 1)
        scaler  = RobustScaler()
        scaler.fit(tr_vals)
        X_tr_re[:, ci] = scaler.transform(tr_vals).ravel()
        X_te_re[:, ci] = scaler.transform(te_vals).ravel()
        replaced.append(col_name)

print(f"  Replaced {len(replaced)} entity columns with 3mo stats")

# ── 9-month window mask ───────────────────────────────────────────────────────
win9_cutoff = np.quantile(train_dates.astype(np.int64), 0.30)   # 30th pct = oldest 30%
win9_cutoff = pd.Timestamp(win9_cutoff)                          # keep newest 70% = ~9 months
win9_mask   = train_dates >= win9_cutoff
print(f"\n9-month window: {win9_cutoff.date()} -> {train_dates.max().date()}")
print(f"  Training rows in 9mo window: {win9_mask.sum():,} (of {len(train_dates):,})")

# ── Focal loss objective (Anshul's implementation) ────────────────────────────
def focal_loss(alpha=0.75, gamma=1.0):
    """
    Focal loss for LightGBM.
    - Downweights easy negatives via (1-pt)^gamma term
    - alpha=0.75: positive class gets 3x more weight than negative
    - gamma=1.0: mild focusing (gamma=2 is stronger)

    WHY we use alpha=0.75 (not scale_pos_weight):
      scale_pos_weight multiplies the loss for ALL positive examples uniformly.
      Focal loss additionally downweights easy negatives that the model is already
      confident about — so the model focuses gradient budget on hard cases.
      At 1:20 ratio (9mo window), alpha=0.75 keeps positives dominant.
      At 1:35 ratio (full train), you'd need alpha~0.97 for the same effect.
    """
    def _obj(preds, dataset):
        labels = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))          # sigmoid(raw)
        pt      = np.where(labels == 1, p, 1 - p)  # prob of correct class
        alpha_t = np.where(labels == 1, alpha, 1 - alpha)
        fw      = alpha_t * (1.0 - pt) ** gamma    # focal weight

        grad = fw * (p - labels)
        hess = fw * p * (1.0 - p) * (gamma * (1.0 - pt) * np.log(pt + 1e-8) + 1.0)
        hess = np.abs(hess)                        # must be positive
        return grad, hess
    return _obj


# ── Training helpers ──────────────────────────────────────────────────────────
LGBM_BASE = dict(
    learning_rate=0.05, num_leaves=63, max_depth=7,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
)

def train_spw_lgbm(X_tr, y_tr, X_te, y_te, sw=None, tag=""):
    """Single LGBM with scale_pos_weight."""
    spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    t   = time.time()
    m   = lgb.LGBMClassifier(
        **LGBM_BASE,
        n_estimators=800, scale_pos_weight=spw,
        random_state=RANDOM_STATE,
    )
    m.fit(X_tr, y_tr, sample_weight=sw)
    pr = average_precision_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"  {tag:<40} PR-AUC = {pr:.4f}  ({time.time()-t:.0f}s)")
    return pr


def train_focal_lgbm_multiseed(X_tr, y_tr, X_te, y_te, sw=None, n_seeds=15, tag=""):
    """Focal LGBM averaged over N seeds (early stopping per seed)."""
    t    = time.time()
    preds_all = []
    best_rounds = []

    for seed in range(n_seeds):
        X_es, X_val, y_es, y_val, sw_es, _ = train_test_split(
            X_tr, y_tr, sw if sw is not None else np.ones(len(y_tr)),
            test_size=0.15, random_state=seed + 42, stratify=y_tr
        )
        params = {**LGBM_BASE,
                  'seed': seed + 42, 'bagging_seed': seed + 42,
                  'feature_fraction_seed': seed + 42,
                  'objective': focal_loss(alpha=0.75, gamma=1.0),
                  'metric': 'average_precision'}

        ds_tr  = lgb.Dataset(X_es,  label=y_es,  weight=sw_es)
        ds_val = lgb.Dataset(X_val, label=y_val)
        m_es = lgb.train(params, ds_tr, num_boost_round=1000,
                         valid_sets=[ds_val],
                         callbacks=[lgb.early_stopping(50, verbose=False),
                                    lgb.log_evaluation(-1)])
        best_rounds.append(m_es.best_iteration)

    # Train final models on full train with median best_round
    best_n = int(np.median(best_rounds))
    print(f"  {tag} — median best_round: {best_n}")
    preds_all = []
    for seed in range(n_seeds):
        params = {**LGBM_BASE,
                  'seed': seed + 42, 'bagging_seed': seed + 42,
                  'feature_fraction_seed': seed + 42,
                  'objective': focal_loss(alpha=0.75, gamma=1.0),
                  'metric': 'average_precision'}
        ds_full = lgb.Dataset(X_tr, label=y_tr,
                              weight=sw if sw is not None else None)
        m_full = lgb.train(params, ds_full, num_boost_round=best_n,
                           callbacks=[lgb.log_evaluation(-1)])
        raw = m_full.predict(X_te)
        preds_all.append(1.0 / (1.0 + np.exp(-raw)))   # sigmoid (custom obj = raw output)

    avg_pred = np.mean(preds_all, axis=0)
    pr = average_precision_score(y_te, avg_pred)
    print(f"  {tag:<40} PR-AUC = {pr:.4f}  ({time.time()-t:.0f}s)")
    return pr


# ── Run all configurations ────────────────────────────────────────────────────
results = {}

for target_label, y_tr_full, y_te in [
    ("INVESTIGATION", y_train_inv,   y_test_inv),
    ("FRAUD",         y_train_fraud, y_test_fraud),
]:
    print(f"\n{'='*65}")
    print(f"TARGET: {target_label}")
    print(f"{'='*65}")

    # Config A: Baseline RE (scale_pos_weight, full 415K)
    pr = train_spw_lgbm(X_tr_re, y_tr_full, X_te_re, y_te,
                         sw=sample_weight, tag="A  RE + SPW  (full 415K)")
    results[f"A_{target_label}"] = pr

    # Config B: RE + 9mo window + scale_pos_weight
    X_9mo   = X_tr_re[win9_mask]
    y_9mo   = y_tr_full[win9_mask]
    sw_9mo  = sample_weight[win9_mask]
    pr = train_spw_lgbm(X_9mo, y_9mo, X_te_re, y_te,
                         sw=sw_9mo, tag="B  RE + SPW  + 9mo window")
    results[f"B_{target_label}"] = pr

    # Config C: RE + Focal loss (full 415K, 15 seeds)
    pr = train_focal_lgbm_multiseed(X_tr_re, y_tr_full, X_te_re, y_te,
                                     sw=sample_weight, n_seeds=15,
                                     tag="C  RE + Focal (full 415K, 15s)")
    results[f"C_{target_label}"] = pr

    # Config D: RE + 9mo window + Focal loss (15 seeds)  <- KEY EXPERIMENT
    pr = train_focal_lgbm_multiseed(X_9mo, y_9mo, X_te_re, y_te,
                                     sw=sw_9mo, n_seeds=15,
                                     tag="D  RE + Focal + 9mo (15s)  ***")
    results[f"D_{target_label}"] = pr

    # Config E: Anshul-style (no rolling entity, 9mo window, focal, 15 seeds)
    X_9mo_orig = X_tr_ckpt[win9_mask]
    pr = train_focal_lgbm_multiseed(X_9mo_orig, y_9mo, X_te_ckpt, y_te,
                                     sw=sw_9mo, n_seeds=15,
                                     tag="E  Anshul-style (no RE, 9mo, 15s)")
    results[f"E_{target_label}"] = pr

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"\n{'Config':<42} {'INV':>8} {'FRAUD':>8}")
print("-" * 60)
labels = {
    'A': 'A  RE + scale_pos_weight (full 415K)',
    'B': 'B  RE + scale_pos_weight + 9mo win',
    'C': 'C  RE + Focal LGBM (full, 15 seeds)',
    'D': 'D  RE + Focal LGBM + 9mo  (15 seeds) ***',
    'E': 'E  Anshul-style (no RE, 9mo, focal, 15s)',
}
for k, label in labels.items():
    inv   = results.get(f"{k}_INVESTIGATION", float('nan'))
    fraud = results.get(f"{k}_FRAUD",         float('nan'))
    print(f"{label:<42} {inv:>8.4f} {fraud:>8.4f}")

print(f"\nAnshul best (Investigation): 0.3343  (9mo FocalLGB, single seed)")
print(f"Our best before (Investigation): 0.2686  (RE-LGBM, scale_pos_weight)")
print(f"\nKey question: Does D beat both?")
