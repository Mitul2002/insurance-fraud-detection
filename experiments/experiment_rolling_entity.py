"""
experiment_rolling_entity.py
==============================
Rolling Window Entity Features — targeted fix for Fraud temporal shift.

Root cause (confirmed in MEMORY):
  Entity features (hosp_fraud_rate, doc_fraud_rate, emp_fraud_rate, etc.)
  are computed over the FULL training history (Jan 2024 - Jun 2025 = 18 months).
  When training on a 3-5mo window, these stats reflect OLD patterns that
  predate the window → entity features conflict with recent behavior.
  This is why window training HURTS Fraud RF (but not Investigation, since
  structural features like hosp_diag_hhi are more stable).

Fix (Approach A — decouple entity lookback from training window):
  Recompute entity aggregations using only last N months of training data.
  Keep ALL training rows (415K), but entity feature columns are replaced
  with rolling-window versions that reflect recent entity behavior.
  This gives: recent entity statistics + full training data volume.

Approach B also tested:
  Rolling entity + window training (N-month window for both entity stats
  and training rows). Most aggressive — but less data.

Implementation:
  1. Load checkpoint → get X_tr, X_te, feat_names, train_dates
  2. Load raw CSV → align rows with checkpoint via date mask
  3. Identify surviving entity features in feat_names (after adversarial drop)
  4. For each window [3mo, 6mo, 12mo, full]:
     a. Compute entity aggs from training rows in that window
     b. Build replacement columns (aligned to all train+test rows)
     c. RobustScale using train distribution
     d. Replace entity columns in X_tr/X_te
     e. Train RF + LGBM (full train + matched window)
  5. Compare vs checkpoint baseline
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
CKPT_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
SPLIT_DATE = pd.Timestamp('2025-07-01')

# ── Load checkpoint ────────────────────────────────────────────────────────────
print("Loading checkpoint...")
ck = np.load(CKPT_PATH, allow_pickle=True)
X_tr_orig  = ck['X_tr'].copy()
X_te_orig  = ck['X_te'].copy()
y_tr_inv   = ck['y_train_inv'].astype(int)
y_te_inv   = ck['y_test_inv'].astype(int)
y_tr_fr    = ck['y_train_fraud'].astype(int)
y_te_fr    = ck['y_test_fraud'].astype(int)
sw_train   = ck['sample_weight_tr']
spw_inv    = float(ck['spw_inv'][0])
spw_fr     = float(ck['spw_fr'][0])
train_dates = pd.to_datetime(ck['train_dates'])
feat_names  = list(ck['feat_names'])
print(f"  X_tr={X_tr_orig.shape}  X_te={X_te_orig.shape}  features={len(feat_names)}")

# ── Identify entity feature columns in checkpoint ─────────────────────────────
ENTITY_PREFIXES = ['hosp_', 'doc_', 'dis_', 'emp_']
entity_cols = {f: i for i, f in enumerate(feat_names)
               if any(f.startswith(p) for p in ENTITY_PREFIXES)}
print(f"  Entity features surviving adversarial drop: {len(entity_cols)}")
print(f"  {sorted(entity_cols.keys())}")

# ── Load raw CSV ───────────────────────────────────────────────────────────────
print("\nLoading raw CSV...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
df['_date'] = pd.to_datetime(df['data_created_at'], format='ISO8601')
train_mask = (df['_date'] < SPLIT_DATE).values
test_mask  = ~train_mask

# Verify alignment with checkpoint
ck_dates_set = set(pd.to_datetime(ck['train_dates']).astype(str))
df_tr_dates  = df.loc[train_mask, '_date'].astype(str).values
assert len(df_tr_dates) == len(X_tr_orig), \
    f"Row count mismatch: CSV train={len(df_tr_dates)} vs ckpt={len(X_tr_orig)}"
print(f"  {len(df):,} rows | train={train_mask.sum():,} test={test_mask.sum():,}  ({time.time()-t0:.1f}s)")

# Pre-compute hospital stay duration (needed for some entity features)
adm = pd.to_datetime(df['Actual_Date_of_Admission'], errors='coerce')
dis = pd.to_datetime(df['Actual_Date_of_Discharge'], errors='coerce')
df['_hosp_days'] = (dis - adm).dt.days.clip(0, 90).fillna(0)

# Key column names
HID = 'HID_anon'
DR  = 'Treating_Dr'
DIS = 'Disease_Category'
POL = 'Policy_number'
AMT = 'Claimed_Amt'
TFR = 'Target_as_fraud'
TIN = 'Target_as_investigation'

# ── Entity feature computation function ───────────────────────────────────────
def hhi(series):
    if len(series) == 0: return 0.0
    counts = series.value_counts(normalize=True)
    return float((counts ** 2).sum()) if len(counts) else 0.0

def top_pct(series):
    if len(series) == 0: return 0.0
    vc = series.value_counts()
    return float(vc.iloc[0] / len(series)) if len(vc) else 0.0

def compute_entity_features(df_full, window_mask, test_mask):
    """
    Compute entity aggregations from df_full[window_mask] (rolling window of training).
    Returns a DataFrame with one row per row in df_full, filled with entity stats.
    Test rows get the same entity stats as train rows (lookup by entity ID).
    """
    td = df_full[window_mask].copy()  # rolling window training data
    result = pd.DataFrame(index=df_full.index)

    # Hospital features
    if HID in df_full.columns and DIS in df_full.columns:
        hosp_hhi_s = td.groupby(HID)[DIS].apply(hhi).rename('hosp_diag_hhi')
        hosp_agg = td.groupby(HID).agg(
            hosp_fraud_rate    =(TFR,          'mean'),
            hosp_inv_rate      =(TIN,          'mean'),
            hosp_uniq_doctors  =(DR,           'nunique'),
            hosp_uniq_diagnoses=(DIS,          'nunique'),
            hosp_mean_LOS      =('_hosp_days', 'mean'),
        ).join(hosp_hhi_s)
        result = result.join(df_full[[HID]].join(hosp_agg, on=HID).drop(columns=[HID]))

    # Doctor features
    if DR in df_full.columns and HID in df_full.columns:
        doc_excl = td.groupby(DR)[HID].apply(top_pct).rename('doc_exclusivity')
        doc_hhi  = td.groupby(DR)[DIS].apply(hhi).rename('doc_diag_hhi')
        doc_agg = td.groupby(DR).agg(
            doc_fraud_rate  =(TFR,  'mean'),
            doc_inv_rate    =(TIN,  'mean'),
            doc_claim_mean  =(AMT,  'mean'),
            doc_hosp_count  =(HID,  'nunique'),
            doc_claim_count =(AMT,  'count'),
        ).join(doc_excl).join(doc_hhi)
        result = result.join(df_full[[DR]].join(doc_agg, on=DR).drop(columns=[DR]))

    # Disease features
    if DIS in df_full.columns:
        dis_agg = td.groupby(DIS).agg(
            dis_fraud_rate   =(TFR,          'mean'),
            dis_expected_LOS =('_hosp_days', 'median'),
            dis_LOS_std      =('_hosp_days', 'std'),
            dis_mean_bill    =(AMT,          'mean'),
            dis_std_bill     =(AMT,          'std'),
            dis_uniq_hosps   =(HID,          'nunique'),
        )
        result = result.join(df_full[[DIS]].join(dis_agg, on=DIS).drop(columns=[DIS]))

    # Employer features
    if POL in df_full.columns and HID in df_full.columns:
        emp_flocking = td.groupby(POL)[HID].apply(top_pct).rename('emp_top_hosp_pct')
        emp_agg = td.groupby(POL).agg(
            emp_fraud_rate    =(TFR, 'mean'),
            emp_inv_rate      =(TIN, 'mean'),
            emp_claim_count   =(AMT, 'count'),
            emp_uniq_hospitals=(HID, 'nunique'),
            emp_claim_mean    =(AMT, 'mean'),
        ).join(emp_flocking)
        result = result.join(df_full[[POL]].join(emp_agg, on=POL).drop(columns=[POL]))

    result = result.fillna(0)
    return result

# ── Model helpers ──────────────────────────────────────────────────────────────
def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w.astype(np.float32)

def make_rf(spw):
    return RFC(n_estimators=400, max_depth=None, min_samples_leaf=5,
               class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)

def make_lgbm(spw):
    return LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)

def run(X_t, y_t, sw, X_e, y_e, spw, label):
    t = time.time()
    rf = make_rf(spw)
    rf.fit(X_t, y_t, sample_weight=sw)
    pr_rf = average_precision_score(y_e, rf.predict_proba(X_e)[:, 1])
    lgb = make_lgbm(spw)
    lgb.fit(X_t, y_t, sample_weight=sw)
    pr_lgb = average_precision_score(y_e, lgb.predict_proba(X_e)[:, 1])
    print(f"  {label:<60} RF={pr_rf:.4f}  LGBM={pr_lgb:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf, pr_lgb

def get_window_idx(months):
    wstart = SPLIT_DATE - pd.DateOffset(months=months)
    return (train_dates >= wstart)

def window_sw(win_idx, y, spw):
    wdates = train_dates[win_idx]
    wdays  = (wdates - wdates.min()).total_seconds().values / 86400.0
    wdecay = np.exp(0.001 * wdays).astype(np.float32); wdecay /= wdecay.mean()
    return make_sw(y[win_idx], spw, wdecay)

# ── Baseline (checkpoint entity features, full history) ───────────────────────
print("\n" + "="*70)
print("BASELINES (checkpoint, full-history entity features)")
print("="*70)

sw_full_inv = make_sw(y_tr_inv, spw_inv, sw_train)
sw_full_fr  = make_sw(y_tr_fr,  spw_fr,  sw_train)

# Investigation baselines
run(X_tr_orig, y_tr_inv, sw_full_inv, X_te_orig, y_te_inv, spw_inv,
    "INV  | Full train | checkpoint entity (18mo)")
w5_idx = get_window_idx(5)
run(X_tr_orig[w5_idx], y_tr_inv[w5_idx], window_sw(w5_idx, y_tr_inv, spw_inv),
    X_te_orig, y_te_inv, spw_inv,
    "INV  | 5mo window | checkpoint entity (18mo)")

# Fraud baselines
run(X_tr_orig, y_tr_fr, sw_full_fr, X_te_orig, y_te_fr, spw_fr,
    "FRAUD| Full train | checkpoint entity (18mo)")
w3_idx = get_window_idx(3)
run(X_tr_orig[w3_idx], y_tr_fr[w3_idx], window_sw(w3_idx, y_tr_fr, spw_fr),
    X_te_orig, y_te_fr, spw_fr,
    "FRAUD| 3mo window | checkpoint entity (18mo)")

# ── Rolling entity feature experiments ────────────────────────────────────────
WINDOWS = [3, 6, 12, 18]   # months of entity lookback (18 = full ~18mo history)

print("\n" + "="*70)
print("ROLLING ENTITY FEATURES (Approach A: replace entity cols, keep all train rows)")
print("="*70)

results = {}

for ew in WINDOWS:
    label_ew = f"{ew}mo" if ew < 18 else "full"
    print(f"\n--- Entity window: {label_ew} ---")
    t_ew = time.time()

    # Compute rolling entity features
    ew_start   = SPLIT_DATE - pd.DateOffset(months=ew)
    entity_mask = train_mask & (df['_date'] >= ew_start).values  # rolling window for entity stats

    ent_df = compute_entity_features(df, entity_mask, test_mask)
    print(f"  Entity computed from {entity_mask.sum():,} rows  ({time.time()-t_ew:.1f}s)")

    # Align: train rows, test rows
    ent_tr = ent_df[train_mask].values.astype(np.float32)
    ent_te = ent_df[test_mask].values.astype(np.float32)
    ent_feat_names = list(ent_df.columns)
    print(f"  Entity features computed: {len(ent_feat_names)} — {ent_feat_names}")

    # Scale using train distribution
    scaler = RobustScaler()
    ent_tr_sc = scaler.fit_transform(ent_tr)
    ent_te_sc = scaler.transform(ent_te)

    # Build new X: replace entity columns in checkpoint with rolling versions
    X_tr_new = X_tr_orig.copy()
    X_te_new = X_te_orig.copy()

    replaced = 0
    for feat_name, col_idx in entity_cols.items():
        # Find this feature in the new entity df
        if feat_name in ent_feat_names:
            ent_col_idx = ent_feat_names.index(feat_name)
            X_tr_new[:, col_idx] = ent_tr_sc[:, ent_col_idx]
            X_te_new[:, col_idx] = ent_te_sc[:, ent_col_idx]
            replaced += 1

    print(f"  Replaced {replaced}/{len(entity_cols)} entity columns")

    # Approach A: rolling entity + FULL training data
    print(f"  [Approach A: all {len(X_tr_new):,} train rows, {label_ew} entity stats]")
    r_inv_rf, r_inv_lgb = run(X_tr_new, y_tr_inv, sw_full_inv, X_te_new, y_te_inv, spw_inv,
        f"INV  | Full train | rolling entity {label_ew}")
    r_fr_rf, r_fr_lgb = run(X_tr_new, y_tr_fr, sw_full_fr, X_te_new, y_te_fr, spw_fr,
        f"FRAUD| Full train | rolling entity {label_ew}")

    # Approach B: rolling entity + matched window training
    if ew < 18:
        ew_idx = get_window_idx(ew)
        print(f"  [Approach B: {ew_idx.sum():,} train rows ({label_ew} window), {label_ew} entity stats]")
        run(X_tr_new[ew_idx], y_tr_inv[ew_idx], window_sw(ew_idx, y_tr_inv, spw_inv),
            X_te_new, y_te_inv, spw_inv,
            f"INV  | {label_ew} window+entity | rolling entity {label_ew}")
        run(X_tr_new[ew_idx], y_tr_fr[ew_idx], window_sw(ew_idx, y_tr_fr, spw_fr),
            X_te_new, y_te_fr, spw_fr,
            f"FRAUD| {label_ew} window+entity | rolling entity {label_ew}")

    results[ew] = {'inv_rf': r_inv_rf, 'inv_lgb': r_inv_lgb,
                   'fr_rf': r_fr_rf, 'fr_lgb': r_fr_lgb}

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Rolling Entity Features (Approach A, full train)")
print("="*70)
print(f"  {'Entity window':<15} {'INV-RF':>8} {'INV-LGBM':>10} {'FRAUD-RF':>10} {'FRAUD-LGBM':>12}")
for ew, r in results.items():
    label_ew = f"{ew}mo" if ew < 18 else "full(18mo)"
    print(f"  {label_ew:<15} {r['inv_rf']:>8.4f} {r['inv_lgb']:>10.4f} "
          f"{r['fr_rf']:>10.4f} {r['fr_lgb']:>12.4f}")

print("\nDone.")
