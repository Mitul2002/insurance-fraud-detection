"""
experiment_new_features.py
===========================
Tests new shift-resistant features ON TOP of checkpoint features:

  1. Velocity ratios between window sizes
       vel_30d / vel_90d  -> spike detector (high = recent burst)
       vel_7d  / vel_30d  -> acute burst detector
     Source: all 4 research docs; empirically, ratios are time-invariant

  2. Policy inception bins
       days_since_inception -> 4 buckets: 0-30d, 30-90d, 90-180d, 180d+
       Docs: "group plans waive waiting periods, making inception signal crucial"

  3. Claimed-amount ratio vs entity means (complementary to z-scores)
       Claimed_Amt / hosp_mean_bill  -> ratio (vs z-score already in checkpoint)
       Claimed_Amt / doc_claim_mean  -> doctor-level ratio

  4. TTD x Hospital Cash product (add-on fraud signal)
       Already have individual flags; product captures co-occurrence
       Docs: "TTD and Hospital Cash are most fraud-prone add-ons"

Method:
  - Compute new features from raw CSV (not from already-scaled checkpoint)
  - RobustScale new features using train stats
  - Horizontally stack with checkpoint X_tr / X_te
  - Test RF + LGBM: full train + 5mo window + 4mo window
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

DATA_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\CFM_anon_final.csv"
CKPT_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
SPLIT_DATE = pd.Timestamp('2025-07-01')

# ── Load checkpoint ────────────────────────────────────────────────────────────
print("Loading checkpoint...")
ck = np.load(CKPT_PATH, allow_pickle=True)
X_tr      = ck['X_tr']
X_te      = ck['X_te']
y_tr_inv  = ck['y_train_inv'].astype(int)
y_te_inv  = ck['y_test_inv'].astype(int)
y_tr_fr   = ck['y_train_fraud'].astype(int)
y_te_fr   = ck['y_test_fraud'].astype(int)
sw_train  = ck['sample_weight_tr']
spw_inv   = float(ck['spw_inv'][0])
spw_fr    = float(ck['spw_fr'][0])
train_dates = pd.to_datetime(ck['train_dates'])
feat_names  = list(ck['feat_names'])
print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  features={len(feat_names)}")

# Show what velocity features exist in checkpoint
vel_feats = [f for f in feat_names if any(w in f for w in ['_30d', '_90d', '_7d', 'vel'])]
print(f"  Velocity-like features in checkpoint ({len(vel_feats)}): {vel_feats[:12]}")

# ── Load raw data ──────────────────────────────────────────────────────────────
print("\nLoading raw data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
df['date'] = pd.to_datetime(df['data_created_at'], format='ISO8601')
train_mask = df['date'] < SPLIT_DATE
test_mask  = ~train_mask
print(f"  {len(df):,} rows | train={train_mask.sum():,} test={test_mask.sum():,} ({time.time()-t0:.1f}s)")

# ── New Feature 1: Velocity ratios ────────────────────────────────────────────
# We need to find which velocity columns are in the checkpoint feat_names
# and compute their ratio versions from the raw data.
# Strategy: look for pairs like *_30d and *_90d with the same prefix.

def find_vel_pairs(names):
    pairs = []
    for n in names:
        if '_30d' in n:
            base = n.replace('_30d', '')
            n90 = base + '_90d'
            n7  = base + '_7d'
            if n90 in names:
                pairs.append((n, n90, '30d_vs_90d', base))
            if n7 in names:
                pairs.append((n7, n, '7d_vs_30d', base))
    return pairs

vel_pairs = find_vel_pairs(feat_names)
print(f"\nVelocity ratio pairs found: {len(vel_pairs)}")
for p in vel_pairs:
    print(f"  {p[2]}: {p[0]} / {p[1]}")

# Compute velocity ratios from scaled checkpoint features
# Note: features are RobustScaled so ratio is in transformed space.
# This is an approximation; for cleaner results we'd want unscaled.
# However, RobustScaler subtracts median and divides by IQR, so
# the scaled ratio reflects relative changes reasonably well.
new_features_tr = []
new_features_te = []
new_feat_names  = []

for short_col, long_col, ratio_name, base in vel_pairs:
    idx_short = feat_names.index(short_col)
    idx_long  = feat_names.index(long_col)
    # Compute ratio: short_window / long_window
    # Add small epsilon to avoid division by zero in scaled space
    eps = 1e-3
    ratio_tr = X_tr[:, idx_short] / (X_tr[:, idx_long] + eps)
    ratio_te = X_te[:, idx_short] / (X_te[:, idx_long] + eps)
    new_features_tr.append(ratio_tr)
    new_features_te.append(ratio_te)
    new_feat_names.append(f'ratio_{ratio_name}_{base}')

print(f"\nVelocity ratio features created: {len(new_feat_names)}")

# ── New Feature 2: Policy inception bins (from raw data) ──────────────────────
# days_since_inception already in checkpoint — find it
incep_candidates = [f for f in feat_names if 'inception' in f.lower() or 'days_since' in f.lower()]
print(f"\nInception-related features in checkpoint: {incep_candidates}")

if incep_candidates:
    incep_col = incep_candidates[0]
    idx_incep = feat_names.index(incep_col)
    # Create 4 bins from scaled inception days
    # Since it's RobustScaled, approximate boundaries:
    # Look at the distribution and create quantile-based bins
    incep_tr = X_tr[:, idx_incep]
    incep_te = X_te[:, idx_incep]
    # Bin into quartiles based on training distribution
    q25, q50, q75 = np.percentile(incep_tr, [25, 50, 75])
    for q, qval in zip([25, 50, 75], [q25, q50, q75]):
        bin_tr = (incep_tr <= qval).astype(np.float32)
        bin_te = (incep_te <= qval).astype(np.float32)
        new_features_tr.append(bin_tr)
        new_features_te.append(bin_te)
        new_feat_names.append(f'incep_q{q}_flag')
    print(f"  Created 3 inception bin features (quartile-based)")
else:
    # Compute from raw data if not in checkpoint
    print("  Computing inception signal from raw data...")
    if 'Risk_Inception_Date' in df.columns:
        df['risk_inc'] = pd.to_datetime(df['Risk_Inception_Date'], errors='coerce')
        df['days_since_incep'] = (df['date'] - df['risk_inc']).dt.days.clip(0, 3650)
        dsi = df['days_since_incep'].values
        tr_dsi = dsi[train_mask]
        te_dsi = dsi[test_mask]
        for thr, name in [(30, '30d'), (90, '90d'), (180, '180d')]:
            new_features_tr.append((tr_dsi <= thr).astype(np.float32))
            new_features_te.append((te_dsi <= thr).astype(np.float32))
            new_feat_names.append(f'incep_within_{name}')
        print(f"  Created 3 inception bin features (absolute thresholds)")

# ── New Feature 3: Claimed-amount ratios vs entity means ──────────────────────
# Check if relevant features exist in checkpoint
amt_col = next((f for f in feat_names if 'claimed' in f.lower() and 'amt' in f.lower()), None)
hosp_mean_col = next((f for f in feat_names if 'hosp_mean' in f.lower() and ('bill' in f.lower() or 'amt' in f.lower())), None)
doc_mean_col  = next((f for f in feat_names if 'doc_claim_mean' in f.lower() or 'doc_mean' in f.lower()), None)

print(f"\nAmount-related features:")
print(f"  claimed amt: {amt_col}")
print(f"  hosp mean:   {hosp_mean_col}")
print(f"  doc mean:    {doc_mean_col}")

if amt_col and hosp_mean_col:
    idx_amt  = feat_names.index(amt_col)
    idx_hm   = feat_names.index(hosp_mean_col)
    ratio_tr = X_tr[:, idx_amt] / (X_tr[:, idx_hm] + 1e-3)
    ratio_te = X_te[:, idx_amt] / (X_te[:, idx_hm] + 1e-3)
    new_features_tr.append(np.clip(ratio_tr, -10, 10).astype(np.float32))
    new_features_te.append(np.clip(ratio_te, -10, 10).astype(np.float32))
    new_feat_names.append('amt_ratio_vs_hosp_mean')
    print("  Created: amt_ratio_vs_hosp_mean")

if amt_col and doc_mean_col:
    idx_amt  = feat_names.index(amt_col)
    idx_dm   = feat_names.index(doc_mean_col)
    ratio_tr = X_tr[:, idx_amt] / (X_tr[:, idx_dm] + 1e-3)
    ratio_te = X_te[:, idx_amt] / (X_te[:, idx_dm] + 1e-3)
    new_features_tr.append(np.clip(ratio_tr, -10, 10).astype(np.float32))
    new_features_te.append(np.clip(ratio_te, -10, 10).astype(np.float32))
    new_feat_names.append('amt_ratio_vs_doc_mean')
    print("  Created: amt_ratio_vs_doc_mean")

# ── New Feature 4: TTD x HospCash interaction ─────────────────────────────────
ttd_col  = next((f for f in feat_names if 'ttd' in f.lower() and 'hosp' not in f.lower()), None)
hcash_col = next((f for f in feat_names if 'hosp_cash' in f.lower() or ('cash' in f.lower())), None)
has_ttd_col = next((f for f in feat_names if f.lower() in ('has_ttd', 'ttd_flag')), None)
has_hc_col  = next((f for f in feat_names if f.lower() in ('has_hosp_cash', 'hosp_cash_flag')), None)
# Try alternative names
if not has_ttd_col:
    has_ttd_col = next((f for f in feat_names if 'ttd' in f.lower()), None)
if not has_hc_col:
    has_hc_col = next((f for f in feat_names if 'hosp_cash' in f.lower()), None)

print(f"\nAdd-on features: TTD={has_ttd_col}  HospCash={has_hc_col}")
if has_ttd_col and has_hc_col:
    idx_ttd = feat_names.index(has_ttd_col)
    idx_hc  = feat_names.index(has_hc_col)
    prod_tr = (X_tr[:, idx_ttd] * X_tr[:, idx_hc]).astype(np.float32)
    prod_te = (X_te[:, idx_ttd] * X_te[:, idx_hc]).astype(np.float32)
    new_features_tr.append(prod_tr)
    new_features_te.append(prod_te)
    new_feat_names.append('TTD_x_HospCash_product')
    print("  Created: TTD_x_HospCash_product")

# ── Assemble augmented feature matrices ───────────────────────────────────────
if new_features_tr:
    new_tr = np.column_stack(new_features_tr).astype(np.float32)
    new_te = np.column_stack(new_features_te).astype(np.float32)
    # Scale new features using train stats
    scaler = RobustScaler()
    new_tr_sc = scaler.fit_transform(new_tr)
    new_te_sc = scaler.transform(new_te)
    X_tr_aug = np.hstack([X_tr, new_tr_sc])
    X_te_aug = np.hstack([X_te, new_te_sc])
    print(f"\nAugmented shape: X_tr_aug={X_tr_aug.shape}  X_te_aug={X_te_aug.shape}")
    print(f"New features ({len(new_feat_names)}): {new_feat_names}")
else:
    print("\nNo new features created — using checkpoint as-is")
    X_tr_aug, X_te_aug = X_tr, X_te

# ── Training and evaluation ───────────────────────────────────────────────────
def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w

def run(X_t, X_e, y_t, y_e, spw_, sw_, label, n_rf=400, n_lgb=600):
    t = time.time()
    rf = RFC(n_estimators=n_rf, max_depth=None, min_samples_leaf=5,
             class_weight={0:1, 1:int(spw_)}, random_state=42, n_jobs=-1)
    rf.fit(X_t, y_t, sample_weight=sw_)
    pr_rf = average_precision_score(y_e, rf.predict_proba(X_e)[:, 1])

    lgb = LGBMClassifier(n_estimators=n_lgb, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw_, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_t, y_t, sample_weight=sw_)
    pr_lgb = average_precision_score(y_e, lgb.predict_proba(X_e)[:, 1])
    print(f"  {label:<50} RF={pr_rf:.4f}  LGBM={pr_lgb:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf, pr_lgb

def get_window(X, dates, months):
    wstart  = SPLIT_DATE - pd.DateOffset(months=months)
    win_idx = (dates >= wstart)
    Xw      = X[win_idx]
    yw_inv  = y_tr_inv[win_idx]
    yw_fr   = y_tr_fr[win_idx]
    wdates  = dates[win_idx]
    wdays   = (wdates - wdates.min()).total_seconds().values / 86400.0
    wdecay  = np.exp(0.001 * wdays).astype(np.float32); wdecay /= wdecay.mean()
    return Xw, yw_inv, yw_fr, wdecay

# ── Investigation ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("INVESTIGATION")
print("="*70)

sw_full_inv = make_sw(y_tr_inv, spw_inv, sw_train)

# Checkpoint baselines
run(X_tr, X_te, y_tr_inv, y_te_inv, spw_inv, sw_full_inv,
    "Baseline: checkpoint full train")
Xw5, yw5_inv, _, wd5 = get_window(X_tr, train_dates, 5)
run(Xw5, X_te, yw5_inv, y_te_inv, spw_inv, make_sw(yw5_inv, spw_inv, wd5),
    "Baseline: checkpoint 5mo window")

# Augmented features
run(X_tr_aug, X_te_aug, y_tr_inv, y_te_inv, spw_inv, sw_full_inv,
    "AUGMENTED: full train +new_features")
Xw5a, yw5a_inv, _, wd5a = get_window(X_tr_aug, train_dates, 5)
run(Xw5a, X_te_aug, yw5a_inv, y_te_inv, spw_inv, make_sw(yw5a_inv, spw_inv, wd5a),
    "AUGMENTED: 5mo window +new_features")
Xw4a, yw4a_inv, _, wd4a = get_window(X_tr_aug, train_dates, 4)
run(Xw4a, X_te_aug, yw4a_inv, y_te_inv, spw_inv, make_sw(yw4a_inv, spw_inv, wd4a),
    "AUGMENTED: 4mo window +new_features")

# ── Fraud ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FRAUD")
print("="*70)

sw_full_fr = make_sw(y_tr_fr, spw_fr, sw_train)

run(X_tr, X_te, y_tr_fr, y_te_fr, spw_fr, sw_full_fr,
    "Baseline: checkpoint full train")
run(X_tr_aug, X_te_aug, y_tr_fr, y_te_fr, spw_fr, sw_full_fr,
    "AUGMENTED: full train +new_features")
Xw3a, _, yw3a_fr, wd3a = get_window(X_tr_aug, train_dates, 3)
run(Xw3a, X_te_aug, yw3a_fr, y_te_fr, spw_fr, make_sw(yw3a_fr, spw_fr, wd3a),
    "AUGMENTED: 3mo window +new_features")
Xw4af, _, yw4a_fr, wd4af = get_window(X_tr_aug, train_dates, 4)
run(Xw4af, X_te_aug, yw4a_fr, y_te_fr, spw_fr, make_sw(yw4a_fr, spw_fr, wd4af),
    "AUGMENTED: 4mo window +new_features")

print("\nDone.")
