"""
experiment_checkpoint_window.py
================================
Window training experiment using the FULL pipeline checkpoint (X_tr from
notebook after all feature engineering + adversarial drop).

Requires: checkpoint.npz (saved by notebook cell ckpt_save_01)

Tests:
  - Baseline: full train
  - Window: 2, 3, 4, 5, 6, 9 months before split
  - Blend: window RF + full RF

Usage: python experiment_checkpoint_window.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

CKPT_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
SPLIT_DATE = pd.Timestamp('2025-07-01')

# ── Load checkpoint ────────────────────────────────────────────────────────────
print("Loading checkpoint...")
t0 = time.time()
ck = np.load(CKPT_PATH, allow_pickle=True)

X_tr      = ck['X_tr']
X_te      = ck['X_te']
y_tr_inv  = ck['y_train_inv'].astype(int)
y_te_inv  = ck['y_test_inv'].astype(int)
y_tr_fr   = ck['y_train_fraud'].astype(int)
y_te_fr   = ck['y_test_fraud'].astype(int)
sw_train  = ck['sample_weight_tr']  # time-decay weights from notebook
spw_inv   = float(ck['spw_inv'][0])
spw_fr    = float(ck['spw_fr'][0])

train_dates = pd.to_datetime(ck['train_dates'])  # datetime64 array
feat_names  = list(ck['feat_names'])

print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  {time.time()-t0:.1f}s")
print(f"  Inv  train pos={y_tr_inv.sum():,}  test pos={y_te_inv.sum():,}")
print(f"  Fraud train pos={y_tr_fr.sum():,}  test pos={y_te_fr.sum():,}")
print(f"  spw_inv={spw_inv:.1f}  spw_fr={spw_fr:.1f}")
print(f"  sample_weight: min={sw_train.min():.3f}  max={sw_train.max():.3f}")

def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w

def run_rf_lgbm(X_tr_, X_te_, y_tr_, y_te_, spw_, sw_, label, n_rf=400, n_lgb=600):
    t = time.time()
    rf = RFC(n_estimators=n_rf, max_depth=None, min_samples_leaf=5,
             class_weight={0:1, 1:int(spw_)}, random_state=42, n_jobs=-1)
    rf.fit(X_tr_, y_tr_, sample_weight=sw_)
    p_rf  = rf.predict_proba(X_te_)[:, 1]
    pr_rf = average_precision_score(y_te_, p_rf)

    lgb = LGBMClassifier(n_estimators=n_lgb, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw_, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_tr_, y_tr_, sample_weight=sw_)
    p_lgb = lgb.predict_proba(X_te_)[:, 1]
    pr_lgb = average_precision_score(y_te_, p_lgb)

    print(f"  {label:<45} RF={pr_rf:.4f}  LGBM={pr_lgb:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf, pr_lgb, p_rf, p_lgb

# ── Full baseline ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("INVESTIGATION")
print("="*70)
sw_inv_full = make_sw(y_tr_inv, spw_inv, sw_train)
pr_rf_inv_full, pr_lgb_inv_full, p_rf_inv_full, p_lgb_inv_full = run_rf_lgbm(
    X_tr, X_te, y_tr_inv, y_te_inv, spw_inv, sw_inv_full, "Full train")

results_inv = {'full': {'RF': pr_rf_inv_full, 'LGBM': pr_lgb_inv_full}}

for w_mo in [1, 2, 3, 4, 5, 6, 9]:
    wstart  = SPLIT_DATE - pd.DateOffset(months=w_mo)
    win_idx = (train_dates >= wstart)
    n_win   = win_idx.sum()
    y_win   = y_tr_inv[win_idx]
    n_pos   = y_win.sum()
    if n_pos < 20:
        print(f"  {w_mo}mo: skip (pos={n_pos})")
        continue
    X_tr_w  = X_tr[win_idx]
    # Recompute time decay within window
    wdates  = train_dates[win_idx]
    wdays   = (wdates - wdates.min()).total_seconds().values / 86400.0
    wdecay  = np.exp(0.001 * wdays).astype(np.float32)
    wdecay /= wdecay.mean()
    sw_w    = make_sw(y_win, spw_inv, wdecay)

    pr_rf, pr_lgb, p_rf, _ = run_rf_lgbm(
        X_tr_w, X_te, y_win, y_te_inv, spw_inv, sw_w,
        f"{w_mo}mo window (n={n_win:,} pos={n_pos})")
    results_inv[w_mo] = {'RF': pr_rf, 'LGBM': pr_lgb, 'p_rf': p_rf}

# Best window for Inv
best_w_inv = max([k for k in results_inv if k != 'full'], key=lambda k: results_inv[k]['RF'])
print(f"\nBest Inv window: {best_w_inv}mo")
if 'p_rf' in results_inv[best_w_inv]:
    for alpha in [0.3, 0.5, 0.7]:
        p_b = alpha * results_inv[best_w_inv]['p_rf'] + (1-alpha) * p_rf_inv_full
        print(f"  blend {alpha:.0%}win+{1-alpha:.0%}full RF = {average_precision_score(y_te_inv, p_b):.4f}")

print("\n" + "="*70)
print("FRAUD")
print("="*70)
sw_fr_full = make_sw(y_tr_fr, spw_fr, sw_train)
pr_rf_fr_full, pr_lgb_fr_full, p_rf_fr_full, _ = run_rf_lgbm(
    X_tr, X_te, y_tr_fr, y_te_fr, spw_fr, sw_fr_full, "Full train")

results_fr = {'full': {'RF': pr_rf_fr_full, 'LGBM': pr_lgb_fr_full}}

for w_mo in [2, 3, 4, 5, 6, 9]:
    wstart  = SPLIT_DATE - pd.DateOffset(months=w_mo)
    win_idx = (train_dates >= wstart)
    n_win   = win_idx.sum()
    y_win   = y_tr_fr[win_idx]
    n_pos   = y_win.sum()
    if n_pos < 10:
        print(f"  {w_mo}mo: skip (pos={n_pos})")
        continue
    X_tr_w  = X_tr[win_idx]
    wdates  = train_dates[win_idx]
    wdays   = (wdates - wdates.min()).total_seconds().values / 86400.0
    wdecay  = np.exp(0.001 * wdays).astype(np.float32)
    wdecay /= wdecay.mean()
    sw_w    = make_sw(y_win, spw_fr, wdecay)

    pr_rf, pr_lgb, p_rf, _ = run_rf_lgbm(
        X_tr_w, X_te, y_win, y_te_fr, spw_fr, sw_w,
        f"{w_mo}mo window (n={n_win:,} pos={n_pos})")
    results_fr[w_mo] = {'RF': pr_rf, 'LGBM': pr_lgb, 'p_rf': p_rf}

    # Show blends
    for alpha in [0.5, 0.7, 0.9]:
        p_b = alpha * p_rf + (1-alpha) * p_rf_fr_full
        blend_pr = average_precision_score(y_te_fr, p_b)
        print(f"    blend {alpha:.0%}win+{1-alpha:.0%}full RF = {blend_pr:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CHECKPOINT WINDOW SWEEP SUMMARY")
print("="*70)
windows_tested = ['full'] + [k for k in results_inv if k != 'full']
print(f"{'Window':<12} {'Inv-RF':>8} {'Inv-LGBM':>10} {'Fr-RF':>8} {'Fr-LGBM':>10}")
print("-"*50)
for w in windows_tested:
    ri = results_inv.get(w, {})
    rf = results_fr.get(w, {})
    wlabel = 'Full' if w == 'full' else f'{w}mo'
    print(f"  {wlabel:<10} {ri.get('RF', 0):>8.4f} {ri.get('LGBM', 0):>10.4f}"
          f" {rf.get('RF', 0):>8.4f} {rf.get('LGBM', 0):>10.4f}")

print("\nDone.")
