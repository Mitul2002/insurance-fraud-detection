"""
experiment_calibration.py
==========================
Test Platt scaling calibration before ensemble blending (PR-AUC focused).

Uses 5-fold time-sorted cross-validation to get OOF predictions,
fits Platt scaler on OOF predictions, applies calibrated probabilities
when blending — and compares to raw probability blending.

Key question: does calibration improve PR-AUC for our 1:14 / 1:56 imbalanced targets?
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

CKPT_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
SPLIT_DATE = pd.Timestamp('2025-07-01')
N_FOLDS    = 5

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
print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
print(f"  Inv  train pos={y_tr_inv.sum():,}  Fraud train pos={y_tr_fr.sum():,}")

# 5mo window
_wstart  = SPLIT_DATE - pd.DateOffset(months=5)
_win_idx = (train_dates >= _wstart)
X_w5     = X_tr[_win_idx]
y_w5_inv = y_tr_inv[_win_idx]
y_w5_fr  = y_tr_fr[_win_idx]
_wdates  = train_dates[_win_idx]
_wdays   = (_wdates - _wdates.min()).total_seconds().values / 86400.0
_wdecay  = np.exp(0.001 * _wdays).astype(np.float32); _wdecay /= _wdecay.mean()
print(f"  5mo window: n={_win_idx.sum():,}")

def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w

def score(y_true, p):
    return average_precision_score(y_true, p)

# ── Time-sorted 5-fold CV for OOF probabilities ───────────────────────────────
def get_oof_probs(X, y, sw, spw, model_type='rf', folds=N_FOLDS):
    """
    Time-sorted k-fold: fold 0 = first 1/k of data, ..., fold k-1 = last 1/k.
    OOF predictions collected for calibration.
    Test predictions averaged across folds.
    """
    n = len(X)
    fold_size = n // folds
    oof = np.zeros(n, dtype=np.float64)
    test_preds = []

    for f in range(folds):
        val_start = f * fold_size
        val_end   = (f + 1) * fold_size if f < folds - 1 else n
        # train = all except val fold
        train_idx = np.concatenate([np.arange(0, val_start),
                                    np.arange(val_end, n)])
        val_idx   = np.arange(val_start, val_end)
        Xf, yf, swf = X[train_idx], y[train_idx], sw[train_idx]
        Xv, yv       = X[val_idx],   y[val_idx]

        if model_type == 'rf':
            m = RFC(n_estimators=200, max_depth=None, min_samples_leaf=5,
                    class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)
        else:
            m = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xf, yf, sample_weight=swf)
        oof[val_idx] = m.predict_proba(Xv)[:, 1]

    # Final model trained on full data for test predictions
    if model_type == 'rf':
        m_final = RFC(n_estimators=400, max_depth=None, min_samples_leaf=5,
                      class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)
    else:
        m_final = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                                  scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    m_final.fit(X, y, sample_weight=sw)
    p_te = m_final.predict_proba(X_te)[:, 1]
    return oof, p_te

def platt_cal(oof, y_oof, p_te):
    lr = LogisticRegression(C=1.0)
    lr.fit(oof.reshape(-1, 1), y_oof)
    return lr.predict_proba(p_te.reshape(-1, 1))[:, 1]

def isotonic_cal(oof, y_oof, p_te):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(oof, y_oof)
    return ir.predict(p_te)

def blend(probs, label, y_te):
    p = np.mean(probs, axis=0)
    print(f"  {label:<55} {score(y_te, p):.4f}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INVESTIGATION — Calibration experiment")
print("="*70)

sw_inv_full = make_sw(y_tr_inv, spw_inv, sw_train)
sw_inv_w5   = make_sw(y_w5_inv, spw_inv, _wdecay)

# Get OOF + final test predictions for each model type
print("\n  Getting RF OOF predictions (5-fold time-sorted)...")
t = time.time()
oof_rf_inv, p_rf_te_inv = get_oof_probs(X_tr, y_tr_inv, sw_inv_full, spw_inv, 'rf')
pr_rf = score(y_te_inv, p_rf_te_inv)
print(f"    Full RF  raw PR-AUC={pr_rf:.4f}  OOF={score(y_tr_inv, oof_rf_inv):.4f}  ({time.time()-t:.1f}s)")

print("  Getting LGBM OOF predictions...")
t = time.time()
oof_lgb_inv, p_lgb_te_inv = get_oof_probs(X_tr, y_tr_inv, sw_inv_full, spw_inv, 'lgbm')
pr_lgb = score(y_te_inv, p_lgb_te_inv)
print(f"    Full LGBM raw PR-AUC={pr_lgb:.4f}  OOF={score(y_tr_inv, oof_lgb_inv):.4f}  ({time.time()-t:.1f}s)")

print("  Getting window RF OOF predictions (5mo window)...")
t = time.time()
oof_wrf_inv, p_wrf_te_inv = get_oof_probs(X_w5, y_w5_inv, sw_inv_w5, spw_inv, 'rf')
pr_wrf = score(y_te_inv, p_wrf_te_inv)
print(f"    Win RF   raw PR-AUC={pr_wrf:.4f}  OOF={score(y_w5_inv, oof_wrf_inv):.4f}  ({time.time()-t:.1f}s)")

# Calibrate each
p_rf_pl  = platt_cal(oof_rf_inv,  y_tr_inv,  p_rf_te_inv)
p_rf_ir  = isotonic_cal(oof_rf_inv, y_tr_inv, p_rf_te_inv)
p_lgb_pl = platt_cal(oof_lgb_inv, y_tr_inv,  p_lgb_te_inv)
p_lgb_ir = isotonic_cal(oof_lgb_inv, y_tr_inv, p_lgb_te_inv)
p_wrf_pl = platt_cal(oof_wrf_inv, y_w5_inv,  p_wrf_te_inv)
p_wrf_ir = isotonic_cal(oof_wrf_inv, y_w5_inv, p_wrf_te_inv)

print(f"\n  Calibration effects on single models:")
print(f"  {'Model':<25} {'raw':>8} {'platt':>8} {'isotonic':>10}")
print(f"  {'-'*53}")
for name, raw, pl, ir in [
    ('Full RF',    pr_rf,  score(y_te_inv, p_rf_pl),  score(y_te_inv, p_rf_ir)),
    ('Full LGBM',  pr_lgb, score(y_te_inv, p_lgb_pl), score(y_te_inv, p_lgb_ir)),
    ('Window RF',  pr_wrf, score(y_te_inv, p_wrf_pl), score(y_te_inv, p_wrf_ir)),
]:
    print(f"  {name:<25} {raw:>8.4f} {pl:>8.4f} {ir:>10.4f}")

print(f"\n  Ensemble blending (3 models): raw vs calibrated")
blend([p_rf_te_inv, p_lgb_te_inv, p_wrf_te_inv], "Raw blend (RF+LGBM+WinRF)",  y_te_inv)
blend([p_rf_pl,     p_lgb_pl,     p_wrf_pl],     "Platt blend (RF+LGBM+WinRF)", y_te_inv)
blend([p_rf_ir,     p_lgb_ir,     p_wrf_ir],     "Isotonic blend (RF+LGBM+WinRF)", y_te_inv)

# Pairwise blends
blend([p_rf_te_inv, p_wrf_te_inv], "Raw blend (RF+WinRF)",      y_te_inv)
blend([p_rf_pl,     p_wrf_pl],     "Platt blend (RF+WinRF)",    y_te_inv)
blend([p_rf_te_inv, p_lgb_te_inv], "Raw blend (RF+LGBM)",       y_te_inv)
blend([p_rf_pl,     p_lgb_pl],     "Platt blend (RF+LGBM)",     y_te_inv)


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRAUD — Calibration experiment")
print("="*70)

sw_fr_full = make_sw(y_tr_fr, spw_fr, sw_train)

print("\n  Getting RF OOF predictions...")
t = time.time()
oof_rf_fr, p_rf_te_fr = get_oof_probs(X_tr, y_tr_fr, sw_fr_full, spw_fr, 'rf')
pr_rf_fr = score(y_te_fr, p_rf_te_fr)
print(f"    Full RF  raw PR-AUC={pr_rf_fr:.4f}  ({time.time()-t:.1f}s)")

print("  Getting LGBM OOF predictions...")
t = time.time()
oof_lgb_fr, p_lgb_te_fr = get_oof_probs(X_tr, y_tr_fr, sw_fr_full, spw_fr, 'lgbm')
pr_lgb_fr = score(y_te_fr, p_lgb_te_fr)
print(f"    Full LGBM raw PR-AUC={pr_lgb_fr:.4f}  ({time.time()-t:.1f}s)")

p_rf_pl_fr  = platt_cal(oof_rf_fr,  y_tr_fr, p_rf_te_fr)
p_rf_ir_fr  = isotonic_cal(oof_rf_fr, y_tr_fr, p_rf_te_fr)
p_lgb_pl_fr = platt_cal(oof_lgb_fr, y_tr_fr, p_lgb_te_fr)
p_lgb_ir_fr = isotonic_cal(oof_lgb_fr, y_tr_fr, p_lgb_te_fr)

print(f"\n  Calibration effects:")
print(f"  {'Model':<25} {'raw':>8} {'platt':>8} {'isotonic':>10}")
print(f"  {'-'*53}")
for name, raw, pl, ir in [
    ('Full RF',   pr_rf_fr,  score(y_te_fr, p_rf_pl_fr),  score(y_te_fr, p_rf_ir_fr)),
    ('Full LGBM', pr_lgb_fr, score(y_te_fr, p_lgb_pl_fr), score(y_te_fr, p_lgb_ir_fr)),
]:
    print(f"  {name:<25} {raw:>8.4f} {pl:>8.4f} {ir:>10.4f}")

print(f"\n  Ensemble blending:")
blend([p_rf_te_fr, p_lgb_te_fr], "Raw blend (RF+LGBM)",      y_te_fr)
blend([p_rf_pl_fr, p_lgb_pl_fr], "Platt blend (RF+LGBM)",    y_te_fr)
blend([p_rf_ir_fr, p_lgb_ir_fr], "Isotonic blend (RF+LGBM)", y_te_fr)

print("\nDone.")
