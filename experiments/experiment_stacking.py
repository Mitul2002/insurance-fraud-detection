"""
experiment_stacking.py
========================
Idea #6: OOF Stacking / Meta-Learner.

Instead of simple probability averaging for the ensemble, train a
LogisticRegression meta-model on out-of-fold (OOF) predictions from
all base models. The meta-learner can learn optimal per-sample weights.

Base models (using checkpoint features):
  1. RF full train
  2. LGBM full train
  3. Window RF (5mo) — best Investigation
  4. Window LGBM (3mo) — best Fraud window

Method:
  - 5-fold time-sorted CV on training data
  - For each fold: train base models on train folds, predict on val fold
  - Collect OOF predictions (one column per base model per target)
  - Train LogisticRegression / RF meta-model on OOF predictions
  - Final prediction: meta-model on test predictions from models trained on ALL train data
  - Compare vs simple probability averaging of top-3

Note: meta-learner uses only model outputs (not raw features) → fast + regularized
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

CKPT_PATH  = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"
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
print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w.astype(np.float32)

def make_rf(spw):
    return RFC(n_estimators=400, max_depth=None, min_samples_leaf=5,
               class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)

def make_lgbm(spw):
    return LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)

def window_decay(dates):
    days = (dates - dates.min()).total_seconds().values / 86400.0
    w = np.exp(0.001 * days).astype(np.float32); w /= w.mean()
    return w

# ── Time-sorted 5-fold CV ─────────────────────────────────────────────────────
N = len(X_tr)
n_folds = 5
fold_size = N // n_folds

# Sort by date for temporal folds
date_order = np.argsort(train_dates)
folds = []
for k in range(n_folds):
    val_start = k * fold_size
    val_end   = (k + 1) * fold_size if k < n_folds - 1 else N
    val_idx   = date_order[val_start:val_end]
    tr_idx    = np.concatenate([date_order[:val_start], date_order[val_end:]])
    folds.append((tr_idx, val_idx))

print(f"  5-fold time-sorted CV: fold sizes ~{fold_size:,}")

# ── OOF collection ─────────────────────────────────────────────────────────────
# Models per fold:
#   M0: RF full
#   M1: LGBM full
#   M2: Window RF (5mo) — uses window from fold's train portion
#   M3: Window LGBM (3mo)

MODEL_NAMES = ['RF_full', 'LGBM_full', 'RF_win5', 'LGBM_win3']
N_MODELS    = len(MODEL_NAMES)

oof_inv = np.zeros((N, N_MODELS), dtype=np.float32)
oof_fr  = np.zeros((N, N_MODELS), dtype=np.float32)

print(f"\nCollecting OOF predictions ({n_folds} folds × {N_MODELS} models)...")

for fold_k, (tr_idx, val_idx) in enumerate(folds):
    t_fold = time.time()
    X_fold_tr  = X_tr[tr_idx];   X_fold_val = X_tr[val_idx]
    y_fold_inv = y_tr_inv[tr_idx]; y_val_inv = y_tr_inv[val_idx]
    y_fold_fr  = y_tr_fr[tr_idx];  y_val_fr  = y_tr_fr[val_idx]
    sw_fold    = sw_train[tr_idx]
    dates_fold = train_dates[tr_idx]

    sw_inv_f = make_sw(y_fold_inv, spw_inv, sw_fold)
    sw_fr_f  = make_sw(y_fold_fr,  spw_fr,  sw_fold)

    # Window indices within fold's training portion
    fold_split_date = train_dates[val_idx].min()  # start of val = cutoff for window
    w5_start = fold_split_date - pd.DateOffset(months=5)
    w3_start = fold_split_date - pd.DateOffset(months=3)
    win5_idx = (dates_fold >= w5_start)
    win3_idx = (dates_fold >= w3_start)

    # M0: RF full
    m = make_rf(spw_inv); m.fit(X_fold_tr, y_fold_inv, sample_weight=sw_inv_f)
    oof_inv[val_idx, 0] = m.predict_proba(X_fold_val)[:, 1]
    m = make_rf(spw_fr);  m.fit(X_fold_tr, y_fold_fr,  sample_weight=sw_fr_f)
    oof_fr[val_idx, 0]  = m.predict_proba(X_fold_val)[:, 1]

    # M1: LGBM full
    m = make_lgbm(spw_inv); m.fit(X_fold_tr, y_fold_inv, sample_weight=sw_inv_f)
    oof_inv[val_idx, 1] = m.predict_proba(X_fold_val)[:, 1]
    m = make_lgbm(spw_fr);  m.fit(X_fold_tr, y_fold_fr,  sample_weight=sw_fr_f)
    oof_fr[val_idx, 1]  = m.predict_proba(X_fold_val)[:, 1]

    # M2: Window RF (5mo)
    if win5_idx.sum() > 100:
        Xw5 = X_fold_tr[win5_idx]; yw5_inv = y_fold_inv[win5_idx]; yw5_fr = y_fold_fr[win5_idx]
        wd5 = window_decay(dates_fold[win5_idx])
        m = make_rf(spw_inv); m.fit(Xw5, yw5_inv, sample_weight=make_sw(yw5_inv, spw_inv, wd5))
        oof_inv[val_idx, 2] = m.predict_proba(X_fold_val)[:, 1]
        m = make_rf(spw_fr);  m.fit(Xw5, yw5_fr,  sample_weight=make_sw(yw5_fr,  spw_fr,  wd5))
        oof_fr[val_idx, 2]  = m.predict_proba(X_fold_val)[:, 1]
    else:
        oof_inv[val_idx, 2] = oof_inv[val_idx, 0]  # fallback to RF full
        oof_fr[val_idx, 2]  = oof_fr[val_idx, 0]

    # M3: Window LGBM (3mo)
    if win3_idx.sum() > 50:
        Xw3 = X_fold_tr[win3_idx]; yw3_inv = y_fold_inv[win3_idx]; yw3_fr = y_fold_fr[win3_idx]
        wd3 = window_decay(dates_fold[win3_idx])
        m = make_lgbm(spw_inv); m.fit(Xw3, yw3_inv, sample_weight=make_sw(yw3_inv, spw_inv, wd3))
        oof_inv[val_idx, 3] = m.predict_proba(X_fold_val)[:, 1]
        m = make_lgbm(spw_fr);  m.fit(Xw3, yw3_fr,  sample_weight=make_sw(yw3_fr,  spw_fr,  wd3))
        oof_fr[val_idx, 3]  = m.predict_proba(X_fold_val)[:, 1]
    else:
        oof_inv[val_idx, 3] = oof_inv[val_idx, 1]
        oof_fr[val_idx, 3]  = oof_fr[val_idx, 1]

    pr_oof_inv = [average_precision_score(y_val_inv, oof_inv[val_idx, m]) for m in range(N_MODELS)]
    pr_oof_fr  = [average_precision_score(y_val_fr,  oof_fr[val_idx,  m]) for m in range(N_MODELS)]
    print(f"  Fold {fold_k+1}: INV=[{' '.join(f'{p:.3f}' for p in pr_oof_inv)}]  "
          f"FR=[{' '.join(f'{p:.3f}' for p in pr_oof_fr)}]  ({time.time()-t_fold:.0f}s)")

# OOF PR-AUC for each model
print("\n  OOF PR-AUC (full training set OOF):")
for m_i, name in enumerate(MODEL_NAMES):
    pr_inv = average_precision_score(y_tr_inv, oof_inv[:, m_i])
    pr_fr  = average_precision_score(y_tr_fr,  oof_fr[:, m_i])
    print(f"    {name:<15}: INV={pr_inv:.4f}  FR={pr_fr:.4f}")

# ── Train final base models on full training data ──────────────────────────────
print("\nTraining final models on full training data...")
sw_full_inv = make_sw(y_tr_inv, spw_inv, sw_train)
sw_full_fr  = make_sw(y_tr_fr,  spw_fr,  sw_train)

SPLIT_DATE_TS = SPLIT_DATE
w5_start_full = SPLIT_DATE_TS - pd.DateOffset(months=5)
w3_start_full = SPLIT_DATE_TS - pd.DateOffset(months=3)
idx5 = (train_dates >= w5_start_full)
idx3 = (train_dates >= w3_start_full)
wd5_full = window_decay(train_dates[idx5])
wd3_full = window_decay(train_dates[idx3])

test_preds_inv = np.zeros((len(X_te), N_MODELS), dtype=np.float32)
test_preds_fr  = np.zeros((len(X_te), N_MODELS), dtype=np.float32)
ind_pr_inv = []
ind_pr_fr  = []

# M0 RF full
t = time.time()
m = make_rf(spw_inv); m.fit(X_tr, y_tr_inv, sample_weight=sw_full_inv)
test_preds_inv[:, 0] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_inv, test_preds_inv[:, 0])
ind_pr_inv.append(pr); print(f"  RF_full  INV={pr:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = make_rf(spw_fr);  m.fit(X_tr, y_tr_fr,  sample_weight=sw_full_fr)
test_preds_fr[:, 0] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_fr, test_preds_fr[:, 0])
ind_pr_fr.append(pr);  print(f"  RF_full  FR={pr:.4f}  ({time.time()-t:.1f}s)")

# M1 LGBM full
t = time.time()
m = make_lgbm(spw_inv); m.fit(X_tr, y_tr_inv, sample_weight=sw_full_inv)
test_preds_inv[:, 1] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_inv, test_preds_inv[:, 1])
ind_pr_inv.append(pr); print(f"  LGBM_full INV={pr:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = make_lgbm(spw_fr);  m.fit(X_tr, y_tr_fr,  sample_weight=sw_full_fr)
test_preds_fr[:, 1] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_fr, test_preds_fr[:, 1])
ind_pr_fr.append(pr);  print(f"  LGBM_full FR={pr:.4f}  ({time.time()-t:.1f}s)")

# M2 Window RF 5mo
t = time.time()
m = make_rf(spw_inv); m.fit(X_tr[idx5], y_tr_inv[idx5], sample_weight=make_sw(y_tr_inv[idx5], spw_inv, wd5_full))
test_preds_inv[:, 2] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_inv, test_preds_inv[:, 2])
ind_pr_inv.append(pr); print(f"  RF_win5   INV={pr:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = make_rf(spw_fr);  m.fit(X_tr[idx5], y_tr_fr[idx5],  sample_weight=make_sw(y_tr_fr[idx5],  spw_fr,  wd5_full))
test_preds_fr[:, 2] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_fr, test_preds_fr[:, 2])
ind_pr_fr.append(pr);  print(f"  RF_win5   FR={pr:.4f}  ({time.time()-t:.1f}s)")

# M3 Window LGBM 3mo
t = time.time()
m = make_lgbm(spw_inv); m.fit(X_tr[idx3], y_tr_inv[idx3], sample_weight=make_sw(y_tr_inv[idx3], spw_inv, wd3_full))
test_preds_inv[:, 3] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_inv, test_preds_inv[:, 3])
ind_pr_inv.append(pr); print(f"  LGBM_win3 INV={pr:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = make_lgbm(spw_fr);  m.fit(X_tr[idx3], y_tr_fr[idx3],  sample_weight=make_sw(y_tr_fr[idx3],  spw_fr,  wd3_full))
test_preds_fr[:, 3] = m.predict_proba(X_te)[:, 1]
pr = average_precision_score(y_te_fr, test_preds_fr[:, 3])
ind_pr_fr.append(pr);  print(f"  LGBM_win3 FR={pr:.4f}  ({time.time()-t:.1f}s)")

# ── Meta-learner ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("META-LEARNER")
print("="*70)

# Simple average (baseline for comparison)
avg_inv = test_preds_inv.mean(axis=1)
avg_fr  = test_preds_fr.mean(axis=1)
pr_avg_inv = average_precision_score(y_te_inv, avg_inv)
pr_avg_fr  = average_precision_score(y_te_fr,  avg_fr)
print(f"\n  Simple average (all 4):   INV={pr_avg_inv:.4f}  FR={pr_avg_fr:.4f}")

# Top-3 average
top3_inv = sorted(range(N_MODELS), key=lambda i: ind_pr_inv[i], reverse=True)[:3]
top3_fr  = sorted(range(N_MODELS), key=lambda i: ind_pr_fr[i],  reverse=True)[:3]
pr_top3_inv = average_precision_score(y_te_inv, test_preds_inv[:, top3_inv].mean(axis=1))
pr_top3_fr  = average_precision_score(y_te_fr,  test_preds_fr[:,  top3_fr].mean(axis=1))
print(f"  Top-3 average:            INV={pr_top3_inv:.4f}  FR={pr_top3_fr:.4f}")
print(f"    Top-3 models INV: {[MODEL_NAMES[i] for i in top3_inv]}")
print(f"    Top-3 models FR:  {[MODEL_NAMES[i] for i in top3_fr]}")

# LogisticRegression meta-learner
for C in [0.01, 0.1, 1.0]:
    meta_inv = LogisticRegression(C=C, random_state=42, max_iter=1000)
    meta_inv.fit(oof_inv, y_tr_inv)
    pr_meta_inv = average_precision_score(y_te_inv, meta_inv.predict_proba(test_preds_inv)[:, 1])

    meta_fr = LogisticRegression(C=C, random_state=42, max_iter=1000)
    meta_fr.fit(oof_fr, y_tr_fr)
    pr_meta_fr = average_precision_score(y_te_fr, meta_fr.predict_proba(test_preds_fr)[:, 1])

    print(f"  LogReg meta (C={C:<5}):   INV={pr_meta_inv:.4f}  FR={pr_meta_fr:.4f}")
    print(f"    INV weights: {dict(zip(MODEL_NAMES, meta_inv.coef_[0].round(3)))}")
    print(f"    FR  weights: {dict(zip(MODEL_NAMES, meta_fr.coef_[0].round(3)))}")

# RF meta-learner (non-linear combination)
meta_rf_inv = RFC(n_estimators=200, max_depth=3, random_state=42, n_jobs=-1)
meta_rf_inv.fit(oof_inv, y_tr_inv)
pr_rfmeta_inv = average_precision_score(y_te_inv, meta_rf_inv.predict_proba(test_preds_inv)[:, 1])

meta_rf_fr = RFC(n_estimators=200, max_depth=3, random_state=42, n_jobs=-1)
meta_rf_fr.fit(oof_fr, y_tr_fr)
pr_rfmeta_fr = average_precision_score(y_te_fr, meta_rf_fr.predict_proba(test_preds_fr)[:, 1])
print(f"  RF meta (max_depth=3):    INV={pr_rfmeta_inv:.4f}  FR={pr_rfmeta_fr:.4f}")

print("\nDone.")
