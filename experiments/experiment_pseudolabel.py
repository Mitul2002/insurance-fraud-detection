"""
experiment_pseudolabel.py
=========================
Pseudo-labeling on test set to address temporal shift (Jul-Dec 2025 vs pre-Jul 2025).

Strategy:
  1. Train initial model on training data
  2. Apply to X_te — high-confidence predictions become pseudo-labels
  3. Retrain with X_tr + pseudo-labeled X_te
  4. Compare against baseline

Key insight: pseudo-labeling mainly adds LEGIT samples from the test distribution,
forcing the model to learn what "normal" looks like in the test period.
This directly targets the adversarial AUC=0.9961 shift problem.

Thresholds tested: 0.97/0.02  0.90/0.05  0.80/0.10
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
print(f"  Inv  train pos={y_tr_inv.sum():,}  test pos={y_te_inv.sum():,}")
print(f"  Fraud train pos={y_tr_fr.sum():,}  test pos={y_te_fr.sum():,}")

# ── 5mo window (best for Investigation) ───────────────────────────────────────
_wstart  = SPLIT_DATE - pd.DateOffset(months=5)
_win_idx = (train_dates >= _wstart)
X_w5     = X_tr[_win_idx]
y_w5_inv = y_tr_inv[_win_idx]
y_w5_fr  = y_tr_fr[_win_idx]
_wdates  = train_dates[_win_idx]
_wdays   = (_wdates - _wdates.min()).total_seconds().values / 86400.0
_wdecay  = np.exp(0.001 * _wdays).astype(np.float32)
_wdecay /= _wdecay.mean()

def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w

sw_inv_full = make_sw(y_tr_inv, spw_inv, sw_train)
sw_fr_full  = make_sw(y_tr_fr,  spw_fr,  sw_train)
sw_inv_w5   = make_sw(y_w5_inv, spw_inv, _wdecay)
sw_fr_w5    = make_sw(y_w5_fr,  spw_fr,  _wdecay)

# ── Model factory ─────────────────────────────────────────────────────────────
def make_rf(spw):
    return RFC(n_estimators=400, max_depth=None, min_samples_leaf=5,
               class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)

def make_lgbm(spw):
    return LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)

def score(y_true, p): return average_precision_score(y_true, p)

# ── Pseudo-labeling engine ─────────────────────────────────────────────────────
def pseudo_label_experiment(
    X_train, y_train, sw, X_te, y_te, spw,
    label, fraud_thresh=0.90, legit_thresh=0.05, n_iter=2,
    use_lgbm=False
):
    """
    Iterative pseudo-labeling.
    Returns final PR-AUC after retraining with pseudo-labeled test samples.
    """
    t = time.time()
    X_cur, y_cur, sw_cur = X_train.copy(), y_train.copy(), sw.copy()

    p_history = []
    for it in range(n_iter + 1):  # +1 for final eval without adding more labels
        m = make_lgbm(spw) if use_lgbm else make_rf(spw)
        m.fit(X_cur, y_cur, sample_weight=sw_cur)
        p_te = m.predict_proba(X_te)[:, 1]
        pr   = score(y_te, p_te)
        p_history.append(pr)

        if it < n_iter:
            fraud_mask = p_te >= fraud_thresh
            legit_mask = p_te <= legit_thresh
            n_pf = fraud_mask.sum()
            n_pl = legit_mask.sum()
            if n_pf + n_pl == 0:
                break
            # Build pseudo-labeled batch
            masks = fraud_mask | legit_mask
            y_pseudo = np.where(fraud_mask[masks], 1, 0).astype(int)
            # Weight pseudo-labeled samples: normal weight (no time-decay for test period)
            sw_pseudo = np.where(y_pseudo == 1, spw, 1.0).astype(np.float32)
            X_cur  = np.vstack([X_train, X_te[masks]])
            y_cur  = np.concatenate([y_train, y_pseudo])
            sw_cur = np.concatenate([sw, sw_pseudo])
            print(f"    iter {it}: +pseudo_fraud={n_pf} +pseudo_legit={n_pl} "
                  f"| train_size={len(X_cur):,} | PR={pr:.4f}")

    pr_final = p_history[-1]
    model_type = "LGBM" if use_lgbm else "RF"
    print(f"  {label:<55} {model_type} PL={pr_final:.4f}  "
          f"({time.time()-t:.1f}s)")
    return p_te, pr_final


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INVESTIGATION — Pseudo-labeling sweep")
print("="*70)

# Baselines (no pseudo-labeling)
t = time.time()
rf_base_inv = make_rf(spw_inv)
rf_base_inv.fit(X_w5, y_w5_inv, sample_weight=sw_inv_w5)
p_b_inv = rf_base_inv.predict_proba(X_te)[:, 1]
pr_b_inv = score(y_te_inv, p_b_inv)
print(f"  {'Baseline: 5mo window RF':<55} RF  base={pr_b_inv:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
rf_full_inv = make_rf(spw_inv)
rf_full_inv.fit(X_tr, y_tr_inv, sample_weight=sw_inv_full)
p_bfull_inv = rf_full_inv.predict_proba(X_te)[:, 1]
pr_bfull_inv = score(y_te_inv, p_bfull_inv)
print(f"  {'Baseline: full train RF':<55} RF  base={pr_bfull_inv:.4f}  ({time.time()-t:.1f}s)")

results_inv = {'baseline_win5': pr_b_inv, 'baseline_full': pr_bfull_inv}

# Pseudo-labeling sweeps
print("\n  -- Starting from 5mo window --")
for fr_t, lg_t in [(0.97, 0.02), (0.90, 0.05), (0.80, 0.10)]:
    tag = f"5mo-win PL fr>={fr_t} lg<={lg_t}"
    _, pr = pseudo_label_experiment(X_w5, y_w5_inv, sw_inv_w5, X_te, y_te_inv,
                                     spw_inv, tag, fr_t, lg_t, n_iter=2)
    results_inv[f'win5_pl_{fr_t}'] = pr

print("\n  -- Starting from full train --")
for fr_t, lg_t in [(0.97, 0.02), (0.90, 0.05), (0.80, 0.10)]:
    tag = f"full-train PL fr>={fr_t} lg<={lg_t}"
    _, pr = pseudo_label_experiment(X_tr, y_tr_inv, sw_inv_full, X_te, y_te_inv,
                                     spw_inv, tag, fr_t, lg_t, n_iter=2)
    results_inv[f'full_pl_{fr_t}'] = pr

# LGBM pseudo-labeling (best configs)
print("\n  -- LGBM pseudo-labeling --")
for fr_t, lg_t in [(0.90, 0.05), (0.80, 0.10)]:
    tag = f"LGBM full-train PL fr>={fr_t} lg<={lg_t}"
    _, pr = pseudo_label_experiment(X_tr, y_tr_inv, sw_inv_full, X_te, y_te_inv,
                                     spw_inv, tag, fr_t, lg_t, n_iter=2, use_lgbm=True)
    results_inv[f'lgbm_full_pl_{fr_t}'] = pr


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRAUD — Pseudo-labeling sweep")
print("="*70)

t = time.time()
rf_base_fr = make_rf(spw_fr)
rf_base_fr.fit(X_tr, y_tr_fr, sample_weight=sw_fr_full)
p_b_fr = rf_base_fr.predict_proba(X_te)[:, 1]
pr_b_fr = score(y_te_fr, p_b_fr)
print(f"  {'Baseline: full train RF':<55} RF  base={pr_b_fr:.4f}  ({time.time()-t:.1f}s)")

results_fr = {'baseline_full': pr_b_fr}

print("\n  -- RF pseudo-labeling --")
for fr_t, lg_t in [(0.97, 0.02), (0.90, 0.05), (0.80, 0.10)]:
    tag = f"full-train PL fr>={fr_t} lg<={lg_t}"
    _, pr = pseudo_label_experiment(X_tr, y_tr_fr, sw_fr_full, X_te, y_te_fr,
                                     spw_fr, tag, fr_t, lg_t, n_iter=2)
    results_fr[f'rf_pl_{fr_t}'] = pr

print("\n  -- LGBM pseudo-labeling --")
for fr_t, lg_t in [(0.90, 0.05), (0.80, 0.10)]:
    tag = f"LGBM full-train PL fr>={fr_t} lg<={lg_t}"
    _, pr = pseudo_label_experiment(X_tr, y_tr_fr, sw_fr_full, X_te, y_te_fr,
                                     spw_fr, tag, fr_t, lg_t, n_iter=2, use_lgbm=True)
    results_fr[f'lgbm_pl_{fr_t}'] = pr


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PSEUDO-LABELING SUMMARY")
print("="*70)
print(f"\nInvestigation baseline (win5 RF): {results_inv['baseline_win5']:.4f}")
print(f"Investigation baseline (full RF): {results_inv['baseline_full']:.4f}")
best_inv = max(results_inv, key=results_inv.get)
print(f"Investigation BEST: {best_inv} = {results_inv[best_inv]:.4f}")

print(f"\nFraud baseline (full RF): {results_fr['baseline_full']:.4f}")
best_fr = max(results_fr, key=results_fr.get)
print(f"Fraud BEST: {best_fr} = {results_fr[best_fr]:.4f}")

print("\nDone.")
