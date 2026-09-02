"""
experiment_lgbm_tuning.py
==========================
Ideas #4 + #5: LGBM Hyperparameter Tuning + Monotone Constraints.

Targets:
  - Fraud 3mo window LGBM (current best: 0.0591 experiment / 0.0582 notebook)
  - Investigation 5mo window LGBM (current: 0.1012 experiment / ~0.17 notebook)

#4 Hyperparameter sweep (Optuna if available, else grid):
  num_leaves:      [31, 63, 127]
  min_child_samples: [10, 30, 50]
  reg_lambda:      [1, 2, 5, 10]
  colsample_bytree: [0.6, 0.8, 1.0]
  n_estimators:    [600, 1000]

#5 Monotone constraints:
  Force fraud_rate / inv_rate features → positive monotone
  (higher entity fraud rate should increase predicted probability)
  Force velocity features → positive monotone
  This prevents learning spurious non-monotone correlations that don't
  generalize across time periods.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
import warnings, time, itertools
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
feat_names  = list(ck['feat_names'])
print(f"  X_tr={X_tr.shape}  spw_inv={spw_inv:.1f}  spw_fr={spw_fr:.1f}")

# ── Window helpers ─────────────────────────────────────────────────────────────
def get_window(months):
    wstart  = SPLIT_DATE - pd.DateOffset(months=months)
    idx     = (train_dates >= wstart)
    Xw      = X_tr[idx]
    wdates  = train_dates[idx]
    wdays   = (wdates - wdates.min()).total_seconds().values / 86400.0
    wdecay  = np.exp(0.001 * wdays).astype(np.float32); wdecay /= wdecay.mean()
    return Xw, idx, wdecay

def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w.astype(np.float32)

Xw5, idx5, wd5 = get_window(5)
Xw3, idx3, wd3 = get_window(3)

sw_full_inv = make_sw(y_tr_inv, spw_inv, sw_train)
sw_full_fr  = make_sw(y_tr_fr,  spw_fr,  sw_train)
sw_w5_inv   = make_sw(y_tr_inv[idx5], spw_inv, wd5)
sw_w3_fr    = make_sw(y_tr_fr[idx3],  spw_fr,  wd3)

# ── #5 Monotone constraints ────────────────────────────────────────────────────
# +1 = monotone increasing, -1 = monotone decreasing, 0 = unconstrained
# Features where higher value = higher fraud risk → constrain to +1
POSITIVE_KEYWORDS = [
    'fraud_rate', 'inv_rate', 'vel', 'velocity', 'high_claim',
    'risk_inc_flag', 'anomal', 'isoforest', 'dae_recon',
    'addon', 'has_TTD', 'has_hosp_cash', 'TTD_x',
    'short_stay', 'weekend_adm', 'repeat_patient',
    'hosp_high_claim', 'exclusivity', 'hhi',
]
NEGATIVE_KEYWORDS = []  # features where higher = lower risk (none clear)

mono_constraints = []
for f in feat_names:
    fl = f.lower()
    if any(k in fl for k in POSITIVE_KEYWORDS):
        mono_constraints.append(1)
    elif any(k in fl for k in NEGATIVE_KEYWORDS):
        mono_constraints.append(-1)
    else:
        mono_constraints.append(0)

n_constrained = sum(1 for c in mono_constraints if c != 0)
print(f"\nMonotone constraints: {n_constrained}/{len(feat_names)} features constrained")
constrained_feats = [f for f, c in zip(feat_names, mono_constraints) if c != 0]
print(f"  Constrained: {constrained_feats[:15]}{'...' if len(constrained_feats)>15 else ''}")

# ── Baselines ──────────────────────────────────────────────────────────────────
def score(y_t, y_e, model, Xe):
    return average_precision_score(y_e, model.predict_proba(Xe)[:, 1])

def baseline_lgbm(spw, X_t, y_t, sw, X_e, y_e, label):
    t = time.time()
    m = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                       scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_t, y_t, sample_weight=sw)
    pr = score(y_t, y_e, m, X_e)
    print(f"  {label:<60} PR={pr:.4f}  ({time.time()-t:.1f}s)")
    return pr

print("\n" + "="*70)
print("BASELINES")
print("="*70)
base_inv = baseline_lgbm(spw_inv, Xw5, y_tr_inv[idx5], sw_w5_inv, X_te, y_te_inv,
                          "INV  5mo window LGBM (default params)")
base_fr  = baseline_lgbm(spw_fr,  Xw3, y_tr_fr[idx3],  sw_w3_fr,  X_te, y_te_fr,
                          "FRAUD 3mo window LGBM (default params)")

# ── #4 Hyperparameter grid search ─────────────────────────────────────────────
print("\n" + "="*70)
print("HYPERPARAMETER SWEEP")
print("="*70)

param_grid = {
    'num_leaves':         [31, 63, 127],
    'min_child_samples':  [10, 30, 50],
    'reg_lambda':         [1, 5, 10],
    'colsample_bytree':   [0.6, 0.8],
    'n_estimators':       [800],
    'learning_rate':      [0.05],
}

keys   = list(param_grid.keys())
values = list(param_grid.values())
combos = list(itertools.product(*values))
print(f"Grid size: {len(combos)} combinations\n")

best_inv = {'pr': base_inv, 'params': 'default'}
best_fr  = {'pr': base_fr,  'params': 'default'}

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    t = time.time()

    # INV 5mo window
    m_inv = LGBMClassifier(**params, scale_pos_weight=spw_inv,
                            subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_inv.fit(Xw5, y_tr_inv[idx5], sample_weight=sw_w5_inv)
    pr_inv = average_precision_score(y_te_inv, m_inv.predict_proba(X_te)[:, 1])

    # FRAUD 3mo window
    m_fr = LGBMClassifier(**params, scale_pos_weight=spw_fr,
                           subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m_fr.fit(Xw3, y_tr_fr[idx3], sample_weight=sw_w3_fr)
    pr_fr = average_precision_score(y_te_fr, m_fr.predict_proba(X_te)[:, 1])

    marker_inv = " <-- BEST INV" if pr_inv > best_inv['pr'] else ""
    marker_fr  = " <-- BEST FR"  if pr_fr  > best_fr['pr']  else ""
    if pr_inv > best_inv['pr']:
        best_inv = {'pr': pr_inv, 'params': params, 'model': m_inv}
    if pr_fr > best_fr['pr']:
        best_fr  = {'pr': pr_fr,  'params': params, 'model': m_fr}

    if pr_inv > base_inv * 1.01 or pr_fr > base_fr * 1.01 or i % 9 == 0:
        p_str = f"nl={params['num_leaves']} mcs={params['min_child_samples']} " \
                f"rl={params['reg_lambda']} cbt={params['colsample_bytree']}"
        print(f"  [{i+1:3d}/{len(combos)}] {p_str:<50} "
              f"INV={pr_inv:.4f}{marker_inv}  FR={pr_fr:.4f}{marker_fr}  ({time.time()-t:.1f}s)")

# ── #5 Monotone constraints — best params + monotone ──────────────────────────
print("\n" + "="*70)
print("MONOTONE CONSTRAINTS (best params + constraints)")
print("="*70)

best_inv_p = best_inv['params'] if isinstance(best_inv['params'], dict) else \
    dict(num_leaves=31, min_child_samples=10, reg_lambda=2, colsample_bytree=0.8,
         n_estimators=800, learning_rate=0.05)
best_fr_p  = best_fr['params']  if isinstance(best_fr['params'],  dict) else \
    dict(num_leaves=31, min_child_samples=10, reg_lambda=2, colsample_bytree=0.8,
         n_estimators=800, learning_rate=0.05)

# No monotone (best params only)
t = time.time()
m = LGBMClassifier(**best_inv_p, scale_pos_weight=spw_inv, subsample=0.8,
                    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xw5, y_tr_inv[idx5], sample_weight=sw_w5_inv)
pr_inv_best = average_precision_score(y_te_inv, m.predict_proba(X_te)[:, 1])
print(f"  INV  5mo best_params (no mono)    PR={pr_inv_best:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = LGBMClassifier(**best_fr_p, scale_pos_weight=spw_fr, subsample=0.8,
                    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xw3, y_tr_fr[idx3], sample_weight=sw_w3_fr)
pr_fr_best = average_precision_score(y_te_fr, m.predict_proba(X_te)[:, 1])
print(f"  FRAUD 3mo best_params (no mono)   PR={pr_fr_best:.4f}  ({time.time()-t:.1f}s)")

# With monotone constraints
t = time.time()
m = LGBMClassifier(**best_inv_p, scale_pos_weight=spw_inv, subsample=0.8,
                    monotone_constraints=mono_constraints,
                    monotone_constraints_method='advanced',
                    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xw5, y_tr_inv[idx5], sample_weight=sw_w5_inv)
pr_inv_mono = average_precision_score(y_te_inv, m.predict_proba(X_te)[:, 1])
print(f"  INV  5mo best_params + monotone   PR={pr_inv_mono:.4f}  ({time.time()-t:.1f}s)")

t = time.time()
m = LGBMClassifier(**best_fr_p, scale_pos_weight=spw_fr, subsample=0.8,
                    monotone_constraints=mono_constraints,
                    monotone_constraints_method='advanced',
                    random_state=42, n_jobs=-1, verbose=-1)
m.fit(Xw3, y_tr_fr[idx3], sample_weight=sw_w3_fr)
pr_fr_mono = average_precision_score(y_te_fr, m.predict_proba(X_te)[:, 1])
print(f"  FRAUD 3mo best_params + monotone  PR={pr_fr_mono:.4f}  ({time.time()-t:.1f}s)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n  INV  5mo LGBM:")
print(f"    Baseline (default params):   {base_inv:.4f}")
print(f"    Best grid params:            {best_inv['pr']:.4f}  params={best_inv['params']}")
print(f"    Best params + monotone:      {pr_inv_mono:.4f}")

print(f"\n  FRAUD 3mo LGBM:")
print(f"    Baseline (default params):   {base_fr:.4f}")
print(f"    Best grid params:            {best_fr['pr']:.4f}  params={best_fr['params']}")
print(f"    Best params + monotone:      {pr_fr_mono:.4f}")

print("\nDone.")
