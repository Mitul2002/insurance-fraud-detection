"""
experiment_density_reweight.py
================================
Density Ratio Reweighting for Covariate Shift Correction.

Theory:
  Train/test come from different distributions (adversarial AUC=0.9961).
  Optimal fix: reweight each training sample by:
      w(x) = p_test(x) / p_train(x)
            = P(domain=test|x) / P(domain=train|x)   [Bayes ratio]
            = p_shift(x) / (1 - p_shift(x))

  where p_shift(x) is the probability that sample x belongs to the test set,
  estimated by a gradient-boosted classifier trained to separate train from test.

  This re-weights training samples so that "test-like" training samples get
  higher weight, forcing the fraud model to focus on patterns that generalize
  to the test period.

Implementation:
  1. Build domain classifier on X_tr (label=0) + X_te (label=1)
  2. Get p_shift for all training samples
  3. Compute density ratio weights, clip at 99th percentile
  4. Combine with time-decay: w_final = w_density * w_timedecay
  5. Train RF + LGBM with combined weights
  6. Compare vs baseline (time-decay only)

Note: Different from IW (importance weighting) tried before.
  - IW was: w(x) proportional to feature importance (wrong, hurts)
  - DRW is: w(x) = p(test|x) / p(train|x) (theoretically correct)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
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
sw_train  = ck['sample_weight_tr']   # time-decay weights
spw_inv   = float(ck['spw_inv'][0])
spw_fr    = float(ck['spw_fr'][0])
train_dates = pd.to_datetime(ck['train_dates'])
print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")
print(f"  spw_inv={spw_inv:.1f}  spw_fr={spw_fr:.1f}")

# ── Step 1: Build domain classifier (train=0, test=1) ─────────────────────────
print("\nBuilding domain classifier...")
t0 = time.time()
n_tr, n_te = len(X_tr), len(X_te)
X_domain = np.vstack([X_tr, X_te])
y_domain = np.array([0]*n_tr + [1]*n_te, dtype=int)

# Use LGBM (fast, good calibration with monotone)
domain_clf = LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
domain_clf.fit(X_domain, y_domain)
domain_auc = roc_auc_score(y_domain, domain_clf.predict_proba(X_domain)[:, 1])
print(f"  Domain classifier AUC={domain_auc:.4f}  ({time.time()-t0:.1f}s)")

# ── Step 2: Compute density ratio weights for training samples ─────────────────
p_shift_tr = domain_clf.predict_proba(X_tr)[:, 1]  # P(test|x) for each train sample
p_shift_te = domain_clf.predict_proba(X_te)[:, 1]  # sanity check

print(f"\n  p_shift_tr: mean={p_shift_tr.mean():.3f}  "
      f"p10={np.percentile(p_shift_tr,10):.3f}  "
      f"p50={np.percentile(p_shift_tr,50):.3f}  "
      f"p90={np.percentile(p_shift_tr,90):.3f}  "
      f"p99={np.percentile(p_shift_tr,99):.3f}")

# Density ratio: clip to avoid extreme weights
# Prior correction: n_tr/n_te balances class frequencies in domain classifier
prior_ratio = n_tr / n_te
eps = 1e-6
density_ratio = (p_shift_tr / (1 - p_shift_tr + eps)) * prior_ratio

# Clip at 99th percentile to avoid extreme weights dominating
clip_val = np.percentile(density_ratio, 99)
density_ratio_clipped = np.clip(density_ratio, 0, clip_val)
# Normalize to mean=1 so scale is comparable to time-decay
density_ratio_clipped /= density_ratio_clipped.mean()

print(f"  Density ratio (clipped): mean={density_ratio_clipped.mean():.3f}  "
      f"p10={np.percentile(density_ratio_clipped,10):.3f}  "
      f"p50={np.percentile(density_ratio_clipped,50):.3f}  "
      f"p90={np.percentile(density_ratio_clipped,90):.3f}  "
      f"max={density_ratio_clipped.max():.3f}")

# ── Step 3: Build combined weights ─────────────────────────────────────────────
def make_sw(y, spw, base_w):
    """class weight × base_w"""
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w.astype(np.float32)

# Baseline: time-decay only (current approach)
sw_base_inv  = make_sw(y_tr_inv, spw_inv, sw_train)
sw_base_fr   = make_sw(y_tr_fr,  spw_fr,  sw_train)

# DRW: density ratio only (no time-decay)
sw_drw_only_inv = make_sw(y_tr_inv, spw_inv, density_ratio_clipped)
sw_drw_only_fr  = make_sw(y_tr_fr,  spw_fr,  density_ratio_clipped)

# DRW + time-decay combined (multiply)
sw_combined = sw_train * density_ratio_clipped
sw_combined /= sw_combined.mean()
sw_comb_inv = make_sw(y_tr_inv, spw_inv, sw_combined)
sw_comb_fr  = make_sw(y_tr_fr,  spw_fr,  sw_combined)

# ── 5mo window variants ────────────────────────────────────────────────────────
_wstart  = SPLIT_DATE - pd.DateOffset(months=5)
_win_idx = (train_dates >= _wstart)
X_w5     = X_tr[_win_idx]
_wdates  = train_dates[_win_idx]
_wdays   = (_wdates - _wdates.min()).total_seconds().values / 86400.0
_wdecay  = np.exp(0.001 * _wdays).astype(np.float32); _wdecay /= _wdecay.mean()
_wdr     = density_ratio_clipped[_win_idx]
_wdr_norm = _wdr / _wdr.mean()
_wcomb   = _wdecay * _wdr_norm; _wcomb /= _wcomb.mean()

y_w5_inv = y_tr_inv[_win_idx]
y_w5_fr  = y_tr_fr[_win_idx]
sw_w5_base_inv = make_sw(y_w5_inv, spw_inv, _wdecay)
sw_w5_drw_inv  = make_sw(y_w5_inv, spw_inv, _wdr_norm)
sw_w5_comb_inv = make_sw(y_w5_inv, spw_inv, _wcomb)
sw_w5_base_fr  = make_sw(y_w5_fr, spw_fr, _wdecay)
sw_w5_drw_fr   = make_sw(y_w5_fr, spw_fr, _wdr_norm)
sw_w5_comb_fr  = make_sw(y_w5_fr, spw_fr, _wcomb)

# ── Model factories ────────────────────────────────────────────────────────────
def make_rf(spw):
    return RFC(n_estimators=400, max_depth=None, min_samples_leaf=5,
               class_weight={0:1, 1:int(spw)}, random_state=42, n_jobs=-1)

def make_lgbm(spw):
    return LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)

def run_both(X_t, y_t, sw, X_e, y_e, spw, label):
    t = time.time()
    rf = make_rf(spw)
    rf.fit(X_t, y_t, sample_weight=sw)
    pr_rf = average_precision_score(y_e, rf.predict_proba(X_e)[:, 1])

    lgb = make_lgbm(spw)
    lgb.fit(X_t, y_t, sample_weight=sw)
    pr_lgb = average_precision_score(y_e, lgb.predict_proba(X_e)[:, 1])
    print(f"  {label:<55} RF={pr_rf:.4f}  LGBM={pr_lgb:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf, pr_lgb

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INVESTIGATION")
print("="*70)

# Full train
run_both(X_tr, y_tr_inv, sw_base_inv,    X_te, y_te_inv, spw_inv, "Full: baseline (time-decay)")
run_both(X_tr, y_tr_inv, sw_drw_only_inv, X_te, y_te_inv, spw_inv, "Full: DRW only")
run_both(X_tr, y_tr_inv, sw_comb_inv,    X_te, y_te_inv, spw_inv, "Full: DRW + time-decay")

# 5mo window
run_both(X_w5, y_w5_inv, sw_w5_base_inv, X_te, y_te_inv, spw_inv, "5mo window: baseline (time-decay)")
run_both(X_w5, y_w5_inv, sw_w5_drw_inv,  X_te, y_te_inv, spw_inv, "5mo window: DRW only")
run_both(X_w5, y_w5_inv, sw_w5_comb_inv, X_te, y_te_inv, spw_inv, "5mo window: DRW + time-decay")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRAUD")
print("="*70)

# Full train
run_both(X_tr, y_tr_fr, sw_base_fr,    X_te, y_te_fr, spw_fr, "Full: baseline (time-decay)")
run_both(X_tr, y_tr_fr, sw_drw_only_fr, X_te, y_te_fr, spw_fr, "Full: DRW only")
run_both(X_tr, y_tr_fr, sw_comb_fr,    X_te, y_te_fr, spw_fr, "Full: DRW + time-decay")

# 5mo window (less useful for Fraud but test anyway)
run_both(X_w5, y_w5_fr, sw_w5_base_fr, X_te, y_te_fr, spw_fr, "5mo window: baseline (time-decay)")
run_both(X_w5, y_w5_fr, sw_w5_drw_fr,  X_te, y_te_fr, spw_fr, "5mo window: DRW only")
run_both(X_w5, y_w5_fr, sw_w5_comb_fr, X_te, y_te_fr, spw_fr, "5mo window: DRW + time-decay")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\nDone.")
