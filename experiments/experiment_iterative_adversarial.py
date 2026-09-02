"""
experiment_iterative_adversarial.py
=====================================
Idea #3: Iterative adversarial feature dropping.

Problem:
  Current: drop features with adversarial importance > 0.008 → 200 features.
  But adversarial RF AUC on remaining 200 features is STILL ~1.0 (perfect separation).
  The first round of dropping didn't reduce the shift — we need more aggressive pruning.

Method:
  1. Load checkpoint (200 features)
  2. Train new adversarial LGBM on X_tr (label=0) vs X_te (label=1)
  3. Check AUC and feature importances
  4. Drop features with importance > threshold (sweep: 0.008, 0.005, 0.003, 0.001)
  5. For each reduced feature set:
     a. Retrain adversarial classifier → measure new AUC
     b. Train fraud models → measure PR-AUC
     c. See if reducing shift (lower adv AUC) correlates with better fraud PR-AUC
  6. Iterate: after each drop, retrain adversarial and drop again (2 rounds)

Key question:
  Is there a feature subset where adversarial AUC < 0.90 (meaningful shift reduction)?
  If so, do fraud models improve with this subset?
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

CKPT_PATH = r"C:\Users\mitul\OneDrive\Documents\arya\fraud\checkpoint.npz"

# ── Load checkpoint ────────────────────────────────────────────────────────────
print("Loading checkpoint...")
ck = np.load(CKPT_PATH, allow_pickle=True)
X_tr_orig  = ck['X_tr']
X_te_orig  = ck['X_te']
y_tr_inv   = ck['y_train_inv'].astype(int)
y_te_inv   = ck['y_test_inv'].astype(int)
y_tr_fr    = ck['y_train_fraud'].astype(int)
y_te_fr    = ck['y_test_fraud'].astype(int)
sw_train   = ck['sample_weight_tr']
spw_inv    = float(ck['spw_inv'][0])
spw_fr     = float(ck['spw_fr'][0])
feat_names = list(ck['feat_names'])
train_dates = pd.to_datetime(ck['train_dates'])
print(f"  X_tr={X_tr_orig.shape}  X_te={X_te_orig.shape}")

# ── Adversarial classifier ─────────────────────────────────────────────────────
n_tr, n_te = len(X_tr_orig), len(X_te_orig)

def train_adversarial(X_t, X_e, feat_subset=None):
    """Train LGBM to separate train (0) from test (1). Returns AUC + importances."""
    if feat_subset is not None:
        X_t = X_t[:, feat_subset]
        X_e = X_e[:, feat_subset]
    X_dom = np.vstack([X_t, X_e])
    y_dom = np.array([0]*len(X_t) + [1]*len(X_e), dtype=int)
    m = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_dom, y_dom)
    p = m.predict_proba(X_dom)[:, 1]
    auc = roc_auc_score(y_dom, p)
    imp = m.feature_importances_
    return auc, imp, m

# ── Model evaluation ───────────────────────────────────────────────────────────
def make_sw(y, spw, base_w):
    return np.where(y == 1, spw, 1.0).astype(np.float32) * base_w.astype(np.float32)

sw_full_inv = make_sw(y_tr_inv, spw_inv, sw_train)
sw_full_fr  = make_sw(y_tr_fr,  spw_fr,  sw_train)

def eval_models(X_t, X_e, feat_subset, label):
    Xt = X_t[:, feat_subset]; Xe = X_e[:, feat_subset]
    t = time.time()
    # RF
    rf_inv = RFC(n_estimators=300, max_depth=None, min_samples_leaf=5,
                 class_weight={0:1, 1:int(spw_inv)}, random_state=42, n_jobs=-1)
    rf_inv.fit(Xt, y_tr_inv, sample_weight=sw_full_inv)
    pr_rf_inv = average_precision_score(y_te_inv, rf_inv.predict_proba(Xe)[:, 1])

    rf_fr = RFC(n_estimators=300, max_depth=None, min_samples_leaf=5,
                class_weight={0:1, 1:int(spw_fr)}, random_state=42, n_jobs=-1)
    rf_fr.fit(Xt, y_tr_fr, sample_weight=sw_full_fr)
    pr_rf_fr = average_precision_score(y_te_fr, rf_fr.predict_proba(Xe)[:, 1])

    # LGBM
    lg_inv = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                             scale_pos_weight=spw_inv, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lg_inv.fit(Xt, y_tr_inv, sample_weight=sw_full_inv)
    pr_lgb_inv = average_precision_score(y_te_inv, lg_inv.predict_proba(Xe)[:, 1])

    lg_fr = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                            scale_pos_weight=spw_fr, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=2, random_state=42, n_jobs=-1, verbose=-1)
    lg_fr.fit(Xt, y_tr_fr, sample_weight=sw_full_fr)
    pr_lgb_fr = average_precision_score(y_te_fr, lg_fr.predict_proba(Xe)[:, 1])

    print(f"  {label:<55} n_feat={len(feat_subset):<4} "
          f"INV: RF={pr_rf_inv:.4f} LGB={pr_lgb_inv:.4f}  "
          f"FR: RF={pr_rf_fr:.4f} LGB={pr_lgb_fr:.4f}  ({time.time()-t:.1f}s)")
    return pr_rf_inv, pr_lgb_inv, pr_rf_fr, pr_lgb_fr

# ── Round 0: Baseline on all 200 features ─────────────────────────────────────
print("\n" + "="*70)
print("ROUND 0: Baseline (200 features from checkpoint)")
print("="*70)

all_idx = list(range(len(feat_names)))
t0 = time.time()
adv_auc_0, imp_0, _ = train_adversarial(X_tr_orig, X_te_orig)
print(f"  Adversarial AUC (200 features): {adv_auc_0:.4f}  ({time.time()-t0:.1f}s)")
eval_models(X_tr_orig, X_te_orig, all_idx, "Baseline checkpoint (200 feat)")

# Feature importance from adversarial classifier
feat_imp = sorted(zip(feat_names, imp_0), key=lambda x: x[1], reverse=True)
print(f"\n  Top-20 adversarially important features:")
for fname, fimp in feat_imp[:20]:
    print(f"    {fname:<40} {fimp:.4f}")

# ── Round 1: Single-round drops at multiple thresholds ────────────────────────
print("\n" + "="*70)
print("ROUND 1: Single-round adversarial drop")
print("="*70)

results_r1 = {}
for thr in [0.008, 0.005, 0.003, 0.001]:
    keep_mask = imp_0 <= (thr * imp_0.sum())   # relative threshold
    keep_idx  = np.where(keep_mask)[0].tolist()
    if len(keep_idx) < 10:
        print(f"  thr={thr}: only {len(keep_idx)} features — skipping")
        continue

    # Retrain adversarial on kept features
    adv_auc_1, imp_1, _ = train_adversarial(X_tr_orig, X_te_orig, keep_idx)
    print(f"\n  Drop threshold={thr}: kept {len(keep_idx)}/{len(feat_names)} features  "
          f"Adv AUC: {adv_auc_0:.4f} -> {adv_auc_1:.4f}")

    r = eval_models(X_tr_orig, X_te_orig, keep_idx,
                    f"Round1 thr={thr} ({len(keep_idx)} feat)")
    results_r1[thr] = {'n_feat': len(keep_idx), 'adv_auc': adv_auc_1, 'results': r}

# ── Round 2: Iterative drop (drop, retrain, drop again) ───────────────────────
print("\n" + "="*70)
print("ROUND 2: Iterative adversarial drop (2 rounds)")
print("="*70)

# Use threshold 0.005 for round 1, then 0.005 again on remaining
ITER_THR = 0.005

keep_mask_r1 = imp_0 <= (ITER_THR * imp_0.sum())
keep_idx_r1  = np.where(keep_mask_r1)[0].tolist()
adv_auc_r1, imp_r1, _ = train_adversarial(X_tr_orig, X_te_orig, keep_idx_r1)
print(f"  After round 1: {len(keep_idx_r1)} features, adv AUC={adv_auc_r1:.4f}")

# Second round drop on remaining features
keep_mask_r2 = imp_r1 <= (ITER_THR * imp_r1.sum())
keep_idx_r2_local = np.where(keep_mask_r2)[0].tolist()
keep_idx_r2 = [keep_idx_r1[i] for i in keep_idx_r2_local]
if len(keep_idx_r2) >= 10:
    adv_auc_r2, _, _ = train_adversarial(X_tr_orig, X_te_orig, keep_idx_r2)
    print(f"  After round 2: {len(keep_idx_r2)} features, adv AUC={adv_auc_r2:.4f}")

    dropped_names = [feat_names[i] for i in all_idx if i not in keep_idx_r2]
    print(f"  Total dropped: {len(dropped_names)} features")
    print(f"  Remaining: {[feat_names[i] for i in keep_idx_r2[:20]]}...")

    eval_models(X_tr_orig, X_te_orig, keep_idx_r2,
                f"Round2 iterative ({len(keep_idx_r2)} feat, adv={adv_auc_r2:.3f})")
else:
    print(f"  Too few features after round 2 ({len(keep_idx_r2)}) — skipping")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Baseline (200 feat): Adv AUC={adv_auc_0:.4f}")
for thr, r in results_r1.items():
    res = r['results']
    print(f"  Drop thr={thr}: {r['n_feat']} feat, Adv={r['adv_auc']:.4f}  "
          f"INV RF={res[0]:.4f} LGB={res[1]:.4f}  FR RF={res[2]:.4f} LGB={res[3]:.4f}")

print("\nDone.")
