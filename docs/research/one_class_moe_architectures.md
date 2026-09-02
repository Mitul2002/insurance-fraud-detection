# One-class MoE architectures for health insurance fraud detection

**Mixture of Experts (MoE) architectures offer a powerful framework for health insurance claims fraud detection by combining specialized one-class classifiers—each modeling a different facet of normal behavior—through learned gating mechanisms that route claims to the most relevant experts.** This approach directly addresses the core challenges of healthcare fraud: extreme class imbalance (fraud rates of 0.1–3%), heterogeneous fraud typologies (upcoding, unbundling, phantom billing), and the need for interpretable decisions. No paper yet directly combines MoE with one-class classification for health insurance fraud, making this a significant research gap with high practical value. The architecture outlined here synthesizes advances from ADMoE (AAAI 2023), ARGUE (IJCNN 2022), Deep SVDD (ICML 2018), and recent tabular MoE benchmarks (GG MoE 2025) into a cohesive, deployable system.

---

## 1. Structuring an asymmetric MoE for fraud detection

The fundamental design insight is that fraud detection benefits from **asymmetric** expert architectures—different model types for different aspects of the problem—rather than homogeneous experts. Each expert models a specific view of "normal" behavior, and deviations from any expert's learned distribution constitute anomaly signals that the gating network combines into a final fraud score.

### One-class classification experts as MoE components

Four primary one-class methods serve as expert building blocks, each with distinct strengths:

**One-Class SVM** learns a hyperplane with maximum margin separating mapped data from the origin in kernel space. Its decision function f(x) = w·Φ(x) − ρ produces continuous anomaly scores, where negative values indicate anomalies. The ν parameter controls the trade-off between outlier fraction and support vectors. OC-SVM works well with RBF kernels on structured, moderate-dimensional feature subspaces (billing patterns, temporal aggregates) and offers **fast inference** suitable for real-time scoring tiers.

**Deep SVDD** (Ruff et al., ICML 2018) trains a neural network to map data into a minimum-volume hypersphere centered at c, with objective L = (1/n)Σ‖φ(xᵢ; W) − c‖² + (λ/2)Σ‖Wˡ‖²_F. The anomaly score is the Euclidean distance from the embedding to center c. Its semi-supervised extension **DeepSAD** adds a loss term that pushes known anomalies away from the center, making it ideal when a small number of confirmed fraud cases exist. Autoencoder pretraining is standard for initialization, with bias terms removed to prevent hypersphere collapse.

**Autoencoders** model normal claim distributions through reconstruction. Trained exclusively on legitimate claims, they produce anomaly scores via reconstruction error ‖x − x̂‖². Variational Autoencoders (VAEs) provide probabilistic anomaly scores through the evidence lower bound. Autoencoders excel at capturing **high-dimensional correlations** across many features simultaneously—for instance, the complex relationships between diagnosis codes, procedure codes, and billing amounts that characterize legitimate claims within a medical specialty.

**Isolation Forest** isolates anomalies through random partitioning: anomalies, being rare and different, require fewer random splits. The anomaly score s(x, n) = 2^(−E(h(x))/c(n)) is based on average path length. Research in healthcare settings shows Isolation Forest **consistently outperforms OC-SVM** in precision/recall with significantly lower computational cost, making it the recommended default for provider-level and billing-pattern experts.

### Problem space partitioning across experts

The MoE architecture's power comes from partitioning the fraud detection problem into sub-problems, each assigned to a specialized expert:

**Feature subspace partitioning** assigns each expert a different feature group. Juszczak & Duin (2004) demonstrated that ensembles of one-class classifiers trained on separate feature groups handle missing data gracefully and are robust to small sample sizes. A practical configuration routes billing features to one expert, temporal features to another, and network/graph features to a third.

**Claim type partitioning** creates experts specialized by medical domain. ICD-10 chapter-based clustering (infectious diseases, neoplasms, musculoskeletal, etc.) or medical specialty alignment (cardiology, orthopedics, oncology) ensures each expert learns the specific billing patterns of its domain. The MECAD system (Dahmardeh et al., 2025) demonstrated this approach, dynamically assigning experts to categories based on cosine similarity of embeddings, achieving **AUROC of 0.8259 across 15 categories** with a 5-expert configuration.

**Provider type partitioning** assigns different experts to hospitals, individual practitioners, pharmacies, and durable medical equipment suppliers—each having fundamentally different legitimate billing distributions.

**Anomaly type partitioning** trains each expert to detect a specific fraud scheme: one expert for upcoding patterns, another for unbundling, another for phantom billing. GS-MoE (2025) validates this approach for video anomaly detection, training each class-expert on its assigned anomaly type plus normal data.

### The recommended 5-expert asymmetric architecture

```
Claims Input → Feature Engineering Pipeline
    │
    ├──→ Expert 1: Isolation Forest on provider billing patterns
    ├──→ Expert 2: Deep SVDD on ICD/CPT embedding distances
    ├──→ Expert 3: Autoencoder on temporal claim sequences
    ├──→ Expert 4: One-Class SVM on patient behavior features
    ├──→ Expert 5: Supervised LightGBM on known fraud patterns
    │
    ├──→ Calibration Layer (per-expert Platt/Beta scaling)
    ├──→ Gating Network (claim features → expert weights)
    └──→ Meta-learner → Final Fraud Probability
```

This **asymmetric** design pairs different architectures with the feature domains where they excel. The supervised expert (Expert 5) leverages available fraud labels for well-characterized fraud types, while the one-class experts detect novel patterns. Vallarino (2025, arXiv:2504.03750) validated a similar asymmetric design (RNN + Transformer + Autoencoder experts) achieving **98.7% accuracy, 94.3% precision, and 91.5% recall** on financial fraud.

---

## 2. Gating mechanisms and meta-model design determine system performance

The gating mechanism is arguably the most critical component—it determines how expert outputs are combined and which experts are consulted for each claim. The choice between sparse and dense gating, the calibration of expert outputs, and the meta-model architecture all significantly impact fraud detection performance.

### Sparse versus dense gating

**Dense gating** activates all experts for every claim, computing y = Σᵢ gᵢ(x)·Eᵢ(x) where all gate weights gᵢ > 0 via softmax. This is preferred when the expert count is small (≤5), claim types are ambiguous and may exhibit multiple patterns simultaneously, and interpretability of per-expert contributions matters for investigator reporting.

**Sparse top-k gating** (Shazeer et al., 2017) activates only the top-k experts per input. The formulation adds trainable Gaussian noise before top-k selection: H(x)ᵢ = (x·Wg)ᵢ + N(0,1)·Softplus((x·Wₙₒᵢₛₑ)ᵢ), then G(x) = softmax(KeepTopK(H(x), k)). This is preferred when expert count exceeds 5, computational efficiency is critical for production scoring, or experts are highly specialized by fraud type. **Expert Choice Routing** (Zhou et al., 2022) inverts this: experts select their preferred inputs, guaranteeing perfect load balancing and allowing variable expert engagement per claim.

For health insurance fraud with 5 experts, **dense gating with entropy regularization** is recommended. The entropy loss L_entropy = −Σ gᵢ log(gᵢ) prevents gate collapse where one expert dominates. Load balancing loss L_balance = CV(importance)² ensures all experts receive training signal.

### Calibration is non-negotiable before combination

Different expert types produce scores on incompatible scales: autoencoder reconstruction error (unbounded positive), Isolation Forest path length (bounded [0,1]), SVM decision function (unbounded real). **Calibration maps all expert outputs to [0,1] probabilities** before the gating network combines them.

**Platt scaling** fits P(fraud|f(x)) = 1/(1 + exp(Af(x) + B)) on a held-out calibration set. Best for OC-SVM and Isolation Forest outputs. **Temperature scaling** divides neural network logits by learned temperature T before softmax—best for autoencoder and Deep SVDD experts. **Beta calibration** (Kull et al., AISTATS 2017) models classwise scores with Beta distributions and was shown in a 2025 study (arXiv:2601.19944) to more consistently improve calibration for modern tabular models than Platt or isotonic regression. For gradient boosting experts, Platt scaling and isotonic regression can actually **degrade** calibration, making Beta calibration the safer choice.

The practical protocol: (1) train each expert independently, (2) freeze expert weights, (3) calibrate each expert on a separate held-out set using the appropriate method, (4) train the gating network on calibrated outputs.

### Stacking meta-learners on expert scores

The stacking approach treats calibrated expert outputs as features for a secondary meta-learner. Three tiers of complexity apply:

A **logistic regression meta-learner** on the score vector [s₁(x), s₂(x), ..., sₘ(x)] automatically learns optimal weighting and produces calibrated probabilities. This is the simplest approach and provides a strong baseline.

A **gradient boosting meta-learner** (XGBoost/LightGBM) captures non-linear interactions between expert outputs—for instance, when Expert A's high score combined with Expert B's low score is more indicative of fraud than either score alone. SHAP feature importance on the meta-learner reveals which experts matter for which claim patterns.

A **neural meta-learner** concatenates expert scores with raw claim features: [s₁(x), ..., sₘ(x), x₁, ..., xd]. Attention mechanisms can weight experts based on input features. The FEAMOE framework (Sharma et al., IJCAI 2023) proved that SHAP values for MoE decompose as φⱼ(MoE(x)) = Σₘ gₘ(x)×φⱼ(expertₘ(x))—meaning system-level explanations are a **weighted combination of per-expert SHAP values**, preserving interpretability.

### Posterior probability combination strategies

Beyond learned meta-models, principled probabilistic combination methods offer theoretical guarantees:

**Bayesian combination** computes P(fraud|e₁,...,eₘ) ∝ P(fraud)·Πᵢ P(eᵢ|fraud) under conditional independence. This naturally incorporates the fraud base rate as a prior.

**Dempster-Shafer Theory (DST)** explicitly models uncertainty through mass functions over the frame of discernment Θ = {Fraud, Normal}. Dempster's rule of combination m₁₂(A) = [Σ_{B∩C=A} m₁(B)·m₂(C)] / [1 − Σ_{B∩C=∅} m₁(B)·m₂(C)] handles conflicting evidence between experts and represents epistemic uncertainty ("don't know") as mass assigned to the full set Θ. The ECET framework (arXiv:2212.12092) demonstrates DST for ensemble classification with anomaly detection capability.

### Attention-based and memory-augmented gating

Standard attention gates compute αᵢ = softmax(vᵀ·tanh(Wₓ·x + Wₑ·eᵢ + b)), weighting expert contributions based on both input characteristics and expert output. This enables **context-aware routing** where claim metadata (provider specialty, ICD chapter, claim amount percentile) influences expert selection.

GAD-MoRE (Zhao et al., Feb 2026, arXiv:2602.06859) introduces a **memory-based dynamic router** that assigns inputs to experts based on historical reconstruction quality on similar patterns. The router maintains a memory bank of past performance, enabling adaptive inference for unseen fraud patterns. Graph-MoE (AAAI 2025) combines memory-augmented routers with graph neural networks, achieving **2–3.5% consistent improvement** across anomaly detection methods as a plug-in enhancement.

---

## 3. Training strategies that exploit class imbalance structure

### One-class formulation eliminates the imbalance problem

The one-class approach fundamentally sidesteps class imbalance by modeling only the legitimate claim distribution. When fraud constitutes 0.1–3% of claims, traditional binary classifiers struggle with overwhelming negative examples. OCC experts trained exclusively on normal claims treat the problem as density estimation rather than classification—anything outside the learned normal boundary is anomalous.

Leevy et al. (Journal of Big Data, 2023) found that **binary classification outperforms pure OCC** when labels are available (CatBoost achieving AUPRC of 0.8567 vs. substantially lower scores for OCC). This motivates the hybrid MoE design: one-class experts handle novel/unlabeled fraud types while supervised experts handle known patterns. The recommended architecture maintains **at least 3 one-class experts alongside 1–2 supervised experts**, with the gating network learning when to trust each.

### Cost-sensitive learning through the MoE stack

In health insurance, false negatives (missed fraud) cost **10–100× more** than false positives (unnecessary investigations). Typical costs: missed fraud = full claim amount ($10K–$500K+); false positive = investigation cost ($500–$5,000).

**Focal loss** FL(pₜ) = −αₜ(1 − pₜ)^γ·log(pₜ) automatically down-weights easy examples through the focusing parameter γ (typically 2), performing implicit hard example mining without explicit sample selection. Boabang & Gyamerah (arXiv 2508.02283, 2025) proposed a **three-stage focal loss** for insurance fraud: (1) convex surrogate for stable initialization, (2) controlled non-convex intermediate loss for feature discrimination, (3) standard focal loss for minority-class sensitivity refinement.

Cost-sensitivity propagates through the MoE in three ways. First, **per-expert cost-sensitive training** where each expert uses its own cost-weighted objective tailored to its specialization. Second, **cost-sensitive gating loss** that penalizes routing high-risk claims to less capable experts. Third, **cost-weighted load balancing** that distributes expected financial loss rather than sample counts across experts.

### Data augmentation at the expert level

Augmentation is most effective when applied **within each expert's training data** rather than globally, preventing cross-contamination between fraud subtypes. The key methods form a hierarchy of increasing sophistication:

- **SMOTE-ENN** (SMOTE oversampling + Edited Nearest Neighbors cleaning) generates synthetic minority samples then removes noisy borderline cases. Recent work combining SMOTE-ENN with LSTM/GRU + MLP meta-learner achieved **perfect recall** with 0.997 specificity.
- **CTGAN** (Xu et al., NeurIPS 2019) is the dominant GAN for tabular data, using conditional generation with mode-specific normalization for mixed-type columns. CTAB-GAN achieved up to **17% accuracy improvement** on complex financial datasets.
- **Conditional VAE** conditions on class labels during both encoding and decoding: the encoder learns q(z|x, fraud) and the decoder generates p(x|z, fraud), enabling direct sampling of synthetic fraud claims from learned latent distributions.

The recommended practice: apply augmentation **after** the gating network assigns claims to expert domains but **before** individual expert training, ensuring synthetic samples are contextually appropriate for each expert's specialization.

### Curriculum learning and hard negative mining in MoE

Curriculum learning for fraud MoE operates at two levels. At the **expert level**: train on clearly legitimate and clearly fraudulent claims first, then progressively introduce ambiguous near-miss cases. At the **gating level**: first learn to route clearly distinct claim types, then learn to handle ambiguous routing decisions. The Self-Paced Ensemble (SPE) framework (Liu et al., ICDE 2020) iteratively selects the most informative majority-class samples based on classification hardness, working particularly well on large-scale data with imbalance ratios exceeding 100:1.

Hard negative mining in MoE creates two categories of difficult examples: **classification hard negatives** (legitimate claims resembling fraud and vice versa) and **routing hard negatives** (claims difficult to assign to the correct expert). Training the gating network specifically on routing hard negatives improves expert specialization. Focal loss acts as a smooth, differentiable alternative to explicit two-stage hard example mining.

### Semi-supervised and self-supervised extensions

Healthcare fraud labels are expensive and delayed (investigation outcomes take 6–18 months). Semi-supervised methods bridge this gap:

**DeepSAD** extends Deep SVDD with a semi-supervised loss that clusters normal samples near the hypersphere center while pushing known anomalies away. This is the most natural semi-supervised extension for one-class MoE experts.

**GTAN** (Xiang et al., AAAI 2023) constructs temporal transaction graphs and propagates risk through gated temporal attention, achieving excellent performance with only a tiny proportion of labeled data. **Pseudo-labeling** uses confident predictions from initial models as training signal for unlabeled claims, with confidence thresholding to select only high-certainty assignments.

For self-supervised pretraining, **masked feature prediction** (analogous to BERT) randomly masks claim features and trains the encoder to reconstruct them. Amazon's self-supervised pretraining for tabular data (NeurIPS TRL 2022) using Manifold Mixup with noise contrastive estimation achieved **9% relative improvement** in fraud detection over supervised baselines. **SCARF** (Bahri et al., 2021) corrupts random features to create positive pairs for contrastive learning, and VIME (NeurIPS 2020) extends mask-prediction pretext tasks to the tabular domain.

The recommended training pipeline: (1) self-supervised pretrain a shared encoder on all claims, (2) initialize experts from this encoder, (3) train semi-supervisedly with labeled claims + pseudo-labels, (4) deploy and use active learning—selecting claims where experts disagree or gating uncertainty is high—to prioritize investigation labeling.

---

## 4. State-of-the-art papers and benchmarks (2023–2025)

### MoE for fraud and anomaly detection

**ADMoE** (Zhao et al., AAAI 2023, arXiv:2208.11290) is the most directly relevant work: the first framework for anomaly detection from noisy labels using MoE. It shares model parameters among noisy label sources while building expert sub-networks for specialization, and innovatively uses noisy labels as input features. ADMoE achieved **up to 34% performance improvement** over non-MoE baselines on eight datasets including enterprise security data. This is directly applicable to healthcare fraud where initial rule-based flags serve as noisy labels.

**Adapted-MoE** (Lei et al., Sep 2024, arXiv:2409.05611) combines MoE with test-time adaptation for anomaly detection, using a routing network to handle multiple distributions of same-category samples via divide-and-conquer. It achieved **2.18–7.20% I-AUROC improvement** and **1.57–16.30% P-AUROC improvement** over prior SOTA on texture anomaly detection benchmarks.

**Yang et al. (2024, MDPI Big Data and Cognitive Computing)** directly integrates MoE with DNN-SMOTE for credit card fraud detection, demonstrating that multiple specialized expert networks combined with gating outperform single classifiers in balancing precision and recall.

**GG MoE** (Feb 2025, arXiv:2502.03608) introduces Gumbel-Softmax gating for tabular data MoE, combined with piecewise-linear embedding layers. It outperforms standard MoE and MLP models while being **significantly more parameter-efficient**, directly validating MoE advantages for tabular fraud data.

**IME** (Interpretable Mixture of Experts, ICLR submission 2024) provides inherently interpretable MoE for tabular data. Tested on Credit Fraud Detection, S-IMEi outperformed not only interpretable models but also black-box DNN and XGBoost baselines.

### Healthcare fraud detection advances

The most comprehensive survey is **du Preez et al. (Artificial Intelligence in Medicine, Feb 2025)**, covering 137 studies from 2000–2024 on ML for health insurance fraud. Key findings: traditional ML remains dominant but deep learning adoption is rising; 94 supervised, 41 unsupervised, and 12 hybrid approaches were catalogued; **absence of standard benchmark datasets** remains the primary barrier.

**Shekhar, Leder-Luis, & Akoglu (NBER Working Paper)** developed a fully unsupervised, explainable ML approach using an ensemble of 3 detection algorithms with instant-runoff voting for Medicare hospital fraud. It achieved **8-fold lift** over random targeting, with 21 of 50 top-ranked hospitals matching DOJ fraud lawsuits.

A 2024 medRxiv preprint on Medicare ophthalmology fraud used a stacking ensemble (XGBoost + MLP), achieving **AUROC of 0.907** and estimating ~8.6% of ophthalmologists engaged in overutilization with estimated losses of **$437.1M** in 2021 alone.

### Deep one-class classification advances

**DROCC** (Goyal et al., ICML 2020, Microsoft Research) overcomes Deep SVDD's representation collapse through adversarial training, assuming normal data lies on a locally linear low-dimensional manifold. It achieved **up to 20% accuracy improvement** over Deep SVDD on image benchmarks and is effective on tabular data with available code.

**DOC³** (Machine Learning/Springer, 2023) extends DROCC with Universum learning (learning from contradictions), providing theoretical guarantees via Rademacher complexity analysis. **NeuTraL AD** (Qiu et al., ICML 2021, Bosch Research) introduces end-to-end learnable transformations for anomaly detection that work on tabular and time-series data, significantly outperforming existing methods with code available at boschresearch/NeuTraL-AD.

### Kaggle and competition insights

The IEEE-CIS Fraud Detection competition's winning solution (Chris Deotte, AUC 0.9459) relied on massive feature engineering (group aggregations, UID creation) with XGBoost + CatBoost + LightGBM ensembles—notably **no MoE or one-class methods**. This reflects a persistent gap: while academic research advances MoE and one-class methods, production Kaggle solutions favor gradient boosting with engineered features. Bridging this gap is an implementation opportunity.

---

## 5. Implementation patterns, feature engineering, and production deployment

### PyOD and SUOD form the practical foundation

**PyOD** (v2.0+) provides a unified API for 50+ outlier detection algorithms with `fit()`, `decision_function()`, `predict()`, and `predict_proba()` methods. Built-in combination methods include Average of Maximum (AOM), Maximum of Average (MOA), and LSCP (Locally Selective Combination in Parallel) for adaptive local model combination. PyOD v2 integrates 12 neural models into a PyTorch-based framework.

**SUOD** (MLSys 2021) accelerates large-scale heterogeneous outlier detection through random projection for dimensionality reduction, pseudo-supervised approximation for fast offline scoring, and balanced parallel scheduling. Critically, **SUOD has a real-world deployment at IQVIA** for fraudulent healthcare claim analysis, validating its applicability.

```python
from pyod.models.suod import SUOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD

experts = [IForest(n_estimators=200), OCSVM(kernel='rbf', nu=0.05),
           COPOD(), IForest(n_estimators=150)]
clf = SUOD(base_estimators=experts, n_jobs=4, combination='average')
clf.fit(X_train_normal)
scores = clf.decision_function(X_test)
```

The hybrid sklearn-experts + PyTorch-gating pattern trains sklearn one-class models as expert scorers, collects their anomaly scores into a feature matrix, then trains a PyTorch gating network on these scores concatenated with routing features:

```python
class MoEFraudDetector(nn.Module):
    def __init__(self, routing_dim, num_experts, k=3):
        super().__init__()
        self.gate = nn.Linear(routing_dim, num_experts)
        self.meta = nn.Sequential(nn.Linear(num_experts, 32), nn.ReLU(),
                                   nn.Linear(32, 1), nn.Sigmoid())
    
    def forward(self, routing_features, expert_scores):
        weights = F.softmax(self.gate(routing_features), dim=-1)
        weighted_scores = (weights * expert_scores).sum(dim=-1, keepdim=True)
        return self.meta(weights * expert_scores)
```

### Feature engineering for expert routing

**Provider-level features** (for the billing pattern expert) include total services billed, unique beneficiaries served, charge-to-payment ratio, unique HCPCS/CPT code diversity, average allowed amount per service, and specialty consistency scores comparing billing patterns to declared specialty. The NBER working paper demonstrated that peer-based features comparing hospital billing to similar hospitals with comparable patient populations yield strong fraud signals.

**Patient-level features** (for the behavior expert) include claim frequency per rolling window, diagnosis diversity (unique ICD codes over time), provider shopping count, geographic spread, and diagnosis-procedure consistency scores computed using medical code embeddings.

**Temporal features** (for the temporal expert) include claim velocity (rolling counts in 7/30/90-day windows), burst detection through sudden billing spikes, weekend/holiday billing anomalies, and time since last claim for the same patient-provider pair.

**Network features** (for the graph expert) include provider-patient bipartite graph degree centrality, referral network betweenness, shared patient ratios between providers, and community detection for identifying clusters of providers with unusual patient-sharing volumes.

### ICD code fraud signals and embedding techniques

**Upcoding** (billing for more severe diagnoses) is detected through DRG severity distribution analysis, E/M level patterns (providers consistently billing at highest severity codes like 99285), and abnormal complication/comorbidity rates. University of Colorado Health paid **$23M in 2024** for systematic ED visit upcoding.

**Unbundling** (billing separately for bundled services) is detected through CPT code pair analysis against National Correct Coding Initiative (NCCI) bundling rules, date-of-service spreading patterns, and lab panel fragmentation (billing individual tests instead of comprehensive metabolic panel codes).

**Medical code embeddings** map ICD and CPT codes into shared vector spaces using Word2Vec-style CBOW training where context = all codes on a single claim. A fraud heuristic computes minimum embedding distance between each CPT code and all ICD codes on the same claim—large distances indicate procedures unrelated to diagnoses. **Pat2Vec** (JMIR AI, 2023) compresses thousands of ICD-10 codes into ≤100 dimensions for patient profiling. **GRAM** supplements Word2Vec embeddings with ICD hierarchy information from medical ontologies.

For expert specialization, ICD codes can be clustered by chapter (21 ICD-10 chapters), by embedding similarity (k-means on learned representations), by medical specialty alignment, or by fraud vulnerability (grouping commonly upcoded or unbundled code families).

### Evaluation metrics: why AUCPR and KS matter more than AUROC

**AUROC exhibits ceiling effects** in imbalanced fraud detection. A 2025 benchmark study across three rare-event datasets (fraud at 0.17%, protein at 1.35%, ozone at 2.9%) confirmed that models with AUROC = 0.98 can have AUCPR = 0.10. The root cause: when negatives dominate, small absolute changes in false positives barely move FPR.

**AUCPR** is the primary metric, focusing exclusively on positive class performance. Baseline equals the fraud prevalence rate, not 0.5. In sklearn: `average_precision_score(y_true, y_scores)`. XGBoost supports native `eval_metric='aucpr'` with `scale_pos_weight`.

**KS statistic** measures the maximum separation between cumulative score distributions of fraud and non-fraud classes. KS > 50 indicates a good model; KS > 70 is excellent (but may indicate overfitting). KS is also valuable for **production monitoring**: tracking KS between training and production score distributions detects model degradation and concept drift.

**Cost-based evaluation** is essential: Expected_loss = (Cost_FN × FN_rate) + (Cost_FP × FP_rate), weighted by claim amount so that catching a $100K fraud matters more than five $1K cases.

For MoE-specific evaluation, track per-expert AUCPR on each expert's specialized claim subset, expert utilization rates, inter-expert score correlation (low correlation = good diversity), and gating entropy (higher = more balanced routing).

### Production deployment architecture

Health insurance claims suit a **three-tier scoring architecture**. The pre-payment real-time tier (30–100ms latency) uses lightweight rules plus cached provider/patient risk profiles. The near-real-time tier runs full MoE scoring within minutes of claim submission. The batch tier performs deep network analysis, cross-provider collusion detection, and pattern mining on 6-hour cycles.

**Concept drift** in fraud is adversarial—fraudsters actively adapt. PRODEM (Springer, 2025) predicts when the primary fraud model will make errors without requiring ground truth labels, which is critical given healthcare fraud's 6–18 month label delay. Feature stability can be improved by designing ratio and percentile features rather than absolute values. A champion-challenger framework maintains a shadow model being validated alongside production, with quarterly retraining for ensemble experts and monthly for the gating network.

**Explainability** leverages FEAMOE's decomposition: the SHAP value for the full MoE equals the weighted combination of per-expert SHAP values, weighted by gating weights. This enables natural language explanations: "This claim was flagged primarily by the Billing Pattern Expert (weight: 0.72) due to unusually high E/M coding level (SHAP +0.34) and charge-to-payment ratio (SHAP +0.28)."

Tools for monitoring include Evidently AI, Arize AI, and NannyML for feature/prediction drift tracking using PSI and KS statistics on feature distributions.

---

## Conclusion

The one-class MoE architecture for health insurance fraud represents an underexplored but highly promising approach that combines the best of several worlds: **one-class experts eliminate the class imbalance problem** by modeling normal behavior directly, **asymmetric expert architectures** match model types to feature domains where they excel, and **learned gating mechanisms** adaptively route claims to the most relevant specialists. The key architectural insight is that 3–5 heterogeneous experts (Isolation Forest for billing patterns, Deep SVDD for code embeddings, Autoencoder for temporal sequences, OC-SVM for patient behavior, supervised LightGBM for known fraud) with Beta-calibrated outputs and a gradient boosting meta-learner provides the strongest practical configuration.

Three critical implementation decisions separate effective systems from mediocre ones. First, **calibration before combination** is non-negotiable—Beta calibration for tree-based experts and temperature scaling for neural experts. Second, **expert-level augmentation** (SMOTE-ENN or CTGAN applied within each expert's domain) outperforms system-level augmentation by respecting expert specialization boundaries. Third, **AUCPR, not AUROC**, must drive model selection and monitoring, supplemented by cost-weighted metrics that account for the 10–100× asymmetry between missed fraud and false alarms.

The research gap is clear: no existing paper directly combines MoE with one-class classification for healthcare fraud. ADMoE (AAAI 2023) and Adapted-MoE (2024) validate the MoE-for-anomaly-detection paradigm; SUOD's deployment at IQVIA validates heterogeneous ensemble anomaly detection in healthcare; GG MoE (2025) validates MoE advantages on tabular data. Combining these strands into the architecture described here represents a concrete research and engineering opportunity with immediate practical value in a domain where the National Health Care Anti-Fraud Association estimates **$68 billion** in annual losses.