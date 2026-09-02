# Health insurance fraud detection: a practitioner's deep technical guide

**Upcoding and diagnosis-procedure mismatches are the single most discriminating fraud signal in hospitalization claims, provider-level aggregation dominates feature importance rankings, and hierarchical ICD embeddings meaningfully outperform flat frequency features.** These findings emerge from synthesizing academic fraud detection research, CMS Medicare studies, Kaggle competition analyses, and DOJ enforcement data. For group corporate health insurance, provider-side fraud accounts for **60–70% of wasteful spend**, making hospital-level features the highest-priority engineering target. This report covers five practitioner-level questions spanning ICD/DRG signals, network features, velocity windows, add-on cover fraud, and entity-level aggregation — with specific papers, feature importance rankings, and implementable feature definitions.

---

## 1. Diagnosis-procedure mismatch is the strongest ICD-based fraud signal

The research literature converges on a clear hierarchy of ICD/DRG fraud signal types. **Upcoding — the mismatch between diagnosis codes and procedure codes or severity levels billed — is the most discriminating signal**, followed closely by short length-of-stay for high-severity DRGs.

Shekhar, Leder-Luis, and Akoglu (2024, *Journal of Policy Analysis and Management*) built a three-algorithm unsupervised ensemble on Medicare inpatient data: a feature subspace detector on ICD-10 code distributions, a DRG frequency distribution comparison across peer hospitals, and an expenditure residual model. Their DRG distribution comparison was the strongest component. Validated against DOJ enforcement data, **21 of their top 50 flagged hospitals were DOJ-named fraudulent** — an 8× improvement over random targeting.

The HHS-OIG 2021 report found hospitals billing at the highest MS-DRG severity increased 20% from FY2014–2019 while average length-of-stay *decreased*. This divergence is the primary red flag used by OIG auditors. Approximately **30% of MS-DRG 871 (sepsis with MCC) stays had comparatively short LOS**, and similar rates were found for heart failure (DRG 291), pneumonia (DRG 193), and renal failure (DRG 682). The CERT program estimates upcoding causes ~$656 million/year in improper IPPS payments.

Rare diagnoses with high claim amounts (signal type a) are less discriminating because low base rates make statistical anomaly detection unreliable. Known commonly falsified codes (signal type d) serve as filters for deeper investigation but evolve quickly and are gamed. The **optimal approach combines (b) and (c)**: flag claims where the DRG severity level is inconsistent with clinical variables, and where LOS is short relative to the assigned DRG weight.

### Hierarchical ICD embeddings significantly outperform flat features

This is not a marginal improvement. Matta, Suesserman et al. (2023, Deloitte, IEEE ICMLA) compared three ICD-10 representations on **36 million Medicare outpatient claims**:

- Sparse multi-hot encoding (flat frequency, ~1,882 dimensions): R² = 0.127
- BioSentVec word embeddings of ICD descriptions (200–400 dims): R² = 0.229
- **Graph embeddings via node2vec incorporating ICD-10 hierarchy + co-occurrence (32–64 dims): R² = 0.253**

The graph embedding approach — which explicitly encoded parent-child edges in the ICD-10 tree — delivered a **99% improvement** over sparse encoding. Primary diagnosis code alone accounted for **48% of feature importance** in their gradient boosted tree model. Other academic work confirms this: GRAM (Choi et al., KDD 2017) supplemented embeddings with ICD ontology hierarchy for 10% accuracy gains on rare diseases; Finch et al. (2021, *JAMIA Open*) showed hierarchical Word2Vec training improved both clustering and classification.

That said, for **Kaggle-style provider-level** fraud detection where ICD codes are already mapped to ~1,000 categories and labels are at the provider (not claim) level, flat aggregation features (unique ICD codes per provider, mean claim amount per ICD, DRG distribution skewness) remain competitive as a baseline. Hierarchical embeddings provide the most lift at **claim-level** fraud detection with the full 68,000+ ICD-10 vocabulary.

### ICD chapters and codes most associated with fraud

The highest-risk areas concentrate in a handful of chapters. **Chapter 1 (Infectious diseases, A00–B99)** dominates: sepsis code A41.9 maps to MS-DRG 871, the #1 most-billed DRG at **$7.4 billion** in FY2019 Medicare spend. Chapter 10 (Respiratory, J00–J99) follows with pneumonia codes J13–J18, where adding a secondary MCC diagnosis elevates reimbursement by more than 2×. Chapter 14 (Genitourinary, N00–N99) shows the **highest percentage of potential upcoding** among all conditions per Lorence and Ibrahim's research, particularly acute kidney failure (N17.x). Chapter 9 (Circulatory, I00–I99) heart failure codes (I50.x) and Chapter 4 (Endocrine, E00–E89) diabetes codes (E11.x) are frequently used as secondary MCC "boosters." OIG found that **over 50% of highest-severity stays achieved that level due to only one secondary diagnosis** — typically acute respiratory failure (J96.0x), sepsis (A41.x), acute kidney injury (N17.x), or malnutrition (E40–E46).

---

## 2. Fraud rings are hub-and-spoke, and provider-beneficiary networks carry the strongest signal

### Concentrated control with sometimes distributed fronts

DOJ case studies reveal a consistent pattern: **hub-and-spoke organization** where a central entity (clinic owner, billing company, or criminal organization) controls the scheme, with physicians, patient brokers, labs, and pharmacies as spokes. Operation Gold Rush (2025) dismantled a Russia-based transnational organization that purchased dozens of DME companies as shell fronts to submit **$10.6 billion** in fraudulent claims — distributed fronts, centralized control. The Arizona wound care scheme (2024) was more concentrated: one couple submitting $1.2 billion for unnecessary amniotic grafts. For hospitalization claims specifically, the concentrated pattern dominates — a single hospital or small cluster plus a few complicit doctors filing abnormally many claims for the same high-reimbursement diagnoses.

In group corporate insurance, **"flocking"** is a distinctive signal: many employees from the same employer all using the same provider, indicating potential provider recruiting or member collusion. This hospital × employer concentration is a feature worth engineering directly.

### The three most discriminating interaction features

**Hospital × diagnosis distribution** is the strongest. Shekhar et al. represent each hospital by its DRG frequency distribution and use subspace outlier detection to find hospitals anomalous in local subsets of codes — critical because fraud is often concentrated in a few codes, not spread across all. Compute diagnosis concentration entropy per hospital; fraudulent hospitals have lower entropy. The z-score of `fraud_rate(hospital, diagnosis_category)` against global or peer-group averages is highly discriminating.

**Doctor × disease category anomaly** is the second-strongest. Bauder and Khoshgoftaar (2016) predicted provider specialty from billing patterns using Multinomial Naïve Bayes; **mismatch between predicted and actual specialty** flags potentially fraudulent providers. Janssens et al. (BMC Medical Informatics, 2024) found general practitioners anomalously billing for orthopedic surgeries using co-occurrence matrix embeddings with SVD. Key features: doctor diagnosis Herfindahl index, specialty-diagnosis probability score, and per-diagnosis claim amounts versus peer averages.

**Hospital × doctor co-occurrence** completes the triad. Build a bipartite graph weighted by claim count and compute an exclusivity score: `claims(doctor D, hospital H) / total claims(doctor D)`. High exclusivity at a suspicious hospital is a red flag. Community detection (Louvain algorithm) on the hospital-doctor graph reveals tight-knit cliques sharing abnormally many patients.

### Graph features that work in practice

Kim et al. (IEEE Access, 2023) tested graph centralities from bipartite provider-beneficiary and provider-physician networks for Medicare fraud detection. **Provider-beneficiary graph centralities improved recall by 24 percentage points** and F1 by 14 points over GNN baselines. The four centrality metrics — degree, eigenvector, closeness, and PageRank — all contributed. Baker, Skinner et al. (2023, *Social Science & Medicine*) introduced the Bipartite Mixture Index (BMIX) measuring patient-sharing intensity; the top-5 BMIX regions (Miami, Las Vegas, Fort Lauderdale, Houston, Los Angeles) were all known fraud hotspots. For heterogeneous graphs, recent work (MHGSL, Heliyon 2024; MHAMFD, BMC Medical Informatics 2023) uses meta-paths like Patient→Department→Patient and Patient→Medicine→Patient with multi-channel GCN and attention mechanisms, outperforming flat feature approaches on real Chinese health insurance data.

### Provider fraud dominates group health insurance

SmartLight Analytics (2021) analyzed claims from multiple employers with 100,000+ combined members in self-funded plans and found **provider-driven billing schemes account for 60–70% of wasteful spend**. The HHS-OIG/DOJ HCFAC FY2023 report contained **zero beneficiary fraud convictions** — all were providers. A scoping review in *Health & Justice* (2021) across 67 studies identified 13 manifestations of provider fraud versus 7 of beneficiary fraud. While individual patient fraud instances may be numerically more frequent in some datasets, provider fraud is systematic, recurring, and vastly higher-dollar per incident.

---

## 3. Thirty-day and ninety-day windows dominate, and the inception signal is real

### Validated window sizes

The empirical literature converges on **30-day and 90-day rolling windows** as the most informative for velocity features. Hess et al. (2022, *Risks*, MDPI) — working directly in the corporate insurance domain — explicitly construct rolling 90-day claim counts per insured, claimant, and payee as core features. The IJERET paper (2025) uses 30-day sliding windows for LSTM autoencoders and temporal convolutional networks, computing rolling means, cumulative sums, and inter-arrival times. DataRobot's insurance claims triage framework recommends building separate models at **FNOL, 30, 60, 90, and 180 days** as information accumulates.

The optimal feature set uses multiple windows simultaneously — not just a single window. The **velocity ratio** between windows is particularly powerful: `claim_count_30d / claim_count_90d` detects spikes, while `claim_count_7d / claim_count_30d` catches acute bursts. For each entity (employee, hospital, hospital × employer pair), compute claim counts, total amounts, distinct counterparties, and distinct diagnoses across 7, 30, 60, 90, 180, and 365-day windows. Inter-claim interval features (mean, minimum, standard deviation, coefficient of variation) complement count-based velocity.

### The policy inception signal is well-documented but fragmented

The signal goes by multiple overlapping names across disciplines: **adverse selection** in health economics, **inception-to-claim timing** in actuarial practice, **early claim fraud** in SIU operations, and **misrepresentation at inception** in legal contexts. The Umbrex analytics framework explicitly defines `Inception-to-Claim Timing = First Claim Date – Policy Inception Date` as a standard metric and recommends flagging policies with very short timing for additional review.

India's IRDAI mandates a **30-day initial waiting period** for individual health policies specifically because early claims are high-risk — a regulatory acknowledgment of this signal. Critically, **corporate/group health plans often waive waiting periods entirely**, making the inception signal even more important for fraud detection in this context because the contractual safeguard is absent. Key features to engineer: `days_since_policy_inception` at claim time, binary flags for claims within 30/60/90 days of inception, and the ratio `claim_amount_first_30d / claim_amount_policy_year`.

### Fraud rings show lifecycle patterns

Research consistently shows fraud rings exhibit a **characteristic lifecycle arc**: setup, ramp-up, peak exploitation, and wind-down or dissolution. Within the exploitation phase, temporal clustering is the norm. The DOJ June 2024 action charged 193 people across 32 states for $1.6 billion in fraud using telehealth platforms and shell companies. Ethos Risk (2025) advises looking for "claim clustering across locations and identical high-cost services." For group corporate insurance specifically, watch for: onboarding bursts (claims concentrated shortly after enrollment), provider-employee clustering (multiple employees from the same company at the same hospital within narrow windows), and end-of-policy-year acceleration before renewal or expiration.

---

## 4. TTD and Hospital Cash are the most fraud-prone add-on covers

### Stacking is a recognized fraud pattern, but cascading is structurally normal

The RGA/MIB 2024 US Life Insurance Fraud Survey ranks **stacking — pursuing multiple policies or benefits to increase coverage — as the second-highest fraud concern** among carriers (3.0 on a 5-point scale, behind medical misrepresentation at 4.0). In workers' compensation, working while collecting TTD benefits is specifically called "double dipping" and constitutes the **primary category of fraud referrals** that leads to criminal prosecution.

However, in group personal accident products, legitimate multi-cover claims are **structurally embedded in the product design**. An employee has an accident, is hospitalized (Accidental Hospitalization), receives daily allowance (Hospital Cash), is unable to work post-discharge (TTD), and incurs medical expenses — triggering four covers from one incident. The fraud signal is not the combination per se but **anomalous patterns within combinations**: disproportionate hospitalization duration relative to diagnosis severity, TTD duration mismatched with injury type, or mutually exclusive covers claimed simultaneously (e.g., PTD and TTD on the same event, which is medically contradictory since PTD implies permanence while TTD implies temporary).

### TTD is the highest-fraud add-on, followed by Hospital Cash

**TTD (Total Temporary Disability)** is the most fraud-prone cover based on transferable workers' compensation evidence. It involves ongoing weekly payments (typically 1–2% of sum insured per week, up to 100 weeks), relies on subjective assessment of inability to work, and requires costly surveillance to verify. The NH Insurance Department Fraud Unit reports that TTD fraud "typically leads to stronger criminal cases" among workers' comp referrals.

**Hospital Cash Allowance** ranks second. It pays a fixed daily amount per hospitalization day, creating a direct incentive for unnecessary or prolonged stays. Insurance Samadhan identifies "conversion of an outpatient or day care procedure into hospital admission" and "extended length of stay keeping in view the high sum insured" as specific Indian health insurance fraud patterns — both directly exploit Hospital Cash benefits. Because Hospital Cash is a fixed indemnity (pays regardless of actual expenses), the payout is disconnected from actual cost, making it particularly attractive for fraud.

**PPD** is moderately fraud-prone due to medical judgment in disability percentage assignment. **PTD and Death benefits** involve higher verification (disability assessments, death certificates) and tend toward hard fraud (completely fabricated) rather than soft fraud (exaggerated), making them lower-frequency but higher-impact when they occur. **Accidental Hospitalization** is vulnerable to staged accidents and hospital collusion, particularly in Indian markets.

### Recommended feature engineering for add-on covers

For tree-based models (XGBoost, LightGBM, CatBoost — the dominant algorithms), treat each add-on as a binary feature and engineer: `num_addon_covers_claimed` (sum of all binary flags), interaction terms (`TTD_AND_HospitalCash`, `Hospitalization_AND_TTD_AND_HospitalCash`), and **cover-to-severity ratios** combining add-on count with diagnosis codes to detect disproportionate claiming. A 2025 *Computational Economics* paper found that "insurance coverage type" was the **most significant feature** in their fraud detection model with SHAP values of 0.2–0.4. Association rule mining (Apriori) on benefit combinations can discover rare co-occurrence patterns with high confidence that indicate fraud.

No published study provides direct base rates for multi-cover versus single-cover fraud in group health insurance. As a practical proxy, if multi-cover (3+ covers) claims represent ~15–20% of total claims, the fraud enrichment ratio in this segment is likely 3–5×, consistent with the general principle that higher-value, more complex claims carry higher fraud incidence.

---

## 5. Provider-level aggregation gives the strongest signal, with published feature rankings to prove it

### Provider ID is the primary aggregation entity

The canonical Kaggle benchmark — "Healthcare Provider Fraud Detection Analysis" (Rohit Anand Gupta, ~5,410 providers, 506 fraudulent) — places fraud labels at the **provider level**, and the standard pipeline aggregates all claim-level and patient-level data up to provider. Johnson and Khoshgoftaar (2023, *SN Computer Science*) enriched CMS Medicare datasets with 47–58 provider summary features and found these significantly outperformed original datasets across all metrics. Shekhar et al.'s hospital-level anomaly detection validated against DOJ data delivered 8× lift. For group corporate insurance, the hierarchy is clear: **provider/hospital first, physician second, employer group third, individual claimant fourth**.

### Published feature importance rankings

Across the Kaggle Healthcare Provider Fraud Detection dataset, the consistent top features are:

1. **PerProviderAvg_InscClaimAmtReimbursed** — average reimbursement per provider (strongest single feature)
2. **InscClaimAmtReimbursed** — raw claim reimbursement amount
3. **PerAttendingPhysicianAvg_InscClaimAmtReimbursed** — average reimbursement per attending physician
4. **PerOperatingPhysicianAvg_InscClaimAmtReimbursed** — average per operating physician
5. **PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed** — average per admission diagnosis

SHAP analysis from Johnson and Khoshgoftaar's enriched Medicare datasets showed **both original aggregated features and new provider-summary features each contributed 9 of the top 20 most important features**. In Deloitte's payment outlier detection model, primary diagnosis code accounted for 48% importance, secondary diagnosis 24%, business practice state 13.5%, and tertiary diagnosis 13.3%. Patient demographics (age, gender) contributed only 0.4%.

Provider-aggregated financial features dominate across virtually all published analyses, with physician-level features second, diagnosis/procedure frequencies third, and patient demographics fourth. The optimal approach computes patient-level features and then **re-aggregates to provider level** — for example, "average age of patients per provider," "average chronic condition count per provider," and "proportion of patients from different states per provider."

### Employer/group level serves a different purpose

Employer-group-level aggregation is most useful for **eligibility fraud** (ghost employees, non-employees on policy, ineligible dependents) rather than billing or claims fraud. Highmark specifically categorizes "Group Fraud" as a distinct fraud type with different detection methods. For billing fraud, the employer dimension is useful primarily as an interaction feature: `employer × hospital claim concentration` (what percentage of an employer's claims go to a single hospital) detects flocking patterns characteristic of organized fraud.

### Key datasets and benchmarks

- **Kaggle Healthcare Provider Fraud Detection** (~5,410 providers, provider-level labels, inpatient/outpatient/beneficiary data)
- **CMS Medicare Part B/D Public Use Files** (67M+ Part B records, 172M+ Part D records, 2013–2019)
- **OIG LEIE** (List of Excluded Individuals/Entities — real fraud labels at provider NPI level)
- **Chinese municipal health insurance datasets** (used in MHAMFD, MHGSL — non-public, but papers publish results)
- **World Bank India Case Study** (fraud in government-sponsored health insurance: Gujarat, Maharashtra, Tamil Nadu, Telangana)

For GBM models on these datasets, typical AUC ranges from **0.77 to 0.94** depending on feature engineering depth and dataset quality. CatBoost achieves AUC ~0.882 on Medicare data with healthcare provider state as a categorical feature (Hancock and Khoshgoftaar, 2020). Stacking ensembles on the Kaggle dataset achieve the highest published scores.

---

## Conclusion: what this means for building a fraud model on group corporate hospitalization data

The evidence points to a clear implementation strategy. **Start with provider-level aggregated features** — per-hospital and per-doctor averages of claim amounts, claim counts, DRG distributions, and diagnosis code entropy. These consistently top feature importance rankings and capture the dominant fraud type (provider-side, 60–70% of waste). Layer in **diagnosis-procedure mismatch features**: compare billed DRG severity against clinical variables, flag short LOS for high-weight DRGs, and compute ICD-to-procedure consistency scores. If working with the full ICD-10 vocabulary, **hierarchical graph embeddings deliver nearly 2× the predictive power** of flat frequency features — but flat features remain a strong baseline for provider-level classification.

For velocity features, **30-day and 90-day rolling windows** are the empirically validated defaults. Compute velocity ratios between windows for spike detection, and engineer the policy inception signal explicitly — it is well-documented and especially important for group plans that waive waiting periods. Among add-on covers, **TTD and Hospital Cash** deserve the most scrutiny; engineer a `num_addon_covers_claimed` count feature and interaction terms between specific cover combinations. Finally, consider graph features: provider-beneficiary bipartite centralities improved recall by 24 percentage points in Medicare research, and community detection on hospital-doctor graphs surfaces fraud ring structures that flat features miss entirely.