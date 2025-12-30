# EDA Summary - NSL-KDD Dataset

## Dataset Overview
- **Training:** 125,973 rows × 43 columns
- **Test:** 22,544 rows × 43 columns
- **No missing values**, no negative counts detected

## Class Distribution
- Training: ~53% normal, ~47% attack
- Test: Different attack distribution, contains new attack types

## Key Findings

### 1. Discriminative Features
- `serror_rate`, `rerror_rate` - Error rate features show clear separation
- `count`, `srv_count` - Connection count features differ between classes
- `same_srv_rate` - Service pattern indicator

### 2. Heavy-Tailed Features (Need Robust Scaling)
- `src_bytes`, `dst_bytes` - Extremely large ranges (0 to millions)
- `count` - Right-skewed distribution

### 3. Categorical Features
- `protocol_type`: 3 values (tcp, udp, icmp)
- `service`: 70 unique values → Group to top-20 + "other"
- `flag`: 11 values, some flags (S0, REJ) indicate attacks

### 4. Redundant Features (High Correlation > 0.85)
- `serror_rate` ↔ `srv_serror_rate`
- `rerror_rate` ↔ `srv_rerror_rate`
- `dst_host_serror_rate` ↔ `dst_host_srv_serror_rate`

### 5. Low Variance Features
- `num_outbound_cmds` - Nearly constant (0)
- `is_host_login` - Nearly constant

## Preprocessing Decisions

### Categorical Encoding
- `protocol_type`: One-hot encoding
- `flag`: One-hot encoding
- `service`: Top-20 + "other" grouping, then one-hot

### Numeric Scaling
- **Scaler:** RobustScaler
- **Rationale:** Handles outliers and heavy-tailed distributions

### Features to Drop
- `num_outbound_cmds` (near-zero variance)
- `is_host_login` (near-zero variance)
- `srv_serror_rate` (redundant with serror_rate)
- `srv_rerror_rate` (redundant with rerror_rate)

### Training Strategy
- Fit preprocessing on **normal-only** training data
- Transform all data using the fitted pipeline

## Next Steps

1. Build preprocessing pipeline in `modeling_unsupervised.ipynb`
2. Train baseline models:
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - One-Class SVM
   - Autoencoder
3. Evaluate using ROC-AUC, PR-AUC, Precision@k
4. Optimize and create ensemble
