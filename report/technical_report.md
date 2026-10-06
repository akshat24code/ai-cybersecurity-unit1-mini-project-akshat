# TECHNICAL RESEARCH REPORT: AI-DRIVEN CYBER THREAT AWARENESS AND DETECTION PROTOTYPE

**Course:** AI in Cyber Security Lab (`ENSP355`) — Unit 1 Mini Project  
**Institution:** K.R. Mangalam University — Faculty of Engineering & Technology  
**Faculty Evaluator:** Dr. Amar Saraswat  
**Author / Student:** Cybersecurity Research Assistant  
**Repository:** `ai-cybersecurity-unit1-mini-project-akshat`  
**Date:** October 2026  

---

## 1. Problem Overview
Modern organizational networks process gigabytes of telemetry per second, exposing infrastructure to sophisticated, adaptive cyber attacks including Denial of Service (DoS), Remote Code Execution (Exploits), Reconnaissance scans, Backdoors, and Fuzzing attacks. Traditional Intrusion Detection Systems (IDS) rely heavily on static signature databases (e.g. Snort rules), which are inherently reactive and ineffective against zero-day vulnerabilities or obfuscated traffic.

Artificial Intelligence (AI) and Machine Learning (ML) empower Next-Generation Intrusion Detection Systems (NIDS) to learn baseline behavioral patterns from raw packet characteristics, identifying subtle anomalies and malicious activity in near real-time. This project presents an end-to-end AI prototype developed on the UNSW-NB15 dataset, evaluating classical machine learning models (Logistic Regression, Decision Trees, Random Forests) against Deep Learning neural network architectures (Multi-Layer Perceptron), while rigorously evaluating operational latency, detection coverage, false positive risks, and ethical considerations.

---

## 2. Dataset Description
The **UNSW-NB15 dataset** was created by the Cyber Security Lab at the University of New South Wales (UNSW Canberra) using the IXIA PerfectStorm tool. It synthesizes real modern network traffic containing synthetic attack behaviors and normal user activity.

### 2.1 Dataset Parameters
- **Predefined Training Set (`UNSW_NB15_training-set.csv`)**: 175,341 records
- **Predefined Testing Set (`UNSW_NB15_testing-set.csv`)**: 82,332 records
- **Features**: 45 columns (including transaction duration, protocol, service, state, byte rates, packet counts, TTL values, TCP window sizes, HTTP payload lengths, and aggregated flow count metrics).
- **Target Variable**: `label` (Binary: `0` = Normal Traffic, `1` = Cyber Attack / Malicious Flow).
- **Attack Metadata**: `attack_cat` (9 distinct attack types: *Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms*).

---

## 3. Methodology & Research Architecture
To prevent data contamination and guarantee reproducible research standards, the pipeline follows strict data hygiene principles:

```text
+-----------------------------------------------------------------------+
|                 UNSW-NB15 Predefined Data Pipeline                     |
+-----------------------------------------------------------------------+
|  Training Data (175,341)              Untouched Testing Data (82,332)   |
|          |                                      |                     |
|  [Data Inspection & Audit]                      |                     |
|          |                                      |                     |
|  [Fit Preprocessor]                             |                     |
|  - Drop `id` & `attack_cat`                     |                     |
|  - LabelEncode Categoricals                     |                     |
|  - StandardScaler Numericals                     |                     |
|          |                                      |                     |
|  [Transform Training Features]        [Transform Testing Features]    |
|          |                                      |                     |
|  +-------+-------+                              |                     |
|  | Supervised ML |                              |                     |
|  | - Decision Tree                              |                     |
|  | - Logistic Reg|                              |                     |
|  | - Random Forest                             |                     |
|  +-------+-------+                              |                     |
|  | Deep Learning |                              |                     |
|  | - MLP (128x64)|                              |                     |
|  +-------+-------+                              |                     |
|          |                                      |                     |
|  [Trained Models] -------------------> [Evaluate on Test Set]        |
|                                                 |                     |
|                                       [Generate Metrics & ROC]        |
|                                       [Threshold & Bias Audit]        |
+-----------------------------------------------------------------------+
```

### Key Methodological Safeguards:
1. **No Retrospective Resplitting**: The official training set is used strictly for model fitting, and the official test set remains completely untouched until final evaluation.
2. **Target Leakage Prevention**: The `attack_cat` column indicates attack types and is removed from input feature matrices ($X$) during binary training (`label`).
3. **Train-Only Scaling**: Standard Scaling parameters ($\mu, \sigma$) and Label Encoding mappings are computed **exclusively on the training set** to avoid test data leakage.

---

## 4. Data Preprocessing & Feature Engineering

### 4.1 Feature Removal & Justification
- **`id`**: Dropped. This is an arbitrary primary key index assigned by the dataset creators. Including it would create shortcut learning artifacts.
- **`attack_cat`**: Dropped from feature set. Used exclusively for post-hoc ethical bias and per-category recall auditing.

### 4.2 Encoding & Scaling
- **Categorical Columns (`proto`, `service`, `state`)**: Encoded using `LabelEncoder`.
- **Numerical Features (40 attributes)**: Normalized using `StandardScaler` ($\mu=0, \sigma=1$) to stabilize gradient updates for Logistic Regression and Deep Learning MLP models.

---

## 5. Exploratory Data Analysis (EDA) & Security Observations

### 5.1 Class Distribution
- **Training Set Target Split**: Normal Traffic = 56,000 (31.9%), Attack Traffic = 119,341 (68.1%).
- **Testing Set Target Split**: Normal Traffic = 37,000 (44.9%), Attack Traffic = 45,332 (55.1%).

### 5.2 Attack Category Breakdown
In the training data, malicious traffic is distributed across 9 attack vectors:
1. **Generic** (58,871 records - 33.6%): Automated rule-based attack patterns.
2. **Exploits** (44,525 records - 25.4%): Known software vulnerability exploits.
3. **Fuzzers** (24,246 records - 13.8%): Random boundary testing packets.
4. **DoS** (16,353 records - 9.3%): Denial of Service flooding.
5. **Reconnaissance** (13,987 records - 8.0%): Port scanning and host discovery.
6. **Analysis** (2,677 records - 1.5%): Web traffic probes and port scans.
7. **Backdoor** (2,329 records - 1.3%): Stealth remote access channels.
8. **Shellcode** (1,511 records - 0.9%): Executable payload injections.
9. **Worms** (174 records - 0.1%): Self-propagating malware (Rare attack class).

### 5.3 Key Feature Security Indicators
- **Source Time-To-Live (`sttl`)**: Normal traffic predominantly exhibits TTL values around 31 or 60, whereas attack packets display extreme TTL values (e.g. 254), corresponding to spoofed packet origins or automated scanners.
- **Connection Counts (`ct_state_ttl`, `ct_srv_dst`)**: High concurrent connection rates to identical destination services within short time windows serve as a primary indicator for DoS and Port Scanning activity.

---

## 6. Model Implementations

### 6.1 Machine Learning Models (Task 3)
1. **Logistic Regression (Linear Baseline)**: Configured with L2 penalty, `max_iter=1000`, solving via `lbfgs`.
2. **Decision Tree Classifier (Primary Task 3 Model)**: Non-linear decision tree constrained to `max_depth=12` to prevent overfitting.
3. **Random Forest Classifier (Advanced Ensemble)**: Ensemble of 100 estimator trees with `max_depth=15`.

### 6.2 Deep Learning Prototype (Task 4)
- **Architecture**: `MLPClassifier` with 2 hidden layers:
  - Input Layer $\to$ Hidden Layer 1: **128 Units (ReLU)**
  - Hidden Layer 1 $\to$ Hidden Layer 2: **64 Units (ReLU)**
  - Output Layer: **1 Sigmoid Neuron**
- **Optimizer**: Adam ($\alpha = 0.0001$, initial learning rate $\eta = 0.001$).
- **Batch Size**: 256 packets per mini-batch.
- **Regularization**: Early stopping enabled with `n_iter_no_change=5` on a 10% internal validation split.

---

## 7. Empirical Results & Comparative Benchmark

Evaluated on the **official untouched testing dataset (82,332 records)**:

| Model | Accuracy | Precision | Recall (TPR) | F1-Score | ROC-AUC | FPR | Training Time | Inference Latency (10k packets) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.34% | 74.88% | 96.76% | 0.8442 | 0.9336 | 39.78% | 2.64s | **0.90 ms** |
| **Decision Tree** | 85.36% | 80.72% | 96.46% | 0.8789 | 0.9625 | 28.23% | **2.05s** | 1.62 ms |
| **Random Forest** | **86.47%** | 80.81% | **98.92%** | **0.8895** | **0.9831** | 28.78% | 3.17s | 14.58 ms |
| **Deep Learning (MLP)** | 85.71% | **80.89%** | 96.94% | 0.8819 | 0.9686 | **28.06%** | 79.81s | 11.66 ms |

---

## 8. Cybersecurity Interpretation & Feasibility Analysis

### 8.1 Why Recall Trumps Accuracy in Intrusion Detection
In network security, a **False Negative (FN)** represents an undetected attack passing through the firewall, potentially allowing lateral movement, ransomware execution, or data exfiltration. Random Forest achieved a **98.92% Recall (Detection Rate)**, missing only 1.08% of malicious packets, making it superior for threat mitigation.

### 8.2 Practical Feasibility: ML vs DL
- **Training Efficiency**: Decision Tree trained in **2.05 seconds**, whereas MLP Deep Learning required **79.81 seconds** (~39x longer) for comparable accuracy (85.36% vs 85.71%).
- **Inference Latency**: Decision Tree achieved **1.62 ms per 10,000 packets**, making it lightweight and ideal for deployment on high-throughput edge routers and FPGA firewalls.

---

## 9. Ethical Analysis & Operational Threshold Simulation (Task 5)

### 9.1 Scenario: False Positive Impact & Legitimate User Disruption
At the default decision threshold $\tau = 0.50$, the models exhibit a **False Positive Rate of ~28%**, incorrectly flagging 10,648 legitimate normal connections as cyber threats. In a corporate environment, automatically blocking these IPs creates an **Operational Denial of Service**, locking out legitimate staff and overloading SOC helpdesks.

### 9.2 Decision Threshold Tuning ($\tau$)
By adjusting the classification threshold from $\tau = 0.50 \to 0.82$:
- **False Positive Rate (FPR)** drops drastically from **28.78% down to 5.57%** (a 5x reduction in false alarms).
- **Precision** increases from **80.81% to 95.22%**.
- **Recall** remains robust at **90.67%**.

### 9.3 Dataset Bias & Rare Attack Audit
Per-attack category evaluation on the test set reveals:
- **Frequent Attacks (Generic, Exploits, DoS, Recon, Backdoor, Worms)**: Achieved **99.5% to 100.0% Detection Rate**.
- **Fuzzers**: Achieved **92.73% Detection Rate** (441 missed packets due to subtle, randomized packet payloads mimicking normal noise).

---

## 10. Limitations & Future Work
1. **Dataset Age**: UNSW-NB15 reflects 2015 threat signatures; future work should benchmark models on modern datasets (e.g. CICIDS2017 / TON_IOT).
2. **Multi-Class Classification**: Extending the binary pipeline to multi-class prediction to output specific attack category signatures directly.

---

## 11. Conclusion
This project successfully implemented an end-to-end AI cyber threat awareness and detection prototype. Supervised ML models (Decision Trees, Random Forests) demonstrated high detection coverage (**98.92% Recall**) with extremely fast inference speeds (**1.62 ms per 10k packets**). Decision threshold optimization ($\tau = 0.82$) proved essential to mitigate false positive user disruptions while maintaining defensive security posture.

---

## 12. References
1. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*. IEEE Military Communications and Information Systems Conference (MilCIS), 1-6.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
3. NIST Special Publication 800-94, *Guide to Intrusion Detection and Prevention Systems (IDPS)*.
