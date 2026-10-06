# AI-Driven Cyber Threat Awareness & Detection Prototype

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-red.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course:** AI in Cyber Security Lab (`ENSP355`) — Unit 1 Mini Project  
> **Institution:** K.R. Mangalam University — Faculty of Engineering & Technology  
> **Faculty:** Dr. Amar Saraswat  
> **Repository:** `ai-cybersecurity-unit1-mini-project-akshat`  
> **Dataset:** UNSW-NB15 Network Intrusion Detection Benchmark  

---

## Project Overview
Modern organizational networks process gigabytes of telemetry per second, exposing infrastructure to sophisticated, adaptive cyber attacks including Denial of Service (DoS), Exploits, Fuzzers, Backdoors, and automated reconnaissance scans. Traditional static rule-based Intrusion Detection Systems (IDS) often fail against zero-day vulnerabilities or obfuscated traffic.

This project implements an end-to-end Machine Learning (ML) and Deep Learning (DL) prototype to detect malicious network events using the **UNSW-NB15 dataset**. It evaluates supervised linear, rule-based, ensemble, and neural network architectures on standard cybersecurity metrics (Accuracy, Precision, Recall, F1-Score, False Positive Rate, Training Time, and Real-Time Latency).

---

## Repository Architecture
```text
ai-cybersecurity-unit1-mini-project-akshat/
├── .gitignore                      # Git exclusion file (venv, checkpoints, binaries)
├── requirements.txt                # Verified Python environment dependencies
├── README.md                       # Master project documentation & user guide
├── data/                           # Dataset repository
│   ├── UNSW_NB15_training-set.csv  # Official training dataset (175,341 records)
│   ├── UNSW_NB15_testing-set.csv   # Official untouched test dataset (82,332 records)
│   └── UNSW-NB15_features.csv      # Feature dictionary & metadata description
├── src/                            # Modular Python source code
│   ├── validate_env.py             # Automated environment & file verification script
│   ├── data_preprocessing.py       # Data cleaning, encoding & train-only scaling pipeline
│   ├── eda_analysis.py             # Threat landscape analysis & figure generation
│   ├── ml_models.py                # Supervised ML classifiers (Decision Tree, Logistic Reg, RF)
│   ├── dl_models.py                # Deep Learning prototype (scikit-learn MLPClassifier)
│   ├── ethical_sim.py              # Operational threshold tuning & per-attack bias audit
│   ├── create_notebook.py          # Automated Jupyter notebook builder script
│   └── main.py                     # Full end-to-end orchestrator script
├── notebooks/                      # Interactive Jupyter Notebooks
│   ├── UNSW_NB15_Analysis.ipynb    # Executed master notebook with outputs & inline plots
│   └── UNSW_NB15_Analysis.html     # Rendered standalone HTML web report
├── outputs/                        # High-resolution figures, metric CSVs & models
│   ├── 01_class_distribution.png
│   ├── 02_attack_categories.png
│   ├── 03_top_protocols_services.png
│   ├── 04_feature_distributions.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_feature_importance.png
│   ├── 07_ml_vs_dl_comparison.png
│   ├── 08_threshold_tradeoff.png
│   ├── 09_ml_roc_curves.png
│   ├── model_comparison_results.csv
│   ├── ethical_bias_attack_category_performance.csv
│   └── best_threat_detector_rf.joblib
└── report/                         # Technical Research Report
    └── technical_report.md         # Comprehensive 14-section research document
```

---

## MITRE ATT&CK Taxonomy Mapping

| Cyber Attack Category | Training Frequency | MITRE ATT&CK Tactic | Technique ID & Name | Security Description |
| :--- | :---: | :--- | :--- | :--- |
| **Generic** | 58,871 (33.6%) | Impact / Defense Evasion | **T1499 / T1027** | Automated protocol collision & payload obfuscation. |
| **Exploits** | 44,525 (25.4%) | Initial Access / Execution | **T1190** | Software vulnerability exploitation against public servers. |
| **Fuzzers** | 24,246 (13.8%) | Discovery / Defense Evasion | **T1499** | Boundary testing with randomized illegal packet structures. |
| **DoS** | 16,353 (9.3%) | Impact | **T1498** | High-volume connection flooding targeting service availability. |
| **Reconnaissance** | 13,987 (8.0%) | Reconnaissance / Discovery | **T1046** | Active port scanning and network topology mapping. |
| **Analysis** | 2,677 (1.5%) | Reconnaissance | **T1595** | Web crawler probes, directory listing, and HTML injection scans. |
| **Backdoor** | 2,329 (1.3%) | Persistence / Command & Control | **T1071** | Stealth remote access shell channels bypassing firewalls. |
| **Shellcode** | 1,511 (0.9%) | Execution | **T1055** | Malicious binary shellcode payload injection into memory. |
| **Worms** | 174 (0.1%) | Lateral Movement | **T1080** | Self-propagating autonomous network malware. |

---

## Empirical Results & Model Comparison

Evaluated on the **official untouched test dataset (82,332 packet samples)**:

| Model Architecture | Accuracy | Precision | Recall (Detection Rate) | F1-Score | ROC-AUC | False Positive Rate (FPR) | Training Time (s) | Inference Latency (ms/10k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.34% | 74.88% | 96.76% | 0.8442 | 0.9336 | 39.78% | **3.27s** | **1.20 ms** |
| **Decision Tree (Primary ML)** | 85.36% | 80.72% | 96.46% | 0.8789 | 0.9625 | 28.23% | **2.47s** | **2.65 ms** |
| **Random Forest (Ensemble)** | **86.47%** | 80.81% | **98.92%** | **0.8895** | **0.9831** | 28.78% | 3.45s | 16.18 ms |
| **Deep Learning (MLP 128x64)** | 85.71% | **80.89%** | 96.94% | 0.8819 | 0.9686 | **28.06%** | 78.51s | 12.39 ms |

---

## Quickstart & Execution

```bash
# 1. Environment Diagnostic Validation
python src/validate_env.py

# 2. Run Full End-to-End Pipeline
python src/main.py

# 3. Launch Interactive Jupyter Notebook
jupyter notebook notebooks/UNSW_NB15_Analysis.ipynb
```

---

## Ethical Considerations & SOC Operational Tuning
- **Decision Threshold Tuning ($\tau = 0.82$)**: Shifting probability threshold reduces the False Positive Rate from **28.78% down to 5.57%** (a 5x decrease in false alarms), shielding legitimate users while preserving **90.67% Detection Coverage** and **95.22% Precision**.
- **Explainability**: Decision Trees provide transparent, rule-based logic required by SOC analysts to validate mitigation actions during incident response.

---
*Created as part of ENSP355 Lab Assessment, K.R. Mangalam University.*
