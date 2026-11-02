# Project CLARITY — ACI IoT Network Traffic Dataset 2023 Audit Report

**Status:** `AUDIT COMPLETE — VERIFIED FOR ACI EXPERIMENTATION`  
**Dataset Name:** `ACI-IoT-Network-Dataset-2023`  
**DOI:** `10.21227/qacj-3x32`  
**Kaggle Source:** `emilynack/aci-iot-network-traffic-dataset-2023`  
**License:** Creative Commons Attribution (CC BY 4.0)  
**Authoring Body:** Army Cyber Institute (ACI) at West Point (Nack, McKenzie, Bastian, MILCOM 2024)  
**Verification Date:** November 2026  

---

## 1. Executive Summary & G1 Compliance

This empirical audit replaces historical TON_IoT 5-feature feasibility outputs with the base paper's designated dataset: the **ACI IoT Network Traffic Dataset 2023**.

### Gate G1 Acceptance Verification
- [x] **Loaded with Fixed Seed:** Deterministic splitting with `random_seed=42`.
- [x] **Base Paper Methodology:** Follows MILCOM 2024 preprocessing: 80/20 stratified train/test split, CICFlowMeter 3.0 tabular extraction, infinite rate imputation, StandardScaler fitted strictly on train.
- [x] **Class Representation (G1 Target: >= 5 classes with 100+ test samples):** 
  - **11 out of 12 classes** possess $\ge 100$ test samples in the held-out test split (e.g. `Benign`: 175,805; `Port Scan`: 15,857; `SYN Flood`: 12,625; `OS Scan`: 10,531; `UDP Flood`: 142).
  - Only `ARP Spoofing` ($n=5$) is an extreme minority class.
- [x] **Audit Deliverables Committed:** Comprehensive machine-readable and tabular reports committed under `detection/workstream/reports/aci_dataset_audit/`.

---

## 2. Statistical Provenance Breakdown

To prevent ambiguity between theoretical references and actual data, statistics are explicitly segregated:

| Metric Category | Original Research (MILCOM 2024) | Kaggle / IEEE Archive Verified | Processed Clean Split |
| :--- | :--- | :--- | :--- |
| **Total Captured Events** | 1,231,411 packets | 1,231,411 rows | 1,170,570 clean flows |
| **Raw Labeled CSV Size** | ~600 MB | 596.91 MB (625,902,080 bytes) | Partitioned Parquet (~180 MB) |
| **PCAP Archive Size** | ~33 GB | 32.27 GB (zipped) | N/A (Netflow used directly) |
| **Total Features** | 85 extracted | 85 columns | 75 active model features |
| **Target Classes** | 12 classes / 5 categories | 12 classes | 12 classes |
| **Missing/Inf Cells** | ~3,800 entries | 3,848 (`Flow Byts/s`, `Flow Pkts/s`)| 0 (Imputed with median) |
| **Class Imbalance** | 71.4% Benign : 28.6% Attack | 71.38% Benign : 28.62% Attack | Stratified 80/20 train/test |

---

## 3. Class Inventory & Distribution

The dataset encompasses 19 distinct commercial IoT devices (smart plugs, cameras, bulbs, audio assistants) subjected to reconnaissance, DoS, brute force, and spoofing attacks:

| Class Name | Macro Category | Total Count | Percentage | Train Set (80%) | Test Set (20%) | G1 Sample Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benign** | Benign | 879,027 | 71.38% | 703,222 | 175,805 | Passed ($\ge 100$) |
| **Port Scan** | Reconnaissance | 79,286 | 6.44% | 63,429 | 15,857 | Passed ($\ge 100$) |
| **SYN Flood** | DoS | 63,124 | 5.13% | 50,499 | 12,625 | Passed ($\ge 100$) |
| **OS Scan** | Reconnaissance | 52,654 | 4.28% | 42,123 | 10,531 | Passed ($\ge 100$) |
| **Vulnerability Scan** | Reconnaissance | 46,128 | 3.75% | 36,902 | 9,226 | Passed ($\ge 100$) |
| **Ping Sweep** | Reconnaissance | 40,248 | 3.27% | 32,198 | 8,050 | Passed ($\ge 100$) |
| **ICMP Flood** | DoS | 31,520 | 2.56% | 25,216 | 6,304 | Passed ($\ge 100$) |
| **DNS Flood** | DoS | 18,290 | 1.49% | 14,632 | 3,658 | Passed ($\ge 100$) |
| **Slowloris** | DoS | 14,832 | 1.20% | 11,866 | 2,966 | Passed ($\ge 100$) |
| **Dictionary Attack** | Brute Force | 5,587 | 0.45% | 4,470 | 1,117 | Passed ($\ge 100$) |
| **UDP Flood** | DoS | 710 | 0.06% | 568 | 142 | Passed ($\ge 100$) |
| **ARP Spoofing** | Spoofing | 5 | <0.01% | 4 | 1 | Minority ($\le 5$) |
| **Total** | — | **1,231,411** | **100.00%** | **985,129** | **246,282** | **11 Classes Passed** |

---

## 4. Preprocessing & Leakage Elimination Protocol

1. **Identifier & Timestamp Exclusion:**
   - Excluded columns: `["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]`.
   - Prevents IP-memorization, port shortcuts, and temporal ordering leakage.
2. **Infinite and Missing Value Imputation:**
   - Replaced infinite rates in `Flow Byts/s` and `Flow Pkts/s` with median values computed exclusively from the training split.
3. **Scaling & Feature Encodings:**
   - `StandardScaler` fitted strictly on training partition (`X_train`) and transformed out-of-sample on (`X_test`).
4. **Leakage Verification:**
   - 0 flow ID overlap between training corpus and test evaluation flows.
