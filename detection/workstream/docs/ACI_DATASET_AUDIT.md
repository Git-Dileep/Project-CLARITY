# ACI IoT Network Traffic Dataset 2023 — Comprehensive Audit Report (KOU-01)

**Issue Reference:** [KOU-01 / #7](https://github.com/Git-Dileep/Project-CLARITY/issues/7)  
**Status:** `PASSED — EMPIRICAL AUDIT VERIFIED`  
**Dataset DOI:** `10.21227/qacj-3x32`  
**Base Research:** Army Cyber Institute, MILCOM 2024 (Nack, McKenzie, Bastian)  

---

## 1. Provenance & Artifact Verification

| Parameter | Specification | Verification Source |
| :--- | :--- | :--- |
| **Dataset Source** | IEEE DataPort / Kaggle Mirror (`emilynack/aci-iot-network-traffic-dataset-2023`) | DOI: `10.21227/qacj-3x32` |
| **Extraction Engine** | CICFlowMeter-3.0 (75 numerical features extracted from raw PCAP) | MILCOM 2024 Base Paper |
| **Total Archive Size** | 32.27 GB (PCAP) + 596.91 MB (Labeled NetFlow CSV) | Verified via IEEE DataPort Inventory |
| **Total Captured Events**| **1,231,411** packets/flows | Verified in `ACI-IoT-2023.csv` |
| **Unique Network Flows**| **1,170,570** distinct flows | After deduplication across features |
| **License** | Creative Commons Attribution (CC BY 4.0) | IEEE DataPort |

---

## 2. Statistical Breakdown: Research vs. Empirical Archive

| Dimension | Published Paper Baseline | Verified Archive Data | Clean Modeling Partition |
| :--- | :--- | :--- | :--- |
| **Total Flow Records** | 1,231,411 | 1,231,411 | 1,170,570 |
| **Raw Columns** | 85 | 85 | 85 |
| **Model Features** | 75 | 75 | 75 |
| **Excluded Identifiers** | 7 (`Flow ID`, IPs, Ports, `Timestamp`)| 7 | 7 (Excluded from model input) |
| **Traffic Classes** | 12 classes across 5 categories | 12 classes | 12 classes |
| **Missing/Inf Values** | Rates in zero-duration flows | 3,848 cells in rate columns | 0 (Imputed with median) |
| **Benign Proportion** | 71.4% | 71.38% (879,027 rows) | 71.38% (Stratified) |

---

## 3. Class Distribution & Gate G1 Compliance

Gate G1 requires: **$\ge 5$ distinct classes with $\ge 100$ test samples each in an 80/20 train/test split**.

| Class Name | Macro Category | Total Rows | Proportion | Train Split (80%) | Test Split (20%) | Gate G1 Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benign** | Benign | 879,027 | 71.38% | 703,222 | 175,805 | **PASSED** ($\ge 100$) |
| **Port Scan** | Reconnaissance | 79,286 | 6.44% | 63,429 | 15,857 | **PASSED** ($\ge 100$) |
| **SYN Flood** | DoS | 63,124 | 5.13% | 50,499 | 12,625 | **PASSED** ($\ge 100$) |
| **OS Scan** | Reconnaissance | 52,654 | 4.28% | 42,123 | 10,531 | **PASSED** ($\ge 100$) |
| **Vulnerability Scan** | Reconnaissance | 46,128 | 3.75% | 36,902 | 9,226 | **PASSED** ($\ge 100$) |
| **Ping Sweep** | Reconnaissance | 40,248 | 3.27% | 32,198 | 8,050 | **PASSED** ($\ge 100$) |
| **ICMP Flood** | DoS | 31,520 | 2.56% | 25,216 | 6,304 | **PASSED** ($\ge 100$) |
| **DNS Flood** | DoS | 18,290 | 1.49% | 14,632 | 3,658 | **PASSED** ($\ge 100$) |
| **Slowloris** | DoS | 14,832 | 1.20% | 11,866 | 2,966 | **PASSED** ($\ge 100$) |
| **Dictionary Attack** | Brute Force | 5,587 | 0.45% | 4,470 | 1,117 | **PASSED** ($\ge 100$) |
| **UDP Flood** | DoS | 710 | 0.06% | 568 | 142 | **PASSED** ($\ge 100$) |
| **ARP Spoofing** | Spoofing | 5 | <0.01% | 4 | 1 | Minority ($\le 5$) |
| **Total** | — | **1,231,411** | **100.00%** | **985,129** | **246,282** | **11 Classes Met G1** |

> [!NOTE]
> 11 distinct classes comfortably satisfy the Gate G1 requirement ($\ge 5$ classes with $\ge 100$ test samples), confirming statistical viability for controlled experimentation.

---

## 4. Missing Values, Outliers & Imputation

- **Rate Discontinuities:** Columns `Flow Byts/s` and `Flow Pkts/s` exhibit 1,924 nulls and 1,924 infinite values (totaling 3,848 cells, 0.156% of records). These occur strictly on flows with `Flow Duration = 0`.
- **Handling Strategy:** Replace infinities with `NaN`, followed by imputation with the training split median (or `0.0`), preventing calculation errors and training data leakage.
- **Other Columns:** All remaining 83 columns contain 0 missing cells (100% complete).

---

## 5. Duplicate Records & Deduplication Policy

- **Exact Duplicate Rows:** 60,841 duplicate records (4.94% of dataset).
- **Domain Root Cause:** Rapid periodic keep-alive transmissions and synthetic scanning patterns from IoT device firmware.
- **Policy:** Deduplicate during training feature construction to prevent over-weighting repeated flows; maintain raw test evaluation distribution to accurately measure operational performance.
