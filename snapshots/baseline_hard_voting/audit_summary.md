# Dataset Audit Summary: ACI-IoT-Network-2023

> [!NOTE]
> Audit completed for dataset `ACI-IoT-Network-2023`. Synthetic Mode: `False`.

## Overview
- **Total Rows:** 1,231,411
- **Total Columns:** 85
- **Unique Attack/Traffic Classes:** 12
- **Class Imbalance Ratio (Max:Min):** 88256.4:1
- **Total Missing Values:** 1,009
- **Total Infinite Values:** 2,839
- **Duplicate Files Found:** 0

## Per-Class Distribution
| Class | Record Count | Percentage |
| :--- | :--- | :--- |
| Port Scan | 441,282 | 35.8355% |
| Benign | 329,295 | 26.7413% |
| ICMP Flood | 225,234 | 18.2907% |
| Ping Sweep | 71,928 | 5.8411% |
| DNS Flood | 46,935 | 3.8115% |
| Vulnerability Scan | 39,537 | 3.2107% |
| OS Scan | 37,524 | 3.0472% |
| Slowloris | 18,643 | 1.514% |
| SYN Flood | 13,857 | 1.1253% |
| Dictionary Attack | 6,380 | 0.5181% |
| UDP Flood | 791 | 0.0642% |
| ARP Spoofing | 5 | 0.0004% |

## Metadata & Excluded Columns
- **Suspicious ID/Time/IP Columns:** `Bwd Packets/s, Dst IP, Dst Port, Flow ID, Flow Packets/s, Fwd Act Data Pkts, Fwd Packets/s, Idle Max, Idle Mean, Idle Min, Idle Std, Src IP, Src Port, Subflow Bwd Packets, Subflow Fwd Packets, Timestamp, Total Bwd packets`
- **Configured Preprocessing Exclusions:** `Source IP, Destination IP, Flow ID, Timestamp`