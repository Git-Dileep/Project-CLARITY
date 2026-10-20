# Dataset Audit Summary: Reference-Synthetic-Network

> [!NOTE]
> Audit completed for dataset `Reference-Synthetic-Network`. Synthetic Mode: `True`.

## Overview
- **Total Rows:** 5,000
- **Total Columns:** 24
- **Unique Attack/Traffic Classes:** 6
- **Class Imbalance Ratio (Max:Min):** 6.72:1
- **Total Missing Values:** 0
- **Total Infinite Values:** 0
- **Duplicate Files Found:** 0

## Per-Class Distribution
| Class | Record Count | Percentage |
| :--- | :--- | :--- |
| Benign | 2,284 | 45.68% |
| DDoS_UDP | 765 | 15.3% |
| PortScan | 711 | 14.22% |
| Mirai_Botnet | 495 | 9.9% |
| Infiltration | 405 | 8.1% |
| BruteForce | 340 | 6.8% |

## Metadata & Excluded Columns
- **Suspicious ID/Time/IP Columns:** `device_id, dst_ip, dst_pkts, dst_port, flow_id, src_ip, src_pkts, src_port, timestamp`
- **Configured Preprocessing Exclusions:** `src_ip, dst_ip, flow_id, timestamp`