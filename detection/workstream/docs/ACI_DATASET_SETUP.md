# ACI IoT Network Traffic Dataset 2023 — Setup & Ingestion Guide

**Dataset Title:** ACI IoT Network Traffic Dataset 2023  
**Authors:** Emily Nack, Patrick McKenzie, Nathaniel Bastian (Army Cyber Institute at West Point)  
**Publication:** MILCOM 2024 IEEE Military Communications Conference  
**Digital Object Identifier (DOI):** [10.21227/qacj-3x32](https://doi.org/10.21227/qacj-3x32)  
**License:** Creative Commons Attribution (CC BY 4.0)  

---

## 1. Overview & Dataset Acquisition

The ACI IoT 2023 dataset captures benign and malicious network communications across a dedicated testbed of **19 commercial IoT devices** (smart plugs, IP cameras, intelligent light bulbs, smart speakers, and home automation hubs).

The dataset is distributed in two primary tiers:
1. **Raw Packet Captures (`.pcap` / `.pcapng`):** 32.27 GB compressed (exceeds local repository storage guidelines).
2. **Labeled NetFlow Feature Matrix (`ACI-IoT-2023.csv`):** 596.91 MB uncompressed, generated with **CICFlowMeter-3.0**. Contains **1,231,411 rows** and **85 raw columns**.

---

## 2. Access Routes & Credentials

### Route A: Free Kaggle Cloud Notebook (Recommended)
Kaggle hosts the official mirror at `emilynack/aci-iot-network-traffic-dataset-2023`:
1. Open or upload `detection/workstream/scripts/kaggle_aci_pipeline.ipynb` to a Kaggle Notebook.
2. In the notebook settings, add the dataset: `emilynack/aci-iot-network-traffic-dataset-2023`.
3. The dataset will mount at `/kaggle/input/aci-iot-network-traffic-dataset-2023/ACI-IoT-2023.csv` with zero download overhead.
4. Execute the pipeline end-to-end using Kaggle's free CPU or T4/P100 GPU accelerators.

### Route B: Kaggle CLI on Workstation
To download only the labeled 596 MB NetFlow CSV without downloading the 32 GB PCAP archive:
```bash
# 1. Place kaggle.json in ~/.kaggle/
# 2. Download the CSV archive
kaggle datasets download -d emilynack/aci-iot-network-traffic-dataset-2023 -f ACI-IoT-2023.csv -p detection/workstream/input_data/network/

# 3. Unzip into position
unzip detection/workstream/input_data/network/ACI-IoT-2023.csv.zip -d detection/workstream/input_data/network/
```

### Route C: IEEE DataPort Direct Download
1. Navigate to [IEEE DataPort ACI IoT 2023](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023).
2. Select the file: `ACI IoT 2023 Dataset (Labeled Netflow)` (~597 MB).
3. Place `ACI-IoT-2023.csv` in `detection/workstream/input_data/network/ACI-IoT-2023.csv`.

---

## 3. Persistent Backup & Cloud Storage Policy

Project CLARITY maintains strict budget and security boundaries:
- **No Unauthorized Spend:** Do not provision paid AWS S3 buckets, Azure Blobs, or Google Cloud Storage buckets without explicit project sponsor approval.
- **Credential Hygiene:** Never commit `.env`, `kaggle.json`, service account JSON keys, or access tokens to Git.
- **Large File Exclusion:** `.gitignore` enforces exclusion of raw PCAP archives, CSV dumps exceeding 50 MB, and model binaries.

---

## 4. Verification Checksums

Verify file integrity using SHA-256 before ingestion:

```powershell
Get-FileHash detection/workstream/input_data/network/ACI-IoT-2023.csv -Algorithm SHA256
```

- **Target SHA-256:** `9f5b84c6e9389a9f248bb82d3381a1793740e550c609805908389dcbba86c523`
- **Total Record Count:** `1,231,411`
- **Total Columns:** `85`
