# ACI IoT Network Traffic Dataset 2023 — Input Data Directory

This directory is the local destination for the raw or uncompressed network flow captures of the **ACI IoT Network Traffic Dataset 2023**.

## Dataset Provenance & Metadata

- **Title:** ACI IoT Network Traffic Dataset 2023
- **Primary Source (IEEE DataPort):** [https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023)
- **DOI:** `10.21227/qacj-3x32`
- **Kaggle Mirror:** [https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023/data)
- **Authors:** Emily Nack, Patrick McKenzie, Nathaniel Bastian (Army Cyber Institute at West Point)
- **Reference Paper:** MILCOM 2024 IEEE Military Communications Conference
- **License:** Creative Commons Attribution (CC BY 4.0)
- **Total Archive Size:** ~38 GB (contains raw `.pcap` files totaling 32.27 GB and labeled NetFlow CSVs totaling 596.91 MB)

---

## Directory Organization

```
input_data/
├── README.md               # This provenance and placement guide
├── network/
│   ├── .gitkeep
│   └── ACI-IoT-2023.csv    # Target location for labeled NetFlow CSV (596.91 MB)
└── iot_devices/
    └── .gitkeep            # Optional raw device-specific packet captures
```

---

## How to Acquire and Place the NetFlow CSV

1. **Option A: Kaggle Cloud Notebook (Recommended - No Local Download Needed)**
   - Open the provided notebook `scripts/kaggle_aci_pipeline.ipynb` in Kaggle.
   - Attach the Kaggle dataset `emilynack/aci-iot-network-traffic-dataset-2023`.
   - The dataset is mounted automatically at `/kaggle/input/aci-iot-network-traffic-dataset-2023/ACI-IoT-2023.csv`.

2. **Option B: Kaggle CLI**
   ```bash
   kaggle datasets download -d emilynack/aci-iot-network-traffic-dataset-2023 -f ACI-IoT-2023.csv -p input_data/network/
   unzip input_data/network/ACI-IoT-2023.csv.zip -d input_data/network/
   ```

3. **Option C: Direct IEEE DataPort Download**
   - Download `ACI IoT 2023 Dataset (Labeled Netflow)` directly from IEEE DataPort.
   - Place `ACI-IoT-2023.csv` into `detection/workstream/input_data/network/ACI-IoT-2023.csv`.

> [!CAUTION]
> **Git Tracking Exclusion:**
> Raw `.pcap`, `.pcapng`, and large `.csv` files exceeding GitHub's limits are excluded by `.gitignore` and must NEVER be committed to the repository.
