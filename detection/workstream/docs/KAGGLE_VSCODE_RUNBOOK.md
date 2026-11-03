# Project CLARITY — Kaggle VS Code CLI Execution Runbook

**Workstream:** `detection/workstream`  
**Target Branch:** `Koushik`  
**Dataset:** ACI IoT Network Traffic Dataset 2023 (`emilynack/aci-iot-network-traffic-dataset-2023`)  

---

## 1. Overview & Free-Tier Strategy

This runbook documents how to control data processing and model training on Kaggle's free tier directly from your local VS Code terminal, with **$0 cost** and **zero local download** of the raw 38 GB dataset.

### Verified Free Tier Limits:
- **Cost:** $0.00 (No credit card or paid services).
- **GPU Quota:** ~30 hours per rolling week (Nvidia P100 or 2x T4).
- **Session Timeout:** 12 hours max execution per batch run.
- **Output Storage:** 20 GB persistent output in `/kaggle/working`.
- **Attached Datasets:** Read-only mount at `/kaggle/input/`, zero consumption of output quota.

---

## 2. One-Time Setup: Kaggle CLI Authentication

### Step 1: Install Kaggle CLI
In your local VS Code terminal (or Python virtual environment):
```bash
pip install kaggle
```

### Step 2: Configure API Credentials
1. Log in to [kaggle.com](https://www.kaggle.com).
2. Go to **Account Settings** $\rightarrow$ complete **Phone Verification** (required to enable GPU and outbound internet in kernels).
3. Scroll down to the **API** section and click **Create New Token**. This downloads `kaggle.json`.
4. Place `kaggle.json` in your user profile:
   - **Linux / macOS:**
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - **Windows PowerShell:**
     ```powershell
     New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"
     Move-Item -Force "$HOME\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json"
     ```
5. Verify authentication:
   ```bash
   kaggle datasets list --search "aci-iot"
   ```

> [!CAUTION]
> Never commit `kaggle.json` or your credentials to Git. Both `.kaggle/` and `kaggle.json` are listed in `.gitignore`.

---

## 3. Configuring the Kernel Metadata

Open `detection/workstream/scripts/kernel-metadata.json` and replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/clarity-aci-detection-pipeline",
  "title": "CLARITY ACI Detection Pipeline",
  "code_file": "kaggle_aci_pipeline.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": [
    "emilynack/aci-iot-network-traffic-dataset-2023"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

---

## 4. Launching, Monitoring, and Fetching Outputs

### Step 1: Push and Launch the Notebook
From the repository root in VS Code:
```bash
kaggle kernels push -p detection/workstream/scripts
```
Output:
`Kernel version 1 successfully pushed. URL: https://www.kaggle.com/code/<YOUR_USERNAME>/clarity-aci-detection-pipeline`

### Step 2: Check Kernel Execution Status
Check whether the kernel is `queued`, `running`, or `complete`:
```bash
kaggle kernels status YOUR_KAGGLE_USERNAME/clarity-aci-detection-pipeline
```

### Step 3: Download Generated Outputs and Reports
When the status reports `complete`, download all output artifacts:
```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/clarity-aci-detection-pipeline -p detection/workstream/reports/aci_experiment
```

The downloaded directory will contain:
- `aci_detection_reports.zip` (Comprehensive archive of all outputs)
- `traditional_metrics.json`
- `dl_metrics.json`
- `confusion_matrix_traditional.csv`
- `confusion_matrix_dl.csv`
- `pilot_traditional_alerts_10.json`
- `pilot_dl_alerts_10.json`
- `split_manifest.json`
- `experiment_summary.md`
- `checkpoints/` (Atomic epoch checkpoints: `1D-CNN_best.pt`, `LSTM_best.pt`, `Transformer_best.pt`)

---

## 5. Resumable Training Across Kaggle Sessions

If a Kaggle kernel times out or is stopped mid-run:

1. **Automatic In-Session Checkpoints:**
   Each epoch is saved atomically to `/kaggle/working/checkpoints/<MODEL>_epoch_<N>.pt` with temporary file atomic rename (`.pt.tmp` $\rightarrow$ `.pt`), preventing corrupted files.
   The best validation loss weights are preserved separately in `<MODEL>_best.pt` and are never overwritten on resume.

2. **Cross-Session Resuming:**
   - On Kaggle, create a private dataset from your previous run's output:
     `kaggle datasets create -p detection/workstream/reports/aci_experiment/checkpoints` or attach the previous notebook run output via notebook settings (**+ Add Input** $\rightarrow$ **Your Work** $\rightarrow$ select `clarity-aci-detection-pipeline`).
   - In `kernel-metadata.json`, add your previous output dataset under `kernel_sources` or `dataset_sources`.
   - The pipeline in `kaggle_aci_pipeline.ipynb` automatically searches attached paths (`/kaggle/input/clarity-aci-checkpoints/` and `../input/clarity-aci-checkpoints/`), detects the latest epoch checkpoint, and resumes training seamlessly from where it stopped.
