# Embodied AI Smart Pillow — Dataset & Code Repository

> **Pre-clinical data and analysis code accompanying the paper:**  
> *"Embodied AI for Closed-Loop Pneumatic Posture Regulation: A Multimodal Smart Pillow Platform for Pre-Clinical Sleep Support Research"*  
> Chanprasert et al., Chulalongkorn University, Bangkok, Thailand

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Pre-clinical](https://img.shields.io/badge/status-pre--clinical-orange.svg)]()

---

## Overview

This repository contains the dataset and analysis scripts used in the pre-clinical validation of an **embodied AI cyber–physical system** for real-time closed-loop posture regulation via a soft pneumatic smart pillow. The system integrates:

- **Multimodal sensing** — pressure mat (512-point force array) + physiological signals
- **AI inference** — hybrid TCN-Transformer model for sleep apnea event detection
- **Distributed pneumatic actuation** — 10-channel independently controlled air chambers

> ⚠️ **Scope notice:** All data were collected under **controlled, non-human (surrogate head) conditions**. This repository supports engineering validation only and makes no claims of clinical efficacy.

---

## Repository Structure

```
ai-pillow-repo/
│
├── data/
│   ├── garmin_sleep/          # Sleep physiological data from Garmin wearable
│   ├── ocr_realtime/          # Real-time OCR-extracted physiological signals (CSV)
│   ├── head_rotation/         # Head orientation & displacement arrays
│   └── pressure_sensing/      # Pressure mat spatial distribution recordings
│
├── scripts/
│   ├── preprocessing/         # Signal filtering, resampling, normalization
│   ├── model/                 # TCN-Transformer inference pipeline
│   ├── control/               # Closed-loop actuation logic
│   └── visualization/         # Heatmaps, scatter plots, time-series plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_physiological_signal_analysis.ipynb
│   ├── 03_pressure_spatial_analysis.ipynb
│   └── 04_closed_loop_evaluation.ipynb
│
├── results/                   # Figures and tables from the paper
├── docs/                      # Extended documentation
│   ├── DATA_COLLECTION.md
│   ├── HARDWARE_SETUP.md
│   └── MODEL_ARCHITECTURE.md
│
├── assets/                    # System diagrams and illustrations
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## Data Description

See [`DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for full field definitions and units.

### 1. Garmin Sleep Data — `data/garmin_sleep/`

Physiological signals exported from a Garmin smartwatch, captured during surrogate-head test sessions.

| File | Description |
|------|-------------|
| `hrv_series.csv` | Heart rate variability (HRV) time series |
| `respiration_series.csv` | Respiration rate (breaths/min) over time |
| `spo2_series.csv` | Blood oxygen saturation (SpO₂, %) |
| `sleep_stages.csv` | Sleep stage annotations (if applicable) |
| `session_metadata.csv` | Session ID, date, duration, conditions |

**Key signals and their temporal behavior during apnea events:**

| Signal | Apnea pattern |
|--------|---------------|
| SpO₂ | Gradual decrease → desaturation → recovery |
| Heart Rate | Slight decrease → rapid compensatory increase |
| Respiration Rate | Cessation / reduction → rebound increase |

---

### 2. OCR Real-Time Data — `data/ocr_realtime/`

Physiological values extracted in real-time via the vision-based OCR pipeline (screen mirroring → Tesseract OCR → JSON/CSV). These are the live signals fed to the AI inference module during closed-loop operation.

| File | Description |
|------|-------------|
| `realtime_log_<session_id>.csv` | Timestamped HR, RR, SpO₂ per session |
| `ocr_raw_frames/` | (Optional) Raw frame captures for reproducibility |

**CSV schema:**

```
timestamp, heart_rate, respiration_rate, spo2, session_id, ocr_confidence
```

| Field | Unit | Notes |
|-------|------|-------|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Wall-clock time |
| `heart_rate` | bpm | Garmin wearable |
| `respiration_rate` | brpm | Garmin wearable |
| `spo2` | % | Garmin wearable |
| `ocr_confidence` | 0–1 | Tesseract confidence score |

---

### 3. Head Rotation Data — `data/head_rotation/`

Orientation angles and displacement measurements captured during closed-loop actuation trials using a surrogate head with an embedded IMU.

| File | Description |
|------|-------------|
| `orientation_trials.csv` | Pitch, roll, yaw per actuation trial |
| `displacement_array.csv` | Absolute horizontal displacement (|ΔX|, mm) |
| `delta_theta_array.csv` | Orientation change (Δθ, degrees) per trial |
| `trial_metadata.csv` | Chamber activation pattern, initial position |

**CSV schema — `orientation_trials.csv`:**

```
trial_id, chamber_activated, initial_pitch, initial_roll, final_pitch, final_roll, delta_theta, abs_displacement_mm
```

**Observed ranges (pre-clinical):**

| Metric | Range | Notes |
|--------|-------|-------|
| Δθ (orientation change) | 30–65° | Depends on initial contact position |
| \|ΔX\| (horizontal displacement) | 0–120 mm | Positive trend with Δθ |

---

### 4. Pressure Sensing Data — `data/pressure_sensing/`

Spatial contact force distributions from the 512-point piezoresistive pressure mat (reconstructed as 2D arrays).

| File | Description |
|------|-------------|
| `pressure_maps/` | NumPy `.npy` arrays — shape `(N_frames, H, W)` |
| `pressure_summary.csv` | Max-pressure coordinates, session-level stats |
| `contact_regions.csv` | Detected dominant contact region per frame |

**Array shape:** Each pressure map is a `(32 × 16)` grid (512 sensing points), values in raw ADC units or normalized 0–1.

---

## AI Model

The decision-making module uses a **hybrid TCN-Transformer** architecture:

- **TCN block** — 3 dilated causal conv layers (dilation: 1, 2, 4); receptive field ~49 s
- **Transformer block** — 2 encoder layers, 4-head self-attention, 128-dim
- **Input** — 60-second windows, 3 channels (airflow / SpO₂ / HR), 1 Hz, 50% overlap
- **Output** — Binary classification: `Normal` / `Apnea` with probability score

See [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) for full architecture details and hyperparameters.

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/<your-org>/ai-pillow-repo.git
cd ai-pillow-repo

# Using conda (recommended)
conda env create -f environment.yml
conda activate ai-pillow

# Or using pip
pip install -r requirements.txt
```

### 2. Explore the Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Run Preprocessing

```bash
python scripts/preprocessing/preprocess_physiological.py \
    --input data/ocr_realtime/ \
    --output data/processed/ \
    --window 60 \
    --overlap 0.5
```

### 4. Run Model Inference

```bash
python scripts/model/run_inference.py \
    --input data/processed/ \
    --threshold 0.45 \
    --output results/inference_output.csv
```

### 5. Reproduce Paper Figures

```bash
python scripts/visualization/plot_all_figures.py --results results/
```

---

## System Performance Summary

| Category | Metric | Observed Value |
|----------|--------|---------------|
| Pressure sensing | Sensor resolution | 512 sensing points |
| Orientation estimation | Δθ range | 30–65° |
| Pneumatic actuation | Response time | 1–3 s |
| AI anomaly detection | Output | Normal / Apnea |
| Overall system | Closed-loop update period | 3–5 s |

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{chanprasert2025embodied,
  title     = {Embodied AI for Closed-Loop Pneumatic Posture Regulation: 
               A Multimodal Smart Pillow Platform for Pre-Clinical Sleep Support Research},
  author    = {Chanprasert, Chalinee and Duangdee, Nutnaree and Trakulmaykee, Kirana 
               and Rattanachaipong, Waritsanant and Chancharoen, Ratchatin},
  journal   = {[Journal Name]},
  year      = {2025},
  note      = {Pre-print / Under Review}
}
```

---

## License

- **Code:** [MIT License](LICENSE)  
- **Data:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

**Corresponding author:** Prof. Ratchatin Chancharoen  
Department of Mechanical Engineering, Faculty of Engineering, Chulalongkorn University  
Bangkok 10330, Thailand  
📧 Ratchatin.c@chula.ac.th

Human–Robot Collaboration and Systems Integration Research Unit, Chulalongkorn University
