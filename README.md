# Embodied AI Smart Pillow — Dataset & Code Repository

> **Pre-clinical data and analysis code accompanying the paper:**  
> *"Embodied AI for Closed-Loop Pneumatic Posture Regulation: A Multimodal Smart Pillow Platform for Pre-Clinical Sleep Support Research"*  
>
> **Author:** Chalinee Chanprasert, Nutnaree Duangdee, Kirana Trakulmaykee, Waritsanant Rattanachaipong, Ratchatin Chancharoen  
> **Affiliation:** Robotics and AI, International School of Engineering, Chulalongkorn University, Bangkok, Thailand  

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Pre-clinical](https://img.shields.io/badge/status-pre--clinical-orange.svg)]()
[![Language: Python 100%](https://img.shields.io/badge/language-Python%20100%25-brightgreen.svg)]()

---

## 📋 Overview

This repository contains the **dataset, preprocessing pipelines, AI inference models, and analysis scripts** used in the pre-clinical validation of an **embodied AI cyber–physical system** for real-time closed-loop posture regulation via a soft pneumatic smart pillow.

### Key System Capabilities

- **Multimodal sensing** — 512-point pressure-sensitive array + real-time physiological signals (HR, RR, SpO₂)
- **AI inference** — Hybrid TCN-Transformer model for sleep apnea event detection and localization
- **Distributed pneumatic actuation** — 10-channel independently controlled air chambers for targeted posture adjustment
- **Closed-loop control** — Real-time decision-making with 3–5 second system latency
- **Edge-compatible** — Quantized model (~0.35 MB) deployable on Raspberry Pi 4

### Scope & Disclaimers

⚠️ **This repository contains:**
- **Pre-clinical engineering validation data** collected under controlled conditions using a surrogate (mannequin) head
- **No human subjects or clinical data** — all experiments non-invasive and using synthetic loads
- **Research-grade datasets** (SHHS, MESA) for model training and validation
- **Reproducible AI pipeline** from raw signal to closed-loop decision

⚠️ **This repository does NOT contain:**
- Clinical efficacy claims or human-subject outcomes
- Real patient physiological data or medical records
- FDA/regulatory clearance documentation

---

## 📁 Repository Structure

```
ai-pillow-system/
│
├── README.md                          # This file
├── LICENSE                            # MIT (code) + CC-BY-4.0 (data)
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
│
├── data/
│   ├── garmin_sleep/                 # Sleep physiological time series from Garmin wearable
│   │   ├── hrv                       # Heart rate variability (HRV) — 5-min intervals
│   │   ├── respiration               # Respiration rate (brpm) — 1-min intervals
│   │   ├── spo2                      # Blood oxygen saturation (%) — 1-min intervals
│   │   ├── sleep_stages              # Sleep stage annotations (light/deep/REM/awake)
│   │
│   ├── ocr_realtime/                 # Real-time OCR-extracted physiological signals
│   │   ├── realtime_log_S001.csv     # Per-session: timestamp, HR, RR, SpO₂, OCR confidence
│   │   ├── realtime_log_S002.csv
│   │   └── ...
│   │
│   ├── head_rotation/                # IMU-based head orientation & displacement
│   │   ├── imu_data_session_*.csv    # Raw accelerometer + derived pitch/roll angles
│   │
│   ├── pressure_sensing/             # 512-point pressure mat spatial recordings
│   │   ├── session_A.csv             # Time series: timestamp + p0–p511 pressure values
│   │   ├── session_B.csv
│   │
│   └── shhs_processed/               # SHHS/MESA training datasets (processed)
│       ├── all_processed.npz         # Combined SHHS + MESA: X(N, 3, T), y(N, T)
│       ├── simulation_samples.npz    # Pre-selected windows for closed-loop simulation
│       └── simulation_results.csv    # Model predictions on simulation samples
│
├── scripts/
│   │
│   ├── preprocessing/
│   │   ├── preprocess_physiological.py   # OCR signal → windowed, normalized tensors
│   │   ├── shhs_preprocessor.py          # SHHS EDF download + preprocessing
│   │   └── mesa_preprocessor.py          # MESA local EDF + XML preprocessing
│   │
│   ├── model/
│   │   ├── sleep_apnea_model.py          # Core: TCN, Transformer, classifier, quantizer
│   │   ├── tcn_transformer.py            # Standalone hybrid TCN-Transformer architecture
│   │   ├── train_with_shhs.py            # Full training pipeline with threshold sweep
│   │   ├── sample_for_simulation.py      # Extract model-ready samples for testing
│   │   ├── combinebatch.py               # Merge SHHS batches + MESA into single file
│   │   └── best_model.pth                # Pre-trained model weights (if available)
│   │
│   ├── control/
│   │   ├── closed_loop_main.py           # (Placeholder) Main control loop orchestration
│   │   ├── pico_firmware.py              # (Placeholder) Raspberry Pi Pico firmware
│   │   └── pneumatic_control.py          # (Placeholder) Valve actuation logic
│   │
│   ├── OCR/
│   │   ├── main2.py                      # Real-time screen OCR: HR, RR, SpO₂ extraction
│   │   └── README.md                     # OCR pipeline setup & troubleshooting
│   │
│   └── visualization/
│       └── plot_all_figures.py           # Reproduce paper figures from processed data
│
│
├── results/                              # Saved figures and inference outputs
│   ├── fig9a_delta_theta_heatmap.png
│   ├── fig9b_displacement_vs_orientation.png
│   ├── fig9c_pressure_heatmap.png
│   └── simulation_results.csv
│
├── docs/                                 # Extended technical documentation
│   ├── DATA_COLLECTION.md                # Experimental protocol & session types
│   ├── DATA_DICTIONARY.md                # Complete field definitions & units for all CSVs
│   ├── HARDWARE_SETUP.md                 # System architecture, wiring, BOM
│   └── MODEL_ARCHITECTURE.md             # TCN-Transformer design, training config
│
└── assets/                               # (Placeholder) System diagrams, photos
    └── system_diagram.png
```

---

## 📊 Data Description

All datasets are collected under **controlled, non-human (surrogate head) pre-clinical conditions**. For complete field definitions, units, and quality notes, see [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

### 1. Garmin Sleep Data — `data/garmin_sleep/`

Physiological signals exported from a Garmin smartwatch via the **Labfront research platform**. Used as reference for real-time signal validation.

| File | Sampling | Description | Obs. range |
|------|----------|-------------|-----------|
| `garmin-connect-hrv-values.csv` | 5 min | Heart rate variability (RMSSD, ms) | 42–102 ms |
| `garmin-connect-sleep-respiration.csv` | 1 min | Respiration rate (breaths/min) | 11.7–20.3 brpm |
| `garmin-connect-sleep-pulse-ox.csv` | 1 min | Blood oxygen saturation (%) | 83–100% |
| `garmin-connect-sleep-stage.csv` | Variable | Sleep stage annotations (light/deep/REM/awake) | — |

**Key apnea indicators:**
- SpO₂ shows gradual decrease → transient desaturation (83–88%)
- HR slight decrease → rapid compensatory increase
- RR cessation/reduction → rebound increase

See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for Labfront CSV format & header parsing.

---

### 2. OCR Real-Time Data — `data/ocr_realtime/`

Physiological values extracted live via vision-based OCR pipeline from Garmin companion app display. These signals feed the AI inference module during closed-loop operation.

| File | Format | Sampling | Description |
|------|--------|----------|-------------|
| `realtime_log_<session_id>.csv` | CSV | ~1 Hz | Per-session: timestamp, HR, RR, SpO₂, OCR confidence |

**CSV schema:**
```
timestamp, heart_rate, respiration_rate, spo2, session_id, ocr_confidence
```

| Field | Unit | Range | Notes |
|-------|------|-------|-------|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | — | Wall-clock time |
| `heart_rate` | bpm | 30–200 | From Garmin wearable |
| `respiration` | brpm | 4–60 | From Garmin wearable |
| `spo2` | % | 70–100 | From Garmin wearable |

**OCR pipeline:**
```
Smartwatch → Companion App → Screen Mirror (MSS) →
Frame Capture → ROI Extraction → Tesseract OCR → Value Parsing → CSV
```

**Known issues:**
- ~1 second latency from display refresh + OCR processing
- Rows with confidence < 0.5 should be excluded from model inputs
- Sessions with >20% low-confidence readings flagged for review

See [`scripts/OCR/main2.py`](scripts/OCR/main2.py) for pipeline implementation.

---

### 3. Head Rotation Data — `data/head_rotation/`

IMU-based orientation angles and displacement measurements from a surrogate head with embedded 3-axis accelerometer during closed-loop actuation trials.

| File | Sampling | Description |
|------|----------|-------------|
| `imu_data_<session_id>.csv` | ~10 Hz | Raw accelerometer [X, Y, Z] + derived pitch/roll |

**Observed ranges (pre-clinical):**

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Δθ (orientation change) | 30° | 65° | Depends on initial contact position |
| \|ΔX\| (horizontal displacement) | 0 mm | 120 mm | Positive trend with Δθ |
| Pitch angle | -12.56° | +3.18° | Derived from accelerometer |
| Roll angle | -9.99° | -6.14° | Derived from accelerometer |

**CSV schema — `imu_data_<session_id>.csv`:**
```
Timestamp, Raw_X, Raw_Y, Raw_Z, Pitch_Deg, Roll_Deg
2026-05-07 17:16:46.123, -5.3, -9.8, -250.1, -2.34, -8.12
```

---

### 4. Pressure Sensing Data — `data/pressure_sensing/`

Spatial contact force distributions from the 512-point piezoresistive pressure mat (32 rows × 16 columns). Stored as both CSV time series and NumPy 2D arrays.

| File | Format | Sampling | Description |
|------|--------|----------|-------------|
| `session_<id>.csv` | CSV | ~5 Hz | Time series: timestamp + 512 pressure cell values |

**Array shape:** Each pressure map is a `(32 × 16)` grid, values in raw ADC units or normalized 0–1.

**CSV schema — `session_A.csv`:**
```
timestamp,p0,p1,p2,...,p511
2026-05-07 17:16:46.736,0,0,0,...,0
2026-05-07 17:16:46.839,0,0,0,...,0
```

**Observed patterns:**
- Most cells: 0 (no contact)
- Active cells during head contact: 0–255+ ADC counts
- Contact regions form coherent spatial clusters
- Temporal continuity expected within contact; isolated spikes = noise

**Loading in Python:**
```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("data/pressure_sensing/session_A.csv")
# df.shape → (N_frames, 513)  # timestamp + 512 cells

# Extract pressure matrix for frame i
frame_i = df.iloc[i, 1:].values.reshape(32, 16)  # shape (32, 16)

# Or load pre-computed .npy arrays
pressure_array = np.load("data/pressure_sensing/pressure_maps/session_A.npy")
# pressure_array.shape → (N_frames, 32, 16)
```

---

## 🤖 AI Model — Hybrid TCN-Transformer

The decision-making module uses a **hybrid Temporal Convolutional Network (TCN) + Transformer architecture** for binary sleep apnea classification:

### Architecture Overview

```
Input: (batch, 3 channels, 60 timesteps @ 1 Hz)
       [Heart Rate, Respiration Rate, SpO₂]
         ↓
┌─────────────────────────────────────────────────────┐
│  TCN Block (3 layers, dilation: 1, 2, 4)            │
│  - Receptive field: ~15 seconds                     │
│  - Output channels: 32 → 64 → 128                   │
│  - Causal padding (no information leakage)          │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Transformer Encoder (2 layers, 4-head attention)   │
│  - d_model=128, feedforward_dim=256                 │
│  - Captures long-range dependencies                 │
│  - ~49 second receptive field from TCN + Transformer│
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Classification Head                                │
│  - Global average pooling → FC(128→64) → FC(64→2)  │
│  - Softmax: [P(Normal), P(Apnea)]                   │
└─────────────────────────────────────────────────────┘
Output: [P(Normal), P(Apnea)]
```

### Model Hyperparameters

| Component | Configuration | Value |
|-----------|---------------|-------|
| **Input** | Channels / Timesteps / Rate | 3 / 60 / 1 Hz |
| **TCN** | Layers / Dilations / Channels | 3 / [1, 2, 4] / [32, 64, 128] |
| **Transformer** | Layers / Heads / d_model / FF | 2 / 4 / 128 / 256 |
| **Dropout** | All layers | 0.1 |
| **Parameters** | Total | ~360 K |
| **Size** | Full precision / Quantized | 1.4 MB / 0.35 MB |

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Loss** | CrossEntropyLoss | Unweighted; class balance via sampler |
| **Optimizer** | AdamW | lr=2e-4, betas=(0.9, 0.999) |
| **Learning rate schedule** | CosineAnnealingLR | Warm down from lr to lr/100 |
| **Batch size** | 32 | With WeightedRandomSampler |
| **Epochs** | 50 | Early stopping if val loss plateaus (patience=10) |
| **Gradient clipping** | max_norm=1.0 | Prevent exploding gradients |
| **Data imbalance** | WeightedRandomSampler | Oversample apnea (~20% of sleep) |

### Inference & Thresholding

```python
from scripts.model.sleep_apnea_model import SleepApneaDetector

model = SleepApneaDetector(...)
model.load_state_dict(torch.load("best_model.pth"))

# Preprocess a 60-second window
window = preprocess(signal)  # shape: (3, 60)

# Forward pass
with torch.no_grad():
    logits = model(window.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)
    p_normal, p_apnea = probs[0, 0].item(), probs[0, 1].item()

# Classify with configurable threshold
threshold = 0.5  # Tunable for sensitivity/specificity trade-off
prediction = "APNEA" if p_apnea >= threshold else "NORMAL"
```

**Threshold guidance:**
- **High sensitivity (catch all events):** threshold ≈ 0.40
- **Balanced:** threshold ≈ 0.50 (default)
- **High specificity (fewer false alarms):** threshold ≈ 0.60

See [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) for full architecture details, training hyperparameters, and threshold sweep results.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/chaalee/ai-pillow-system.git
cd ai-pillow-system

# Using conda (recommended)
conda env create -f environment.yml
conda activate ai-pillow

# Or using pip
pip install -r requirements.txt
```

### 2. Explore the Data

```bash
# Start Jupyter and open notebooks
jupyter notebook

# Or run exploration script
python notebooks/01_data_exploration.ipynb
```

### 3. Preprocess Physiological Signals

```bash
# From real-time OCR logs
python scripts/preprocessing/preprocess_physiological.py \
    --input data/ocr_realtime/ \
    --output data/processed/ \
    --window 60 \
    --overlap 0.5 \
    --confidence_threshold 0.5
```

### 4. Preprocess SHHS Training Data

```bash
# Download and preprocess SHHS dataset (requires NSRR token)
python scripts/preprocessing/shhs_preprocessor.py \
    --token YOUR_NSRR_TOKEN \
    --max-subjects 150 \
    --output-dir data/shhs_processed/

# Or preprocess local MESA EDF files
python scripts/preprocessing/mesa_preprocessor.py \
    --raw-dir data/raw \
    --out data/shhs_processed/mesa_processed.npz
```

### 5. Train the Model

```bash
# Train on SHHS + MESA combined dataset
python scripts/model/train_with_shhs.py \
    --processed-file data/shhs_processed/all_processed.npz \
    --max-subjects None \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --device mps  # or cuda / cpu
```

### 6. Run Model Inference on Sample Data

```bash
# Extract sample windows and run inference
python scripts/model/sample_for_simulation.py \
    --npz data/shhs_processed/all_processed.npz \
    --weights best_model.pth \
    --n-apnea 5 \
    --n-normal 5 \
    --threshold 0.5 \
    --device cpu
```

### 7. Reproduce Paper Figures

```bash
python scripts/visualization/plot_all_figures.py \
    --results results/ \
    --data data/
```

---

## 📈 System Performance Summary

| Category | Metric | Observed Value | Notes |
|----------|--------|----------------|-------|
| **Pressure Sensing** | Resolution | 512 points (32×16 grid) | ~17 mm spatial resolution |
| **Orientation Estimation** | Δθ range | 30–65° | Depends on initial head position |
| **Pneumatic Actuation** | Response time | 1–3 s | Proportional valve latency + physics |
| **AI Inference** | Model size (quantized) | 0.35 MB | 8-bit dynamic quantization |
| **AI Inference** | Latency (Raspberry Pi 4) | 40–60 ms | ONNX Runtime, int8 |
| **AI Performance** | F1-score (SHHS test set) | ~0.85 | Threshold-dependent |
| **System Closed-Loop** | Update period | 3–5 s | Physiological signal latency + compute |

---

## 📚 Documentation

- **[`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md)** — Experimental protocol, session types, quality control
- **[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)** — Complete field definitions, units, value ranges for all CSVs
- **[`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md)** — System architecture, wiring diagrams, component list, pneumatic design
- **[`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md)** — Detailed TCN-Transformer design, training config, threshold sweep, quantization

---

## 📖 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{chanprasert2025embodied,
  title={Embodied AI for Closed-Loop Pneumatic Posture Regulation: 
         A Multimodal Smart Pillow Platform for Pre-Clinical Sleep Support Research},
  author={Chanprasert, Chalinee and Duangdee, Nutnaree and Trakulmaykee, Kirana 
          and Rattanachaipong, Waritsanant and Chancharoen, Ratchatin},
  journal={[Journal Name]},
  year={2025},
  note={Pre-clinical validation dataset. Pre-print / Under Review}
}
```

---

## 📜 License

- **Code:** [MIT License](LICENSE) — Free use, modification, distribution with attribution
- **Data:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — Free use with attribution; share-alike optional

---

## ⚠️ Scope & Limitations

### Pre-Clinical Validation Only
- All experiments conducted on a surrogate (mannequin) head, **not human subjects**
- Controlled laboratory environment (no realistic sleep dynamics)
- Results do not generalize to clinical populations without further validation

### Known Limitations
| Issue | Impact | Mitigation |
|-------|--------|-----------|
| Surrogate vs. human anatomy | Limited generalizability | Clearly scoped as engineering validation |
| Memory foam attenuation | Reduced pressure spatial resolution | Region-based (not point-wise) strategy |
| OCR display lag (~1 s) | Temporal offset in physiological data | Documented in metadata; aligned via timestamps |
| Inter-channel pneumatic coupling | Variable pressure during simultaneous actuation | Sequential actuation enforced in control logic |
| Garmin wearable accuracy | ±2–3% typical accuracy (SpO₂) | Used for relative variation, not absolute thresholds |

---

## 🔗 Acknowledgments

This research was supported by Robotics and AI, International School of Engineering, Bangkok, Thailand.

**Data sources:**
- **SHHS:** [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs) (NSRR, NIH)
- **MESA:** [Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa) (NSRR, NIH)
- **Garmin:** Garmin Connect companion app & Labfront research platform

---

## 📧 Contact

**Corresponding author:** Prof. Ratchatin Chancharoen  
Department of Mechanical Engineering, Faculty of Engineering, Chulalongkorn University  
Bangkok 10330, Thailand  
📧 Ratchatin.c@chula.ac.th

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please see the repository for guidelines.

---

**Last updated:** 2026-05-08  
**Repository:** [github.com/chaalee/ai-pillow-system](https://github.com/chaalee/ai-pillow-system)
