# AI Model Architecture

This document details the hybrid TCN-Transformer model used for sleep apnea event classification in the closed-loop control system. The architecture and all hyperparameters documented here reflect the actual implementation in `scripts/model/sleep_apnea_model.py`, `train_with_shhs.py`, and preprocessing pipelines.

---

## Overview

The model performs **binary temporal classification** of physiological time-series data to detect sleep apnea events. It is designed for both research-grade datasets (SHHS, MASS) and edge deployment on edge devices (Raspberry Pi 4, Apple Silicon).

**Input:** 60-second windows of 3-channel physiological signals @ 1 Hz → tensor shape `(batch, 3, 60)`  
**Output:** `[P(Normal), P(Apnea)]` — softmax probability pair (shape: `(batch, 2)`)

### Key Signals
- **Heart Rate (HR):** Beats per minute, extracted from ECG or pulse signals
- **Respiration Rate (RR):** Breaths per minute, derived from airflow or impedance signals
- **SpO₂:** Arterial oxygen saturation percentage, from pulse oximetry

---

## Architecture Diagram

```
Input: (batch, 3 channels, 60 time steps @ 1 Hz)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              TCN Feature Extractor                  │
│  (TemporalConvolutionalNetwork - TemporalBlock)    │
│                                                     │
│  Layer 1: TemporalBlock                             │
│  ├─ Conv1D(3 → 32, kernel=3, dilation=1)           │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  ├─ Conv1D(32 → 32, kernel=3, dilation=1)         │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  └─ Residual: 1×1 Conv if channels change          │
│                                                     │
│  Layer 2: TemporalBlock                             │
│  ├─ Conv1D(32 → 64, kernel=3, dilation=2)         │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  ├─ Conv1D(64 → 64, kernel=3, dilation=2)         │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  └─ Residual: 1×1 Conv                              │
│                                                     │
│  Layer 3: TemporalBlock                             │
│  ├─ Conv1D(64 → 128, kernel=3, dilation=4)        │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  ├─ Conv1D(128 → 128, kernel=3, dilation=4)       │
│  ├─ BatchNorm1d + ReLU + Dropout(0.1)              │
│  └─ Residual: 1×1 Conv                              │
│                                                     │
│  Output: (batch, 128, 60)                           │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│          Transformer Encoder Block                  │
│                                                     │
│  Input transpose: (batch, 128, 60)                  │
│                → (batch, 60, 128)                   │
│                                                     │
│  TransformerEncoderLayer ×2                         │
│  ├─ 4-head self-attention (d_model=128)            │
│  ├─ Feed-forward network (dim_ff=256)              │
│  ├─ Dropout(0.1) + Layer normalization             │
│  └─ Residual connections                            │
│                                                     │
│  Output: (batch, 60, 128)                           │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│          Classification Head                        │
│                                                     │
│  AdaptiveAvgPool1d: (batch, 60, 128)               │
│                    → (batch, 128)                   │
│                                                     │
│  FC(128 → 64) + ReLU + Dropout(0.1)                │
│  FC(64 → 2)                                         │
│  Softmax → [P(Normal), P(Apnea)]                   │
└─────────────────────────────────────────────────────┘
```

---

## Actual Model Configuration

The model is instantiated in code as:

```python
model = SleepApneaDetector(
    input_channels=3,
    tcn_channels=[32, 64, 128],      # 3-layer TCN with channel progression
    transformer_heads=4,              # 4 attention heads
    transformer_layers=2,             # 2 transformer encoder layers
    num_classes=2,                    # Binary: Normal vs Apnea
    dropout=0.1
)
```

### TCN Block Implementation

Each **TemporalBlock** contains:
1. **First Conv1D:** `n_inputs → n_outputs`, kernel_size=3, with dilation
2. **Normalization & Activation:** BatchNorm1d → ReLU
3. **Dropout:** probability=0.1
4. **Second Conv1D:** `n_outputs → n_outputs`, kernel_size=3, with same dilation
5. **Normalization & Activation:** BatchNorm1d → ReLU
6. **Dropout:** probability=0.1
7. **Residual Connection:** Downsamples input if channel mismatch via 1×1 conv

The causal cropping ensures the output matches input temporal length:
```python
out = out[:, :, :x.size(2)]  # Keep only past/present, discard future
```

### TCN Dilation Schedule

| Layer | In Channels | Out Channels | Dilation | Kernel | Receptive Field |
|-------|-------------|-------------|---------|--------|-----------------|
| Block 1 | 3 | 32 | 1 | 3 | 3 s |
| Block 2 | 32 | 64 | 2 | 3 | 7 s |
| Block 3 | 64 | 128 | 4 | 3 | 15 s |

**Receptive field calculation:** `RF = 1 + 2 × Σ(dilation_i × (kernel - 1))`  
For Block 3: `1 + 2×(1×2 + 2×2 + 4×2) = 1 + 2×14 = 29s` (approximate for all blocks combined)

### Transformer Block Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input dimension (`d_model`) | 128 | From TCN output |
| Number of encoder layers | 2 | Stacked TransformerEncoderLayers |
| Attention heads | 4 | Head dimension = 128/4 = 32 |
| Feedforward dimension | 256 | 2× input dimension |
| Dropout | 0.1 | Applied in MHA and FFN |
| Positional encoding | **None** (implicit) | Transformer learns relative positions |

The transformer captures long-range dependencies across the 60-second window, essential for detecting desaturation cascades and irregular breathing patterns.

---

## Input Preprocessing Pipeline

The preprocessing pipeline is implemented in `scripts/preprocessing/shhs_preprocessor.py` and `scripts/model/sleep_apnea_model.py` (class `SignalPreprocessor`).

### Step 1: Raw Signal Extraction

**From EDF files (SHHS/MASS datasets):**
```
AIRFLOW (or THOR RES / ABDO RES)     @ variable fs (typically 32 Hz)
SAO2 (SpO2)                           @ variable fs (typically 1 Hz)
H.R. (Heart Rate)                     @ variable fs (typically 1 Hz)
```

### Step 2: Per-Channel Artifact Removal

```python
# Airflow → Respiration Rate
- Butterworth bandpass filter: 0.1–3.0 Hz (low-pass: remove high-freq noise)
- Peak detection to find breath intervals
- Convert peak intervals to respiration rate (breaths/min)
- Interpolate to 1 Hz uniform grid

# SpO2
- Remove artifacts: clip to [70, 100]
- Interpolate missing values via linear interpolation
- Resample to 1 Hz

# Heart Rate
- Clean and clip to [30, 200] bpm
- Resample to 1 Hz
```

**From airflow peaks:**
```
intervals = np.diff(peaks) / fs
resp_rate = 60.0 / intervals  # Convert to breaths/min
```

### Step 3: Temporal Windowing

Create 60-second windows with 50% overlap (30-second step):

```python
window_samples = 60        # @ 1 Hz = 60 samples
step_samples = 30          # 50% overlap
n_windows = (signal_length - window_samples) // step_samples + 1

# For each window:
window = signals[:, start:start+60]   # Shape: (3, 60)
label = majority_vote(labels[start:start+60])  # Binary: 0 or 1
```

### Step 4: Z-Score Normalization

```python
for channel in [heart_rate, respiration, spo2]:
    channel_norm = (channel - channel.mean()) / (channel.std() + 1e-8)
```

Applied **independently per channel** to preserve physiological differences between channels.

---

## Dataset Preparation

### SHHS Preprocessing (`shhs_preprocessor.py`)

**Input:** Raw EDF + NSRR XML annotations from sleepdata.org

**Processing:**
1. Extract AIRFLOW, SAO2, H.R. signals
2. Parse ScoredEvent XML for apnea/hypopnea events
3. Apply per-channel preprocessing (artifact removal, resampling to 1 Hz)
4. Create binary labels at 1 Hz (1 = apnea event ongoing, 0 = normal)
5. Save as `.npz` files (compressed, efficient)

**Output format:**
```python
{
    'X': [array(3, N1), array(3, N2), ...],  # Per-subject signal arrays
    'y': [array(N1,), array(N2,), ...],      # Per-subject labels
    'subject_ids': ['shhs1-200001', ...]
}
```

### MESA Preprocessing (`mesa_preprocessor.py`)

**Input:** Local EDF + NSRR XML files (same as SHHS, but downloaded separately from NSRR)

**Processing:** Identical to SHHS pipeline  
**Output:** Same `.npz` format, compatible for combined training

### Training Data Split

```python
# Step 1: Create windows from all subjects
X_windows, y_windows = create_windows(X_processed, y_labels)

# Step 2: Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_windows, y_windows, test_size=0.3, stratified
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratified
)

# Step 3: Balance training data (oversampling apnea)
sampler = WeightedRandomSampler(
    weights=[n_normal/n_apnea if y[i]==1 else 1.0 for i in range(len(y))],
    num_samples=len(y_train),
    replacement=True
)
```

**Rationale:** Apnea events are rare (~20% of sleep in typical patients). Balanced sampling ensures the model doesn't become biased toward the majority class.

---

## Training Configuration

All training is performed via `train_with_shhs.py`:

```python
train_with_shhs(
    processed_file='data/shhs_processed/shhs_processed.npz',
    max_subjects=None,          # Use all available
    epochs=50,
    batch_size=32,
    learning_rate=0.0002,       # AdamW with low LR
    device='mps'  # or 'cuda' / 'cpu'
)
```

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Loss function | `CrossEntropyLoss` (unweighted) | Weighted sampling handles class imbalance |
| Optimizer | `AdamW` | lr=2e-4, betas=(0.9, 0.999) |
| Learning rate schedule | `CosineAnnealingLR` | Warm down from lr to lr/100 |
| Batch size | 32 | With `WeightedRandomSampler` for balance |
| Epochs | 50 | Early stopping if val loss doesn't improve for 10 epochs |
| Gradient clipping | max_norm=1.0 | Prevent exploding gradients |

### Early Stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pth')
else:
    patience_counter += 1
    if patience_counter >= 10:
        break  # Stop training
```

---

## Inference & Classification

### Real-Time Inference (60-second window)

```python
# Preprocess a 60-second window
window = preprocess(signal)  # Shape: (3, 60)

# Forward pass
with torch.no_grad():
    logits = model(window.unsqueeze(0))  # Shape: (1, 2)
    probs = torch.softmax(logits, dim=-1)  # Shape: (1, 2)
    p_normal, p_apnea = probs[0, 0].item(), probs[0, 1].item()

# Classify with threshold
threshold = 0.5  # Configurable
prediction = 'APNEA' if p_apnea >= threshold else 'NORMAL'
confidence = max(p_normal, p_apnea)
```

### Threshold Optimization

The script `train_with_shhs.py` performs **threshold sweep analysis** to find optimal operating points:

```python
thresholds = np.arange(0.05, 0.96, 0.05)
for t in thresholds:
    preds = (all_probs >= t).astype(int)
    precision, recall, f1 = compute_metrics(y_true, preds)
    # Identify best F1, balanced precision-recall, etc.
```

**Metrics reported:**
- **Precision:** Of predicted apneas, how many were correct?
- **Recall (Sensitivity):** Of true apneas, how many were detected?
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under receiver-operating-characteristic curve

**Clinical guidance:**
- **High sensitivity (recall):** Use lower threshold (~0.40) to catch all events (fewer missed apneas)
- **Balanced:** Use ~0.50 (default)
- **High specificity:** Use higher threshold (~0.60) to reduce false alarms

---

## Model Size & Inference Latency

### Parameter Count

```
TCNEncoder:    ~150 K parameters
TransformerEncoder: ~200 K parameters
Classification head: ~10 K parameters
─────────────────────────────
Total:         ~360 K parameters
```

### Inference Latency (measured on test hardware)

| Device | Framework | Mode | Latency |
|--------|-----------|------|---------|
| Mac M1 | PyTorch | float32 | ~5 ms |
| Mac M1 | PyTorch | MPS | ~3 ms |
| CPU (typical) | PyTorch | float32 | ~25–50 ms |
| Raspberry Pi 4 | PyTorch | int8 quantized | ~100–150 ms |
| Raspberry Pi 4 | ONNX Runtime | int8 quantized | ~40–60 ms |

### Quantization for Edge Deployment

The `TinyMLQuantizer` class applies **8-bit dynamic quantization** to reduce model size and inference latency:

```python
quantizer = TinyMLQuantizer()
quantized_model = quantizer.quantize_model(original_model, test_loader)

# Results (typical):
# - Size reduction: 75% (from ~1.4 MB → ~0.35 MB)
# - Speed improvement: 2–3× faster
# - Accuracy drop: <1–2%
```

---

## Explainable AI (XAI)

The `ExplainableAI` class provides interpretability for predictions:

```python
xai = ExplainableAI(model)

explanation = xai.explain_prediction(window_data)
# Returns:
# {
#     'prediction': 'APNEA',
#     'confidence': 0.87,
#     'probabilities': {'Normal': 0.13, 'Apnea': 0.87},
#     'channel_importance': {
#         'Heart Rate': 0.42,
#         'Respiration': 0.35,
#         'SpO2': 0.23
#     },
#     'temporal_importance': array(3, 60)  # Per-timestep importance
# }
```

**Method:** Gradient-based feature importance (saliency maps)

---

## Integration with Control Loop

The model operates as a **decision-support module** within the pneumatic pillow's closed-loop control system:

```
Every 30 s (sliding window updates)
    ↓
Preprocess new 60-second window
    ↓
model.predict(window) → P(Apnea)
    ↓
if P(Apnea) > threshold (0.5 default):
    ├─ Extract dominant signal frequencies (respiratory, cardiac)
    ├─ Select appropriate pneumatic chamber
    ├─ Issue inflate-hold-deflate command sequence
    ├─ Monitor pressure response for 15 s
    └─ Re-assess
else:
    └─ Continue monitoring (low-power state)
```

**Key design principles:**
1. **Temporal awareness:** 60-second input captures event evolution
2. **Real-time:** ~5 ms inference on edge devices (non-blocking)
3. **Graceful degradation:** Falls back to heuristic control if model unavailable
4. **Explainability:** Medical device requirement (must justify decisions)

---

## Model Checkpoints

| Checkpoint | Purpose | Size |
|-----------|---------|------|
| `best_model.pth` | Primary inference model (saved during training) | ~1.4 MB |
| `quantized_model.pth` | Edge deployment (8-bit) | ~0.35 MB |

Load checkpoints:
```python
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
```

---

## Code References

| File | Purpose |
|------|---------|
| `scripts/model/sleep_apnea_model.py` | Core model classes (TCNEncoder, TransformerEncoder, SleepApneaDetector, SignalPreprocessor, ExplainableAI, TinyMLQuantizer) |
| `train_with_shhs.py` | SHHS training pipeline with evaluation & threshold sweep |
| `scripts/preprocessing/shhs_preprocessor.py` | SHHS dataset download & preprocessing |
| `scripts/preprocessing/mesa_preprocessor.py` | MESA dataset preprocessing (local EDF files) |
| `docs/MODEL_ARCHITECTURE.md` | This document |

---

## Future Work

1. **Multi-task learning:** Simultaneously predict AHI (Apnea-Hypopnea Index), oxygen desaturation index, etc.
2. **Attention visualization:** Generate interpretable heatmaps showing which parts of the signal drove predictions
3. **Transfer learning:** Pre-train on large datasets, fine-tune for individual users
4. **Confidence calibration:** Ensure predicted probabilities match observed accuracy
5. **Event localization:** Predict exact time window within 60-second segment where apnea occurs
