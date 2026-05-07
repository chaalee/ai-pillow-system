# AI Model Architecture

This document details the hybrid TCN-Transformer model used for sleep apnea event classification in the closed-loop control system.

---

## Overview

The model performs **binary temporal classification** of physiological time-series data to detect sleep apnea events. It is designed for edge deployment on a Raspberry Pi 4 and must produce inference results within the 30-second overlap window.

**Input:** 60-second windows of 3-channel physiological signals @ 1 Hz → tensor shape `(3, 60)`  
**Output:** `P(Normal)`, `P(Apnea)` — softmax probability pair

---

## Architecture Diagram

```
Input: (batch, 3 channels, 60 time steps)
         │
         ▼
┌─────────────────────────────────────┐
│         TCN Block                   │
│  ┌──────────────────────────────┐   │
│  │ DilatedCausalConv1D          │   │
│  │ in=3,  out=64,  dilation=1   │   │
│  │ BN → ReLU → Dropout(0.1)     │   │
│  │ + Residual connection        │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ DilatedCausalConv1D          │   │
│  │ in=64, out=128, dilation=2   │   │
│  │ BN → ReLU → Dropout(0.1)     │   │
│  │ + Residual connection        │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ DilatedCausalConv1D          │   │
│  │ in=128,out=256, dilation=4   │   │
│  │ BN → ReLU → Dropout(0.1)     │   │
│  │ + Residual connection        │   │
│  └──────────────────────────────┘   │
│  Output: (batch, 256, 60)            │
└─────────────────────────────────────┘
         │  Linear projection → 128 dim
         ▼
┌─────────────────────────────────────┐
│       Transformer Encoder Block     │
│                                     │
│  Sinusoidal Positional Encoding     │
│         │                           │
│  TransformerEncoderLayer ×2         │
│  - 4-head self-attention            │
│  - d_model = 128                    │
│  - dim_feedforward = 256            │
│  - dropout = 0.1                    │
│                                     │
│  Output: (batch, 60, 128)           │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Classification Head           │
│                                     │
│  GlobalAveragePooling → (batch,128) │
│  FC(128 → 64) → ReLU → Dropout(0.1)│
│  FC(64 → 2)                         │
│  Softmax → [P(Normal), P(Apnea)]    │
└─────────────────────────────────────┘
```

---

## TCN Block Details

| Layer | In channels | Out channels | Dilation | Receptive field |
|-------|-------------|-------------|---------|----------------|
| Conv1 | 3 | 64 | 1 | 7 s |
| Conv2 | 64 | 128 | 2 | 21 s |
| Conv3 | 128 | 256 | 4 | 49 s |

- **Kernel size:** 7 (causal padding to preserve temporal causality)
- **Residual connections:** 1×1 convolution used when channel dimensions change
- **Normalization:** BatchNorm1d after each convolution
- **Activation:** ReLU
- **Dropout:** 0.1 probability

The 49-second receptive field is critical: it spans the typical apnea event duration (10–30 s) while allowing some context before onset.

---

## Transformer Block Details

| Parameter | Value |
|-----------|-------|
| Input dimension (`d_model`) | 128 |
| Number of encoder layers | 2 |
| Attention heads | 4 |
| Feedforward dimension | 256 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal (fixed) |

Sinusoidal positional encoding preserves temporal order information, which is essential for capturing the **30–60 second delay** between apnea onset and SpO₂ desaturation.

---

## Input Preprocessing

```python
# Per-channel preprocessing
signals = {
    'airflow': butterworth_bandpass(x, low=0.1, high=3.0, fs=original_fs),
    'heart_rate': butterworth_bandpass(x, low=0.5, high=40.0, fs=original_fs),
    'spo2': median_filter(x, kernel_size=3)
}

# Resample all channels to 1 Hz
for key in signals:
    signals[key] = polyphase_resample(signals[key], target_fs=1.0)

# Z-score normalization per channel
for key in signals:
    signals[key] = (signals[key] - signals[key].mean()) / (signals[key].std() + 1e-8)

# Windowing
windows = sliding_window(signals, window_size=60, step_size=30)
# Output shape: (N_windows, 3, 60)
```

---

## Classification and Thresholding

```python
logits = model(input_tensor)          # (batch, 2)
probs = softmax(logits, dim=-1)       # [P(Normal), P(Apnea)]
p_apnea = probs[:, 1]

# Configurable threshold (default 0.45)
prediction = 'APNEA' if p_apnea > threshold else 'NORMAL'
```

**Threshold guidance:**

| Threshold | Effect |
|-----------|--------|
| 0.40 | Higher sensitivity — fewer missed apnea events; more false positives |
| 0.45 | Balanced (default) |
| 0.50 | Higher specificity — fewer false positives; more missed events |

Clinical requirements should guide threshold selection. The system uses `0.45` as the default for the pre-clinical validation experiments.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | PhysioNet / custom synthetic apnea sequences |
| Loss function | CrossEntropyLoss |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Batch size | 32 |
| Epochs | 50 (early stopping, patience=10) |
| Train/val split | 80/20 |
| Class weighting | Balanced (apnea events underrepresented) |

---

## Model Files

| File | Description |
|------|-------------|
| `scripts/model/tcn_transformer.py` | Model class definition (PyTorch) |
| `scripts/model/run_inference.py` | Inference script for batch or real-time use |
| `scripts/model/train.py` | Training script |
| `scripts/model/evaluate.py` | Evaluation metrics computation |

---

## Inference Latency (Raspberry Pi 4)

| Mode | Latency |
|------|---------|
| PyTorch (float32) | ~120 ms per window |
| ONNX Runtime | ~45 ms per window |
| TorchScript | ~80 ms per window |

ONNX Runtime is recommended for edge deployment. Export the model:

```bash
python scripts/model/export_onnx.py --checkpoint model.pth --output model.onnx
```

---

## Integration with Control Loop

The model operates as a **decision-support module** within the closed-loop system:

```
Every 30 s (new window available)
    ↓
model.predict(window) → P(Apnea)
    ↓
if P(Apnea) > threshold:
    → identify dominant pressure region
    → select target pneumatic chamber
    → issue inflate-hold-deflate command sequence
    → wait 15 s for physical response
    → re-assess
else:
    → continue monitoring
```

The AI component handles **temporal state estimation** (apnea detection), while the control layer handles **spatial reasoning** (region selection) and **physical constraints** (rate limiting, amplitude limiting, sequencing).
