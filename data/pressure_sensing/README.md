# Pressure Sensing Data

Spatial contact force distributions from the 512-point piezoresistive pressure mat.

## Structure

```
pressure_sensing/
├── pressure_maps/          # NumPy .npy arrays — shape (N_frames, 32, 16)
├── pressure_summary.csv    # Per-frame max-pressure coordinates and stats
└── contact_regions.csv     # Dominant contact region per frame
```

## Array Format

Each `.npy` file: `pressure_<session_id>_<trial_id>.npy`

- **Shape:** `(N_frames, 32, 16)` — 32 rows × 16 columns = 512 sensing points
- **Values:** float32, normalized 0.0–1.0
- **Sampling rate:** ~5 Hz

## Loading Example

```python
import numpy as np
data = np.load("pressure_maps/pressure_S001_T003.npy")
# data.shape → (150, 32, 16)  # ~30 seconds at 5 Hz

# Get mean frame
mean_map = data.mean(axis=0)  # (32, 16)

# Get max pressure coordinate per frame
max_idx = data.reshape(data.shape[0], -1).argmax(axis=1)
row_idx = max_idx // 16
col_idx = max_idx % 16
```

See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) for full schema.

> Data available from the corresponding author upon reasonable request.
