# Pressure Sensing Data

Spatial contact force distributions from the 512-point piezoresistive pressure mat, recorded as CSV time series.

## Structure

```
pressure_sensing/
├── session_A.csv           # Pressure matrix time series for session A
├── session_B.csv           # Pressure matrix time series for session B
└── README.md               # This file
```

## CSV Format

Each session file contains timestamped pressure readings across all 512 cells.

**File format:** `session_<id>.csv`

- **Rows:** One per timestamped frame (~5 Hz sampling)
- **Columns:** `timestamp` + 512 pressure cells (`p0` through `p511`)
- **Sampling rate:** ~5 Hz (approximately 100–150 ms between frames)
- **Total frames:** Varies by session (100s to 1000s of frames)

## Pressure Cell Mapping

The 512 cells are arranged in a **32 row × 16 column** grid:

- Cell index `i` maps to grid coordinates: `row = i // 16`, `col = i % 16`
- Physical layout: rows span head-to-foot direction, columns span side-to-side
- Grid orientation: Row 0 = head end, Row 31 = foot end; Col 0 = left, Col 15 = right

**Example mappings:**
- `p0` → (0, 0) = top-left
- `p15` → (0, 15) = top-right
- `p496` → (31, 0) = bottom-left
- `p511` → (31, 15) = bottom-right

## Data Types and Ranges

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| `timestamp` | datetime | — | `YYYY-MM-DD HH:MM:SS.mmm` format, millisecond precision |
| `p0`–`p511` | int | 0–255+ | Raw pressure ADC counts; 0 = no contact, higher values = increased pressure |

**Observed patterns:**
- Most cells are 0 during inactivity or no contact
- Active contact regions show localized clusters of non-zero values (typically 30–50+ ADC counts)
- Temporal coherence: pressure values change smoothly over adjacent frames during contact

## Loading Example

```python
import pandas as pd
import numpy as np

# Load session data
df = pd.read_csv("session_A.csv")
# df.shape → (N_frames, 513)  # timestamp + 512 pressure columns

# Extract pressure matrix for frame i
frame_i_flat = df.iloc[i, 1:].values  # (512,) array
frame_i_grid = frame_i_flat.reshape(32, 16)  # (32, 16) grid

# Find all contact points (above threshold)
contact_threshold = 5
contact_mask = frame_i_grid > contact_threshold

# Get centroid of contact region
contact_rows, contact_cols = np.where(contact_mask)
if len(contact_rows) > 0:
    centroid_row = contact_rows.mean()
    centroid_col = contact_cols.mean()
    print(f"Contact centroid: ({centroid_row:.1f}, {centroid_col:.1f})")

# Stack consecutive frames into a tensor for model input
frames_subset = df.iloc[100:160, 1:].values  # 60 frames
tensor = frames_subset.reshape(60, 32, 16)  # shape (60, 32, 16)
```

## Real Data Examples

From **session_B.csv** (2026-05-07):

| timestamp | p0–p83 | p84 | p85–p511 | Interpretation |
|-----------|--------|-----|----------|-----------------|
| 2026-05-07 17:16:46.736 | 0 | 33 | 0 | Contact at cell p84 (row 5, col 4) |
| 2026-05-07 17:16:46.839 | 0 | 33 | 0 | Contact persists |
| 2026-05-07 17:16:46.936 | 0 | 35 | 0 | Slight pressure increase |

From **session_A.csv** (2026-05-07):

- **2026-05-07 17:12:53.932 – 17:12:58.937**: All cells = 0 (no contact/idle period)
- **No active contact events recorded** in visible frames

## Preprocessing Notes

Before using pressure data in the AI model, apply:
1. **Temporal smoothing:** Median filter with 3-frame window
2. **Normalization:** Z-score per frame
3. **Resampling:** 1 Hz polyphase resampling (downfrom ~5 Hz native rate)
4. **Spatial filtering:** Optional 3×3 median kernel on 32×16 grid

See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) for complete preprocessing pipeline.

---

> Data available from the corresponding author upon reasonable request.
