# Data Dictionary

This document provides complete field definitions, units, collection protocols, and quality notes for all datasets in this repository.

---

## 1. Garmin Sleep Data (`data/garmin_sleep/`)

Data were exported from a Garmin wearable device via its companion smartphone app. Values were subsequently extracted using the vision-based OCR pipeline described in Section 4.3 of the paper and cross-referenced with manual annotations.

---

### `hrv_series.csv`

Heart rate variability time series.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Recording time |
| `session_id` | string | — | Unique session identifier |
| `rr_interval_ms` | float | ms | R-R interval between successive heartbeats |
| `hrv_rmssd` | float | ms | Root mean square of successive RR differences |
| `hrv_sdnn` | float | ms | Standard deviation of NN intervals |
| `quality_flag` | int | 0/1 | 1 = valid, 0 = noisy / artifact |

**Notes:**
- Butterworth bandpass filter applied: 0.5–40 Hz before feature extraction
- `quality_flag = 0` rows should be excluded from model training

---

### `respiration_series.csv`

Respiration rate over time.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Recording time |
| `session_id` | string | — | Unique session identifier |
| `respiration_rate_brpm` | float | breaths/min | Instantaneous respiration rate |
| `airflow_proxy` | float | normalized | Wrist-derived airflow proxy signal (0–1) |
| `quality_flag` | int | 0/1 | 1 = valid |

**Notes:**
- Butterworth bandpass filter applied: 0.1–3 Hz (airflow channel)
- Resampled to 1 Hz using polyphase antialiasing
- Apnea events manifest as `respiration_rate_brpm ≈ 0` for ≥10 s

---

### `spo2_series.csv`

Blood oxygen saturation over time.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Recording time |
| `session_id` | string | — | Unique session identifier |
| `spo2_pct` | float | % | SpO₂ percentage (0–100) |
| `quality_flag` | int | 0/1 | 1 = valid |

**Notes:**
- 3-point median filter applied to reduce motion artifact
- Typical baseline: 95–100 %
- Desaturation events (SpO₂ < 90 %) occur 30–60 s after apnea onset

---

### `sleep_stages.csv`

Sleep stage annotations (where applicable).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Epoch start time |
| `session_id` | string | — | Unique session identifier |
| `stage` | string | — | `WAKE`, `LIGHT`, `DEEP`, `REM` |
| `epoch_duration_s` | int | s | Duration of this epoch (typically 30 s) |
| `source` | string | — | `garmin_auto` or `manual_annotation` |

---

### `session_metadata.csv`

Top-level session information.

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | string | Unique identifier (e.g., `S001`) |
| `date` | date | Collection date |
| `duration_min` | int | Total session duration (minutes) |
| `condition` | string | `surrogate_head` / `static_load` / `baseline` |
| `chamber_layout` | string | Active chamber configuration |
| `notes` | string | Free-text notes |

---

## 2. OCR Real-Time Data (`data/ocr_realtime/`)

These CSV files capture physiological values as extracted live by the Tesseract OCR pipeline from the mirrored Garmin smartphone display during closed-loop system operation. One file is generated per session.

---

### `realtime_log_<session_id>.csv`

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Wall-clock extraction time |
| `session_id` | string | — | Session identifier |
| `heart_rate` | int | bpm | HR as displayed on companion app |
| `respiration` | int | brpm | RR as displayed on companion app |
| `spo2` | int | % | SpO₂ as displayed |

**OCR Pipeline:**
1. Screen mirroring via MSS frame capture
2. ROI extraction based on fixed app layout coordinates
3. Grayscale → threshold → noise reduction
4. Tesseract OCR → numeric extraction
5. Post-processing: range clamping (HR: 30–200, RR: 4–60, SpO₂: 70–100)

**Known limitations:**
- OCR confidence < 0.6 rows may contain misread values
- Display refresh lag (~1 s) introduces slight temporal offset vs. true physiological state
- Rows where `ocr_confidence < 0.5` are flagged; consider excluding from model inputs

---

## 3. Head Rotation Data (`data/head_rotation/`)

Inertial Measurement Unit (IMU) measurements from a surrogate head (mannequin with embedded 3-axis accelerometer) resting on the smart pillow. Raw acceleration and derived orientation angles were recorded continuously across multiple sessions.

---

### `imu_data_<session_id>.csv`

IMU sensor readings and derived orientation angles.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Timestamp` | datetime | `YYYY-MM-DD HH:MM:SS.mmm` | Wall-clock recording time |
| `Raw_X` | float | m/s² | X-axis accelerometer reading |
| `Raw_Y` | float | m/s² | Y-axis accelerometer reading |
| `Raw_Z` | float | m/s² | Z-axis accelerometer reading |
| `Pitch_Deg` | float | ° | Pitch angle (rotation about X-axis) |
| `Roll_Deg` | float | ° | Roll angle (rotation about Y-axis) |

**Observed ranges:**

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Raw_X | ~-13 | ~45 | m/s² |
| Raw_Y | ~-35 | ~-27 | m/s² |
| Raw_Z | ~-260 | ~-245 | m/s² (gravity component dominant) |
| Pitch_Deg | ~-12.6 | ~3.0 | ° |
| Roll_Deg | ~-9.9 | ~-6.1 | ° |

**Notes:**
- Sampling rate: ~10 Hz (100 ms nominal intervals)
- Accelerometer range: ±16 g (typical for embedded IMU sensors)
- Z-axis includes gravity component (~-9.81 m/s² ≈ -250 raw units at rest)
- Pitch and roll derived using complementary filter or similar orientation algorithm
- All timestamps synchronized to wall-clock time (UTC-based)

---

### `orientation_trials.csv` (if present)

Summary statistics per trial.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `trial_id` | string | — | Unique trial ID (e.g., `T001`) |
| `session_id` | string | — | Parent session |
| `chamber_activated` | string | — | Comma-separated list of activated chamber IDs |
| `actuation_pattern` | string | — | `inflate` / `deflate` / `inflate-hold-deflate` |
| `initial_pitch_deg` | float | ° | Pitch before actuation |
| `initial_roll_deg` | float | ° | Roll before actuation |
| `final_pitch_deg` | float | ° | Pitch after actuation settled |
| `final_roll_deg` | float | ° | Roll after actuation settled |
| `delta_theta_deg` | float | ° | Net orientation change (Δθ) |

---

### `displacement_array.csv` (if present)

Flattened array of displacement measurements for scatter plot reproduction (Fig. 9b in paper).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `abs_displacement_mm` | float | mm | \|ΔX\| per measurement |
| `delta_theta_deg` | float | ° | Corresponding Δθ |
| `initial_contact_x_mm` | float | mm | Horizontal sensor position at start |
| `initial_contact_y_mm` | float | mm | Vertical sensor position at start |

---

### `delta_theta_array.csv` (if present)

Orientation change values with spatial context for heatmap reproduction (Fig. 9a in paper).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `sensor_x_mm` | float | mm | Horizontal sensor position |
| `sensor_y_mm` | float | mm | Vertical sensor position |
| `delta_theta_deg` | float | ° | Orientation change at this location |
| `trial_id` | string | — | Source trial |

---

## 4. Pressure Sensing Data (`data/pressure_sensing/`)

Spatial pressure distribution data from the 32×16 (512-point) piezoresistive pressure mat. Raw data are provided as NumPy arrays; summary statistics are provided as CSV.

---

### `pressure_maps/` — NumPy `.npy` files

Each file: `pressure_<session_id>_<trial_id>.npy`

- **Shape:** `(N_frames, 32, 16)` where N_frames varies by session
- **Values:** Raw normalized float32, range 0.0–1.0 (0 = no contact, 1 = max measurable load)
- **Sampling rate:** ~5 Hz (200 ms per frame)

---

### `pressure_summary.csv`

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `frame_id` | int | — | Sequential frame index |
| `session_id` | string | — | Session identifier |
| `trial_id` | string | — | Trial identifier |
| `max_pressure_x` | int | grid index | Column index of max pressure point |
| `max_pressure_y` | int | grid index | Row index of max pressure point |
| `max_pressure_val` | float | 0–1 | Maximum pressure value in frame |
| `contact_area_cells` | int | # cells | Number of cells above threshold (0.1) |
| `centroid_x` | float | grid index | Pressure centroid, x-axis |
| `centroid_y` | float | grid index | Pressure centroid, y-axis |

---

### `contact_regions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int | Frame index |
| `session_id` | string | Session identifier |
| `region_label` | string | `top-left`, `top-center`, `top-right`, `mid-left`, etc. |
| `dominant_chamber` | string | Nearest pneumatic chamber ID |

---

## Signal Preprocessing Summary

All signals fed to the AI model undergo the following preprocessing (see `scripts/preprocessing/`):

| Signal | Filter | Resampling | Normalization |
|--------|--------|-----------|---------------|
| IMU Acceleration | Butterworth bandpass 0.5–20 Hz | 1 Hz polyphase | z-score per channel |
| Pitch / Roll | Low-pass 1 Hz | 1 Hz polyphase | z-score per channel |
| Pressure Map | Median 3×3 kernel | 1 Hz polyphase | z-score per frame |

**Windowing:** 60-second sliding windows, 50% overlap (30-second step), yielding tensors of shape `(3, 60)` for IMU or `(32, 16, 60)` for pressure maps.

---

## Data Availability

All data in this repository were collected under controlled pre-clinical (non-human) conditions at Chulalongkorn University. Raw data files are available from the corresponding author upon reasonable request.
