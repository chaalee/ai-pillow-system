# Data Dictionary

This document provides complete field definitions, units, collection protocols, and quality notes for all datasets in this repository.

---

## 1. Garmin Sleep Data (`data/garmin_sleep/`)

Data were collected via a Garmin wearable device and exported through the **Labfront** research platform ([help.labfront.com/data-documentation](https://help.labfront.com/data-documentation)). Each CSV file contains physiological time series with a standardized header structure.

### File Naming Convention

Files follow the Labfront export naming pattern:

```
<YYMMDD>_garmin-connect-<data-type>_<participantInsignia>_<participantId-prefix>.csv
```

Example: `260324_garmin-connect-sleep-pulse-ox_A000_2bb4e22b.csv`

---

### Common Labfront File Header (rows 1–5, present in all files)

| Row | Content | Example |
|-----|---------|---------|
| 1 | `Header Length,5` | Signals 5 header rows before data |
| 2 | `Powered by Labfront` | Platform identifier |
| 3 | `Documentation,<url>` | Link to Labfront data docs |
| 4 | Column labels: `projectId,projectTitle,participantId,participantInsignia` | — |
| 5 | Participant metadata values | `105b2a61-...,Garmin Sleep Test,2bb4e22b-...,A000` |

Row 6 is blank. Row 7 onward is the actual data header + rows.

**Loading in Python (skip the header):**
```python
import pandas as pd
df = pd.read_csv("garmin_sleep/260324_garmin-connect-hrv-values_A000_2bb4e22b.csv",
                 skiprows=6)
```

---

### Common Columns (present in all four files)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `*SummaryId` | string | — | Session-level unique ID linking all rows of the same sleep session (e.g., `sleepSummaryId`, `hrvSummaryId`). Same value across all rows of one night's recording. |
| `timezoneOffsetInMs` | int | ms | UTC offset of the recording device. `25200000` = UTC+7 (Bangkok, ICT) |
| `unixTimestampInMs` | int | ms | Unix epoch timestamp in milliseconds |
| `isoDate` | string | ISO 8601 | Human-readable local datetime with timezone offset, e.g. `2026-03-24T00:34:00.000+07:00` |

---

### `garmin-connect-hrv-values` — HRV Time Series

**Sampling interval:** 5 minutes  
**Rows per night (example):** ~62 rows

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `hrvSummaryId` | string | — | Session ID (shorter format vs. sleep files, e.g. `x5f7febc-69c17951`) |
| `timezoneOffsetInMs` | int | ms | See common columns |
| `unixTimestampInMs` | int | ms | See common columns |
| `isoDate` | string | — | See common columns |
| `hrv` | int | ms | HRV value (RMSSD — root mean square of successive RR differences), recorded every 5 minutes during sleep |

**Observed value range:** 42–102 ms (in sample session)  
**Typical baseline:** 60–100 ms during sleep; lower values may indicate sympathetic arousal or sleep disruption

**Example rows:**
```
hrvSummaryId,timezoneOffsetInMs,unixTimestampInMs,isoDate,hrv
x5f7febc-69c17951,25200000,1774287248000,2026-03-24T00:34:08.000+07:00,72
x5f7febc-69c17951,25200000,1774287548000,2026-03-24T00:39:08.000+07:00,96
```

---

### `garmin-connect-sleep-pulse-ox` — SpO₂ Time Series

**Sampling interval:** 1 minute  
**Rows per night (example):** ~312 rows

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `sleepSummaryId` | string | — | Sleep session ID (longer format, e.g. `x5f7febc-69c17951-4344`) |
| `timezoneOffsetInMs` | int | ms | See common columns |
| `unixTimestampInMs` | int | ms | See common columns |
| `isoDate` | string | — | See common columns |
| `spo2` | int | % | Blood oxygen saturation, integer percentage |

**Observed value range:** 83–100% (in sample session)  
**Notable:** Values dropping to 83–88% visible in sample data (~01:56–02:03 and other periods), consistent with transient desaturation events. Typical baseline: 95–100%.

**Example rows:**
```
sleepSummaryId,timezoneOffsetInMs,unixTimestampInMs,isoDate,spo2
x5f7febc-69c17951-4344,25200000,1774287240000,2026-03-24T00:34:00.000+07:00,97
x5f7febc-69c17951-4344,25200000,1774292160000,2026-03-24T01:56:00.000+07:00,88
```

---

### `garmin-connect-sleep-respiration` — Respiration Rate Time Series

**Sampling interval:** 1 minute  
**Rows per night (example):** ~310 rows

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `sleepSummaryId` | string | — | Sleep session ID |
| `timezoneOffsetInMs` | int | ms | See common columns |
| `unixTimestampInMs` | int | ms | See common columns |
| `isoDate` | string | — | See common columns |
| `breathsPerMinute` | float | breaths/min | Instantaneous respiration rate, reported to 2 decimal places |

**Observed value range:** 11.70–20.33 breaths/min (in sample session)  
**Notable:** Values dipping to ~11–13 breaths/min visible at multiple timepoints (e.g., 01:43, 03:00, 03:48), consistent with periodic breathing reduction. Normal sleep range: 12–20 breaths/min.

**Gaps in timestamps:** Minor gaps exist in the data (e.g., 00:58 → 01:01, skipping 00:59–01:00). These may reflect periods where the device could not compute a valid respiration estimate.

**Example rows:**
```
sleepSummaryId,timezoneOffsetInMs,unixTimestampInMs,isoDate,breathsPerMinute
x5f7febc-69c17951-4344,25200000,1774287240000,2026-03-24T00:34:00.000+07:00,17.58
x5f7febc-69c17951-4344,25200000,1774291380000,2026-03-24T01:43:00.000+07:00,11.70
```

---

### `garmin-connect-sleep-stage` — Sleep Stage Annotations

**Epoch duration:** Variable (not fixed 30 s — Garmin uses variable-length epochs)  
**Rows per night (example):** ~25 rows (one row per stage transition)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `sleepSummaryId` | string | — | Sleep session ID |
| `timezoneOffsetInMs` | int | ms | See common columns |
| `unixTimestampInMs` | int | ms | Epoch start time |
| `isoDate` | string | — | Epoch start datetime |
| `durationInMs` | int | ms | Duration of this sleep stage epoch |
| `type` | string | — | Sleep stage label: `light`, `deep`, `rem`, or `awake` |

**Stage label mapping:**

| `type` value | Description |
|-------------|-------------|
| `light` | Light NREM sleep (N1/N2) |
| `deep` | Deep / slow-wave sleep (N3) |
| `rem` | REM sleep |
| `awake` | Wakefulness during sleep period |

**Duration notes:** `durationInMs` values range from 60,000 ms (1 min) to 3,300,000 ms (55 min) in the sample. Convert to minutes: `durationInMs / 60000`.

**Example rows:**
```
sleepSummaryId,timezoneOffsetInMs,unixTimestampInMs,isoDate,durationInMs,type
x5f7febc-69d553f4-59ed,25200000,1775588340000,2026-04-08T01:59:00.000+07:00,360000,light
x5f7febc-69d553f4-59ed,25200000,1775588700000,2026-04-08T02:05:00.000+07:00,1800000,deep
x5f7febc-69d553f4-59ed,25200000,1775590500000,2026-04-08T02:35:00.000+07:00,120000,awake
```

**Sleep architecture in sample session (2026-04-08):**

| Stage | Total duration |
|-------|---------------|
| Light | ~149 min |
| Deep | ~39 min |
| REM | ~40 min |
| Awake | ~25 min |

---

## 2. OCR Real-Time Data (`data/ocr_realtime/`)

These CSV files capture physiological values as extracted live by the Tesseract OCR pipeline from the mirrored Garmin smartphone display during closed-loop system operation. One file is generated per session.

---

### `realtime_log_<session_id>.csv`

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS` | Wall-clock extraction time |
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

Inertial Measurement Unit (IMU) measurements from a surrogate head (mannequin with embedded 3-axis accelerometer) resting on the smart pillow. Raw acceleration and derived orientation angles were recorded.

---

### `imu_data_<session_id>.csv`

IMU sensor readings and derived orientation angles.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Timestamp` | datetime | `YYYY-MM-DD HH:MM:SS.mmm` | Wall-clock recording time with millisecond precision |
| `Raw_X` | float | m/s² | X-axis linear acceleration |
| `Raw_Y` | float | m/s² | Y-axis linear acceleration |
| `Raw_Z` | float | m/s² | Z-axis linear acceleration |
| `Pitch_Deg` | float | ° | Pitch angle (rotation about X-axis) |
| `Roll_Deg` | float | ° | Roll angle (rotation about Y-axis) |

**Observed ranges:**

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Raw_X | -13.0 | +45.0 | m/s² — varies with head motion and pneumatic actuation |
| Raw_Y | -35.0 | +0.0 | m/s² — gravity component dominant at rest (~-9.8 m/s²) |
| Raw_Z | -260.0 | -245.0 | m/s² — Z-axis stabilized by mounting orientation; includes gravity |
| Pitch_Deg | -12.56 | +3.18 | ° — derived from accelerometer via complementary filter |
| Roll_Deg | -9.99 | -6.14 | ° — derived from accelerometer via complementary filter |

**Notes:**
- Sampling rate: ~10 Hz (100 ms nominal intervals, with occasional variance)
- Accelerometer range: ±16 g (full scale typical for embedded 6-axis IMUs)
- Z-axis baseline: ~-250 m/s² reflects gravity component plus sensor mounting orientation
- Pitch and Roll angles computed using complementary or Kalman filter from raw acceleration data
- All timestamps synchronized to wall-clock time (YYYY-MM-DD HH:MM:SS.mmm format)
- Use millisecond precision for sub-second temporal resolution in analysis

---

## 4. Pressure Sensing Data (`data/pressure_sensing/`)

Spatial pressure distribution data from the 32×16 (512-point) piezoresistive pressure mat. Raw readings are provided as CSV time series, with one column per sensing cell.

---

### `session_<id>.csv` — Pressure Matrix Time Series

Each session file contains per-frame pressure values across all 512 cells.

**File structure:**
- **Rows:** Timestamped pressure frames, one per row
- **Columns:** `timestamp` + 512 pressure cells (`p0` through `p511`)
- **Sampling rate:** ~5 Hz (approximately 100–150 ms between frames)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | datetime | `YYYY-MM-DD HH:MM:SS.mmm` | Wall-clock frame capture time with millisecond precision |
| `p0` – `p511` | int | pressure units | Raw pressure reading from cell index (row, col) = (i // 16, i % 16). Cell indices map to physical 32-row × 16-col grid. |

**Pressure cell mapping:**
- Cell index `i` maps to grid position: `row = i // 16`, `col = i % 16`
- Grid spans 32 rows (head-to-foot) × 16 columns (side-to-side)
- Total cells: 512

**Observed value ranges:**
- Most cells: 0 (no contact)
- Active cells during contact: 0–255+ (raw ADC counts or normalized pressure)
- Example from session_B.csv: p84=33–35 during contact events

**Notable patterns:**
- Long periods of all-zero readings indicate no contact or inactivity
- Non-zero clusters indicate localized pressure contact (e.g., head, shoulders)
- Temporal continuity expected within contact regions; isolated spikes may be noise

**Example rows (session_B.csv):**
```
timestamp,p0,p1,...,p83,p84,p85,...,p511
2026-05-07 17:16:46.736,0,0,...,0,33,0,...,0
2026-05-07 17:16:46.839,0,0,...,0,33,0,...,0
2026-05-07 17:16:46.936,0,0,...,0,35,0,...,0
```

**Loading in Python:**
```python
import pandas as pd
df = pd.read_csv("data/pressure_sensing/session_A.csv")
# df.shape → (N_frames, 513)  # timestamp + 512 pressure cells
# df.columns → ['timestamp', 'p0', 'p1', ..., 'p511']

# Extract pressure matrix for frame i
frame_i = df.iloc[i, 1:].values.reshape(32, 16)  # shape (32, 16)

# Find contact centroid
contact_coords = (df.iloc[i, 1:] > 0).nonzero()[0]
```

---

## Signal Preprocessing Summary

All signals fed to the AI model undergo the following preprocessing (see `scripts/preprocessing/`):

| Signal | Filter | Resampling | Normalization |
|--------|--------|-----------|---------------|
| IMU Acceleration (X, Y, Z) | Butterworth bandpass 0.5–20 Hz | 1 Hz polyphase | z-score per channel |
| Pitch / Roll | Low-pass 1 Hz | 1 Hz polyphase | z-score per channel |
| Pressure Map | Median 3×3 kernel | 1 Hz polyphase | z-score per frame |

**Windowing:** 60-second sliding windows, 50% overlap (30-second step), yielding tensors of shape `(6, 60)` for IMU (acceleration + angles) or `(32, 16, 60)` for pressure maps.

---

## Data Availability

All data in this repository were collected under controlled pre-clinical (non-human) conditions at Chulalongkorn University. Raw data files are available from the corresponding author upon reasonable request.
