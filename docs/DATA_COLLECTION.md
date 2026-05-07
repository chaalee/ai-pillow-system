# Data Collection Protocol

This document describes the procedures used to collect all datasets in this repository under controlled pre-clinical conditions.

> **Ethics note:** All experiments were conducted using a surrogate head (mannequin). No human subjects were involved in data collection for this study. Human-subject studies require separate ethical approval and are identified as future work.

---

## General Experimental Setup

All experiments were conducted in the Human–Robot Collaboration and Systems Integration Research Unit, Department of Mechanical Engineering, Chulalongkorn University, Bangkok, Thailand.

**Environmental conditions:**
- Indoor, temperature-controlled room (~25°C)
- Consistent ambient lighting for OCR reliability
- No external vibration sources during pressure sensing

**Surrogate head:** A rigid mannequin head (~1.5 kg) was used to simulate head contact loading on the pillow surface. An IMU (inertial measurement unit) was embedded to record orientation angles during actuation trials.

---

## Session Types

| Session type | Purpose | Notes |
|-------------|---------|-------|
| `baseline` | Pressure map with no load | System zero-point calibration |
| `static_load` | Surrogate head, no actuation | Sensing characterization |
| `surrogate_head` | Full closed-loop trial | Primary experimental condition |

---

## Garmin Sleep Data Collection

1. Garmin smartwatch was placed on a static fixture (no wrist) during non-human tests, or worn by the operator during baseline physiological recordings.
2. Companion Garmin Connect app was opened on a paired Android smartphone.
3. Screen mirroring was initiated via software (MSS Python library) to a nearby laptop.
4. The OCR pipeline (`scripts/preprocessing/ocr_acquisition.py`) was started to log HR, RR, and SpO₂ at ~1 Hz.
5. Sessions ran for a minimum of 10 minutes to capture sufficient signal variation.
6. Garmin raw data were additionally exported post-session via the Garmin Connect web portal.

**Quality checks:**
- Sessions with >20% OCR confidence below 0.5 were flagged for review
- HR values outside 30–200 bpm, RR outside 4–60 brpm, SpO₂ outside 70–100% were treated as OCR errors and excluded

---

## Pressure Sensing Data Collection

1. Pressure mat was placed directly beneath the silicone layer in the pillow stack.
2. Bluetooth or UART serial connection to Raspberry Pi 4 was established.
3. Acquisition script (`scripts/preprocessing/pressure_acquisition.py`) streamed data at ~5 Hz.
4. Frames were stored as serialized 2D arrays and reconstructed offline into `.npy` format.
5. For each trial, a 30-second baseline (no load) recording was taken before placing the surrogate head.

**Spatial calibration:**
- The mat was re-zeroed at the start of each session
- Known weights (0.5 kg, 1.0 kg, 1.5 kg) were used to verify linearity

---

## Head Rotation / Actuation Trials

1. Surrogate head was placed on the pillow at a defined starting position (verified by pressure centroid).
2. IMU orientation was recorded at 50 Hz (pitch, roll, yaw).
3. The closed-loop controller issued an actuation sequence (triggered manually or by AI apnea detection).
4. The system waited 15 seconds for physical settling before the next reading.
5. Final orientation was recorded after the pressure distribution stabilized (Δpressure < 5% over 3 s).
6. The surrogate head was returned to starting position between trials.

**Trial conditions:**
- Minimum 5 repeated trials per chamber configuration
- 3 starting positions tested: center, left-offset, right-offset
- Actuation patterns: inflate only; inflate-hold-deflate; partial (3/10 chambers)

---

## Naming Conventions

### Session IDs
Format: `S<NNN>` — e.g., `S001`, `S042`

### Trial IDs
Format: `T<NNN>` within each session — e.g., `T001`, `T012`

### File naming
- `realtime_log_S001.csv` — OCR real-time log for session S001
- `pressure_S001_T003.npy` — Pressure map array for session S001, trial T003
- `orientation_S001.csv` — All orientation trials for session S001

---

## Known Issues and Limitations

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| OCR display lag (~1 s) | Temporal offset in physiological signals | Document in metadata; use timestamps for alignment |
| Inter-channel pneumatic coupling | Variable pressure during simultaneous actuation | Sequential actuation enforced in control logic |
| Surrogate head vs. human anatomy | Limited generalizability | Clearly scoped as pre-clinical engineering validation |
| Memory foam attenuation | Reduced pressure spatial resolution | Region-based (not point-wise) sensing strategy |
| Garmin SpO₂ accuracy | ±2–3% typical wearable accuracy | Used for relative variation, not absolute clinical thresholds |
