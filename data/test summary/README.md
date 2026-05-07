# Test Summary Data

Summary results from closed-loop pillow control experiments testing head rotation tracking and pressure distribution control.

## Overview

This directory contains aggregated experimental results from multiple test runs of the AI pillow system. Data includes head position tracking accuracy, orientation angle changes, and spatial displacement metrics.

## Files

- **`experiment_data.csv`** — Summary statistics and computed metrics from individual experiments

## `experiment_data.csv` Schema

Experiment results with initial state, final state, and computed deltas (changes).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `No` | int | — | Experiment number (1–20, sequential run identifier) |
| **Initial Position (Time t₁)** |
| `X1_mm` | float | mm | Initial head position, X-axis (left-right) |
| `Y1_mm` | float | mm | Initial head position, Y-axis (front-back) |
| `Pitch1` | float | ° | Initial pitch angle (rotation about X-axis) |
| `Roll1` | float | ° | Initial roll angle (rotation about Y-axis) |
| **Final Position (Time t₂)** |
| `X2_mm` | float | mm | Final head position, X-axis |
| `Y2_mm` | float | mm | Final head position, Y-axis |
| `Pitch2` | float | ° | Final pitch angle |
| `Roll2` | float | ° | Final roll angle |
| **Computed Deltas (Changes)** |
| `dPitch` | float | ° | Change in pitch: `Pitch2 - Pitch1` |
| `dRoll` | float | ° | Change in roll: `Roll2 - Roll1` |
| `dTheta` | float | ° | Total angular displacement (magnitude): `√(dPitch² + dRoll²)` |
| `dX_mm` | float | mm | Linear displacement (magnitude): `√((X2-X1)² + (Y2-Y1)²)` |

## Observed Ranges

### Positional Data (mm)
| Metric | Min | Max | Range | Notes |
|--------|-----|-----|-------|-------|
| X1_mm, X2_mm | 28.5 | 305.0 | 276.5 | Lateral movement; 305 mm ≈ pillow width |
| Y1_mm, Y2_mm | 0.0 | 142.5 | 142.5 | Longitudinal movement; 142.5 mm ≈ pillow depth |

### Orientation Data (degrees)
| Metric | Min | Max | Range | Notes |
|--------|-----|-----|-------|-------|
| Pitch1, Pitch2 | -60.95 | 53.53 | 114.48 | Rotation about X-axis (forward/backward tilt) |
| Roll1, Roll2 | -14.76 | 11.59 | 26.35 | Rotation about Y-axis (left/right tilt) |

### Delta Metrics (changes)
| Metric | Min | Max | Mean | Notes |
|--------|-----|-----|------|-------|
| dPitch | -59.22 | 62.23 | 1.56 | Pitch changes up to 62° per experiment |
| dRoll | -0.29 | 22.29 | 5.75 | Roll changes typically smaller |
| dTheta | 31.4 | 65.1 | 45.6 | Total angular displacement per test |
| dX_mm | 19.0 | 96.0 | 66.4 | Linear displacement 19–96 mm |

## Data Patterns

**Angular Motion:** 
- Pitch exhibits the largest changes (±60°), consistent with forward/backward head movements
- Roll changes are generally smaller (±22°), reflecting side-to-side tilts
- Combined angular displacements (dTheta) range 31–65°

**Linear Motion:**
- X-axis (lateral) shows wider range (up to 305 mm) than Y-axis (up to 142.5 mm)
- Spatial displacements (dX_mm) range 19–96 mm, averaging ~66 mm per test

**Correlation Patterns:**
- Experiments with large pitch changes typically show large linear displacements
  - Example: Row 2: dPitch=62.23° → dX_mm=77 mm
  - Example: Row 5: dPitch=-59.62° → dX_mm=95 mm
- Smaller dRoll values suggest pitch dominates head kinematics in this system

## Example Calculations

**Experiment 1 (Row 1):**
- Initial: (133.0, 28.5) mm, Pitch=5.3°, Roll=-10.7°
- Final: (191.0, 19.0) mm, Pitch=-37.7°, Roll=11.59°
- **Changes:**
  - dPitch = -37.7 − 5.3 = -43.0°
  - dRoll = 11.59 − (−10.7) = 22.29°
  - dTheta = √(43.0² + 22.29²) = 48.43°
  - dX_mm = √((191−133)² + (19−28.5)²) = √(58² + 9.5²) = 58.77 mm ≈ 58.0 mm (rounded)

## Test Conditions

- **System:** Smart pillow with pneumatic actuation and head-mounted IMU
- **Controlled variable:** Pillow pressure distribution (via chamber control)
- **Measured output:** Head position (X, Y) and orientation (Pitch, Roll)
- **Duration:** ~20 experiments per session
- **Head surrogate:** Mannequin with embedded 3-axis accelerometer

## Usage Notes

- Deltas represent cumulative changes during each individual experiment
- dTheta is computed as magnitude only; use dPitch/dRoll for signed directional information
- Linear displacements are measured in millimeters; assume pillow surface reference frame
- Data collected under controlled pre-clinical conditions (non-human, Chulalongkorn University)

## Related Documentation

- Detailed data collection protocols: See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md)
- Head rotation data (IMU time series): See [`../head_rotation/`](../head_rotation/)
- Pressure sensing data: See [`../pressure_sensing/`](../pressure_sensing/)

---

> Data available from the corresponding author upon reasonable request.
