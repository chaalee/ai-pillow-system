# Head Rotation Data

Kinematic measurements from surrogate head actuation trials.

## Files

| File | Description |
|------|-------------|
| `orientation_trials.csv` | Pitch, roll, Δθ per trial |
| `displacement_array.csv` | \|ΔX\| and Δθ pairs for scatter plot (Fig. 9b) |
| `delta_theta_array.csv` | Δθ with spatial sensor coordinates (Fig. 9a) |
| `trial_metadata.csv` | Chamber patterns, starting positions |

## Key Columns

- `delta_theta_deg` — Net orientation change (degrees), range: 30–65°
- `abs_displacement_mm` — Horizontal displacement magnitude (mm), range: 0–120 mm
- `sensor_x_mm`, `sensor_y_mm` — Contact location on pressure mat

See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) for full schema.

> Data available from the corresponding author upon reasonable request.
