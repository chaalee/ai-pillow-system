# Head Rotation Data

Kinematic measurements from surrogate head actuation trials.

## Files

`imu_data_<session_id>.csv`

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Timestamp` | datetime | `YYYY-MM-DD HH:MM:SS.mmm` | Wall-clock recording time with millisecond precision |
| `Raw_X` | float | m/s² | X-axis linear acceleration |
| `Raw_Y` | float | m/s² | Y-axis linear acceleration |
| `Raw_Z` | float | m/s² | Z-axis linear acceleration |
| `Pitch_Deg` | float | ° | Pitch angle (rotation about X-axis) |
| `Roll_Deg` | float | ° | Roll angle (rotation about Y-axis) |

See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) for full schema.

> Data available from the corresponding author upon reasonable request.
