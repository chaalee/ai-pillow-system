# OCR Real-Time Data

Place session CSV files here, following the naming convention:

```
realtime_log_<session_id>.csv
```

e.g., `realtime_log_S001.csv`, `realtime_log_S042.csv`

## CSV Schema

```
timestamp,heart_rate,respiration,spo2
```

See [`../../docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) for full field descriptions.

## How these files were generated

The OCR pipeline captures Garmin companion app display in real time via screen mirroring.  
See [`../../docs/DATA_COLLECTION.md`](../../docs/DATA_COLLECTION.md) for the full collection protocol.

> Data available from the corresponding author upon reasonable request.
