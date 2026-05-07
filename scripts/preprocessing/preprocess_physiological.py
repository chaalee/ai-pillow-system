"""
preprocess_physiological.py
----------------------------
Preprocesses raw OCR-extracted physiological signals (HR, RR, SpO2) for
input to the TCN-Transformer apnea detection model.

Steps:
  1. Load raw CSV log(s)
  2. Quality filtering (OCR confidence, value range clamping)
  3. Signal-specific filtering (Butterworth bandpass / median)
  4. Resample to 1 Hz
  5. Z-score normalization per channel
  6. Sliding window segmentation (60 s, 50% overlap)
  7. Save processed tensors as .npy

Usage:
    python preprocess_physiological.py \
        --input ../../data/ocr_realtime/ \
        --output ../../data/processed/ \
        --window 60 \
        --overlap 0.5 \
        --confidence_threshold 0.5
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from scipy.signal import medfilt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────── Signal parameters ────────────────────────────── #

SIGNAL_PARAMS = {
    "heart_rate_bpm": {
        "filter": "bandpass",
        "low": 0.5,
        "high": 40.0,
        "order": 4,
    },
    "respiration_rate_brpm": {
        "filter": "bandpass",
        "low": 0.1,
        "high": 3.0,
        "order": 4,
    },
    "spo2_pct": {
        "filter": "median",
        "kernel_size": 3,
    },
}

VALID_RANGES = {
    "heart_rate_bpm":       (30,  200),
    "respiration_rate_brpm": (4,   60),
    "spo2_pct":             (70,  100),
}

TARGET_FS = 1.0  # Hz


# ─────────────────────────── Filtering helpers ────────────────────────────── #

def butterworth_bandpass(signal: np.ndarray, low: float, high: float,
                         order: int, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def apply_filter(signal: np.ndarray, params: dict, fs: float) -> np.ndarray:
    if params["filter"] == "bandpass":
        return butterworth_bandpass(signal, params["low"], params["high"],
                                    params["order"], fs)
    elif params["filter"] == "median":
        return medfilt(signal, kernel_size=params["kernel_size"])
    else:
        raise ValueError(f"Unknown filter type: {params['filter']}")


def resample_to_target(signal: np.ndarray, original_fs: float,
                        target_fs: float = TARGET_FS) -> np.ndarray:
    """Polyphase resampling with antialiasing."""
    if original_fs == target_fs:
        return signal
    up = int(target_fs)
    down = int(original_fs)
    from math import gcd
    g = gcd(up, down)
    return resample_poly(signal, up // g, down // g)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std < 1e-8:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


# ─────────────────────────── Windowing ───────────────────────────────────── #

def sliding_window(data: np.ndarray, window_size: int,
                   step_size: int) -> np.ndarray:
    """
    Args:
        data: (n_channels, n_samples)
        window_size: samples per window
        step_size: step between windows
    Returns:
        (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = data.shape
    windows = []
    start = 0
    while start + window_size <= n_samples:
        windows.append(data[:, start:start + window_size])
        start += step_size
    return np.stack(windows, axis=0) if windows else np.empty((0, n_channels, window_size))


# ─────────────────────────── Main processing ─────────────────────────────── #

def process_session(df: pd.DataFrame, session_id: str,
                    window_s: int, overlap: float,
                    confidence_threshold: float) -> dict:
    """Process one session's DataFrame into model-ready windows."""

    # 1. Quality filter
    if "ocr_confidence" in df.columns:
        n_before = len(df)
        df = df[df["ocr_confidence"] >= confidence_threshold].copy()
        logger.info(f"  OCR filter: {n_before} → {len(df)} rows "
                    f"(removed {n_before - len(df)})")

    # 2. Clamp values
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # 3. Interpolate missing values (forward fill then linear)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in SIGNAL_PARAMS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    if len(df) < window_s:
        logger.warning(f"  Session {session_id}: too short ({len(df)} rows), skipping.")
        return {}

    # 4. Estimate original sampling rate
    timestamps = pd.to_datetime(df["timestamp"])
    dt = timestamps.diff().median().total_seconds()
    original_fs = 1.0 / dt if dt and dt > 0 else 1.0
    logger.info(f"  Estimated fs: {original_fs:.2f} Hz")

    # 5. Filter + resample + normalize each channel
    channels = []
    channel_names = []
    for col, params in SIGNAL_PARAMS.items():
        if col not in df.columns:
            continue
        raw = df[col].to_numpy(dtype=np.float32)
        filtered = apply_filter(raw, params, fs=original_fs)
        resampled = resample_to_target(filtered, original_fs, TARGET_FS)
        normalized = zscore_normalize(resampled)
        channels.append(normalized)
        channel_names.append(col)

    if not channels:
        logger.warning(f"  No valid channels found for {session_id}.")
        return {}

    data = np.stack(channels, axis=0).astype(np.float32)  # (C, T)

    # 6. Sliding windows
    step_size = max(1, int(window_s * (1 - overlap)))
    windows = sliding_window(data, window_s, step_size)
    logger.info(f"  → {windows.shape[0]} windows of shape {windows.shape[1:]}")

    return {
        "session_id": session_id,
        "windows": windows,
        "channel_names": channel_names,
        "original_fs": original_fs,
        "n_rows": len(df),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess physiological signals")
    parser.add_argument("--input",  required=True, help="Path to raw CSV directory")
    parser.add_argument("--output", required=True, help="Path to save processed .npy files")
    parser.add_argument("--window", type=int,   default=60,  help="Window size (seconds)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum OCR confidence")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("realtime_log_*.csv"))
    if not csv_files:
        logger.error(f"No realtime_log_*.csv files found in {input_dir}")
        return

    manifest = []

    for csv_path in csv_files:
        session_id = csv_path.stem.replace("realtime_log_", "")
        logger.info(f"Processing session: {session_id}")

        df = pd.read_csv(csv_path)
        result = process_session(df, session_id,
                                  args.window, args.overlap,
                                  args.confidence_threshold)
        if not result:
            continue

        out_path = output_dir / f"windows_{session_id}.npy"
        np.save(out_path, result["windows"])
        logger.info(f"  Saved: {out_path}")

        manifest.append({
            "session_id": session_id,
            "n_windows": int(result["windows"].shape[0]),
            "n_channels": int(result["windows"].shape[1]),
            "window_size": args.window,
            "overlap": args.overlap,
            "channel_names": result["channel_names"],
            "original_fs_hz": round(result["original_fs"], 3),
            "output_file": str(out_path.name),
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nManifest saved: {manifest_path}")
    logger.info(f"Total sessions processed: {len(manifest)}")


if __name__ == "__main__":
    main()
