"""
MESA (Multi-Ethnic Study of Atherosclerosis) Preprocessing Pipeline

Processes local MESA EDF + XML files downloaded from NSRR into the same
.npz format used by SHHS, so both datasets can be combined for training.

Usage:
    # Place EDF and XML files in data/raw/ as:
    #   mesa-sleep-XXXX.edf
    #   mesa-sleep-XXXX-nsrr.xml
    #
    python mesa_preprocessor.py

Output:
    data/shhs_processed/mesa_processed.npz  (same format as SHHS batches)
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt, resample
from scipy.signal import find_peaks
from pathlib import Path
import os
from typing import Dict, List, Optional
import argparse
import pyedflib as edf


RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/shhs_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Signal extraction ────────────────────────────────────────────────────────

def extract_signals(edf_path: Path) -> Dict[str, np.ndarray]:
    """Extract Flow, SpO2, HR from a MESA EDF file."""
    try:
        with edf.EdfReader(str(edf_path)) as reader:
            labels = reader.getSignalLabels()
            fs_list = reader.getSampleFrequencies()
            print(f"  Signals: {labels}")

            indices = {}
            for i, label in enumerate(labels):
                lu = (label or '').upper()
                if lu in ('FLOW', 'AIRFLOW') or 'THOR' in lu or 'ABDO' in lu:
                    if 'AIRFLOW' not in indices:   # prefer Flow over Thor/Abdo
                        indices['AIRFLOW'] = i
                elif lu == 'SPO2' or 'SAO2' in lu:
                    indices['SAO2'] = i
                elif lu == 'HR':
                    indices['HR'] = i

            if not indices:
                print("  ✗ No target signals found")
                return {}

            signals = {}
            for name, idx in indices.items():
                signals[name]            = reader.readSignal(idx)
                signals[f'{name}_fs']    = float(fs_list[idx])
            return signals

    except Exception as e:
        print(f"  Error reading EDF: {e}")
        return {}


# ── Annotation parsing ───────────────────────────────────────────────────────

def parse_annotations(xml_path: Path) -> List[Dict]:
    """Parse MESA NSRR XML — identical format to SHHS."""
    try:
        root = ET.parse(xml_path).getroot()
        events = []
        for event in root.findall('.//ScoredEvent'):
            name_elem = (event.find('Name') or
                         event.find('EventType') or
                         event.find('EventConcept'))
            if name_elem is None or not name_elem.text:
                continue
            name = name_elem.text.strip()
            if not any(k in name.upper() for k in ['APNEA', 'HYPOPNEA']):
                continue
            try:
                start = float(event.find('Start').text)
                dur   = float(event.find('Duration').text)
                events.append({'type': name, 'start': start,
                               'duration': dur, 'end': start + dur})
            except (AttributeError, TypeError, ValueError):
                continue
        return events
    except Exception as e:
        print(f"  Error parsing XML: {e}")
        return []


# ── Signal preprocessing ─────────────────────────────────────────────────────

def preprocess_signals(raw: Dict) -> Dict[str, np.ndarray]:
    """Resample/derive all signals to 1 Hz — same logic as SHHS preprocessor."""
    processed = {}

    # Airflow → respiration rate at 1 Hz
    if 'AIRFLOW' in raw:
        airflow = raw['AIRFLOW']
        fs      = raw['AIRFLOW_fs']
        nyq     = fs / 2
        low, high = 0.1 / nyq, min(3.0 / nyq, 0.99)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, airflow)

        peaks, _ = find_peaks(filtered, distance=int(2 * fs))
        if len(peaks) > 1:
            intervals  = np.diff(peaks) / fs
            resp_rate  = np.clip(60.0 / intervals, 5, 40)
            time_peaks = peaks[1:] / fs
            time_uni   = np.arange(0, len(airflow) / fs, 1.0)
            processed['respiration'] = np.interp(time_uni, time_peaks, resp_rate)
        else:
            processed['respiration'] = np.ones(int(len(airflow) / fs)) * 16.0

    # SpO2 — MESA already records at 1 Hz, just clean it
    if 'SAO2' in raw:
        spo2 = raw['SAO2'].copy().astype(float)
        fs   = raw['SAO2_fs']
        spo2[(spo2 < 70) | (spo2 > 100)] = np.nan
        mask = ~np.isnan(spo2)
        if mask.any():
            idx  = np.arange(len(spo2))
            spo2 = np.interp(idx, idx[mask], spo2[mask])
        if fs != 1.0:
            spo2 = resample(spo2, int(len(spo2) / fs))
        processed['spo2'] = spo2

    # HR — MESA already at 1 Hz
    if 'HR' in raw:
        hr = raw['HR'].copy().astype(float)
        fs = raw['HR_fs']
        if fs != 1.0:
            hr = resample(hr, int(len(hr) / fs))
        processed['heart_rate'] = np.clip(hr, 30, 200)

    return processed


# ── Label creation ───────────────────────────────────────────────────────────

def create_labels(events: List[Dict], duration: int) -> np.ndarray:
    labels = np.zeros(duration, dtype=np.int32)
    for ev in events:
        s, e = int(ev['start']), int(ev['end'])
        labels[s:min(e, duration)] = 1
    return labels


# ── Per-subject processing ───────────────────────────────────────────────────

def process_subject(subject_id: str) -> Optional[Dict]:
    edf_path = RAW_DIR / f"{subject_id}.edf"
    xml_path = RAW_DIR / f"{subject_id}-nsrr.xml"

    if not edf_path.exists() or not xml_path.exists():
        print(f"  ✗ Files not found for {subject_id}")
        return None

    print(f"\nProcessing {subject_id}...")
    raw      = extract_signals(edf_path)
    if not raw:
        return None

    events   = parse_annotations(xml_path)
    print(f"  Found {len(events)} apnea/hypopnea events")

    proc     = preprocess_signals(raw)
    if 'respiration' not in proc:
        print("  ✗ Missing respiration signal")
        return None

    duration = min(len(proc[k]) for k in proc)
    labels   = create_labels(events, duration)

    apnea_pct = labels.mean() * 100
    print(f"  Duration: {duration}s  |  Apnea: {labels.sum()}s ({apnea_pct:.1f}%)")

    return {
        'subject_id': subject_id,
        'signals':    proc,
        'labels':     labels,
        'duration':   float(duration),
    }


# ── Save ─────────────────────────────────────────────────────────────────────

def save_batch(all_data: List[Dict], out_path: Path):
    X_list, y_list, ids = [], [], []

    for d in all_data:
        sig = d['signals']
        hr   = sig.get('heart_rate',  np.zeros(int(d['duration'])))
        resp = sig.get('respiration', np.zeros(int(d['duration'])))
        spo2 = sig.get('spo2',        np.zeros(int(d['duration'])))

        n = min(len(hr), len(resp), len(spo2), len(d['labels']))
        X_list.append(np.stack([hr[:n], resp[:n], spo2[:n]], axis=0))
        y_list.append(d['labels'][:n])
        ids.append(d['subject_id'])

    # Pad to same length
    max_len = max(x.shape[1] for x in X_list)
    X_pad = [np.pad(x, ((0,0),(0, max_len - x.shape[1]))) for x in X_list]
    y_pad = [np.pad(y, (0, max_len - len(y))) for y in y_list]

    np.savez_compressed(out_path, X=X_pad, y=y_pad, subject_ids=ids)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved {len(ids)} subjects → {out_path}  ({size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Preprocess local MESA EDF files')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Folder containing mesa-sleep-XXXX.edf files')
    parser.add_argument('--out',     default='data/shhs_processed/mesa_processed.npz',
                        help='Output .npz path')
    args = parser.parse_args()

    raw_dir  = Path(args.raw_dir)
    out_path = Path(args.out)

    # Auto-detect all MESA EDF files in raw_dir
    edf_files   = sorted(raw_dir.glob("mesa-sleep-*.edf"))
    subject_ids = [f.stem for f in edf_files]   # e.g. "mesa-sleep-0001"

    if not subject_ids:
        print(f"No mesa-sleep-*.edf files found in {raw_dir}")
        return

    print(f"Found {len(subject_ids)} MESA subjects: {subject_ids}")

    all_data = []
    for sid in subject_ids:
        result = process_subject(sid)
        if result and result['labels'].sum() > 0:
            all_data.append(result)
        elif result:
            print(f"  ⚠ Skipping {sid} — 0 apnea labels")

    print(f"\nSuccessfully processed: {len(all_data)}/{len(subject_ids)}")

    if all_data:
        save_batch(all_data, out_path)
    else:
        print("No valid subjects to save.")


if __name__ == '__main__':
    main()
