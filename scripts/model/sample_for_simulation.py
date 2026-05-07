"""
sample_for_simulation.py
────────────────────────
Extract ready-to-run apnea and non-apnea sample windows from
shhs_processed.npz, run them through a trained SleepApneaDetector,
and save both the selected samples and model predictions to disk.

Saved files
───────────
  simulation_samples.npz   – raw windows + ground-truth labels + metadata
  simulation_results.csv   – per-window predictions table (human-readable)

  NOTE: actuation mechanism is NOT implemented here.
        Wire it in separately once you are ready.

Usage:
    python sample_for_simulation.py                        # defaults
    python sample_for_simulation.py --npz shhs_processed.npz \
        --weights best_model.pth --n-apnea 5 --n-normal 5 --threshold 0.4
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── paste / import your model ────────────────────────────────────────────────
# If sleep_apnea_model.py is in the same folder, just do:
from sleep_apnea_model import SleepApneaDetector, SignalPreprocessor


# ═══════════════════════════════════════════════════════════════════════════
# 1.  WINDOW EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_windows(X_subject, y_subject,
                    window_samples: int = 60,
                    step_samples: int = 30):
    """
    Slide a window over one subject's continuous signals.

    Parameters
    ----------
    X_subject : ndarray (3, T)   – three channels: HR, Resp, SpO2
    y_subject : ndarray (T,)     – sample-level binary labels
    window_samples : int         – window width  (matches training: 60 samples)
    step_samples   : int         – hop size      (matches training: 30 samples)

    Returns
    -------
    windows : ndarray (N, 3, window_samples)
    labels  : ndarray (N,)   – majority-vote label per window
    """
    T = X_subject.shape[1]
    windows, labels = [], []

    for start in range(0, T - window_samples + 1, step_samples):
        end = start + window_samples
        win = X_subject[:, start:end]           # (3, window_samples)
        lbl = int(np.bincount(y_subject[start:end]).argmax())
        windows.append(win)
        labels.append(lbl)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING  (mirrors train_with_shhs.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_subject(X_subject: np.ndarray) -> np.ndarray:
    """
    Apply artifact removal + z-score normalisation to each channel.
    Mirrors the per-subject loop in train_with_shhs.py → Step 3.
    """
    preprocessor = SignalPreprocessor(window_size=30, sampling_rate=1, overlap=0.5)
    processed_channels = []
    for ch in range(X_subject.shape[0]):
        cleaned    = preprocessor.remove_artifacts_ppg(X_subject[ch])
        normalised = preprocessor.normalize_signal(cleaned)
        processed_channels.append(normalised)
    return np.stack(processed_channels, axis=0)   # (3, T)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SAMPLE PICKER
# ═══════════════════════════════════════════════════════════════════════════

def pick_samples(npz_path: str,
                 n_apnea: int = 5,
                 n_normal: int = 5,
                 seed: int = 42) -> tuple:
    """
    Load shhs_processed.npz, preprocess, window, and return balanced samples.

    Returns
    -------
    X_samples : ndarray  (n_apnea + n_normal, 3, 60)
    y_samples : ndarray  (n_apnea + n_normal,)        – ground-truth labels
    meta      : list of dicts with subject_id and window index
    """
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path, allow_pickle=True)

    X_all          = data['X']           # (150, 3, T)
    y_all          = data['y']           # (150, T)
    subject_ids    = data['subject_ids'] # (150,)

    apnea_pool, normal_pool = [], []

    print(f"Processing {len(X_all)} subjects …")
    for idx in range(len(X_all)):
        X_proc   = preprocess_subject(X_all[idx])
        wins, lbls = extract_windows(X_proc, y_all[idx])

        for w, (win, lbl) in enumerate(zip(wins, lbls)):
            entry = {'window': win, 'label': lbl,
                     'subject_id': subject_ids[idx], 'window_idx': w}
            if lbl == 1:
                apnea_pool.append(entry)
            else:
                normal_pool.append(entry)

    print(f"  Pool → apnea={len(apnea_pool)}, normal={len(normal_pool)}")

    # random draw without replacement
    apnea_idx  = rng.choice(len(apnea_pool),  size=min(n_apnea,  len(apnea_pool)),  replace=False)
    normal_idx = rng.choice(len(normal_pool), size=min(n_normal, len(normal_pool)), replace=False)

    chosen = ([apnea_pool[i]  for i in apnea_idx] +
              [normal_pool[i] for i in normal_idx])

    X_samples = np.stack([c['window'] for c in chosen], axis=0).astype(np.float32)
    y_samples = np.array([c['label']  for c in chosen], dtype=np.int32)
    meta      = [{'subject_id': c['subject_id'],
                  'window_idx': c['window_idx'],
                  'true_label': c['label']} for c in chosen]

    return X_samples, y_samples, meta


# ═══════════════════════════════════════════════════════════════════════════
# 4.  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════

def load_model(weights_path: str, device: str = 'cpu') -> SleepApneaDetector:
    """
    Rebuild the exact architecture used in train_with_shhs.py and load weights.
    """
    model = SleepApneaDetector(
        input_channels=3,
        tcn_channels=[32, 64, 128],   # ← must match what you trained
        transformer_heads=4,
        transformer_layers=2,
        num_classes=2,
        dropout=0.1
    )
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✓ Weights loaded from {weights_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE  (no actuation — log predictions only)
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(model: SleepApneaDetector,
                  X_samples: np.ndarray,
                  meta: list,
                  threshold: float = 0.5,
                  device: str = 'cpu') -> tuple:
    """
    Pass each window through the model and return predictions.
    Actuation is NOT performed here — wire it in separately.

    Returns
    -------
    predictions : ndarray (N,)   – 0=normal, 1=apnea
    probs       : ndarray (N,)   – P(apnea) for each window
    """
    tensor = torch.FloatTensor(X_samples).to(device)   # (N, 3, 60)

    with torch.no_grad():
        logits = model(tensor)                          # (N, 2)
        probs  = F.softmax(logits, dim=1)[:, 1]        # P(apnea)

    predictions = (probs.cpu().numpy() >= threshold).astype(int)
    probs_np    = probs.cpu().numpy()

    label_str = lambda l: "APNEA" if l == 1 else "normal"

    print(f"\n{'─'*72}")
    print(f"{'#':>3}  {'Subject':15}  {'Win':>5}  {'True':>6}  "
          f"{'P(apnea)':>9}  {'Pred':>7}  {'Match':>5}")
    print(f"{'─'*72}")

    n_correct = 0
    for i, (pred, prob, m) in enumerate(zip(predictions, probs_np, meta)):
        true_lbl  = m['true_label']
        match     = (pred == true_lbl)
        n_correct += int(match)
        print(f"{i+1:>3}  {m['subject_id']:15}  {m['window_idx']:>5}  "
              f"{label_str(true_lbl):>6}  {prob:>9.4f}  "
              f"{label_str(pred):>7}  {'✓' if match else '✗':>5}")

    print(f"{'─'*72}")
    n = len(predictions)
    print(f"\nSummary  : {n_correct}/{n} correct  ({100*n_correct/n:.1f}%)")
    print(f"Threshold: {threshold}")

    return predictions, probs_np


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_samples(X_samples: np.ndarray,
                 y_gt: np.ndarray,
                 meta: list,
                 out_npz: str = 'simulation_samples.npz'):
    """
    Save the selected windows + ground-truth labels + metadata to an .npz file.

    Saved arrays
    ─────────────
      X           : (N, 3, 60)  float32 – preprocessed signal windows
                                           channels: [HR, Resp, SpO2]
      y_true      : (N,)        int32   – ground-truth labels (0=normal, 1=apnea)
      subject_ids : (N,)        str     – SHHS subject identifier
      window_idxs : (N,)        int32   – window index within that subject
    """
    subject_ids = np.array([m['subject_id'] for m in meta])
    window_idxs = np.array([m['window_idx'] for m in meta], dtype=np.int32)

    np.savez(
        out_npz,
        X=X_samples,
        y_true=y_gt,
        subject_ids=subject_ids,
        window_idxs=window_idxs,
    )
    print(f"\n✓ Samples saved  → {out_npz}")
    print(f"   Arrays: X{X_samples.shape}, y_true{y_gt.shape}, "
          f"subject_ids({len(subject_ids)},), window_idxs({len(window_idxs)},)")


def save_results(meta: list,
                 predictions: np.ndarray,
                 probs: np.ndarray,
                 threshold: float,
                 out_csv: str = 'simulation_results.csv'):
    """
    Save per-window inference results to a human-readable CSV.

    Columns
    ───────
      #, subject_id, window_idx, true_label, p_apnea, predicted_label, correct
    """
    label_str = lambda l: 'apnea' if l == 1 else 'normal'

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#', 'subject_id', 'window_idx',
                         'true_label', 'p_apnea', 'predicted_label', 'correct'])
        for i, (pred, prob, m) in enumerate(zip(predictions, probs, meta)):
            writer.writerow([
                i + 1,
                m['subject_id'],
                m['window_idx'],
                label_str(m['true_label']),
                f"{prob:.6f}",
                label_str(pred),
                pred == m['true_label'],
            ])

    print(f"✓ Results saved  → {out_csv}  (threshold={threshold})")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Simulation sampler for sleep apnea model")
    p.add_argument('--npz',       default='data/shhs_processed/shhs_processed.npz',
                   help='Path to shhs_processed.npz')
    p.add_argument('--weights',   default='best_model.pth',
                   help='Path to trained model weights (.pth)')
    p.add_argument('--n-apnea',   type=int, default=5,
                   help='Number of apnea windows to sample')
    p.add_argument('--n-normal',  type=int, default=5,
                   help='Number of normal windows to sample')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Apnea probability threshold (0–1)')
    p.add_argument('--seed',      type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--device',    default='cpu',
                   help='cpu | cuda | mps')
    p.add_argument('--out-npz',   default='data/shhs_processed/simulation_samples.npz',
                   help='Output path for the saved sample windows (.npz)')
    p.add_argument('--out-csv',   default='data/shhs_processed/simulation_results.csv',
                   help='Output path for the inference results table (.csv)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ── Step A: pick balanced samples ──────────────────────────────────────
    print("=" * 70)
    print("STEP A  Sampling windows from SHHS dataset")
    print("=" * 70)
    X_samples, y_gt, meta = pick_samples(
        npz_path=args.npz,
        n_apnea=args.n_apnea,
        n_normal=args.n_normal,
        seed=args.seed
    )
    print(f"Selected {len(X_samples)} windows  "
          f"(apnea={args.n_apnea}, normal={args.n_normal})")
    print(f"Window shape: {X_samples[0].shape}   [channels=3 × samples=60]")

    # ── Step B: save selected samples ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP B  Saving selected samples")
    print("=" * 70)
    save_samples(X_samples, y_gt, meta, out_npz=args.out_npz)

    # ── Step C: load model ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP C  Loading trained model")
    print("=" * 70)
    model = load_model(args.weights, device=args.device)

    # ── Step D: run inference ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP D  Running inference  (actuation: not yet implemented)")
    print("=" * 70)
    preds, probs = run_inference(
        model=model,
        X_samples=X_samples,
        meta=meta,
        threshold=args.threshold,
        device=args.device
    )

    # ── Step E: save results ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP E  Saving inference results")
    print("=" * 70)
    save_results(meta, preds, probs, threshold=args.threshold, out_csv=args.out_csv)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Samples  → {args.out_npz}")
    print(f"  Results  → {args.out_csv}")
    print("  Next step: implement actuation by reading predictions from")
    print(f"             {args.out_csv}  or the 'preds' array above.")
    print("=" * 70)