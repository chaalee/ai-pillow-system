# Run this once to combine all SHHS batches + MESA into one file
import numpy as np
from pathlib import Path

processed_dir = Path('data/shhs_processed')

X_all, y_all, ids_all = [], [], []

# Load SHHS batch files
shhs_files = sorted(processed_dir.glob('shhs_processed_shhs1-*.npz'))
for f in shhs_files:
    d = np.load(f, allow_pickle=True)
    X_all.extend(d['X'])
    y_all.extend(d['y'])
    ids_all.extend(d['subject_ids'])
    print(f'SHHS  {f.name}: {len(d["subject_ids"])} subjects')

# Load MESA processed file (if it exists)
mesa_file = processed_dir / 'mesa_processed.npz'
if mesa_file.exists():
    d = np.load(mesa_file, allow_pickle=True)
    X_all.extend(d['X'])
    y_all.extend(d['y'])
    ids_all.extend(d['subject_ids'])
    print(f'MESA  {mesa_file.name}: {len(d["subject_ids"])} subjects')
else:
    print('MESA  mesa_processed.npz not found — skipping')

if not X_all:
    print('No data found.')
    exit(1)

# Pad all subjects to the same time length
max_len = max(x.shape[1] for x in X_all)
X_pad = [np.pad(x, ((0, 0), (0, max_len - x.shape[1]))) for x in X_all]
y_pad = [np.pad(y, (0, max_len - len(y))) for y in y_all]

out_path = processed_dir / 'all_processed.npz'
np.savez_compressed(out_path, X=X_pad, y=y_pad, subject_ids=ids_all)
print(f'\nMerged {len(ids_all)} subjects → {out_path}')
