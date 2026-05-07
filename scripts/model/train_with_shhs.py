"""
Training Script for SHHS Dataset
Uses preprocessed SHHS data with the same pipeline as manual UCD data

Usage:
    # First, preprocess SHHS data:
    python shhs_preprocessor.py --token YOUR_TOKEN --max-subjects 20
    
    # Then train:
    python train_with_shhs.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
from pathlib import Path

# Import model components
from sleep_apnea_model import (
    SleepApneaDetector,
    ModelTrainer,
    SleepApneaDataset,
    SignalPreprocessor,
    ExplainableAI,
    TinyMLQuantizer
)

# Import SHHS loader
from shhs_preprocessor import SHHSDatasetLoader


def plot_evaluation_metrics(all_labels, all_preds, all_probs, train_losses, val_losses, save_path='shhs_evaluation_metrics.png'):
    """Save a 2×3 grid of evaluation charts."""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('Sleep Apnea Detection — Model Evaluation', fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'],
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    # ── 2. Per-class Precision / Recall / F1 ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1])
    x = np.arange(2)
    w = 0.25
    bars_p = ax2.bar(x - w, precision, w, label='Precision', color='#4C72B0')
    bars_r = ax2.bar(x,      recall,    w, label='Recall',    color='#DD8452')
    bars_f = ax2.bar(x + w,  f1,        w, label='F1-Score',  color='#55A868')
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Normal', 'Apnea'])
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel('Score')
    ax2.set_title('Precision / Recall / F1 per Class', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # ── 3. Overall Metrics Bar ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    overall_acc  = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_p      = precision.mean()
    macro_r      = recall.mean()
    macro_f1     = f1.mean()
    metrics      = [overall_acc, macro_p, macro_r, macro_f1]
    labels_      = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1']
    colors       = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
    bars3 = ax3.bar(labels_, metrics, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars3, metrics):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 1.12)
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Metrics', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # ── 4. ROC Curve ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color='#C44E52', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax4.fill_between(fpr, tpr, alpha=0.1, color='#C44E52')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(alpha=0.3)

    # ── 5. Training & Validation Loss ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if train_losses and val_losses:
        epochs_range = range(1, len(train_losses) + 1)
        ax5.plot(epochs_range, train_losses, label='Train Loss', color='#4C72B0', lw=2)
        ax5.plot(epochs_range, val_losses,   label='Val Loss',   color='#DD8452', lw=2, linestyle='--')
        best_epoch = int(np.argmin(val_losses)) + 1
        ax5.axvline(best_epoch, color='gray', linestyle=':', lw=1.5, label=f'Best epoch {best_epoch}')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Training history\nnot available\n(eval-only mode)',
                 ha='center', va='center', fontsize=12, color='gray',
                 transform=ax5.transAxes)
        ax5.axis('off')
    ax5.set_title('Training History', fontsize=13, fontweight='bold')

    # ── 6. Prediction Distribution ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    normal_probs = np.array(all_probs)[np.array(all_labels) == 0]
    apnea_probs  = np.array(all_probs)[np.array(all_labels) == 1]
    bins = np.linspace(0, 1, 41)
    ax6.hist(normal_probs, bins=bins, alpha=0.6, color='#4C72B0', label='Normal', density=True)
    ax6.hist(apnea_probs,  bins=bins, alpha=0.6, color='#C44E52', label='Apnea',  density=True)
    ax6.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold 0.5')
    ax6.set_xlabel('Predicted Apnea Probability')
    ax6.set_ylabel('Density')
    ax6.set_title('Prediction Score Distribution', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Evaluation metrics saved → {save_path}")


def plot_threshold_sweep(all_true, all_probs, save_path='shhs_threshold_sweep.png'):
    """Sweep thresholds 0.05–0.95 and plot precision/recall/F1/accuracy vs threshold."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    rows = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        tp = int(((preds == 1) & (all_true == 1)).sum())
        fp = int(((preds == 1) & (all_true == 0)).sum())
        fn = int(((preds == 0) & (all_true == 1)).sum())
        tn = int(((preds == 0) & (all_true == 0)).sum())
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc    = (tp + tn) / len(all_true)
        rows.append((t, prec, rec, f1, acc, tp, fp, fn, tn))

    rows = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows],
                    dtype=[('t','f4'),('prec','f4'),('rec','f4'),('f1','f4'),('acc','f4')])

    best_f1_idx  = np.argmax(rows['f1'])
    best_rec_idx = np.argmax(rows['rec'])          # highest recall
    balanced_idx = np.argmin(np.abs(rows['prec'] - rows['rec']))  # prec ≈ recall

    # ── Print table ──────────────────────────────────────────────────────────
    print("\nThreshold Sweep (Apnea class):")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print("-" * 55)
    highlights = {best_f1_idx: '← best F1', balanced_idx: '← balanced', int(np.where(thresholds >= 0.5)[0][0]): '← default'}
    for idx, row in enumerate(rows):
        tag = highlights.get(idx, '')
        print(f"  {row['t']:>6.2f}     {row['prec']:>8.3f}   {row['rec']:>8.3f}   {row['f1']:>8.3f}   {row['acc']:>8.3f}  {tag}")

    print(f"\n★ Best F1       threshold = {rows['t'][best_f1_idx]:.2f}  "
          f"(P={rows['prec'][best_f1_idx]:.3f}, R={rows['rec'][best_f1_idx]:.3f}, F1={rows['f1'][best_f1_idx]:.3f})")
    print(f"★ Balanced P≈R  threshold = {rows['t'][balanced_idx]:.2f}  "
          f"(P={rows['prec'][balanced_idx]:.3f}, R={rows['rec'][balanced_idx]:.3f}, F1={rows['f1'][balanced_idx]:.3f})")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Threshold Sweep — Apnea Class', fontsize=14, fontweight='bold')

    # Left: metrics vs threshold
    ax = axes[0]
    ax.plot(rows['t'], rows['prec'], 'o-', color='#4C72B0', lw=2, label='Precision')
    ax.plot(rows['t'], rows['rec'],  's-', color='#DD8452', lw=2, label='Recall')
    ax.plot(rows['t'], rows['f1'],   '^-', color='#55A868', lw=2, label='F1-Score')
    ax.plot(rows['t'], rows['acc'],  'D-', color='#C44E52', lw=2, label='Accuracy', alpha=0.7)
    ax.axvline(0.5,                          color='gray',    lw=1.5, linestyle='--', label='Default (0.5)')
    ax.axvline(rows['t'][best_f1_idx],       color='#55A868', lw=1.5, linestyle=':',  label=f'Best F1 ({rows["t"][best_f1_idx]:.2f})')
    ax.axvline(rows['t'][balanced_idx],      color='purple',  lw=1.5, linestyle=':',  label=f'Balanced ({rows["t"][balanced_idx]:.2f})')
    ax.set_xlabel('Classification Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs Threshold', fontsize=12, fontweight='bold')
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: precision-recall curve (threshold as colour)
    ax2 = axes[1]
    sc = ax2.scatter(rows['rec'], rows['prec'], c=rows['t'], cmap='RdYlGn_r',
                     s=80, zorder=3, edgecolors='white', linewidths=0.5)
    ax2.plot(rows['rec'], rows['prec'], '-', color='gray', lw=1, alpha=0.5, zorder=2)
    plt.colorbar(sc, ax=ax2, label='Threshold')
    ax2.scatter(rows['rec'][best_f1_idx], rows['prec'][best_f1_idx],
                s=200, marker='*', color='#55A868', zorder=4, label=f'Best F1 (t={rows["t"][best_f1_idx]:.2f})')
    ax2.scatter(rows['rec'][int(np.where(thresholds >= 0.5)[0][0])],
                rows['prec'][int(np.where(thresholds >= 0.5)[0][0])],
                s=150, marker='D', color='gray', zorder=4, label='Default (t=0.50)')
    ax2.set_xlabel('Apnea Recall (Sensitivity)', fontsize=11)
    ax2.set_ylabel('Apnea Precision', fontsize=11)
    ax2.set_title('Precision–Recall Curve', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Threshold sweep saved → {save_path}")


def train_with_shhs(
    processed_file='data/shhs_processed/shhs_processed.npz',
    max_subjects=None,
    epochs=50,
    batch_size=32,
    learning_rate=0.0002,
    device='auto',
    eval_only=False
):
    """
    Training pipeline for SHHS dataset
    """
    
    print("=" * 70)
    print("SLEEP APNEA DETECTION - TRAINING WITH SHHS DATASET")
    print("=" * 70)
    print()
    
    # Check if processed file exists
    if not Path(processed_file).exists():
        print(f"ERROR: Processed SHHS data not found: {processed_file}")
        print()
        print("Please run preprocessing first:")
        print(f"  python shhs_preprocessor.py --token YOUR_TOKEN --max-subjects 20")
        print()
        return
    
    # Set device (prioritize MPS for M-series Macs)
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    if device == 'mps':
        print("  ✓ Metal Performance Shaders enabled (M-series GPU)")
    print()
    
    # ==================== STEP 1: LOAD DATA ====================
    print("STEP 1: Loading Preprocessed SHHS Data")
    print("-" * 70)
    
    try:
        loader = SHHSDatasetLoader(processed_file)
        all_data = loader.load_all_subjects(max_subjects=max_subjects)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    print(f"✓ Loaded {len(all_data)} subjects")
    
    # Calculate statistics
    total_duration = sum(d['duration'] for d in all_data)
    total_apnea = sum(np.sum(d['labels']) for d in all_data)
    
    print(f"✓ Total duration: {total_duration/3600:.1f} hours")
    print(f"✓ Total apnea: {total_apnea/3600:.1f} hours ({total_apnea/total_duration*100:.1f}%)")
    print()
    
    # ==================== STEP 2: PREPARE DATA ====================
    print("STEP 2: Preparing Data")
    print("-" * 70)
    
    X_list, y_list = loader.prepare_for_training(all_data)
    
    print(f"Prepared {len(X_list)} subjects")
    print(f"Signal shape (example): {X_list[0].shape}")
    print(f"Label shape (example): {y_list[0].shape}")
    print()
    
    # ==================== STEP 3: PREPROCESSING ====================
    print("STEP 3: Preprocessing Signals")
    print("-" * 70)
    
    preprocessor = SignalPreprocessor(window_size=30, sampling_rate=1, overlap=0.5)
    
    # Preprocess each subject
    X_processed = []
    for i, signals in enumerate(X_list):
        print(f"Processing subject {i+1}/{len(X_list)}...", end='\r')
        
        # Apply artifact removal and normalization
        processed_channels = []
        for channel_idx in range(signals.shape[0]):
            channel_data = signals[channel_idx]
            cleaned = preprocessor.remove_artifacts_ppg(channel_data)
            normalized = preprocessor.normalize_signal(cleaned)
            processed_channels.append(normalized)
        
        X_processed.append(np.stack(processed_channels, axis=0))
    
    print(f"✓ Preprocessed {len(X_processed)} subjects" + " " * 20)
    print()
    
    # ==================== STEP 4: CREATE WINDOWS ====================
    print("STEP 4: Creating Training Windows")
    print("-" * 70)
    
    all_windows = []
    all_labels = []
    
    for i, (signals, labels) in enumerate(zip(X_processed, y_list)):
        print(f"Windowing subject {i+1}/{len(X_processed)}...", end='\r')
        
        window_samples = 60
        step_samples = 30
        
        signal_length = signals.shape[1]
        n_windows = (signal_length - window_samples) // step_samples + 1
        
        for j in range(n_windows):
            start_idx = j * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > signal_length:
                break
            
            window = signals[:, start_idx:end_idx]
            label = np.bincount(labels[start_idx:end_idx]).argmax()
            
            all_windows.append(window)
            all_labels.append(label)
    
    X = np.array(all_windows)
    y = np.array(all_labels)
    
    print(f"✓ Created {len(X)} windows" + " " * 30)
    print(f"  Shape: {X.shape}")
    print(f"  Normal: {np.sum(y==0)}, Apnea: {np.sum(y==1)}")
    print(f"  Class balance: {np.sum(y==1)/len(y)*100:.1f}% apnea")
    print()
    
    # ==================== STEP 5: SPLIT DATA ====================
    print("STEP 5: Splitting Data")
    print("-" * 70)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]} windows")
    print(f"  Normal: {np.sum(y_train==0)}, Apnea: {np.sum(y_train==1)}")
    print(f"Val: {X_val.shape[0]} windows")
    print(f"  Normal: {np.sum(y_val==0)}, Apnea: {np.sum(y_val==1)}")
    print(f"Test: {X_test.shape[0]} windows")
    print(f"  Normal: {np.sum(y_test==0)}, Apnea: {np.sum(y_test==1)}")
    print()
    
    # Balanced sampler: oversample apnea so each batch is ~50/50
    # Do NOT also use class weights in the loss — that would double-penalise Normal
    n_normal = int(np.sum(y_train == 0))
    n_apnea  = int(np.sum(y_train == 1))
    sample_weights = np.where(y_train == 1, n_normal / n_apnea, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f"Sampler ratio — Normal:Apnea oversampling = 1:{n_normal/n_apnea:.1f}")
    print()

    # Create datasets
    train_dataset = SleepApneaDataset(X_train, y_train)
    val_dataset = SleepApneaDataset(X_val, y_val)
    test_dataset = SleepApneaDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ==================== STEP 6: TRAIN MODEL ====================
    model = SleepApneaDetector(
        input_channels=3,
        tcn_channels=[32, 64, 128],
        transformer_heads=4,
        transformer_layers=2,
        num_classes=2,
        dropout=0.1
    )
    total_params = sum(p.numel() for p in model.parameters())

    train_losses, val_losses = None, None

    if eval_only:
        print("STEP 6: Skipping Training (--eval-only)")
        print("-" * 70)
        if not Path('best_model.pth').exists():
            print("ERROR: best_model.pth not found. Run without --eval-only first.")
            return
        print(f"Model parameters: {total_params:,}")
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))
        print("✓ Loaded weights from best_model.pth")
        print()
        trainer = ModelTrainer(model, device=device, class_weights=None)
    else:
        print("STEP 6: Training Model")
        print("-" * 70)
        print(f"Model parameters: {total_params:,}")
        print()
        trainer = ModelTrainer(model, device=device, class_weights=None)
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            epochs=epochs,
            lr=learning_rate
        )
    
    # ==================== STEP 7: EVALUATE ====================
    print("\nSTEP 7: Evaluating Model")
    print("-" * 70)
    
    if not eval_only:
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.validate(test_loader, criterion)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print()

    model.eval()
    all_preds = []
    all_true  = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)[:, 1]
            all_true.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

    threshold = 0.5
    all_probs_arr = np.array(all_probs)
    all_preds = (all_probs_arr >= threshold).astype(int).tolist()

    print(f"Classification threshold: {threshold}")
    print("Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=['Normal', 'Apnea'], digits=3))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true, all_preds)
    print(f"               Predicted")
    print(f"              Normal  Apnea")
    print(f"Actual Normal  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Apnea   {cm[1,0]:6d}  {cm[1,1]:6d}")
    print()

    plot_evaluation_metrics(all_true, all_preds, all_probs, train_losses, val_losses)
    plot_threshold_sweep(all_true, all_probs)

    # ==================== STEP 8: SAVE ====================
    print("STEP 8: Saving Results")
    print("-" * 70)
    
    # # XAI
    # xai = ExplainableAI(model)
    # sample_idx = np.random.choice(len(X_test))
    # explanation = xai.explain_prediction(X_test[sample_idx])
    # fig_xai = xai.visualize_explanation(X_test[sample_idx], explanation)
    # plt.savefig('shhs_xai_explanation.png', dpi=150, bbox_inches='tight')
    # print("✓ XAI explanation saved")
    
    # # Quantization (move to CPU since MPS doesn't support quantization ops)
    # print("Quantizing model...")
    # model_cpu = model.cpu()  # Move to CPU for quantization
    # quantizer = TinyMLQuantizer()
    # quantized_model = quantizer.quantize_model(model_cpu, test_loader)
    # torch.save(quantized_model.state_dict(), 'shhs_quantized_model.pth')
    # print("✓ Quantized model saved")
    
    # Dataset info
    import json
    dataset_info = {
        'dataset': 'SHHS',
        'num_subjects': len(all_data),
        'total_duration_hours': total_duration / 3600,
        'total_apnea_hours': total_apnea / 3600,
        'apnea_percentage': (total_apnea / total_duration) * 100,
        'num_windows': len(X),
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    with open('shhs_dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print("✓ Dataset info saved")
    print()
    
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nDataset: SHHS ({len(all_data)} subjects, {total_duration/3600:.1f} hours)")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train on SHHS dataset')
    parser.add_argument('--processed-file', default='data/shhs_processed/shhs_processed.npz',
                       help='Path to preprocessed SHHS data')
    parser.add_argument('--max-subjects', type=int, default=None,
                       help='Maximum subjects to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training, load best_model.pth and evaluate only')

    args = parser.parse_args()

    train_with_shhs(
        processed_file=args.processed_file,
        max_subjects=args.max_subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_only=args.eval_only
    )
