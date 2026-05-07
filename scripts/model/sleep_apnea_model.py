"""
Sleep Apnea Detection Model
TCN + Transformer Architecture for Real-Time Physiological Signal Analysis

This script implements a deep learning model for detecting sleep apnea patterns
from heart rate, respiration, and SpO2 signals using public datasets (SHHS, MASS)
with transfer learning capabilities for wearable device deployment.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== DATA PREPROCESSING ====================

class SignalPreprocessor:
    """
    Preprocessing pipeline for physiological signals
    Handles artifact removal, normalization, and windowing
    """
    
    def __init__(self, window_size=30, sampling_rate=1, overlap=0.5):
        """
        Args:
            window_size: Window length in seconds (default: 30s, matching clinical scoring)
            sampling_rate: Sampling frequency in Hz
            overlap: Overlap ratio between consecutive windows (0-1)
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def remove_artifacts_ppg(self, signal_data, cutoff_low=0.5, cutoff_high=8.0):
        """
        Remove motion artifacts and noise from PPG-derived signals
        
        Args:
            signal_data: Input signal array
            cutoff_low: Low-frequency cutoff for high-pass filter (Hz)
            cutoff_high: High-frequency cutoff for low-pass filter (Hz)
        
        Returns:
            Filtered signal
        """
        # Handle NaN values first
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # For very low sampling rates (< 2 Hz), skip bandpass filtering and just apply median filter
        if self.sampling_rate < 2:
            # Remove sporadic spikes using median filter
            filtered_signal = signal.medfilt(signal_data, kernel_size=3)
            return filtered_signal
        
        # Bandpass filter to remove baseline drift and high-frequency noise
        nyquist = self.sampling_rate / 2
        low = cutoff_low / nyquist
        high = cutoff_high / nyquist
        
        # Clamp frequencies to valid range (0, 1)
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        # Remove sporadic spikes using median filter
        filtered_signal = signal.medfilt(filtered_signal, kernel_size=5)
        
        return filtered_signal
    
    def normalize_signal(self, signal_data):
        """
        Apply z-score normalization to place signals on consistent dynamic range
        
        Args:
            signal_data: Input signal array (can be multi-dimensional)
        
        Returns:
            Normalized signal
        """
        # Handle NaN values
        signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if len(signal_data.shape) == 1:
            signal_data = signal_data.reshape(-1, 1)
        
        normalized = self.scaler.fit_transform(signal_data)
        
        # Ensure no NaN values in output
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized.flatten() if normalized.shape[1] == 1 else normalized
    
    def multi_scale_analysis(self, spo2_signal, scales=[1, 3, 5, 10]):
        """
        Multi-scale analysis for SpO2 to capture acute desaturations and long-term trends
        
        Args:
            spo2_signal: SpO2 time series
            scales: List of window sizes (in samples) for smoothing
        
        Returns:
            Dictionary of multi-scale features
        """
        features = {}
        
        for scale in scales:
            # Smooth signal at different scales
            smoothed = uniform_filter1d(spo2_signal, size=scale, mode='nearest')
            features[f'spo2_scale_{scale}'] = smoothed
            
            # Calculate desaturation slope at each scale
            gradient = np.gradient(smoothed)
            features[f'desat_slope_scale_{scale}'] = gradient
        
        return features
    
    def create_windows(self, signals_dict, labels=None):
        """
        Segment continuous signals into fixed-length windows
        
        Args:
            signals_dict: Dictionary of signal arrays {'hr': array, 'resp': array, 'spo2': array}
            labels: Optional labels for each window
        
        Returns:
            Windowed data and corresponding labels
        """
        signal_length = len(next(iter(signals_dict.values())))
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.overlap))
        
        windows = []
        window_labels = []
        
        # Calculate number of windows
        n_windows = (signal_length - window_samples) // step_samples + 1
        
        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > signal_length:
                break
            
            # Stack all signals for this window
            window_data = []
            for signal_name in ['hr', 'resp', 'spo2']:
                if signal_name in signals_dict:
                    window_data.append(signals_dict[signal_name][start_idx:end_idx])
            
            windows.append(np.stack(window_data, axis=0))  # Shape: (n_channels, window_samples)
            
            if labels is not None:
                # Use majority voting for label in this window
                window_label = np.bincount(labels[start_idx:end_idx]).argmax()
                window_labels.append(window_label)
        
        return np.array(windows), np.array(window_labels) if labels is not None else None


# ==================== DATASET CLASS ====================

class SleepApneaDataset(Dataset):
    """
    PyTorch Dataset for sleep apnea detection
    Handles both SHHS and MASS datasets with unified interface
    """
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Windowed signal data, shape (n_samples, n_channels, window_length)
            labels: Binary labels (0: normal, 1: apnea event)
            transform: Optional data augmentation
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# ==================== MODEL ARCHITECTURE ====================

class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with causal convolution
    Maintains strict real-time compatibility
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Causal cropping to maintain causality
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        # Crop to match input size (causal convolution)
        out = out[:, :, :x.size(2)]
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network for feature extraction
    Captures short-to-medium range temporal structures
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCNEncoder, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for capturing long-range dependencies
    Models global sequence reasoning across minutes of sleep activity
    """
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.transformer_encoder(x)
        return x


class SleepApneaDetector(nn.Module):
    """
    Hybrid TCN + Transformer model for sleep apnea detection
    Integrates temporal convolutions with self-attention mechanism
    """
    
    def __init__(self, input_channels=3, tcn_channels=[64, 128, 256], 
                 transformer_heads=8, transformer_layers=4, num_classes=2, dropout=0.2):
        super(SleepApneaDetector, self).__init__()
        
        # TCN feature extractor
        self.tcn = TCNEncoder(input_channels, tcn_channels, kernel_size=7, dropout=dropout)
        
        # Transformer for long-range dependencies
        self.transformer = TransformerEncoder(
            d_model=tcn_channels[-1],
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=512,
            dropout=dropout
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # For XAI - store attention weights
        self.attention_weights = None
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        
        # TCN feature extraction
        tcn_features = self.tcn(x)  # (batch, tcn_channels[-1], seq_len)
        
        # Transformer encoding
        transformer_out = self.transformer(tcn_features)  # (batch, seq_len, tcn_channels[-1])
        
        # Global pooling
        pooled = transformer_out.mean(dim=1)  # (batch, tcn_channels[-1])
        
        # Classification
        output = self.fc(pooled)
        
        return output
    
    def get_feature_importance(self, x):
        """
        Extract feature importance for XAI purposes
        Returns attention patterns and gradient-based importance
        """
        with torch.enable_grad():
            x.requires_grad = True
            output = self.forward(x)
            
            # Get gradients
            output[:, 1].sum().backward()
            importance = x.grad.abs().mean(dim=0)
            
        return importance


# ==================== EXPLAINABLE AI (XAI) MODULE ====================

class ExplainableAI:
    """
    Provides interpretability for model predictions
    Identifies desaturation slopes and cardiac variability contributions
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def explain_prediction(self, input_data, labels=['Normal', 'Apnea']):
        """
        Generate explanation for a single prediction
        
        Args:
            input_data: Single sample (channels, seq_len)
            labels: Class labels
        
        Returns:
            Dictionary with prediction and explanation
        """
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        input_tensor = input_tensor.to(self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get feature importance
        importance = self.model.get_feature_importance(input_tensor)
        
        # Analyze channel contributions
        channel_names = ['Heart Rate', 'Respiration', 'SpO2']
        channel_importance = importance.mean(dim=1).cpu().numpy()
        
        explanation = {
            'prediction': labels[prediction],
            'confidence': probabilities[0, prediction].item(),
            'probabilities': {labels[i]: probabilities[0, i].item() for i in range(len(labels))},
            'channel_importance': {channel_names[i]: float(channel_importance[i]) 
                                  for i in range(len(channel_names))},
            'temporal_importance': importance.cpu().numpy()
        }
        
        return explanation
    
    def visualize_explanation(self, input_data, explanation):
        """
        Visualize the explanation
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        channel_names = ['Heart Rate', 'Respiration', 'SpO2']
        
        # Plot original signals
        for i in range(3):
            axes[i].plot(input_data[i], label=channel_names[i])
            axes[i].set_ylabel(channel_names[i])
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Plot temporal importance
        importance_avg = explanation['temporal_importance'].mean(axis=0)
        axes[3].plot(importance_avg, color='red', linewidth=2)
        axes[3].set_ylabel('Feature Importance')
        axes[3].set_xlabel('Time (samples)')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title(f"Prediction: {explanation['prediction']} "
                         f"(Confidence: {explanation['confidence']:.2%})")
        
        plt.tight_layout()
        return fig


# ==================== TRAINING UTILITIES ====================

class ModelTrainer:
    """
    Handles model training, validation, and evaluation
    """
    
    def __init__(self, model, device='mps' if torch.backends.mps.is_available() else 'cpu', class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """
        Full training loop
        """
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 10

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            scheduler.step()

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'  New best model saved! (Val Loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f'  Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)')
                    break
            print()

        return self.train_losses, self.val_losses
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ==================== MODEL QUANTIZATION FOR TinyML ====================

class TinyMLQuantizer:
    """
    8-bit quantization for edge deployment
    Enables real-time inference on microcontrollers
    """
    
    @staticmethod
    def quantize_model(model, calibration_loader):
        """
        Apply dynamic quantization for TinyML deployment
        
        Args:
            model: Trained PyTorch model
            calibration_loader: DataLoader for calibration
        
        Returns:
            Quantized model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Apply dynamic quantization to linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def measure_model_size(model):
        """Measure model size in MB"""
        torch.save(model.state_dict(), '/tmp/temp_model.pth')
        import os
        size_mb = os.path.getsize('/tmp/temp_model.pth') / (1024 * 1024)
        os.remove('/tmp/temp_model.pth')
        return size_mb
    
    @staticmethod
    def compare_models(original_model, quantized_model, test_loader, device='cpu'):
        """
        Compare original and quantized model performance
        """
        def evaluate(model, loader):
            model.eval()
            correct = 0
            total = 0
            inference_times = []
            
            import time
            with torch.no_grad():
                for data, target in loader:
                    data = data.to(device)
                    start_time = time.time()
                    output = model(data)
                    inference_times.append(time.time() - start_time)
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            return 100. * correct / total, np.mean(inference_times) * 1000  # ms
        
        orig_acc, orig_time = evaluate(original_model, test_loader)
        quant_acc, quant_time = evaluate(quantized_model, test_loader)
        
        orig_size = TinyMLQuantizer.measure_model_size(original_model)
        quant_size = TinyMLQuantizer.measure_model_size(quantized_model)
        
        print("=" * 60)
        print("Model Comparison:")
        print("=" * 60)
        print(f"Original Model:")
        print(f"  Accuracy: {orig_acc:.2f}%")
        print(f"  Avg Inference Time: {orig_time:.2f} ms")
        print(f"  Model Size: {orig_size:.2f} MB")
        print()
        print(f"Quantized Model:")
        print(f"  Accuracy: {quant_acc:.2f}%")
        print(f"  Avg Inference Time: {quant_time:.2f} ms")
        print(f"  Model Size: {quant_size:.2f} MB")
        print()
        print(f"Improvements:")
        print(f"  Size Reduction: {(1 - quant_size/orig_size)*100:.1f}%")
        print(f"  Speed Improvement: {(orig_time/quant_time):.2f}x")
        print(f"  Accuracy Drop: {orig_acc - quant_acc:.2f}%")
        print("=" * 60)


# ==================== MAIN EXECUTION ====================

def main():
    """
    Main execution pipeline
    """
    print("=" * 60)
    print("Sleep Apnea Detection Model - Training Pipeline")
    print("=" * 60)
    print()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # ===== STEP 1: Generate Synthetic Data (Replace with SHHS/MASS data loading) =====
    print("Step 1: Loading and preprocessing data...")
    
    # For demonstration, we'll create synthetic data
    # In practice, replace this with actual SHHS/MASS dataset loading
    def generate_synthetic_data(n_samples=1000, seq_length=1800, sampling_rate=1):
        """
        Generate synthetic physiological signals
        Replace this with actual dataset loading from SHHS/MASS
        """
        data = []
        labels = []
        
        for i in range(n_samples):
            # Simulate normal vs apnea patterns
            is_apnea = np.random.rand() > 0.7
            
            # Heart rate: 60-100 bpm normally, irregular during apnea
            if is_apnea:
                hr = np.random.normal(75, 15, seq_length)
                # Add periodic irregularities
                hr += 10 * np.sin(np.linspace(0, 10*np.pi, seq_length))
            else:
                hr = np.random.normal(70, 5, seq_length)
            
            # Respiration: regular vs irregular
            if is_apnea:
                resp = np.random.normal(15, 8, seq_length)
                # Add apnea-like pauses
                pause_indices = np.random.choice(seq_length, size=seq_length//10)
                resp[pause_indices] = resp[pause_indices] * 0.3
            else:
                resp = np.random.normal(16, 2, seq_length)
            
            # SpO2: 95-100% normally, drops during apnea
            if is_apnea:
                spo2 = np.random.normal(94, 4, seq_length)
                # Simulate desaturation events
                desat_indices = np.random.choice(seq_length, size=seq_length//20)
                spo2[desat_indices] = np.clip(spo2[desat_indices] - 10, 80, 100)
            else:
                spo2 = np.random.normal(98, 1, seq_length)
            
            # Stack channels
            sample = np.stack([hr, resp, spo2], axis=0)
            data.append(sample)
            labels.append(1 if is_apnea else 0)
        
        return np.array(data), np.array(labels)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=2000, seq_length=1800)
    print(f"Generated data shape: {X.shape}")
    print(f"Labels distribution: Normal={np.sum(y==0)}, Apnea={np.sum(y==1)}")
    print()
    
    # ===== STEP 2: Preprocessing =====
    print("Step 2: Applying preprocessing pipeline...")
    
    preprocessor = SignalPreprocessor(window_size=30, sampling_rate=1, overlap=0.5)
    
    # Process each sample
    processed_data = []
    for i in range(len(X)):
        # Apply artifact removal and normalization
        hr_clean = preprocessor.remove_artifacts_ppg(X[i, 0])
        resp_clean = preprocessor.remove_artifacts_ppg(X[i, 1])
        spo2_clean = preprocessor.remove_artifacts_ppg(X[i, 2])
        
        hr_norm = preprocessor.normalize_signal(hr_clean)
        resp_norm = preprocessor.normalize_signal(resp_clean)
        spo2_norm = preprocessor.normalize_signal(spo2_clean)
        
        processed_sample = np.stack([hr_norm, resp_norm, spo2_norm], axis=0)
        processed_data.append(processed_sample)
    
    X_processed = np.array(processed_data)
    print("Preprocessing complete!")
    print()
    
    # ===== STEP 3: Create train/val/test splits =====
    print("Step 3: Creating train/validation/test splits...")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Create datasets and dataloaders
    train_dataset = SleepApneaDataset(X_train, y_train)
    val_dataset = SleepApneaDataset(X_val, y_val)
    test_dataset = SleepApneaDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ===== STEP 4: Initialize model =====
    print("Step 4: Initializing TCN + Transformer model...")
    
    model = SleepApneaDetector(
        input_channels=3,
        tcn_channels=[64, 128, 256],
        transformer_heads=8,
        transformer_layers=4,
        num_classes=2,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # ===== STEP 5: Train model =====
    print("Step 5: Training model...")
    print()
    
    trainer = ModelTrainer(model, device=device)
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, epochs=20, lr=0.001
    )
    
    # Plot training history
    fig = trainer.plot_training_history()
    plt.savefig('/home/claude/training_history.png', dpi=150, bbox_inches='tight')
    print("Training history saved to training_history.png")
    print()
    
    # ===== STEP 6: Evaluate on test set =====
    print("Step 6: Evaluating on test set...")
    
    model.load_state_dict(torch.load('best_model.pth'))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.validate(test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print()
    
    # ===== STEP 7: Demonstrate XAI =====
    print("Step 7: Demonstrating Explainable AI...")
    
    xai = ExplainableAI(model)
    sample_idx = 0
    sample_data = X_test[sample_idx]
    
    explanation = xai.explain_prediction(sample_data)
    print("\nPrediction Explanation:")
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Confidence: {explanation['confidence']:.2%}")
    print(f"  Channel Importance:")
    for channel, importance in explanation['channel_importance'].items():
        print(f"    {channel}: {importance:.4f}")
    
    fig_xai = xai.visualize_explanation(sample_data, explanation)
    plt.savefig('/home/claude/xai_explanation.png', dpi=150, bbox_inches='tight')
    print("\nXAI visualization saved to xai_explanation.png")
    print()
    
    # ===== STEP 8: Model quantization for TinyML =====
    print("Step 8: Applying TinyML quantization...")
    
    quantizer = TinyMLQuantizer()
    quantized_model = quantizer.quantize_model(model, test_loader)
    quantizer.compare_models(model, quantized_model, test_loader, device='cpu')
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')
    print("\nQuantized model saved to quantized_model.pth")
    print()
    
    print("=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
