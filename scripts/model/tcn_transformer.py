"""
tcn_transformer.py
-------------------
Hybrid TCN-Transformer model for binary sleep apnea classification.

Architecture:
  - TCN block: 3 dilated causal conv layers (dilation 1, 2, 4)
  - Transformer encoder: 2 layers, 4-head self-attention, d_model=128
  - Classification head: GAP → FC(64) → FC(2) → Softmax

Input:  (batch, 3, 60) — 3 channels × 60 time steps @ 1 Hz
Output: (batch, 2)     — [P(Normal), P(Apnea)]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── Dilated Causal Conv Block ───────────────────────── #

class CausalConvBlock(nn.Module):
    """Single residual dilated causal convolution block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 7, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: pad only left

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1×1 projection for residual when channels differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        out = out[:, :, :x.size(2)]   # trim causal padding
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


# ─────────────────────────── TCN Block ───────────────────────────────────── #

class TCNBlock(nn.Module):
    """3-layer dilated causal TCN."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConvBlock(in_channels,  64,  dilation=1, dropout=dropout),
            CausalConvBlock(64,          128,  dilation=2, dropout=dropout),
            CausalConvBlock(128,         256,  dilation=4, dropout=dropout),
        )
        # Project to d_model=128 for Transformer
        self.proj = nn.Conv1d(256, 128, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → out: (B, 128, T)
        return self.proj(self.layers(x))


# ──────────────────── Sinusoidal Positional Encoding ────────────────────── #

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ───────────────────────── Full Model ────────────────────────────────────── #

class TCNTransformer(nn.Module):
    """
    Hybrid TCN-Transformer for sleep apnea binary classification.

    Args:
        in_channels:  Number of input signal channels (default: 3)
        d_model:      Transformer model dimension (default: 128)
        n_heads:      Attention heads (default: 4)
        n_layers:     Transformer encoder layers (default: 2)
        dim_ff:       Feedforward dimension (default: 256)
        dropout:      Dropout probability (default: 0.1)
        n_classes:    Output classes — 2 for Normal/Apnea (default: 2)
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.tcn = TCNBlock(in_channels=in_channels, dropout=dropout)

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — batch × channels × time
        Returns:
            logits: (B, n_classes)
        """
        # TCN: (B, C, T) → (B, 128, T)
        tcn_out = self.tcn(x)

        # Rearrange for Transformer: (B, T, 128)
        tf_in = tcn_out.permute(0, 2, 1)
        tf_in = self.pos_enc(tf_in)

        # Transformer: (B, T, 128)
        tf_out = self.transformer(tf_in)

        # Global average pooling: (B, 128)
        pooled = tf_out.mean(dim=1)

        # Classification head: (B, 2)
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, threshold: float = 0.45) -> list[str]:
        """Returns list of 'NORMAL' / 'APNEA' strings."""
        probs = self.predict_proba(x)
        p_apnea = probs[:, 1]
        return ["APNEA" if p > threshold else "NORMAL" for p in p_apnea]


# ──────────────────────────── Model summary ──────────────────────────────── #

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TCNTransformer()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Smoke test
    dummy = torch.randn(8, 3, 60)
    logits = model(dummy)
    probs  = model.predict_proba(dummy)
    preds  = model.predict(dummy)

    print(f"\nInput shape:  {dummy.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape:  {probs.shape}")
    print(f"Predictions:  {preds}")
