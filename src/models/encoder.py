"""
Flow Encoder — Neural network architectures for encoding flow features into embeddings.

Primary: Transformer Encoder (2-layer, 4-head attention)
Ablation: CNN + BiLSTM with attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.config import (
    D_MODEL, N_HEADS, N_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    EMBEDDING_DIM, NUM_INPUT_FEATURES,
    CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS,
)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer.
    Adds position information to feature "tokens".
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerFlowEncoder(nn.Module):
    """
    Transformer-based flow encoder.

    Architecture:
        Input [batch, num_features]
        → Linear projection to d_model
        → Treat each feature as a "token" (seq_len = num_features)
        → Positional encoding
        → 2-layer Transformer encoder (4 heads, feedforward=512)
        → CLS token pooling or global average pooling
        → Output [batch, embedding_dim]

    This architecture treats each flow feature as a "token" in a sequence,
    allowing the self-attention mechanism to learn feature-feature interactions
    (e.g., the relationship between packet size and inter-arrival time).
    """

    def __init__(
        self,
        input_dim: int = NUM_INPUT_FEATURES,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        embedding_dim: int = EMBEDDING_DIM,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # Project each feature to d_model dimensions
        # Each feature value becomes a d_model-dimensional token
        self.feature_embedding = nn.Linear(1, d_model)

        # Learnable CLS token for classification pooling
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim + 1, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Final projection to embedding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flow features [batch, num_features]

        Returns:
            Embedding [batch, embedding_dim]
        """
        batch_size = x.size(0)

        # Reshape: [batch, num_features] → [batch, num_features, 1]
        x = x.unsqueeze(-1)

        # Project each feature to d_model: [batch, num_features, d_model]
        x = self.feature_embedding(x)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_features+1, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Pooling
        if self.use_cls_token:
            # Use CLS token output
            x = x[:, 0]  # [batch, d_model]
        else:
            # Global average pooling over all tokens
            x = x.mean(dim=1)  # [batch, d_model]

        # Project to embedding space
        embedding = self.output_projection(x)  # [batch, embedding_dim]

        return embedding


class CNNBiLSTMEncoder(nn.Module):
    """
    CNN + BiLSTM flow encoder (ablation alternative).

    Architecture:
        Input [batch, num_features]
        → Reshape to [batch, 1, num_features]
        → Conv1D(1, 64, kernel=3) → ReLU → BatchNorm
        → Conv1D(64, 128, kernel=3) → ReLU → BatchNorm
        → BiLSTM(128, hidden=128)
        → Attention pooling
        → Output [batch, embedding_dim]
    """

    def __init__(
        self,
        input_dim: int = NUM_INPUT_FEATURES,
        cnn_channels: list = None,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = CNN_CHANNELS

        # CNN feature extractor
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention layer for weighted pooling of LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_hidden * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flow features [batch, num_features]

        Returns:
            Embedding [batch, embedding_dim]
        """
        # Reshape for Conv1D: [batch, 1, num_features]
        x = x.unsqueeze(1)

        # CNN feature extraction: [batch, cnn_channels[-1], num_features]
        x = self.cnn(x)

        # Permute for LSTM: [batch, num_features, cnn_channels[-1]]
        x = x.permute(0, 2, 1)

        # BiLSTM: [batch, num_features, lstm_hidden*2]
        lstm_out, _ = self.lstm(x)

        # Attention pooling
        attn_weights = self.attention(lstm_out)  # [batch, num_features, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # [batch, lstm_hidden*2]

        # Project to embedding space
        embedding = self.output_projection(context)

        return embedding


class MLPFlowEncoder(nn.Module):
    """
    MLP-based flow encoder — simple, stable, and effective for tabular flow data.

    Architecture:
        Input [batch, num_features]
        → Linear(num_features, 256) → BatchNorm → GELU → Dropout
        → Linear(256, 256) → BatchNorm → GELU → Dropout + Skip Connection
        → Linear(256, 256) → BatchNorm → GELU → Dropout + Skip Connection
        → Linear(256, embedding_dim)
        → Output [batch, embedding_dim]

    Why MLP over Transformer for flow features:
    - Tabular data with ~60 features doesn't benefit from self-attention's
      O(n²) complexity — features aren't sequential like tokens
    - MLPs with BatchNorm are more resistant to embedding collapse
    - Skip connections prevent gradient vanishing in contrastive training
    - Much faster training: ~10x fewer parameters
    """

    def __init__(
        self,
        input_dim: int = NUM_INPUT_FEATURES,
        hidden_dim: int = 256,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
        num_layers: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.residual_blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for GELU activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flow features [batch, num_features]

        Returns:
            Embedding [batch, embedding_dim]
        """
        # Input projection
        h = self.input_proj(x)

        # Residual blocks with skip connections
        for block in self.residual_blocks:
            h = h + block(h)  # Skip connection

        # Output projection
        embedding = self.output_proj(h)
        return embedding


def get_encoder(
    encoder_type: str = "transformer",
    input_dim: int = NUM_INPUT_FEATURES,
    embedding_dim: int = EMBEDDING_DIM,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get the appropriate encoder.

    Args:
        encoder_type: "transformer", "cnn_bilstm", or "mlp".
        input_dim: Number of input flow features.
        embedding_dim: Output embedding dimension.

    Returns:
        Encoder module.
    """
    if encoder_type == "transformer":
        return TransformerFlowEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    elif encoder_type == "cnn_bilstm":
        return CNNBiLSTMEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    elif encoder_type == "mlp":
        return MLPFlowEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'transformer', 'cnn_bilstm', or 'mlp'.")
