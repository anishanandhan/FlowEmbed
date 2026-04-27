"""
Configuration — All hyperparameters, paths, and constants in one place.
"""

import os
import torch
from pathlib import Path


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
for d in [PROCESSED_DIR, SPLITS_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
def get_device():
    """Get best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ─────────────────────────────────────────────
# Data Pipeline
# ─────────────────────────────────────────────
# Number of first packet sizes to capture as a sequence feature
FIRST_N_PACKETS = 20

# Features to extract from flow data (our novel feature set)
FLOW_FEATURES = [
    # === Basic flow statistics ===
    "flow_duration_sec", "total_packets", "total_bytes",
    "packets_per_sec", "bytes_per_sec",

    # === Packet size statistics ===
    "pkt_size_min", "pkt_size_max", "pkt_size_mean", "pkt_size_std",
    "pkt_size_median", "pkt_size_q25", "pkt_size_q75", "pkt_size_iqr",
    "pkt_size_skewness", "pkt_size_kurtosis",

    # === Inter-arrival time statistics ===
    "iat_min_ms", "iat_max_ms", "iat_mean_ms", "iat_std_ms", "iat_median_ms",
    "iat_q25_ms", "iat_q75_ms",

    # === Protocol encoding ===
    "protocol_encoded", "is_encrypted",

    # === NOVELTY 1: Entropy features ===
    # High entropy = unpredictable sizes (browsing)
    # Low entropy = regular sizes (streaming, VoIP)
    "pkt_size_entropy", "iat_entropy",

    # === NOVELTY 2: Burst detection features ===
    # Captures the "rhythm" of traffic
    "burst_count", "burst_mean_size", "burst_mean_duration_ms",
    "burst_rate", "inter_burst_mean_ms", "burst_bytes_ratio",

    # === NOVELTY 3: First-N packet sizes (handshake fingerprint) ===
    *[f"first_pkt_{i}" for i in range(FIRST_N_PACKETS)],

    # === NOVELTY 4: Context-aware host features (FlowXpert-inspired) ===
    # Captures behavioral patterns of the source HOST, not just the flow
    "ctx_dst_ip_count", "ctx_protocol_diversity",
    "ctx_connection_rate", "ctx_packets_per_sec", "ctx_bytes_per_sec",

    # === Derived ratios ===
    "avg_payload_ratio", "iat_coefficient_of_variation",
    "pkt_size_coefficient_of_variation",
]

# Number of features after preprocessing
NUM_INPUT_FEATURES = len(FLOW_FEATURES)  # ~56 features

# Class label mapping
TRAFFIC_CLASSES = {
    0: "streaming",       # YouTube, Netflix
    1: "gaming",          # Mobile gaming, cloud gaming
    2: "voip",            # WhatsApp call, Zoom
    3: "social_media",    # Instagram, TikTok
    4: "browsing",        # General web browsing
    5: "file_transfer",   # Downloads, cloud sync
    6: "vpn",             # VPN tunnel traffic
    7: "malware_c2",      # Command & control traffic
    8: "xr_ar",           # XR/AR traffic (few-shot target)
}

NUM_CLASSES = len(TRAFFIC_CLASSES)

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ─────────────────────────────────────────────
# Model — Transformer Encoder
# ─────────────────────────────────────────────
# Transformer architecture
D_MODEL = 128           # Internal dimension of transformer
N_HEADS = 4             # Number of attention heads
N_ENCODER_LAYERS = 2    # Number of transformer encoder layers
DIM_FEEDFORWARD = 512   # Feedforward network dimension
DROPOUT = 0.1           # Dropout rate

# Embedding output
EMBEDDING_DIM = 256     # Final embedding dimension

# CNN-BiLSTM alternative
CNN_CHANNELS = [64, 128]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2


# ─────────────────────────────────────────────
# Contrastive Learning
# ─────────────────────────────────────────────
TEMPERATURE = 0.07       # NT-Xent temperature parameter (τ)
TRIPLET_MARGIN = 0.5     # Triplet loss margin

# Augmentation parameters
AUG_NOISE_STD = 0.01     # Gaussian noise standard deviation
AUG_MASK_RATIO = 0.15    # Feature masking ratio
AUG_JITTER_RATIO = 0.05  # Temporal jitter ratio


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE = 512         # Larger batches = more negatives = better contrastive learning
LEARNING_RATE = 3e-4     # AdamW learning rate
WEIGHT_DECAY = 1e-4      # AdamW weight decay
NUM_EPOCHS = 100         # Training epochs
WARMUP_EPOCHS = 10       # Linear warmup epochs
SCHEDULER = "cosine"     # Learning rate scheduler: "cosine" or "step"

# Early stopping
PATIENCE = 15            # Stop if no improvement for N epochs
MIN_DELTA = 1e-4         # Minimum improvement threshold


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────
KNN_K = 5                # k for k-NN classifier
KNN_METRIC = "cosine"    # Distance metric for FAISS
SVM_KERNEL = "rbf"       # SVM kernel
SVM_C = 1.0              # SVM regularization


# ─────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────
ADWIN_DELTA = 0.002      # ADWIN sensitivity (lower = more sensitive)
DRIFT_WINDOW_SIZE = 100  # Number of recent flows to consider for drift
DRIFT_REFRESH_SIZE = 50  # Number of flows to add to index on drift


# ─────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────
OLLAMA_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"  # Quantized for speed
OLLAMA_HOST = "http://localhost:11434"
SHAP_BACKGROUND_SIZE = 100   # Number of background samples for SHAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


# ─────────────────────────────────────────────
# KPI Targets
# ─────────────────────────────────────────────
KPI_INTRA_CLASS_COSINE = 0.7    # Target: cosine similarity > 0.7 for same class
KPI_INTER_CLASS_COSINE = 0.3    # Target: cosine similarity < 0.3 for different classes
KPI_ACCURACY = 0.90             # Target: ≥ 90% classification accuracy
KPI_GENERALIZATION = 0.85       # Target: ≥ 85% on unseen traffic types
KPI_LATENCY_MS = 100            # Target: < 100ms per flow
