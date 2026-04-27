"""
FastAPI Application — Main entry point for the backend API.

Serves real-time traffic classification, embedding visualization,
drift detection status, and LLM alerts via REST + WebSocket.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import CORS_ORIGINS, CHECKPOINTS_DIR, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)

# Global state for loaded models
app_state = {
    "encoder": None,
    "knn_classifier": None,
    "scaler": None,
    "label_encoder": None,
    "drift_detector": None,
    "online_updater": None,
    "explainer": None,
    "umap_coords": None,
    "umap_labels": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("🚀 Loading models...")

    try:
        _load_models()
        logger.info("✅ All models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        logger.info("API will start but classification endpoints will return errors.")

    yield

    logger.info("👋 Shutting down...")


def _load_models():
    """Load encoder, classifier, scaler from saved checkpoints."""
    import torch
    import joblib
    import numpy as np

    from src.config import DEVICE, SPLITS_DIR
    from src.models.encoder import get_encoder
    from src.classifier.knn_classifier import FAISSKNNClassifier
    from src.drift.drift_detector import DriftDetector
    from src.drift.online_updater import OnlineUpdater
    from src.explainability.llm_alerter import preload_cache

    # Load encoder
    checkpoint_path = CHECKPOINTS_DIR / "best_encoder.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        encoder = get_encoder(
            encoder_type=checkpoint.get("encoder_type", "transformer"),
            input_dim=checkpoint.get("input_dim", 40),
        )
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.to(DEVICE)
        encoder.eval()
        app_state["encoder"] = encoder
        logger.info("Loaded encoder checkpoint")

    # Load k-NN classifier
    knn = FAISSKNNClassifier()
    faiss_path = EMBEDDINGS_DIR / "faiss_index.bin"
    if faiss_path.exists():
        knn.load(str(faiss_path))
        app_state["knn_classifier"] = knn
        logger.info("Loaded FAISS index")

    # Load scaler and label encoder
    scaler_path = SPLITS_DIR / "scaler.pkl"
    if scaler_path.exists():
        app_state["scaler"] = joblib.load(scaler_path)

    le_path = SPLITS_DIR / "label_encoder.pkl"
    if le_path.exists():
        app_state["label_encoder"] = joblib.load(le_path)

    # Initialize drift detector
    drift_detector = DriftDetector()
    knn = app_state.get("knn_classifier")
    if knn:
        updater = OnlineUpdater(knn)
        drift_detector.on_drift_callback = updater.refresh_index
        app_state["online_updater"] = updater
    app_state["drift_detector"] = drift_detector

    # Pre-load UMAP coordinates if available
    umap_path = EMBEDDINGS_DIR / "umap_coords.npy"
    if umap_path.exists():
        app_state["umap_coords"] = np.load(umap_path)
        app_state["umap_labels"] = np.load(EMBEDDINGS_DIR / "umap_labels.npy")

    # Pre-load LLM alert cache
    preload_cache()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FlowEmbed — Network Traffic Classifier",
        description=(
            "Context-Aware Flow Embeddings for Adaptive AI-Based "
            "Network Traffic Classification"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routes
    from src.api.routes import router
    from src.api.websocket import ws_router

    app.include_router(router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")

    return app


# Create the app instance
app = create_app()
