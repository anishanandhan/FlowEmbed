"""
API Routes — REST endpoints for classification, embeddings, drift, and alerts.
"""

import numpy as np
import torch
import time
from typing import Optional, List
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from src.config import DEVICE, TRAFFIC_CLASSES
from src.explainability.llm_alerter import generate_alert

router = APIRouter()


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────

class FlowFeaturesRequest(BaseModel):
    """Input flow features for classification."""
    features: List[float]
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    protocol: Optional[str] = None

class ClassificationResponse(BaseModel):
    """Classification result."""
    traffic_class: str
    class_id: int
    confidence: float
    latency_ms: float
    embedding: Optional[List[float]] = None
    is_anomalous: bool = False

class AlertRequest(BaseModel):
    """Request for LLM alert generation."""
    traffic_class: str
    confidence: float
    flow_features: dict
    is_anomalous: bool = False

class KPIResponse(BaseModel):
    """Current KPI metrics."""
    intra_class_cosine: float
    inter_class_cosine: float
    classification_accuracy: float
    avg_latency_ms: float
    total_flows_classified: int
    drift_events: int


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from src.api.app import app_state
    return {
        "status": "healthy",
        "encoder_loaded": app_state["encoder"] is not None,
        "classifier_loaded": app_state["knn_classifier"] is not None,
        "scaler_loaded": app_state["scaler"] is not None,
    }


@router.post("/classify", response_model=ClassificationResponse)
async def classify_flow(request: FlowFeaturesRequest):
    """
    Classify a single network flow from its features.

    Pipeline: features → scale → encode → k-NN → class label + confidence
    """
    from src.api.app import app_state

    encoder = app_state.get("encoder")
    knn = app_state.get("knn_classifier")
    scaler = app_state.get("scaler")
    label_encoder = app_state.get("label_encoder")
    drift_detector = app_state.get("drift_detector")
    online_updater = app_state.get("online_updater")

    if encoder is None or knn is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first.")

    start_time = time.perf_counter()

    # Scale features
    features = np.array(request.features).reshape(1, -1).astype(np.float32)
    if scaler:
        features = scaler.transform(features)

    # Encode
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(DEVICE)
        embedding = encoder(features_tensor).cpu().numpy()

    # Classify
    predictions, confidences = knn.predict(embedding)

    latency_ms = (time.perf_counter() - start_time) * 1000

    pred_label = int(predictions[0])
    confidence = float(confidences[0])

    # Get class name
    if label_encoder:
        traffic_class = label_encoder.inverse_transform([pred_label])[0]
    else:
        traffic_class = TRAFFIC_CLASSES.get(pred_label, f"unknown_{pred_label}")

    # Check for anomaly (low confidence = potentially novel traffic)
    is_anomalous = confidence < 0.5

    # Update drift detector
    if drift_detector:
        drift_detector.update(confidence)

    # Update online learner buffer
    if online_updater:
        online_updater.add_to_buffer(embedding[0], pred_label)

    return ClassificationResponse(
        traffic_class=traffic_class,
        class_id=pred_label,
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
        embedding=embedding[0].tolist(),
        is_anomalous=is_anomalous,
    )


@router.post("/classify/batch")
async def classify_batch(flows: List[FlowFeaturesRequest]):
    """Classify multiple flows in a batch."""
    from src.api.app import app_state

    encoder = app_state.get("encoder")
    knn = app_state.get("knn_classifier")
    scaler = app_state.get("scaler")
    label_encoder = app_state.get("label_encoder")

    if encoder is None or knn is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    start_time = time.perf_counter()

    # Batch processing
    features = np.array([f.features for f in flows]).astype(np.float32)
    if scaler:
        features = scaler.transform(features)

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(DEVICE)
        embeddings = encoder(features_tensor).cpu().numpy()

    predictions, confidences = knn.predict(embeddings)
    latency_ms = (time.perf_counter() - start_time) * 1000

    results = []
    for i in range(len(flows)):
        pred_label = int(predictions[i])
        if label_encoder:
            traffic_class = label_encoder.inverse_transform([pred_label])[0]
        else:
            traffic_class = TRAFFIC_CLASSES.get(pred_label, f"unknown_{pred_label}")

        results.append({
            "traffic_class": traffic_class,
            "class_id": pred_label,
            "confidence": float(confidences[i]),
        })

    return {
        "results": results,
        "total_latency_ms": round(latency_ms, 2),
        "avg_latency_ms": round(latency_ms / len(flows), 2),
    }


@router.get("/embeddings")
async def get_embeddings():
    """Get UMAP-reduced embedding coordinates for visualization."""
    from src.api.app import app_state
    from src.explainability.umap_visualizer import get_plotly_data

    coords = app_state.get("umap_coords")
    labels = app_state.get("umap_labels")

    if coords is None or labels is None:
        raise HTTPException(status_code=404, detail="No embedding data available.")

    label_encoder = app_state.get("label_encoder")
    label_names = {}
    if label_encoder:
        for i, name in enumerate(label_encoder.classes_):
            label_names[i] = name
    else:
        label_names = TRAFFIC_CLASSES

    plotly_data = get_plotly_data(coords, labels, label_names=label_names)
    return plotly_data


@router.get("/drift")
async def get_drift_status():
    """Get concept drift detector status."""
    from src.api.app import app_state

    drift_detector = app_state.get("drift_detector")
    online_updater = app_state.get("online_updater")

    if drift_detector is None:
        raise HTTPException(status_code=404, detail="Drift detector not initialized.")

    status = drift_detector.get_status()
    if online_updater:
        status["updater"] = online_updater.get_status()

    return status


@router.post("/alert")
async def generate_alert_endpoint(request: AlertRequest):
    """Generate an LLM-powered security alert."""
    alert = generate_alert(
        traffic_class=request.traffic_class,
        confidence=request.confidence,
        flow_features=request.flow_features,
        is_anomalous=request.is_anomalous,
    )

    return {"alert": alert}


@router.get("/kpis")
async def get_kpis():
    """Get current KPI metrics."""
    from src.api.app import app_state

    drift_detector = app_state.get("drift_detector")

    # These would be computed from evaluation results in production
    return {
        "intra_class_cosine": 0.0,
        "inter_class_cosine": 0.0,
        "classification_accuracy": 0.0,
        "avg_latency_ms": 0.0,
        "total_flows_classified": drift_detector.observation_count if drift_detector else 0,
        "drift_events": len(drift_detector.drift_events) if drift_detector else 0,
    }


@router.get("/classes")
async def get_classes():
    """Get available traffic class labels."""
    from src.api.app import app_state

    label_encoder = app_state.get("label_encoder")
    if label_encoder:
        classes = {i: name for i, name in enumerate(label_encoder.classes_)}
    else:
        classes = TRAFFIC_CLASSES

    return {"classes": classes}
