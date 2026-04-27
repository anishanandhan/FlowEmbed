"""
Live Classification — Real-time traffic capture and classification pipeline.

Captures live network traffic via nfstream, encodes flows with the trained
Transformer encoder, classifies via FAISS k-NN, detects drift, generates
LLM alerts, and pushes results to the dashboard via FastAPI/WebSocket.

Usage:
    python scripts/run_live.py --interface en0  # WiFi
    python scripts/run_live.py --pcap demo.pcap  # Replay PCAP
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEVICE, CHECKPOINTS_DIR, SPLITS_DIR, EMBEDDINGS_DIR, TRAFFIC_CLASSES
from src.models.encoder import get_encoder
from src.classifier.knn_classifier import FAISSKNNClassifier
from src.drift.drift_detector import DriftDetector
from src.drift.online_updater import OnlineUpdater
from src.explainability.llm_alerter import generate_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Live traffic classification")
    parser.add_argument("--interface", type=str, default="en0", help="Network interface")
    parser.add_argument("--pcap", type=str, default=None, help="PCAP file to replay instead of live")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_encoder.pt"))
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--alert-threshold", type=float, default=0.5,
                        help="Confidence threshold below which to generate alerts")
    return parser.parse_args()


def load_pipeline(checkpoint_path: str):
    """Load the full classification pipeline."""
    # Encoder
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    encoder = get_encoder(
        encoder_type=checkpoint.get("encoder_type", "transformer"),
        input_dim=checkpoint.get("input_dim", 40),
        embedding_dim=checkpoint.get("embedding_dim", 256),
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(DEVICE)
    encoder.eval()

    # Scaler
    scaler = joblib.load(SPLITS_DIR / "scaler.pkl")

    # Label encoder
    label_encoder = joblib.load(SPLITS_DIR / "label_encoder.pkl")

    # k-NN
    knn = FAISSKNNClassifier()
    knn.load()

    # Drift
    drift_detector = DriftDetector()
    updater = OnlineUpdater(knn)
    drift_detector.on_drift_callback = updater.refresh_index

    return {
        "encoder": encoder,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "knn": knn,
        "drift_detector": drift_detector,
        "updater": updater,
    }


def classify_flow(pipeline: dict, flow_features: np.ndarray) -> dict:
    """Classify a single flow through the full pipeline."""
    start = time.perf_counter()

    # Scale
    features = pipeline["scaler"].transform(flow_features.reshape(1, -1))

    # Encode
    with torch.no_grad():
        tensor = torch.FloatTensor(features).to(DEVICE)
        embedding = pipeline["encoder"](tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        embedding_np = embedding.cpu().numpy()

    # Classify
    predictions, confidences = pipeline["knn"].predict(embedding_np)

    latency_ms = (time.perf_counter() - start) * 1000

    pred_label = int(predictions[0])
    confidence = float(confidences[0])
    traffic_class = pipeline["label_encoder"].inverse_transform([pred_label])[0]

    # Drift detection
    drift = pipeline["drift_detector"].update(confidence)

    # Buffer for online learning
    pipeline["updater"].add_to_buffer(embedding_np[0], pred_label)

    result = {
        "traffic_class": traffic_class,
        "class_id": pred_label,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
        "is_anomalous": confidence < 0.5,
        "drift_detected": drift,
        "timestamp": time.time(),
    }

    return result


def run_live(args):
    """Run live classification loop."""
    logger.info("🚀 Loading classification pipeline...")
    pipeline = load_pipeline(args.checkpoint)
    logger.info("✅ Pipeline loaded. Starting live classification...")

    source = args.pcap or args.interface

    try:
        from nfstream import NFStreamer

        streamer = NFStreamer(
            source=source,
            statistical_analysis=True,
            idle_timeout=30 if args.pcap else 15,
            active_timeout=120,
        )

        flow_count = 0
        for flow in streamer:
            flow_count += 1

            # Extract features from nfstream flow
            feature_names = pipeline["scaler"].feature_names_in_ if hasattr(pipeline["scaler"], "feature_names_in_") else None

            # Build feature vector from flow object
            flow_dict = flow.to_dict() if hasattr(flow, "to_dict") else vars(flow)
            features = []

            if feature_names:
                for fname in feature_names:
                    features.append(flow_dict.get(fname, 0.0))
            else:
                # Fallback: use numeric attributes
                for key, value in flow_dict.items():
                    if isinstance(value, (int, float)) and key not in ("id", "expiration_id"):
                        features.append(float(value))

            if not features:
                continue

            features = np.array(features, dtype=np.float32)

            # Classify
            result = classify_flow(pipeline, features)

            # Log
            icon = "🔴" if result["is_anomalous"] else "🟢"
            drift_str = " ⚠️ DRIFT!" if result["drift_detected"] else ""
            logger.info(
                f"{icon} Flow #{flow_count}: {result['traffic_class']} "
                f"({result['confidence']:.1%}) {result['latency_ms']:.1f}ms{drift_str}"
            )

            # Generate alert for suspicious flows
            if result["is_anomalous"] or result["traffic_class"] == "malware_c2":
                alert = generate_alert(
                    traffic_class=result["traffic_class"],
                    confidence=result["confidence"],
                    flow_features={
                        "src_ip": flow_dict.get("src_ip", "N/A"),
                        "dst_ip": flow_dict.get("dst_ip", "N/A"),
                        "protocol": flow_dict.get("protocol", "N/A"),
                        "duration": flow_dict.get("bidirectional_duration_ms", 0) / 1000,
                    },
                    is_anomalous=result["is_anomalous"],
                )
                logger.warning(f"🚨 ALERT: {alert}")

    except ImportError:
        logger.error("nfstream not installed. Install with: pip install nfstream")
    except KeyboardInterrupt:
        logger.info("\n👋 Stopping live classification.")
        logger.info(f"Total flows classified: {flow_count}")
        logger.info(f"Drift events: {len(pipeline['drift_detector'].drift_events)}")


if __name__ == "__main__":
    args = parse_args()
    run_live(args)
