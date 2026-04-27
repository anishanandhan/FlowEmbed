"""
Live Real-Time Inference Pipeline.

Sniffs packets from the network interface, extracts the exact 60 features required
by FlowEmbed using a custom NFPlugin, runs the PyTorch MLP encoder for classification,
monitors for concept drift using ADWIN, and alerts via LLM if anomalous.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINTS_DIR, SPLITS_DIR, DEVICE
from src.models.encoder import get_encoder
from src.drift.drift_detector import DriftDetector
from src.explainability.llm_alerter import generate_alert, preload_cache

# Attempt to load nfstream
try:
    from nfstream import NFStreamer
except ImportError:
    print("❌ NFStream is required for live capture. Please install it:")
    print("pip install nfstream")
    sys.exit(1)

from src.data.nfstream_plugin import FlowEmbedPlugin, extract_features_from_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("LiveInference")

# Extracted to src.data.nfstream_plugin

# ─────────────────────────────────────────────────────────────
# 2. Main Live Pipeline
# ─────────────────────────────────────────────────────────────
def run_live_inference(interface="en0"):
    logger.info("Initializing Live Inference Pipeline...")
    
    # Load Label Encoder
    import joblib
    try:
        le = joblib.load(SPLITS_DIR / "label_encoder.pkl")
        label_names = list(le.classes_)
    except:
        label_names = ["gaming", "streaming", "voip", "xr_ar"]

    # Load Model
    ckpt_path = CHECKPOINTS_DIR / "best_encoder.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
        
    logger.info(f"Loading PyTorch model from {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    encoder = get_encoder(
        encoder_type=checkpoint.get("encoder_type", "mlp"),
        input_dim=60,
        embedding_dim=48,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(DEVICE)
    encoder.eval()
    
    num_classes = len(checkpoint["ce_head_state_dict"]["bias"]) if "ce_head_state_dict" in checkpoint else 4
    ce_head = torch.nn.Linear(48, num_classes)
    if "ce_head_state_dict" in checkpoint:
        ce_head.load_state_dict(checkpoint["ce_head_state_dict"])
    ce_head.to(DEVICE)
    ce_head.eval()

    # Slice label names to match the model output shape
    label_names = label_names[:num_classes]

    # Initialize Drift Detector
    def drift_alert(event):
        logger.warning("🚨 [ADWIN DRIFT DETECTOR] Statistical anomaly detected in prediction confidence!")
        
    drift_detector = DriftDetector(on_drift_callback=drift_alert)
    
    # Preload LLM alerts
    preload_cache()
    
    # Start capturing
    logger.info(f"🚀 Starting live packet capture on interface: {interface}")
    logger.info("Waiting for flows to expire (timeout=10s)...")
    
    streamer = NFStreamer(
        source=interface,
        udps=FlowEmbedPlugin(),
        active_timeout=10,
        idle_timeout=5
    )
    
    for flow in streamer:
        # Ignore tiny uninteresting flows
        if flow.bidirectional_packets < 5:
            continue
            
        t0 = time.perf_counter()
        
        # 1. Extract Features
        features_np = extract_features_from_flow(flow)
        tensor = torch.FloatTensor(features_np).unsqueeze(0).to(DEVICE)
        
        # 2. Inference
        with torch.no_grad():
            h = encoder(tensor)
            logits = ce_head(h)
            probs = torch.softmax(logits, dim=1)[0]
            
            conf, pred_idx = torch.max(probs, dim=0)
            conf = conf.item()
            pred_class = label_names[pred_idx.item()]
            
        latency = (time.perf_counter() - t0) * 1000
        
        # 3. Drift Detection
        drift_detected = drift_detector.update(conf)
        
        # 4. LLM Alerting if anomalous (low confidence or drift)
        is_anomalous = conf < 0.65 or drift_detected
        
        # Format output
        src = f"{flow.src_ip}:{flow.src_port}"
        dst = f"{flow.dst_ip}:{flow.dst_port}"
        
        if is_anomalous:
            flow_dict = {
                "src_ip": flow.src_ip, "dst_ip": flow.dst_ip,
                "protocol": flow.protocol, "duration": flow.bidirectional_duration_ms/1000,
                "avg_pkt_size": int(features_np[7]), "iat_mean": int(features_np[17]),
                "total_packets": flow.bidirectional_packets
            }
            alert = generate_alert(pred_class, conf, flow_dict, is_anomalous=True)
            logger.warning(f"🔴 ANOMALY | {src} -> {dst} | Pred: {pred_class} ({conf:.1%}) | {latency:.2f}ms")
            logger.warning(f"    ↳ SOC Alert: {alert}")
        else:
            logger.info(f"🟢 OK      | {src} -> {dst} | Pred: {pred_class} ({conf:.1%}) | {latency:.2f}ms")

if __name__ == "__main__":
    # Use loopback 'lo0' or 'en0' depending on environment
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", default="en0", help="Network interface to sniff")
    args = parser.parse_args()
    
    run_live_inference(args.interface)
