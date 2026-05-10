import os
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from nfstream import NFStreamer
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SPLITS_DIR, CHECKPOINTS_DIR, DEVICE, EMBEDDINGS_DIR
from src.models.encoder import get_encoder
from scripts.merge_datasets import FEATURE_COLS

from src.data.nfstream_plugin import FlowEmbedPlugin, extract_features_from_flow

def extract_features(pcap_path):
    print(f"📡 Processing live PCAP with Full 60-Feature Plugin: {pcap_path} ...")
    streamer = NFStreamer(source=pcap_path, udps=FlowEmbedPlugin(), statistical_analysis=True)
    all_flows = []
    
    for flow in streamer:
        if flow.bidirectional_packets < 10:
            continue
            
        feat_array = extract_features_from_flow(flow)
        
        # Convert to dict using FEATURE_COLS
        flow_data = {FEATURE_COLS[i]: feat_array[i] for i in range(len(FEATURE_COLS))}
        all_flows.append(flow_data)
        
    df = pd.DataFrame(all_flows)
    
    if len(df) == 0:
        print("❌ No valid flows found in PCAP.")
        return None
        
    df = df[FEATURE_COLS].copy()
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # Scale features
    print("📏 Scaling live features...")
    scaler = joblib.load(SPLITS_DIR / "scaler.pkl")
    scaled_features = scaler.transform(df)
    return scaled_features

def main():
    if len(sys.argv) > 1:
        pcap_path = sys.argv[1]
    else:
        pcap_path = "/Users/anishanan/Samsung/data/raw/samsung_s23/samsung_live.pcapng"
        
    X_live = extract_features(pcap_path)
    
    if X_live is None:
        return
        
    out_path = Path("/Users/anishanan/Samsung/data/processed/samsung_live_features.npy")
    np.save(out_path, X_live)
    print(f"💾 Saved live features to {out_path} for dashboard.")
        
    print("🧠 Loading model and label encoder...")
    le = joblib.load(SPLITS_DIR / "label_encoder.pkl")
    
    ckpt = torch.load(CHECKPOINTS_DIR / "best_encoder.pt", map_location=DEVICE, weights_only=False)
    enc = get_encoder('mlp', input_dim=ckpt['input_dim'], embedding_dim=ckpt['embedding_dim']).to(DEVICE)
    enc.load_state_dict(ckpt['encoder_state_dict'])
    enc.eval()

    print("🧬 Generating Embeddings...")
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_live), 4096):
            b = torch.tensor(X_live[i:i+4096]).float().to(DEVICE)
            embs.append(enc(b).cpu().numpy())
    embs = np.concatenate(embs)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    print("🤖 Running Scikit-Learn k-NN Classifier (Local Fallback)...")
    from sklearn.neighbors import KNeighborsClassifier
    
    train_df = pd.read_csv(SPLITS_DIR / 'train.csv')
    X_train = train_df.drop(columns=['label']).values.astype(np.float32)
    y_train = train_df['label'].values.astype(int)
    
    train_embs = []
    with torch.no_grad():
        for i in range(0, len(X_train), 4096):
            b = torch.tensor(X_train[i:i+4096]).float().to(DEVICE)
            train_embs.append(enc(b).cpu().numpy())
    train_embs = np.concatenate(train_embs)
    train_embs = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    
    # Use Scikit-learn locally to avoid the faiss-cpu MacOS segmentation fault!
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')
    knn.fit(train_embs, y_train)
    
    preds = knn.predict(embs)
    pred_labels = le.inverse_transform(preds)
    
    print('='*60)
    print('🔴 LIVE SAMSUNG S23 TRAFFIC PREDICTIONS 🔴')
    print('='*60)
    
    counts = Counter(pred_labels)
    total = sum(counts.values())
    
    for label, count in counts.most_common():
        pct = (count / total) * 100
        print(f"  {label.upper():<12} : {count:>5} flows ({pct:>5.1f}%)")
    print('='*60)

if __name__ == "__main__":
    main()
