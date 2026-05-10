import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from src.config import SPLITS_DIR, CHECKPOINTS_DIR, DEVICE, EMBEDDINGS_DIR, PROCESSED_DIR
from src.models.encoder import get_encoder
from src.classifier.knn_classifier import FAISSKNNClassifier
from scripts.merge_datasets import FEATURE_COLS

def main():
    print("=" * 60)
    print("🥽 XR/AR FEW-SHOT GENERALIZATION DEMO")
    print("=" * 60)
    
    # 1. Load XR/AR data that was held out from training
    df = pd.read_csv(PROCESSED_DIR / "kaggle_5g_flows.csv")
    xr_df = df[df["label"] == "xr_ar"].copy()
    print(f"Loaded {len(xr_df)} unseen XR/AR flows.")
    
    for col in FEATURE_COLS:
        if col not in xr_df.columns:
            xr_df[col] = 0.0
            
    X_xr = xr_df[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    
    # 2. Scale features
    scaler = joblib.load(SPLITS_DIR / "scaler.pkl")
    X_xr = scaler.transform(X_xr)
    
    # 3. Load Model
    ckpt = torch.load(CHECKPOINTS_DIR / "best_encoder.pt", map_location=DEVICE, weights_only=False)
    enc = get_encoder('mlp', input_dim=ckpt['input_dim'], embedding_dim=ckpt['embedding_dim']).to(DEVICE)
    enc.load_state_dict(ckpt['encoder_state_dict'])
    enc.eval()
    
    # 4. Generate Embeddings
    print("Generating embeddings through frozen encoder...")
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_xr), 4096):
            b = torch.tensor(X_xr[i:i+4096]).float().to(DEVICE)
            embs.append(enc(b).cpu().numpy())
    embs = np.concatenate(embs)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    
    # 5. Few-shot setup
    num_shots = 20
    support_embs = embs[:num_shots]
    query_embs = embs[num_shots:]
    
    # We assign a new label ID '4' for XR_AR
    NEW_CLASS_ID = 4
    support_labels = np.full(num_shots, NEW_CLASS_ID)
    true_query_labels = np.full(len(query_embs), NEW_CLASS_ID)
    
    # 6. Load FAISS Index and inject few-shot examples
    print(f"Injecting {num_shots} XR/AR examples into the FAISS index...")
    knn = FAISSKNNClassifier(embedding_dim=ckpt['embedding_dim'])
    knn.load(str(EMBEDDINGS_DIR / 'faiss_index.bin'))
    
    # Check if index is IVFFlat and train it if we add new data
    # Wait, IVFFlat requires retraining if adding new data, but it's already trained!
    # So we can just add to it directly.
    knn.add_embeddings(support_embs, support_labels)
    
    # 7. Evaluate
    print(f"Evaluating on remaining {len(query_embs)} unseen XR/AR flows...")
    preds, confs = knn.predict(query_embs)
    
    acc = accuracy_score(true_query_labels, preds)
    print("=" * 60)
    print(f"XR/AR Few-Shot Accuracy (using only {num_shots} examples): {acc*100:.2f}%")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
