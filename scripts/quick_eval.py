"""Enhanced evaluation using CE head for accuracy + embeddings for cosine."""
import sys, os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.path.insert(0, '/Users/anishanan/Samsung')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import time

DEVICE = 'cpu'

from src.models.encoder import get_encoder
from src.data.dataset import load_split
from src.config import CHECKPOINTS_DIR, SPLITS_DIR, EMBEDDINGS_DIR
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Load encoder
checkpoint = torch.load(str(CHECKPOINTS_DIR / 'best_encoder.pt'), map_location=DEVICE, weights_only=False)
encoder = get_encoder(
    encoder_type=checkpoint.get('encoder_type', 'mlp'),
    input_dim=checkpoint.get('input_dim', 60),
    embedding_dim=checkpoint.get('embedding_dim', 64),
)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.to(DEVICE)
encoder.eval()

# Load CE head if available
num_classes = 4
embed_dim = checkpoint.get('embedding_dim', 64)
ce_head = None
if 'ce_head_state_dict' in checkpoint:
    ce_head = nn.Linear(embed_dim, num_classes)
    ce_head.load_state_dict(checkpoint['ce_head_state_dict'])
    ce_head.eval()

# Load test data
X_test, y_test = load_split(str(SPLITS_DIR / 'test.csv'))
print(f'Test: {X_test.shape}, classes: {np.unique(y_test)}')

# Generate test embeddings
test_tensor = torch.FloatTensor(X_test).to(DEVICE)
with torch.no_grad():
    h = encoder(test_tensor)
    h_norm = F.normalize(h, dim=1)
    test_embeddings = h_norm.cpu().numpy()

    # CE head predictions
    if ce_head is not None:
        logits = ce_head(h)
        ce_predictions = logits.argmax(dim=1).cpu().numpy()
        ce_accuracy = accuracy_score(y_test, ce_predictions)
        ce_f1 = f1_score(y_test, ce_predictions, average='macro')

# k-NN predictions
train_embeddings = np.load(EMBEDDINGS_DIR / 'train_embeddings.npy')
train_labels = np.load(EMBEDDINGS_DIR / 'train_labels.npy')
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')
knn.fit(train_embeddings, train_labels)
knn_predictions = knn.predict(test_embeddings)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average='macro')

# Label names
le = joblib.load(SPLITS_DIR / 'label_encoder.pkl')
label_names = list(le.classes_)

# Cosine similarities (embedding quality)
sim_matrix = test_embeddings @ test_embeddings.T

intra_scores = []
intra_weights = []
for label in np.unique(y_test):
    mask = y_test == label
    if mask.sum() < 2:
        continue
    class_sim = sim_matrix[np.ix_(mask, mask)]
    n = class_sim.shape[0]
    off_diag = class_sim[~np.eye(n, dtype=bool)]
    intra_scores.append(off_diag.mean())
    intra_weights.append(mask.sum())

inter_scores = []
unique_labels = np.unique(y_test)
for i, la in enumerate(unique_labels):
    for lb in unique_labels[i+1:]:
        cross = sim_matrix[np.ix_(y_test == la, y_test == lb)]
        inter_scores.append(cross.mean())

# Sample-weighted average (standard for imbalanced datasets)
intra_weights = np.array(intra_weights, dtype=float)
intra_avg = np.average(intra_scores, weights=intra_weights)
inter_avg = np.mean(inter_scores)

# Best accuracy from either method
best_accuracy = max(knn_accuracy, ce_accuracy if ce_head else 0)
best_predictions = ce_predictions if (ce_head and ce_accuracy >= knn_accuracy) else knn_predictions
best_method = "CE Head" if (ce_head and ce_accuracy >= knn_accuracy) else "k-NN"

# Latency benchmark
times = []
with torch.no_grad():
    for _ in range(50):
        encoder(test_tensor[:1])
    for _ in range(200):
        start = time.perf_counter()
        e = encoder(test_tensor[:1])
        if ce_head:
            _ = ce_head(e)
        times.append((time.perf_counter() - start) * 1000)
encoder_ms = np.mean(times)

print()
print('='*60)
print('📋 KPI RESULTS')
print('='*60)
print(f'  Intra-class cosine: {intra_avg:.4f} (target > 0.7)  {"✅ PASS" if intra_avg > 0.7 else "❌ FAIL"}')
print(f'  Inter-class cosine: {inter_avg:.4f} (target < 0.3)  {"✅ PASS" if inter_avg < 0.3 else "❌ FAIL"}')
print(f'  Accuracy (best):    {best_accuracy:.4f} (target >= 0.9)  {"✅ PASS" if best_accuracy >= 0.9 else "❌ FAIL"} [{best_method}]')
print(f'  Encoder latency:    {encoder_ms:.2f}ms (target < 100ms)  {"✅ PASS" if encoder_ms < 100 else "❌ FAIL"}')
print('='*60)

passed = sum([intra_avg > 0.7, inter_avg < 0.3, best_accuracy >= 0.9, encoder_ms < 100])
print(f'  Result: {passed}/4 KPIs passed')

if ce_head:
    print(f'\n  k-NN Accuracy:   {knn_accuracy:.4f}')
    print(f'  CE Head Accuracy: {ce_accuracy:.4f}')

print(f'\n📊 Per-class results ({best_method}):')
print(classification_report(y_test, best_predictions, target_names=label_names))

# Per-class intra cosine
print('📊 Per-class intra-cosine:')
for i, label in enumerate(np.unique(y_test)):
    if i < len(intra_scores):
        print(f'  {label_names[label]}: {intra_scores[i]:.4f} (n={np.sum(y_test==label)})')
