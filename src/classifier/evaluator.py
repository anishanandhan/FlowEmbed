"""
Evaluator — KPI metrics computation for flow embedding quality and classification.

Computes all official hackathon KPIs:
1. Intra-class cosine similarity (target > 0.7)
2. Inter-class cosine similarity (target < 0.3)
3. Classification accuracy (target ≥ 90%)
4. Generalization to new types (target ≥ 85%)
5. Real-time latency (target < 100ms)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from typing import Dict, Optional, Tuple
import time
import logging

from src.config import (
    KPI_INTRA_CLASS_COSINE, KPI_INTER_CLASS_COSINE,
    KPI_ACCURACY, KPI_GENERALIZATION, KPI_LATENCY_MS,
)

logger = logging.getLogger(__name__)


def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T


def compute_intra_class_cosine(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    Compute average cosine similarity within each class.

    KPI Target: > 0.7 — flows of the same type should have similar embeddings.
    """
    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    unique_labels = np.unique(labels)

    results = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue

        class_sim = sim_matrix[np.ix_(mask, mask)]
        # Exclude diagonal (self-similarity = 1.0)
        n = class_sim.shape[0]
        off_diag = class_sim[~np.eye(n, dtype=bool)]
        results[int(label)] = float(off_diag.mean())

    overall = np.mean(list(results.values())) if results else 0.0
    logger.info(f"Intra-class cosine similarity: {overall:.4f} (target > {KPI_INTRA_CLASS_COSINE})")

    return results


def compute_inter_class_cosine(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute average cosine similarity between different classes.

    KPI Target: < 0.3 — flows of different types should have dissimilar embeddings.
    """
    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    unique_labels = np.unique(labels)

    results = {}
    all_inter = []

    for i, label_a in enumerate(unique_labels):
        for label_b in unique_labels[i + 1:]:
            mask_a = labels == label_a
            mask_b = labels == label_b

            cross_sim = sim_matrix[np.ix_(mask_a, mask_b)]
            mean_sim = float(cross_sim.mean())
            results[f"{int(label_a)}_vs_{int(label_b)}"] = mean_sim
            all_inter.append(mean_sim)

    overall = np.mean(all_inter) if all_inter else 0.0
    logger.info(f"Inter-class cosine similarity: {overall:.4f} (target < {KPI_INTER_CLASS_COSINE})")

    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[list] = None,
) -> Dict:
    """
    Compute classification accuracy, F1 score, and confusion matrix.

    KPI Target: ≥ 90% accuracy.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
    )

    logger.info(f"Classification Accuracy: {accuracy:.4f} (target ≥ {KPI_ACCURACY})")
    logger.info(f"F1 Score (macro): {f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_generalization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    holdout_class_name: str = "unknown",
) -> Dict:
    """
    Compute generalization metrics for few-shot classification on held-out class.

    KPI Target: ≥ 85% on unseen traffic types.
    """
    accuracy = accuracy_score(y_true, y_pred)

    logger.info(
        f"Generalization Accuracy ({holdout_class_name}): {accuracy:.4f} "
        f"(target ≥ {KPI_GENERALIZATION})"
    )

    return {
        "generalization_accuracy": accuracy,
        "holdout_class": holdout_class_name,
    }


def benchmark_latency(
    encoder: torch.nn.Module,
    sample_input: torch.Tensor,
    faiss_classifier,
    num_trials: int = 100,
    device: str = "cpu",
) -> Dict:
    """
    Benchmark end-to-end classification latency.

    KPI Target: < 100ms per flow.
    Measures: encoder inference + FAISS k-NN lookup.
    """
    encoder.eval()
    sample_input = sample_input.to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(sample_input[:1])

    # Benchmark encoder
    encoder_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start = time.perf_counter()
            embedding = encoder(sample_input[:1])
            if device == "mps":
                torch.mps.synchronize()
            encoder_times.append((time.perf_counter() - start) * 1000)

    # Benchmark FAISS lookup
    dummy_embedding = np.random.randn(1, embedding.shape[1]).astype(np.float32)
    faiss_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        faiss_classifier.predict(dummy_embedding)
        faiss_times.append((time.perf_counter() - start) * 1000)

    encoder_ms = np.mean(encoder_times)
    faiss_ms = np.mean(faiss_times)
    total_ms = encoder_ms + faiss_ms

    logger.info(
        f"Latency: encoder={encoder_ms:.2f}ms, FAISS={faiss_ms:.2f}ms, "
        f"total={total_ms:.2f}ms (target < {KPI_LATENCY_MS}ms)"
    )

    return {
        "encoder_latency_ms": encoder_ms,
        "faiss_latency_ms": faiss_ms,
        "total_latency_ms": total_ms,
        "meets_kpi": total_ms < KPI_LATENCY_MS,
    }


def compute_all_kpis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    y_pred: np.ndarray,
    encoder: torch.nn.Module = None,
    sample_input: torch.Tensor = None,
    faiss_classifier=None,
    label_names: list = None,
    device: str = "cpu",
) -> Dict:
    """
    Compute all official KPIs in one call.

    Returns a comprehensive dictionary with all metrics and pass/fail status.
    """
    results = {}

    # 1. Intra-class cosine similarity
    intra_cosine = compute_intra_class_cosine(embeddings, labels)
    intra_avg = np.mean(list(intra_cosine.values())) if intra_cosine else 0.0
    results["intra_class_cosine"] = {
        "per_class": intra_cosine,
        "average": intra_avg,
        "target": KPI_INTRA_CLASS_COSINE,
        "pass": intra_avg > KPI_INTRA_CLASS_COSINE,
    }

    # 2. Inter-class cosine similarity
    inter_cosine = compute_inter_class_cosine(embeddings, labels)
    inter_avg = np.mean(list(inter_cosine.values())) if inter_cosine else 1.0
    results["inter_class_cosine"] = {
        "per_pair": inter_cosine,
        "average": inter_avg,
        "target": KPI_INTER_CLASS_COSINE,
        "pass": inter_avg < KPI_INTER_CLASS_COSINE,
    }

    # 3. Classification accuracy
    cls_metrics = compute_classification_metrics(labels, y_pred, label_names)
    results["classification"] = {
        **cls_metrics,
        "target": KPI_ACCURACY,
        "pass": cls_metrics["accuracy"] >= KPI_ACCURACY,
    }

    # 4. Latency (if encoder and classifier provided)
    if encoder is not None and sample_input is not None and faiss_classifier is not None:
        latency = benchmark_latency(encoder, sample_input, faiss_classifier, device=device)
        results["latency"] = latency

    # Summary
    kpis_passed = sum([
        results["intra_class_cosine"]["pass"],
        results["inter_class_cosine"]["pass"],
        results["classification"]["pass"],
        results.get("latency", {}).get("meets_kpi", True),
    ])
    results["summary"] = {
        "kpis_passed": kpis_passed,
        "total_kpis": 4,
        "all_passed": kpis_passed == 4,
    }

    return results
