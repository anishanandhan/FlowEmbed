"""
k-NN Classifier — FAISS-backed k-Nearest Neighbors on flow embeddings.

FAISS provides sub-millisecond nearest neighbor search, enabling
real-time classification at scale. k-NN naturally supports few-shot
generalization: to classify a new traffic type, just add a few
labelled embeddings to the index.
"""

import numpy as np
import faiss
import joblib
from pathlib import Path
from typing import Optional, Tuple
import logging

from src.config import KNN_K, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)


class FAISSKNNClassifier:
    """
    k-NN classifier backed by a FAISS index for fast cosine similarity search.

    The classifier stores embeddings and their labels in a FAISS index.
    At inference, it finds the k nearest neighbors and uses majority
    voting (optionally distance-weighted) to predict the class.
    """

    def __init__(self, k: int = KNN_K, embedding_dim: int = 256, weighted: bool = True):
        """
        Args:
            k: Number of nearest neighbors.
            embedding_dim: Dimension of embeddings.
            weighted: If True, use distance-weighted voting.
        """
        self.k = k
        self.embedding_dim = embedding_dim
        self.weighted = weighted

        # FAISS IndexIVFFlat (Production-ready inverted file index)
        quantizer = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 10, faiss.METRIC_INNER_PRODUCT)
        self.labels = np.array([], dtype=np.int64)
        self.embeddings = np.array([]).reshape(0, embedding_dim)

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Build the FAISS index from training embeddings.

        Args:
            embeddings: L2-normalized embeddings [N, embedding_dim].
            labels: Integer class labels [N].
        """
        # Ensure L2 normalization (required for cosine similarity via inner product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        embeddings = embeddings.astype(np.float32)

        # Switch to IndexIVFFlat for production scale
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 10, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = 3

        self.labels = np.array(labels, dtype=np.int64)
        self.embeddings = embeddings

        logger.info(f"Built FAISS IndexIVFFlat with {len(labels)} embeddings, k={self.k}")

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels for query embeddings.

        Args:
            embeddings: L2-normalized query embeddings [M, embedding_dim].

        Returns:
            Tuple of (predicted_labels [M], confidence_scores [M]).
        """
        # Normalize queries
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / (norms + 1e-8)).astype(np.float32)

        # Search k nearest neighbors
        distances, indices = self.index.search(embeddings, self.k)

        predictions = []
        confidences = []

        for i in range(len(embeddings)):
            neighbor_labels = self.labels[indices[i]]
            neighbor_dists = distances[i]  # Cosine similarities (higher = closer)

            if self.weighted:
                # Distance-weighted voting
                unique_labels = np.unique(neighbor_labels)
                scores = {}
                for label in unique_labels:
                    mask = neighbor_labels == label
                    scores[label] = neighbor_dists[mask].sum()

                pred_label = max(scores, key=scores.get)
                total_score = sum(scores.values())
                confidence = scores[pred_label] / total_score if total_score > 0 else 0.0
            else:
                # Majority voting
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                pred_label = unique[counts.argmax()]
                confidence = counts.max() / self.k

            predictions.append(pred_label)
            confidences.append(confidence)

        return np.array(predictions), np.array(confidences)

    def predict_with_neighbors(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with full neighbor information (for explainability).

        Returns:
            Tuple of (predictions, confidences, neighbor_distances, neighbor_labels).
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / (norms + 1e-8)).astype(np.float32)

        distances, indices = self.index.search(embeddings, self.k)
        neighbor_labels = self.labels[indices]

        predictions, confidences = self.predict(embeddings)

        return predictions, confidences, distances, neighbor_labels

    def add_embeddings(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Add new embeddings to the index (for online learning / drift adaptation).
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / (norms + 1e-8)).astype(np.float32)

        self.index.add(embeddings)
        self.labels = np.concatenate([self.labels, labels])
        self.embeddings = np.concatenate([self.embeddings, embeddings])

        logger.info(f"Added {len(labels)} embeddings. Index now has {self.index.ntotal} total.")

    def save(self, path: Optional[str] = None):
        """Save the FAISS index and labels."""
        path = path or str(EMBEDDINGS_DIR / "faiss_index.bin")
        faiss.write_index(self.index, path)

        labels_path = path.replace(".bin", "_labels.npy")
        np.save(labels_path, self.labels)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: Optional[str] = None):
        """Load a previously saved FAISS index."""
        path = path or str(EMBEDDINGS_DIR / "faiss_index.bin")
        self.index = faiss.read_index(path)

        labels_path = path.replace(".bin", "_labels.npy")
        self.labels = np.load(labels_path)

        self.embedding_dim = self.index.d
        logger.info(f"Loaded FAISS index with {self.index.ntotal} embeddings")

    def get_latency_ms(self, embeddings: np.ndarray, num_trials: int = 100) -> float:
        """Benchmark classification latency in milliseconds."""
        import time

        norms = np.linalg.norm(embeddings[:1], axis=1, keepdims=True)
        query = (embeddings[:1] / (norms + 1e-8)).astype(np.float32)

        start = time.perf_counter()
        for _ in range(num_trials):
            self.index.search(query, self.k)
        elapsed = (time.perf_counter() - start) / num_trials * 1000

        return elapsed
