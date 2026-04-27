"""
Online Updater — Automatically refresh the k-NN index when concept drift is detected.

When the drift detector fires, this module collects recent flow embeddings
with their predicted labels and adds them to the FAISS index, effectively
adapting the classifier to new traffic patterns without full retraining.
"""

import numpy as np
from collections import deque
from typing import Optional
import logging

from src.config import DRIFT_REFRESH_SIZE

logger = logging.getLogger(__name__)


class OnlineUpdater:
    """
    Online k-NN index updater for concept drift adaptation.

    Maintains a buffer of recent embeddings and their labels.
    When triggered (by drift detection), adds the buffered embeddings
    to the FAISS index to adapt the classifier.
    """

    def __init__(
        self,
        knn_classifier,
        buffer_size: int = 500,
        refresh_size: int = DRIFT_REFRESH_SIZE,
    ):
        """
        Args:
            knn_classifier: FAISSKNNClassifier instance to update.
            buffer_size: Maximum number of recent embeddings to buffer.
            refresh_size: Number of embeddings to add to index on each refresh.
        """
        self.knn_classifier = knn_classifier
        self.refresh_size = refresh_size

        # Buffer for recent (embedding, label) pairs
        self.embedding_buffer = deque(maxlen=buffer_size)
        self.label_buffer = deque(maxlen=buffer_size)

        # Statistics
        self.total_refreshes = 0
        self.total_added = 0

    def add_to_buffer(self, embedding: np.ndarray, label: int):
        """
        Add a new embedding and its label to the buffer.
        Called for every classified flow.
        """
        self.embedding_buffer.append(embedding)
        self.label_buffer.append(label)

    def refresh_index(self, drift_event: Optional[dict] = None):
        """
        Refresh the k-NN index with buffered embeddings.
        Called when drift is detected.
        """
        if len(self.embedding_buffer) < self.refresh_size:
            logger.warning(
                f"Buffer too small ({len(self.embedding_buffer)}) for refresh. "
                f"Need at least {self.refresh_size} samples."
            )
            return

        # Take the most recent embeddings from buffer
        recent_embeddings = np.array(
            list(self.embedding_buffer)[-self.refresh_size:]
        )
        recent_labels = np.array(
            list(self.label_buffer)[-self.refresh_size:]
        )

        # Add to FAISS index
        self.knn_classifier.add_embeddings(recent_embeddings, recent_labels)

        self.total_refreshes += 1
        self.total_added += len(recent_labels)

        logger.info(
            f"🔄 Index refreshed: added {len(recent_labels)} embeddings. "
            f"Total refreshes: {self.total_refreshes}, "
            f"Index size: {self.knn_classifier.index.ntotal}"
        )

    def get_status(self) -> dict:
        """Get updater status."""
        return {
            "buffer_size": len(self.embedding_buffer),
            "total_refreshes": self.total_refreshes,
            "total_added": self.total_added,
            "index_size": self.knn_classifier.index.ntotal,
        }
