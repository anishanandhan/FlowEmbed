"""
SVM Classifier — Support Vector Machine on flow embeddings.

Used as a comparison baseline against k-NN.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
from typing import Optional, Tuple
import logging

from src.config import SVM_KERNEL, SVM_C, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)


class SVMClassifier:
    """
    SVM classifier on flow embeddings with probability calibration.
    """

    def __init__(self, kernel: str = SVM_KERNEL, C: float = SVM_C):
        self.svm = CalibratedClassifierCV(
            SVC(kernel=kernel, C=C, gamma="scale"),
            cv=3,
        )

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Train SVM on embeddings."""
        self.svm.fit(embeddings, labels)
        logger.info(f"Trained SVM ({SVM_KERNEL}) on {len(labels)} samples")

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores."""
        predictions = self.svm.predict(embeddings)
        probabilities = self.svm.predict_proba(embeddings)
        confidences = probabilities.max(axis=1)
        return predictions, confidences

    def save(self, path: Optional[str] = None):
        path = path or str(EMBEDDINGS_DIR / "svm_classifier.pkl")
        joblib.dump(self.svm, path)
        logger.info(f"Saved SVM to {path}")

    def load(self, path: Optional[str] = None):
        path = path or str(EMBEDDINGS_DIR / "svm_classifier.pkl")
        self.svm = joblib.load(path)
        logger.info(f"Loaded SVM from {path}")
