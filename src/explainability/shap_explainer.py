"""
SHAP Explainer — Feature importance for flow classifications.

Uses a surrogate model approach: train a lightweight RandomForest on
the embeddings, then compute SHAP values to show which flow features
drove each classification decision.
"""

import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FlowExplainer:
    """
    SHAP-based feature importance for flow classification.

    Since the encoder is a neural network and SHAP's DeepExplainer can be slow,
    we use a two-step approach:
    1. Train a RandomForest surrogate on (original_features → predicted_labels)
    2. Use SHAP TreeExplainer on the surrogate for fast explanations
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.surrogate = None
        self.explainer = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Train the surrogate model for SHAP explanations.

        Args:
            features: Original flow features (before encoding) [N, feature_dim].
            labels: Predicted class labels [N].
        """
        from sklearn.ensemble import RandomForestClassifier

        self.surrogate = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.surrogate.fit(features, labels)

        try:
            import shap
            self.explainer = shap.TreeExplainer(self.surrogate)
            logger.info("SHAP TreeExplainer initialized")
        except ImportError:
            logger.warning("SHAP not installed. Using feature importance fallback.")

    def explain(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Get top-k most important features for each sample.

        Args:
            features: Flow features to explain [M, feature_dim].
            top_k: Number of top features to return.

        Returns:
            List of lists of (feature_name, importance) tuples.
        """
        if self.explainer is not None:
            return self._explain_shap(features, top_k)
        elif self.surrogate is not None:
            return self._explain_importance(features, top_k)
        else:
            logger.warning("Explainer not fitted. Call .fit() first.")
            return [[] for _ in range(len(features))]

    def _explain_shap(
        self, features: np.ndarray, top_k: int
    ) -> List[List[Tuple[str, float]]]:
        """Use SHAP values for explanation."""
        import shap

        shap_values = self.explainer.shap_values(features)

        results = []
        for i in range(len(features)):
            # Get SHAP values for the predicted class
            if isinstance(shap_values, list):
                # Multi-class: shap_values is a list of arrays per class
                pred = self.surrogate.predict(features[i:i+1])[0]
                sv = np.abs(shap_values[pred][i])
            else:
                sv = np.abs(shap_values[i])

            top_indices = np.argsort(sv)[-top_k:][::-1]

            explanations = []
            for idx in top_indices:
                name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                explanations.append((name, float(sv[idx])))

            results.append(explanations)

        return results

    def _explain_importance(
        self, features: np.ndarray, top_k: int
    ) -> List[List[Tuple[str, float]]]:
        """Fallback: use RandomForest feature importances (global, not per-sample)."""
        importances = self.surrogate.feature_importances_
        top_indices = np.argsort(importances)[-top_k:][::-1]

        explanation = []
        for idx in top_indices:
            name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            explanation.append((name, float(importances[idx])))

        # Same explanation for all samples (global importance)
        return [explanation for _ in range(len(features))]

    def get_top_features_str(
        self,
        features: np.ndarray,
        top_k: int = 3,
    ) -> List[str]:
        """Get top features as formatted strings (for LLM alerter)."""
        explanations = self.explain(features, top_k)
        results = []
        for exp in explanations:
            parts = [f"{name} (importance: {imp:.3f})" for name, imp in exp]
            results.append(", ".join(parts))
        return results
