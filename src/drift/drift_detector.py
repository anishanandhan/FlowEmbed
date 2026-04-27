"""
Drift Detector — Real-time concept drift detection using River's ADWIN.

Monitors classification confidence scores as a proxy metric.
When ADWIN detects a statistically significant change in the confidence
distribution, it triggers a drift alert and k-NN index refresh.
"""

import numpy as np
from river import drift
from collections import deque
from typing import Optional, Callable
import logging
import time

from src.config import ADWIN_DELTA, DRIFT_WINDOW_SIZE

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Concept drift detector using River's ADWIN algorithm.

    ADWIN (Adaptive Windowing) maintains a variable-length window of recent
    observations and detects when the statistical properties of the data
    change significantly. We feed it classification confidence scores —
    a drop in confidence indicates the model is seeing unfamiliar traffic.
    """

    def __init__(
        self,
        delta: float = ADWIN_DELTA,
        window_size: int = DRIFT_WINDOW_SIZE,
        on_drift_callback: Optional[Callable] = None,
    ):
        """
        Args:
            delta: ADWIN sensitivity (lower = more sensitive to drift).
            window_size: Size of the recent observations window.
            on_drift_callback: Optional callback function when drift is detected.
        """
        self.adwin = drift.ADWIN(delta=delta)
        self.window_size = window_size
        self.on_drift_callback = on_drift_callback

        # Track history
        self.confidence_history = deque(maxlen=10000)
        self.drift_events = []
        self.observation_count = 0

        # State
        self.is_drifting = False
        self.last_drift_time = None

    def update(self, confidence: float) -> bool:
        """
        Feed a new classification confidence score to the drift detector.

        Args:
            confidence: Classification confidence (0.0 to 1.0).

        Returns:
            True if drift was detected on this update.
        """
        self.observation_count += 1
        self.confidence_history.append(confidence)

        self.adwin.update(confidence)

        if self.adwin.drift_detected:
            self.is_drifting = True
            self.last_drift_time = time.time()

            drift_event = {
                "observation": self.observation_count,
                "timestamp": self.last_drift_time,
                "confidence_at_drift": confidence,
                "mean_confidence_recent": np.mean(
                    list(self.confidence_history)[-self.window_size:]
                ),
            }
            self.drift_events.append(drift_event)

            logger.warning(
                f"⚠️ CONCEPT DRIFT DETECTED at observation {self.observation_count}. "
                f"Confidence: {confidence:.3f}, "
                f"Recent mean: {drift_event['mean_confidence_recent']:.3f}"
            )

            if self.on_drift_callback:
                self.on_drift_callback(drift_event)

            return True

        self.is_drifting = False
        return False

    def get_status(self) -> dict:
        """Get current drift detector status."""
        recent = list(self.confidence_history)[-self.window_size:]
        return {
            "is_drifting": self.is_drifting,
            "total_observations": self.observation_count,
            "total_drift_events": len(self.drift_events),
            "last_drift_time": self.last_drift_time,
            "mean_confidence": float(np.mean(recent)) if recent else 0.0,
            "std_confidence": float(np.std(recent)) if recent else 0.0,
            "recent_drift_events": self.drift_events[-5:],
        }

    def reset(self):
        """Reset the drift detector."""
        self.adwin = drift.ADWIN(delta=ADWIN_DELTA)
        self.is_drifting = False
        self.observation_count = 0
        self.confidence_history.clear()
        self.drift_events.clear()
        logger.info("Drift detector reset")
