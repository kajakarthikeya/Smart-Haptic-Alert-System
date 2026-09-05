"""
Prediction Result Structures, Confidence Evaluation, and Stability Gating.

Provides:
1. PredictionStatus: Enum representing alert confirmation levels (CONFIRMED, TENTATIVE, LOW_CONFIDENCE).
2. LatencyMetrics: High-precision timing metrics for profiling each inference pipeline stage.
3. PredictionResult: Rich, structured dataclass capturing full inference outcome.
4. PredictionStabilizer: Multi-window consensus buffer to eliminate spurious acoustic noise spikes.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import Config, InferenceConfig, settings
from app.ai.inference.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionStatus(str, Enum):
    """Classification stability and confidence status."""
    CONFIRMED = "CONFIRMED"            # Confident and confirmed by stability consensus
    TENTATIVE = "TENTATIVE"            # Confident single prediction, awaiting buffer consensus
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # Below configurable confidence threshold


@dataclass(frozen=True)
class LatencyMetrics:
    """Breakdown of processing durations in milliseconds."""
    preprocessing_ms: float
    feature_extraction_ms: float
    inference_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "feature_extraction_ms": round(self.feature_extraction_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass(frozen=True)
class PredictionResult:
    """Structured inference outcome for a single audio window."""
    timestamp: str
    predicted_class: str
    predicted_id: int
    raw_class: str
    raw_id: int
    confidence: float
    is_confident: bool
    status: PredictionStatus
    probabilities: Dict[str, float]
    latency: LatencyMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "predicted_class": self.predicted_class,
            "predicted_id": self.predicted_id,
            "raw_class": self.raw_class,
            "raw_id": self.raw_id,
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
            "status": self.status.value,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency": self.latency.to_dict(),
        }


class PredictionStabilizer:
    """
    Maintains a rolling buffer of recent predictions to filter out transient false alarms.

    Requires at least K identical predictions within the last N evaluation windows
    before an alert is promoted from TENTATIVE to CONFIRMED.
    """

    def __init__(
        self,
        buffer_size: int = 3,
        required_agreement: int = 2,
    ) -> None:
        """
        Args:
            buffer_size: Number of recent predictions to track (N, default: 3).
            required_agreement: Number of matching predictions required for consensus (K, default: 2).
        """
        if required_agreement > buffer_size:
            raise ValueError(
                f"required_agreement ({required_agreement}) cannot exceed buffer_size ({buffer_size})"
            )
        self.buffer_size = max(1, buffer_size)
        self.required_agreement = max(1, required_agreement)
        self._history: deque = deque(maxlen=self.buffer_size)

    @property
    def history(self) -> List[str]:
        """Returns current history queue of confident class predictions."""
        return list(self._history)

    def evaluate_stability(
        self,
        predicted_class: str,
        is_confident: bool,
    ) -> PredictionStatus:
        """
        Updates buffer with latest prediction and assesses confirmation consensus.

        Args:
            predicted_class: Sound class label.
            is_confident: True if prediction exceeded the confidence threshold.

        Returns:
            PredictionStatus: CONFIRMED, TENTATIVE, or LOW_CONFIDENCE.
        """
        if not is_confident:
            # Low confidence predictions do not contribute to positive agreement
            return PredictionStatus.LOW_CONFIDENCE

        self._history.append(predicted_class)

        # Count occurrences of current class in recent history
        matches = sum(1 for c in self._history if c == predicted_class)

        if matches >= self.required_agreement:
            return PredictionStatus.CONFIRMED

        return PredictionStatus.TENTATIVE

    def reset(self) -> None:
        """Clears the stability history buffer."""
        self._history.clear()
