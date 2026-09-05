"""
Data Models for Sound Predictions, Alert Policies, and Context Decisions.

Defines:
1. SoundPrediction: Validated acoustic sound event payload.
2. AlertPolicy: Rules mapping PriorityLevel to boolean alert dispatch.
3. DecisionResult: Structured decision outcome containing priority, alert flag, and reason.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Union

from app.context.enums import EnvironmentMode, PriorityLevel
from app.context.exceptions import InvalidConfidenceError


@dataclass(frozen=True)
class SoundPrediction:
    """
    Validated prediction payload representing an AI sound classification event.
    """
    sound: str
    confidence: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate confidence is finite float in [0.0, 1.0]
        try:
            conf = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise InvalidConfidenceError(
                f"Confidence score must be numeric, got '{self.confidence}'"
            ) from exc

        if math.isnan(conf) or math.isinf(conf):
            raise InvalidConfidenceError(f"Confidence score cannot be NaN or infinite: {conf}")

        if not (0.0 <= conf <= 1.0):
            raise InvalidConfidenceError(
                f"Confidence score must be in range [0.0, 1.0], got {conf}"
            )

        # Normalize sound string
        if not isinstance(self.sound, str) or not self.sound.strip():
            raise ValueError("Sound label must be a non-empty string.")

    @classmethod
    def from_prediction_result(cls, result: Any) -> "SoundPrediction":
        """
        Creates a SoundPrediction directly from Phase 7 PredictionResult.

        Args:
            result: Phase 7 PredictionResult instance or duck-typed object.

        Returns:
            SoundPrediction instance.
        """
        sound = getattr(result, "predicted_class", None) or getattr(result, "sound", None)
        confidence = getattr(result, "confidence", 0.0)
        timestamp = getattr(result, "timestamp", None) or datetime.now(timezone.utc).isoformat()
        metadata = {}
        if hasattr(result, "to_dict") and callable(result.to_dict):
            metadata = result.to_dict()
        elif isinstance(result, dict):
            sound = result.get("predicted_class") or result.get("sound")
            confidence = result.get("confidence", 0.0)
            timestamp = result.get("timestamp", timestamp)
            metadata = result

        return cls(
            sound=str(sound),
            confidence=float(confidence),
            timestamp=str(timestamp),
            raw_metadata=metadata,
        )


@dataclass(frozen=True)
class AlertPolicy:
    """
    Policy governing whether a given PriorityLevel triggers an alert.
    """
    high: bool = True
    medium: bool = True
    low: bool = False
    ignore: bool = False

    def should_alert(self, priority: PriorityLevel) -> bool:
        """Determines if priority level warrants an alert."""
        if priority == PriorityLevel.HIGH:
            return self.high
        if priority == PriorityLevel.MEDIUM:
            return self.medium
        if priority == PriorityLevel.LOW:
            return self.low
        return self.ignore


@dataclass(frozen=True)
class DecisionResult:
    """
    Structured context-aware decision outcome.
    """
    sound: str
    confidence: float
    mode: EnvironmentMode
    priority: PriorityLevel
    alert_required: bool
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision into dictionary format."""
        return {
            "sound": self.sound,
            "confidence": round(float(self.confidence), 4),
            "mode": self.mode.value,
            "priority": self.priority.value,
            "alert_required": self.alert_required,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"DecisionResult(sound='{self.sound}', confidence={self.confidence:.2%}, "
            f"mode={self.mode.value}, priority={self.priority.value}, "
            f"alert={self.alert_required}, reason='{self.reason}')"
        )
