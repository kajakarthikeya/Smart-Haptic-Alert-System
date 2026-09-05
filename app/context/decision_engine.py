"""
Context-Aware Decision Engine.

Combines AI acoustic predictions, active operational mode, priority matrix rules,
and confidence gating to produce structured alert decisions with human-readable reasons.
"""

from typing import Any, Optional, Tuple, Union
from app.context.config import get_context_config
from app.context.enums import EnvironmentMode, PriorityLevel
from app.context.exceptions import ContextError, InvalidConfidenceError
from app.context.mode_manager import ModeManager
from app.context.models import AlertPolicy, DecisionResult, SoundPrediction
from app.context.priority_engine import PriorityEngine
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ContextDecisionEngine:
    """
    Coordinates context-aware alert decision evaluation.
    """

    def __init__(
        self,
        mode_manager: Optional[ModeManager] = None,
        priority_engine: Optional[PriorityEngine] = None,
        confidence_threshold: Optional[float] = None,
        alert_policy: Optional[AlertPolicy] = None,
    ) -> None:
        """
        Initializes ContextDecisionEngine.

        Args:
            mode_manager: Optional custom ModeManager instance.
            priority_engine: Optional custom PriorityEngine instance.
            confidence_threshold: Confidence cutoff (defaults to config setting, e.g. 0.70).
            alert_policy: Optional custom AlertPolicy. Defaults to standard policy.
        """
        config = get_context_config()
        self.mode_manager = mode_manager or ModeManager(default_mode=config.default_mode)
        self.priority_engine = priority_engine or PriorityEngine()
        
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(config.confidence_threshold)
        )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise InvalidConfidenceError(
                f"Confidence threshold must be in range [0.0, 1.0], got {self.confidence_threshold}"
            )

        self.alert_policy = alert_policy or AlertPolicy(
            high=config.alert_on_high,
            medium=config.alert_on_medium,
            low=config.alert_on_low,
            ignore=config.alert_on_ignore,
        )

        logger.info(
            "ContextDecisionEngine initialized: default_mode=%s, confidence_threshold=%.2f",
            self.mode_manager.current_mode.value,
            self.confidence_threshold,
        )

    def evaluate(
        self,
        prediction: Union[SoundPrediction, Any, Tuple[str, float]],
        override_mode: Optional[Union[EnvironmentMode, str]] = None,
    ) -> DecisionResult:
        """
        Evaluates a sound prediction in context and determines alert necessity.

        Args:
            prediction: SoundPrediction, Phase 7 PredictionResult, or (sound, confidence) tuple.
            override_mode: Optional mode override for single evaluation without changing state.

        Returns:
            DecisionResult dataclass.

        Raises:
            UnknownSoundError: If sound label is unrecognized.
            InvalidConfidenceError: If confidence is invalid.
            InvalidModeError: If operating mode is invalid.
        """
        # Standardize prediction into SoundPrediction model
        pred_obj = self._coerce_prediction(prediction)
        sound = pred_obj.sound.lower().strip()
        confidence = pred_obj.confidence

        # Determine effective evaluation mode
        active_mode = (
            self.mode_manager._validate_mode(override_mode)
            if override_mode is not None
            else self.mode_manager.current_mode
        )

        # Confidence gating: if below threshold, return safe IGNORE decision
        if confidence < self.confidence_threshold:
            logger.warning(
                "Prediction confidence (%.4f) for sound '%s' below threshold (%.2f). Alert suppressed.",
                confidence,
                sound,
                self.confidence_threshold,
            )
            return DecisionResult(
                sound=sound,
                confidence=confidence,
                mode=active_mode,
                priority=PriorityLevel.IGNORE,
                alert_required=False,
                reason=(
                    f"Prediction confidence ({confidence:.2%}) is below "
                    f"the configured threshold ({self.confidence_threshold:.2%})."
                ),
                timestamp=pred_obj.timestamp,
            )

        # Look up priority rule
        priority = self.priority_engine.get_priority(sound, active_mode)

        # Evaluate alert policy
        alert_required = self.alert_policy.should_alert(priority)

        # Synthesize transparent reason
        sound_display = sound.replace("_", " ").title()
        if alert_required:
            reason = f"{sound_display} has {priority.value} priority in {active_mode.value} mode."
        else:
            reason = (
                f"{sound_display} is classified as {priority.value} priority in "
                f"{active_mode.value} mode (no immediate alert required)."
            )

        logger.info(
            "Sound: %s | Confidence: %.2f | Mode: %s | Priority: %s | Alert: %s",
            sound.upper(),
            confidence,
            active_mode.value,
            priority.value,
            alert_required,
        )

        return DecisionResult(
            sound=sound,
            confidence=confidence,
            mode=active_mode,
            priority=priority,
            alert_required=alert_required,
            reason=reason,
            timestamp=pred_obj.timestamp,
        )

    def _coerce_prediction(
        self,
        raw_pred: Union[SoundPrediction, Any, Tuple[str, float]],
    ) -> SoundPrediction:
        """Coerces various prediction input formats into a validated SoundPrediction."""
        if isinstance(raw_pred, SoundPrediction):
            return raw_pred

        if isinstance(raw_pred, tuple) and len(raw_pred) == 2:
            return SoundPrediction(sound=str(raw_pred[0]), confidence=float(raw_pred[1]))

        # Handle Phase 7 PredictionResult or dict
        return SoundPrediction.from_prediction_result(raw_pred)
