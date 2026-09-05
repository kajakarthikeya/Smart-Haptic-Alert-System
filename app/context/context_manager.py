"""
Context Engine Manager for Environmental Mode Prioritization.

Provides a unified facade integrating the Phase 8 ContextDecisionEngine and ModeManager
while maintaining backwards compatibility for existing services and tests.
"""

from typing import Any, Dict, Optional, Tuple, Union
from app.context.config.mode_profiles import (
    EnvironmentMode as LegacyEnvironmentMode,
    SoundPriority as LegacySoundPriority,
    ModeProfile,
    MODE_PROFILES,
)
from app.context.decision_engine import ContextDecisionEngine
from app.context.enums import EnvironmentMode, PriorityLevel, SoundPriority
from app.context.exceptions import ContextError
from app.context.mode_manager import ModeManager
from app.context.models import DecisionResult, SoundPrediction
from app.context.priority_engine import PriorityEngine
from app.context.rules import TARGET_SOUNDS
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    """
    Unified manager for environmental mode tracking and context-aware sound decisions.
    """

    def __init__(
        self,
        initial_mode: Union[EnvironmentMode, LegacyEnvironmentMode] = EnvironmentMode.HOME,
        decision_engine: Optional[ContextDecisionEngine] = None,
    ) -> None:
        """
        Initializes ContextManager.

        Args:
            initial_mode: Initial active operating environment mode.
            decision_engine: Optional custom ContextDecisionEngine instance.
        """
        # Initialize mode manager
        mode_val = initial_mode.value if hasattr(initial_mode, "value") else str(initial_mode)
        self._mode_manager = ModeManager(default_mode=EnvironmentMode.from_string(mode_val))
        
        # Initialize decision engine
        self._decision_engine = decision_engine or ContextDecisionEngine(
            mode_manager=self._mode_manager,
            priority_engine=PriorityEngine(),
        )

        # Legacy profiles dictionary for backwards compatibility
        self._profiles: Dict[Any, ModeProfile] = dict(MODE_PROFILES)
        logger.info("ContextManager facade initialized with mode: %s", self.active_mode.value)

    @property
    def active_mode(self) -> EnvironmentMode:
        """Returns currently active operating mode."""
        return self._mode_manager.current_mode

    @property
    def mode_manager(self) -> ModeManager:
        """Returns underlying ModeManager instance."""
        return self._mode_manager

    @property
    def decision_engine(self) -> ContextDecisionEngine:
        """Returns underlying ContextDecisionEngine instance."""
        return self._decision_engine

    @property
    def current_profile(self) -> ModeProfile:
        """Legacy access to active ModeProfile object."""
        # Find matching key in _profiles
        for mode_key, profile in self._profiles.items():
            if getattr(mode_key, "value", str(mode_key)) == self.active_mode.value:
                return profile
        return self._profiles.get(LegacyEnvironmentMode[self.active_mode.value])

    def set_mode(self, new_mode: Union[EnvironmentMode, LegacyEnvironmentMode, str]) -> None:
        """
        Switches active operating mode.

        Args:
            new_mode: Target operating mode.
        """
        mode_val = new_mode.value if hasattr(new_mode, "value") else str(new_mode)
        self._mode_manager.set_mode(EnvironmentMode.from_string(mode_val))

    def register_profile(self, profile: ModeProfile) -> None:
        """Legacy method to register custom mode profile."""
        self._profiles[profile.mode] = profile
        logger.info("Registered custom mode profile for mode: %s", profile.mode.value)

    def evaluate(
        self,
        prediction: Union[SoundPrediction, Any, Tuple[str, float]],
        override_mode: Optional[Union[EnvironmentMode, str]] = None,
    ) -> DecisionResult:
        """
        Evaluates a sound prediction using the Phase 8 Decision Engine.

        Args:
            prediction: SoundPrediction or Phase 7 PredictionResult.
            override_mode: Optional mode override.

        Returns:
            DecisionResult dataclass.
        """
        return self._decision_engine.evaluate(prediction, override_mode=override_mode)

    def evaluate_sound(
        self,
        sound_label: str,
        confidence: float,
    ) -> Tuple[bool, Any]:
        """
        Legacy evaluation method returning (should_alert: bool, priority: SoundPriority).

        Args:
            sound_label: Identified sound label.
            confidence: Confidence score between 0.0 and 1.0.

        Returns:
            Tuple of (should_alert: bool, priority: SoundPriority).
        """
        profile = self.current_profile
        priority = profile.get_priority(sound_label)

        p_name = getattr(priority, "name", str(priority))
        if p_name == "IGNORE" or priority == LegacySoundPriority.IGNORE:
            return False, priority

        threshold = getattr(profile, "min_confidence_threshold", 0.60)
        if confidence < threshold and p_name != "CRITICAL":
            return False, priority

        return True, priority
