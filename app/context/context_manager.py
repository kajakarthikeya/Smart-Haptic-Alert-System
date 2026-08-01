"""Context Engine Manager for Environmental Mode Prioritization."""

from typing import Dict, Optional, Tuple
from app.context.config.mode_profiles import (
    EnvironmentMode,
    SoundPriority,
    ModeProfile,
    MODE_PROFILES,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    """Manages active environmental mode and prioritizes detected sounds accordingly."""

    def __init__(self, initial_mode: EnvironmentMode = EnvironmentMode.HOME) -> None:
        """Initializes ContextManager with standard default profiles.

        Args:
            initial_mode: Initial active operating environment mode.
        """
        self._profiles: Dict[EnvironmentMode, ModeProfile] = dict(MODE_PROFILES)
        self._active_mode: EnvironmentMode = initial_mode
        logger.info(f"ContextManager initialized with mode: {self._active_mode.value}")

    @property
    def active_mode(self) -> EnvironmentMode:
        """Returns currently active environment mode."""
        return self._active_mode

    @property
    def current_profile(self) -> ModeProfile:
        """Returns the active ModeProfile object."""
        return self._profiles[self._active_mode]

    def set_mode(self, new_mode: EnvironmentMode) -> None:
        """Switches the active environmental operating mode.

        Args:
            new_mode: The new EnvironmentMode to set.
        """
        if new_mode not in self._profiles:
            raise ValueError(f"Unsupported environment mode: {new_mode}")
        
        old_mode = self._active_mode
        self._active_mode = new_mode
        logger.info(f"Environment mode changed from {old_mode.value} to {new_mode.value}")

    def register_profile(self, profile: ModeProfile) -> None:
        """Registers or overrides a custom mode profile.

        Args:
            profile: Custom ModeProfile instance to register.
        """
        self._profiles[profile.mode] = profile
        logger.info(f"Registered custom mode profile for mode: {profile.mode.value}")

    def evaluate_sound(self, sound_label: str, confidence: float) -> Tuple[bool, SoundPriority]:
        """Evaluates detected sound against active mode profile and confidence thresholds.

        Args:
            sound_label: Identified environmental sound label.
            confidence: Inference confidence score between 0.0 and 1.0.

        Returns:
            Tuple of (should_alert: bool, priority: SoundPriority).
        """
        profile = self.current_profile
        priority = profile.get_priority(sound_label)

        if priority == SoundPriority.IGNORE:
            logger.debug(f"Sound '{sound_label}' ignored under mode {self._active_mode.value}")
            return False, SoundPriority.IGNORE

        if confidence < profile.min_confidence_threshold and priority != SoundPriority.CRITICAL:
            logger.debug(
                f"Sound '{sound_label}' confidence ({confidence:.2f}) below threshold "
                f"({profile.min_confidence_threshold:.2f}) for mode {self._active_mode.value}"
            )
            return False, priority

        logger.info(
            f"Sound '{sound_label}' evaluated under {self._active_mode.value}: "
            f"ALERT TRIGGERED with priority {priority.name}"
        )
        return True, priority
