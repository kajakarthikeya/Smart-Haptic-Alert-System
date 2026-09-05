"""
Priority Rule Engine for Environmental Sounds and Operational Modes.

Looks up priority levels from a configurable rule matrix with strict validation.
"""

from typing import Dict, Mapping, Optional, Union
from app.context.enums import EnvironmentMode, PriorityLevel
from app.context.exceptions import (
    ConfigurationError,
    InvalidModeError,
    PriorityRuleError,
    UnknownSoundError,
)
from app.context.rules import DEFAULT_PRIORITY_MATRIX, TARGET_SOUNDS, validate_priority_matrix
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PriorityEngine:
    """
    Evaluates urgency priority for recognized environmental sounds under the active mode.
    """

    def __init__(
        self,
        matrix: Optional[Mapping[EnvironmentMode, Mapping[str, PriorityLevel]]] = None,
        target_sounds: Optional[tuple[str, ...]] = None,
    ) -> None:
        """
        Initializes the PriorityEngine.

        Args:
            matrix: Optional custom priority matrix. Defaults to DEFAULT_PRIORITY_MATRIX.
            target_sounds: Optional tuple of valid sound names. Defaults to TARGET_SOUNDS.
        """
        self._target_sounds = tuple(s.lower() for s in (target_sounds or TARGET_SOUNDS))
        self._matrix: Dict[EnvironmentMode, Dict[str, PriorityLevel]] = {}
        
        # Load and validate matrix
        source_matrix = matrix if matrix is not None else DEFAULT_PRIORITY_MATRIX
        validate_priority_matrix(source_matrix)
        
        for mode, sound_rules in source_matrix.items():
            self._matrix[mode] = {k.lower(): v for k, v in sound_rules.items()}

        logger.debug("PriorityEngine initialized with %d operating modes", len(self._matrix))

    @property
    def supported_sounds(self) -> tuple[str, ...]:
        """Returns tuple of recognized target sounds."""
        return self._target_sounds

    def get_priority(
        self,
        sound: str,
        mode: Union[EnvironmentMode, str],
    ) -> PriorityLevel:
        """
        Determines priority level for a sound under the specified operating mode.

        Args:
            sound: Environmental sound label.
            mode: Operating environment mode.

        Returns:
            PriorityLevel (HIGH, MEDIUM, LOW, IGNORE).

        Raises:
            UnknownSoundError: If sound is not recognized.
            InvalidModeError: If mode is not valid.
            PriorityRuleError: If rule is missing in configuration.
        """
        # Validate sound
        if not isinstance(sound, str) or not sound.strip():
            raise UnknownSoundError("Sound label must be a non-empty string.")

        clean_sound = sound.strip().lower()
        if clean_sound not in self._target_sounds:
            logger.error("Unknown sound '%s' rejected by PriorityEngine", sound)
            raise UnknownSoundError(
                f"Unknown sound '{sound}'. Expected one of: {self._target_sounds}"
            )

        # Validate mode
        if isinstance(mode, EnvironmentMode):
            env_mode = mode
        elif isinstance(mode, str):
            try:
                env_mode = EnvironmentMode.from_string(mode)
            except ValueError as exc:
                logger.error("Invalid operating mode '%s' in PriorityEngine", mode)
                raise InvalidModeError(str(exc)) from exc
        else:
            raise InvalidModeError(f"Unsupported mode type: {type(mode).__name__}")

        if env_mode not in self._matrix:
            logger.error("Mode '%s' not configured in PriorityEngine matrix", env_mode.value)
            raise ConfigurationError(f"Mode '{env_mode.value}' is not configured in priority matrix.")

        mode_rules = self._matrix[env_mode]
        if clean_sound not in mode_rules:
            logger.error(
                "No priority rule found for sound '%s' in mode '%s'",
                clean_sound,
                env_mode.value,
            )
            raise PriorityRuleError(
                f"No priority rule configured for sound '{clean_sound}' under mode '{env_mode.value}'"
            )

        priority = mode_rules[clean_sound]
        logger.debug(
            "PriorityEngine: sound='%s', mode='%s' -> priority=%s",
            clean_sound,
            env_mode.value,
            priority.value,
        )
        return priority
