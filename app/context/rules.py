"""
Priority Rule Definitions and Configuration Matrix.

Defines the initial default priority matrix mapping environmental sounds and operating
modes to alert urgency tiers.
"""

from typing import Dict, Mapping
from app.context.enums import EnvironmentMode, PriorityLevel
from app.context.exceptions import ConfigurationError

# Recognized AI target sounds
TARGET_SOUNDS = (
    "ambulance",
    "car_horn",
    "fire_alarm",
    "doorbell",
    "dog_bark",
)

# Initial configurable priority matrix mapping: Mode -> Sound -> PriorityLevel
DEFAULT_PRIORITY_MATRIX: Dict[EnvironmentMode, Dict[str, PriorityLevel]] = {
    EnvironmentMode.HOME: {
        "ambulance": PriorityLevel.HIGH,
        "car_horn": PriorityLevel.MEDIUM,
        "fire_alarm": PriorityLevel.HIGH,
        "doorbell": PriorityLevel.HIGH,
        "dog_bark": PriorityLevel.MEDIUM,
    },
    EnvironmentMode.ROAD: {
        "ambulance": PriorityLevel.HIGH,
        "car_horn": PriorityLevel.HIGH,
        "fire_alarm": PriorityLevel.HIGH,
        "doorbell": PriorityLevel.LOW,
        "dog_bark": PriorityLevel.LOW,
    },
    EnvironmentMode.OFFICE: {
        "ambulance": PriorityLevel.HIGH,
        "car_horn": PriorityLevel.LOW,
        "fire_alarm": PriorityLevel.HIGH,
        "doorbell": PriorityLevel.LOW,
        "dog_bark": PriorityLevel.LOW,
    },
}


def validate_priority_matrix(matrix: Mapping[EnvironmentMode, Mapping[str, PriorityLevel]]) -> None:
    """
    Validates that a priority matrix covers all required modes and target sounds.

    Args:
        matrix: Dictionary mapping EnvironmentMode -> sound label -> PriorityLevel.

    Raises:
        ConfigurationError: If any mode or target sound rule is missing.
    """
    if not isinstance(matrix, Mapping):
        raise ConfigurationError("Priority matrix must be a mapping structure.")

    for mode in EnvironmentMode:
        if mode not in matrix:
            raise ConfigurationError(f"Priority matrix missing definition for mode: {mode.value}")
        
        mode_rules = matrix[mode]
        for sound in TARGET_SOUNDS:
            if sound not in mode_rules:
                raise ConfigurationError(
                    f"Priority matrix missing rule for sound '{sound}' under mode '{mode.value}'"
                )
            priority = mode_rules[sound]
            if not isinstance(priority, PriorityLevel):
                raise ConfigurationError(
                    f"Invalid priority level '{priority}' for sound '{sound}' in mode '{mode.value}'"
                )
