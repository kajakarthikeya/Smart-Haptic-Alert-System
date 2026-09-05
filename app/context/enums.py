"""
Enumerations for Environmental Operating Modes and Alert Urgency Priorities.

Defines:
1. EnvironmentMode: Strongly-typed operational modes (HOME, ROAD, OFFICE).
2. PriorityLevel: Hierarchical alert urgency tiers (HIGH, MEDIUM, LOW, IGNORE).
"""

from enum import Enum
from typing import Any


class EnvironmentMode(str, Enum):
    """
    Supported user operating environment modes.

    - HOME: Domestic residential context (doorbells, alarms, appliances).
    - ROAD: Outdoor traffic and transit safety context (car horns, sirens).
    - OFFICE: Professional workplace context (focused work, reduced interruptions).
    """
    HOME = "HOME"
    ROAD = "ROAD"
    OFFICE = "OFFICE"

    @classmethod
    def from_string(cls, value: Any) -> "EnvironmentMode":
        """
        Parses a string or enum into an EnvironmentMode case-insensitively.

        Args:
            value: Mode name (e.g. 'home', 'Home', 'HOME', EnvironmentMode.HOME).

        Returns:
            EnvironmentMode member.

        Raises:
            ValueError: If string does not correspond to a valid mode.
        """
        if isinstance(value, cls):
            return value
        cleaned = str(value).strip().upper()
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        try:
            return cls[cleaned]
        except KeyError:
            valid_modes = [m.value for m in cls]
            raise ValueError(
                f"Invalid environment mode '{value}'. Must be one of: {valid_modes}"
            )


class PriorityLevel(str, Enum):
    """
    Alert urgency prioritization levels.

    - HIGH: Critical or safety-related sounds requiring immediate attention.
    - MEDIUM: Important sounds that should alert the user but are less urgent.
    - LOW: Sounds that may be relevant but generally do not require immediate attention.
    - IGNORE: Sounds that should not generate an alert in the current context.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    IGNORE = "IGNORE"

    @property
    def rank(self) -> int:
        """Numeric rank for comparative sorting (higher = more urgent)."""
        ranks = {
            self.HIGH: 3,
            self.MEDIUM: 2,
            self.LOW: 1,
            self.IGNORE: 0,
        }
        return ranks[self]


# Backwards compatibility alias for Phase 1 code
SoundPriority = PriorityLevel
