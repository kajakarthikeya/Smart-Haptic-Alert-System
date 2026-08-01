"""Mode Profiles and Priority Configurations for Environmental Contexts."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Mapping


class EnvironmentMode(str, Enum):
    """Supported user operating environment modes."""
    HOME = "HOME"
    ROAD = "ROAD"
    OFFICE = "OFFICE"


class SoundPriority(int, Enum):
    """Alert urgency prioritization levels."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    IGNORE = 0


@dataclass(frozen=True)
class ModeProfile:
    """Configurable sound priority mapping profile for a given mode."""
    mode: EnvironmentMode
    description: str
    priority_rules: Mapping[str, SoundPriority]
    min_confidence_threshold: float = 0.60

    def get_priority(self, sound_label: str) -> SoundPriority:
        """Looks up priority level for a recognized sound label.

        Args:
            sound_label: Standardized label of detected environmental sound.

        Returns:
            Mapped SoundPriority or DEFAULT priority (LOW).
        """
        # Universal emergency override
        if sound_label.lower() in {"fire_alarm", "smoke_detector", "explosion"}:
            return SoundPriority.CRITICAL
            
        return self.priority_rules.get(sound_label.lower(), SoundPriority.LOW)


# Default mode profiles dictionary
MODE_PROFILES: Dict[EnvironmentMode, ModeProfile] = {
    EnvironmentMode.HOME: ModeProfile(
        mode=EnvironmentMode.HOME,
        description="Optimized for domestic sounds such as doorbells, alarms, and infants.",
        priority_rules={
            "fire_alarm": SoundPriority.CRITICAL,
            "baby_crying": SoundPriority.HIGH,
            "doorbell": SoundPriority.HIGH,
            "door_knock": SoundPriority.MEDIUM,
            "glass_shatter": SoundPriority.HIGH,
            "dog_bark": SoundPriority.MEDIUM,
            "car_horn": SoundPriority.LOW,
            "speech": SoundPriority.LOW,
        },
        min_confidence_threshold=0.55,
    ),
    EnvironmentMode.ROAD: ModeProfile(
        mode=EnvironmentMode.ROAD,
        description="Optimized for outdoor and traffic safety awareness.",
        priority_rules={
            "fire_alarm": SoundPriority.CRITICAL,
            "siren": SoundPriority.CRITICAL,
            "car_horn": SoundPriority.HIGH,
            "vehicle_engine": SoundPriority.MEDIUM,
            "bicycle_bell": SoundPriority.HIGH,
            "shout_screaming": SoundPriority.HIGH,
            "doorbell": SoundPriority.IGNORE,
            "baby_crying": SoundPriority.LOW,
        },
        min_confidence_threshold=0.65,
    ),
    EnvironmentMode.OFFICE: ModeProfile(
        mode=EnvironmentMode.OFFICE,
        description="Optimized for workspace interactions while filtering non-work ambient noise.",
        priority_rules={
            "fire_alarm": SoundPriority.CRITICAL,
            "door_knock": SoundPriority.HIGH,
            "speech_calling_name": SoundPriority.HIGH,
            "phone_ring": SoundPriority.MEDIUM,
            "alarm_clock": SoundPriority.MEDIUM,
            "keyboard_typing": SoundPriority.IGNORE,
            "car_horn": SoundPriority.IGNORE,
        },
        min_confidence_threshold=0.60,
    ),
}
