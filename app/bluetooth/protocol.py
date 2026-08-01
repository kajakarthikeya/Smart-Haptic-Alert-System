"""Haptic Alert Protocol Serializer for ESP32 Wearable Device."""

from enum import Enum, auto
import struct
from typing import NamedTuple
from app.context.config.mode_profiles import SoundPriority


class HapticPattern(int, Enum):
    """Predefined vibration motor pulse patterns on ESP32."""
    SINGLE_SHORT = 1
    SINGLE_LONG = 2
    DOUBLE_PULSE = 3
    TRIPLE_PULSE = 4
    CONTINUOUS_ALERT = 5


class HapticPacket(NamedTuple):
    """Structured binary payload packet sent over BLE."""
    alert_id_hash: int  # uint16 hash of alert string
    priority: int       # uint8 priority level
    pattern: int        # uint8 pattern ID
    duration_ms: int    # uint16 vibration duration


class HapticPacketSerializer:
    """Serializes alert metadata and priority into binary byte arrays for ESP32 BLE GATT payload."""

    @staticmethod
    def map_priority_to_pattern(priority: SoundPriority) -> Tuple[HapticPattern, int]:
        """Maps SoundPriority level to a corresponding HapticPattern and duration in milliseconds.

        Args:
            priority: Priority level of detected sound.

        Returns:
            Tuple of (HapticPattern, duration_ms).
        """
        if priority == SoundPriority.CRITICAL:
            return HapticPattern.CONTINUOUS_ALERT, 3000
        elif priority == SoundPriority.HIGH:
            return HapticPattern.TRIPLE_PULSE, 1500
        elif priority == SoundPriority.MEDIUM:
            return HapticPattern.DOUBLE_PULSE, 1000
        else:
            return HapticPattern.SINGLE_SHORT, 500

    @classmethod
    def encode(cls, alert_id: str, priority: SoundPriority) -> bytes:
        """Encodes alert information into a binary byte string.

        Packet Format:
        [Byte 0-1]: uint16 Alert ID Hash
        [Byte 2]:   uint8  Priority Level (0-4)
        [Byte 3]:   uint8  Haptic Pattern ID (1-5)
        [Byte 4-5]: uint16 Duration in milliseconds

        Args:
            alert_id: Unique string identifier for alert.
            priority: SoundPriority enum value.

        Returns:
            6-byte binary payload for BLE write payload.
        """
        alert_hash = sum(ord(c) for c in alert_id) % 65535
        pattern, duration = cls.map_priority_to_pattern(priority)

        # Pack format: >HBBH (Big-endian: uint16, uint8, uint8, uint16)
        payload = struct.pack(">HBBH", alert_hash, priority.value, pattern.value, duration)
        return payload
