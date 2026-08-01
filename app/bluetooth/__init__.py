"""Bluetooth hardware interface package initialization."""

from app.bluetooth.ble_manager import BaseBLEManager, ESP32BLEManager
from app.bluetooth.protocol import HapticPacketSerializer, HapticPattern

__all__ = ["BaseBLEManager", "ESP32BLEManager", "HapticPacketSerializer", "HapticPattern"]
