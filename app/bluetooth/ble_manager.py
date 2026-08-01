"""Bluetooth LE Client Manager Interface & Implementation for ESP32 Wearable Device."""

from abc import ABC, abstractmethod
from typing import Optional
from app.context.config.mode_profiles import SoundPriority
from app.bluetooth.protocol import HapticPacketSerializer
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class BaseBLEManager(ABC):
    """Abstract Base Class for BLE Communication Managers."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establishes connection to the BLE target device.

        Returns:
            True if connected successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnects from the active BLE target device."""
        pass

    @abstractmethod
    async def send_haptic_alert(self, alert_id: str, priority: SoundPriority) -> bool:
        """Transmits a haptic alert packet to the wearable device.

        Args:
            alert_id: Correlation ID for alert.
            priority: Urgency level of alert.

        Returns:
            True if packet was written successfully.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Returns connection status."""
        pass


class ESP32BLEManager(BaseBLEManager):
    """Concrete BLE Client Manager handling communication with ESP32 Wearable Device.

    Note: Full Bleak BLE transport implementation will be added in Hardware Integration phase.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        service_uuid: Optional[str] = None,
        char_uuid: Optional[str] = None,
    ) -> None:
        """Initializes BLE settings from config defaults.

        Args:
            device_name: Target ESP32 device name.
            service_uuid: GATT service UUID.
            char_uuid: GATT characteristic UUID for haptic writes.
        """
        self._device_name = device_name or settings.ble.device_name
        self._service_uuid = service_uuid or settings.ble.alert_service_uuid
        self._char_uuid = char_uuid or settings.ble.alert_char_uuid
        self._connected: bool = False
        logger.info(f"ESP32BLEManager initialized for device target '{self._device_name}'")

    async def connect(self) -> bool:
        """Simulates BLE connection handshake to target ESP32 device."""
        logger.info(f"Connecting to ESP32 device '{self._device_name}' [UUID: {self._service_uuid}]...")
        # Placeholder for bleak BleakClient connection logic
        self._connected = True
        logger.info("BLE Connection established successfully (Mock/Starter state)")
        return True

    async def disconnect(self) -> None:
        """Simulates BLE disconnection."""
        if self._connected:
            logger.info(f"Disconnecting from ESP32 device '{self._device_name}'...")
            self._connected = False
            logger.info("BLE Connection closed.")

    async def send_haptic_alert(self, alert_id: str, priority: SoundPriority) -> bool:
        """Serializes alert and writes payload over BLE GATT characteristic.

        Args:
            alert_id: Correlation identifier.
            priority: Urgency priority level.

        Returns:
            True if transmission succeeds.
        """
        if not self._connected:
            logger.warning("Attempted to send BLE alert without active connection. Connecting automatically...")
            connected = await self.connect()
            if not connected:
                logger.error("Failed to establish BLE connection for alert dispatch.")
                return False

        payload = HapticPacketSerializer.encode(alert_id, priority)
        logger.info(
            f"Transmitting BLE Haptic Alert [{alert_id}] to ESP32: "
            f"Payload size={len(payload)} bytes, Priority={priority.name}"
        )
        # Placeholder for await client.write_gatt_char(self._char_uuid, payload)
        return True

    @property
    def is_connected(self) -> bool:
        """Returns True if mock/real connection is active."""
        return self._connected
