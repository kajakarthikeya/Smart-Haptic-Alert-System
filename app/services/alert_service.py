"""Central Alert Processing & Prioritization Dispatcher Service."""

from typing import Dict, List, Optional
from app.context.context_manager import ContextManager
from app.bluetooth.ble_manager import BaseBLEManager, ESP32BLEManager
from app.utils.helpers import generate_alert_id, format_timestamp
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertService:
    """Core Service orchestrating sound detection evaluation and ESP32 haptic alert dispatch."""

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        ble_manager: Optional[BaseBLEManager] = None,
    ) -> None:
        """Initializes AlertService dependencies.

        Args:
            context_manager: Environmental ContextManager instance.
            ble_manager: Bluetooth manager instance for ESP32 target.
        """
        self._context_manager = context_manager or ContextManager()
        self._ble_manager = ble_manager or ESP32BLEManager()
        self._alert_history: List[Dict] = []
        logger.info("AlertService initialized cleanly.")

    async def handle_sound_event(self, sound_label: str, confidence: float) -> Optional[Dict]:
        """Evaluates detected sound against current mode and transmits haptic alert if authorized.

        Args:
            sound_label: Identified environmental sound.
            confidence: Model confidence score (0.0 - 1.0).

        Returns:
            Alert record dictionary if alert was sent, None if ignored.
        """
        should_alert, priority = self._context_manager.evaluate_sound(sound_label, confidence)

        if not should_alert:
            logger.info(f"Sound '{sound_label}' ignored under active profile or confidence threshold.")
            return None

        alert_id = generate_alert_id()
        timestamp = format_timestamp()

        logger.info(
            f"ALERT DISPATCH [{alert_id}]: Label='{sound_label}', Priority={priority.name}, "
            f"Mode='{self._context_manager.active_mode.value}'"
        )

        # Transmit via Bluetooth LE to ESP32 Wearable
        success = await self._ble_manager.send_haptic_alert(alert_id, priority)

        record = {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "sound_label": sound_label,
            "confidence": confidence,
            "priority": priority.name,
            "mode": self._context_manager.active_mode.value,
            "delivered_ble": success,
        }

        self._alert_history.append(record)
        return record

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Returns recent alert event history.

        Args:
            limit: Maximum records to return.

        Returns:
            List of recent alert dictionaries.
        """
        return self._alert_history[-limit:]
