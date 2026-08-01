"""Unit tests for AlertService dispatch logic."""

import unittest
from app.services.alert_service import AlertService
from app.context.context_manager import ContextManager, EnvironmentMode
from app.bluetooth.ble_manager import ESP32BLEManager


class TestAlertService(unittest.IsolatedAsyncioTestCase):
    """Test suite for async AlertService processing."""

    async def test_alert_dispatch_success(self) -> None:
        """Verifies sound event dispatching produces alert record."""
        ctx = ContextManager(initial_mode=EnvironmentMode.HOME)
        ble = ESP32BLEManager()
        await ble.connect()

        service = AlertService(context_manager=ctx, ble_manager=ble)
        record = await service.handle_sound_event("doorbell", confidence=0.92)

        self.assertIsNotNone(record)
        self.assertEqual(record["sound_label"], "doorbell")
        self.assertTrue(record["delivered_ble"])
        self.assertEqual(len(service.get_alert_history()), 1)


if __name__ == "__main__":
    unittest.main()
