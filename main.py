"""Main Application Bootstrap Entrypoint for Smart Haptic Alert System."""

import asyncio
import sys
from typing import NoReturn

from config import settings
from app.utils.logger import setup_logger, get_logger
from app.context.context_manager import ContextManager, EnvironmentMode
from app.bluetooth.ble_manager import ESP32BLEManager
from app.services.alert_service import AlertService
from app.services.audio_service import AudioService
from app.controllers.alert_controller import AlertController
from app.controllers.mode_controller import ModeController

# Setup application logger
logger = setup_logger("SmartHapticAlertSystem")


def print_banner() -> None:
    """Prints application startup banner."""
    banner = f"""
    ======================================================================
       {settings.system.app_name} v0.1.0
       Architecture: Clean Architecture (SOLID Principles)
       Runtime Environment: {settings.system.app_env.upper()}
       Initial Mode: {settings.system.initial_mode}
       BLE Device Target: {settings.ble.device_name}
    ======================================================================
    """
    print(banner)


async def bootstrap_system() -> None:
    """Bootstraps core services, performs health checks, and initializes components."""
    logger.info("Initializing system components...")

    # 1. Initialize Context Manager
    initial_mode = EnvironmentMode[settings.system.initial_mode]
    context_mgr = ContextManager(initial_mode=initial_mode)

    # 2. Initialize BLE Hardware Manager
    ble_mgr = ESP32BLEManager()
    await ble_mgr.connect()

    # 3. Initialize Core Business Services
    alert_service = AlertService(context_manager=context_mgr, ble_manager=ble_mgr)
    audio_service = AudioService()

    # 4. Initialize Controllers
    mode_ctrl = ModeController(context_manager=context_mgr)
    alert_ctrl = AlertController(alert_service=alert_service)

    logger.info(f"System boot complete. Current Mode: {mode_ctrl.get_current_mode()['mode']}")

    # Verification sample alert evaluation
    logger.info("Executing initial boot verification alert check ('doorbell')...")
    test_result = await alert_ctrl.trigger_manual_alert(sound_label="doorbell", confidence=0.92)
    logger.info(f"Boot verification result: {test_result['status']}")

    logger.info("System is operational and ready for continuous audio intake.")


def main() -> None:
    """Entrypoint function."""
    print_banner()
    try:
        asyncio.run(bootstrap_system())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception during system bootstrap: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
