"""Mode Controller for managing environment operating mode changes."""

from typing import Dict, Optional
from app.context.context_manager import ContextManager
from app.context.config.mode_profiles import EnvironmentMode
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModeController:
    """Controller exposing mode switching and context query actions."""

    def __init__(self, context_manager: Optional[ContextManager] = None) -> None:
        """Initializes controller with ContextManager dependency.

        Args:
            context_manager: ContextManager instance.
        """
        self._context_manager = context_manager or ContextManager()
        logger.info("ModeController initialized.")

    def get_current_mode(self) -> Dict[str, str]:
        """Returns details of the active operating mode.

        Returns:
            Dictionary with active mode name and description.
        """
        profile = self._context_manager.current_profile
        return {
            "mode": profile.mode.value,
            "description": profile.description,
            "min_confidence_threshold": str(profile.min_confidence_threshold),
        }

    def switch_mode(self, mode_name: str) -> Dict[str, str]:
        """Switches the environmental mode by name string (HOME, ROAD, OFFICE).

        Args:
            mode_name: Upper/lowercase mode string.

        Returns:
            Success status dictionary or error message.
        """
        try:
            target_mode = EnvironmentMode[mode_name.upper()]
            self._context_manager.set_mode(target_mode)
            logger.info(f"Successfully switched mode to {target_mode.value}")
            return {"status": "success", "mode": target_mode.value}
        except KeyError:
            valid_modes = [m.value for m in EnvironmentMode]
            logger.error(f"Invalid mode requested: '{mode_name}'. Valid options: {valid_modes}")
            return {"status": "error", "message": f"Invalid mode. Choose from {valid_modes}"}
