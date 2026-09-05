"""
Operating Mode Manager for User Environmental Context.

Manages active operational mode (HOME, ROAD, OFFICE), enforces validation,
and dispatches state change events to registered listeners.
"""

from typing import Callable, List, Union
from app.context.enums import EnvironmentMode
from app.context.exceptions import InvalidModeError
from app.utils.logger import get_logger

logger = get_logger(__name__)

ModeChangeCallback = Callable[[EnvironmentMode, EnvironmentMode], None]


class ModeManager:
    """
    State manager for user environmental operating modes.
    """

    def __init__(self, default_mode: Union[EnvironmentMode, str] = EnvironmentMode.HOME) -> None:
        """
        Initializes ModeManager with a safe default mode.

        Args:
            default_mode: Initial operating mode (default: HOME).
        """
        self._default_mode = self._validate_mode(default_mode)
        self._current_mode = self._default_mode
        self._listeners: List[ModeChangeCallback] = []
        logger.info("ModeManager initialized with mode: %s", self._current_mode.value)

    @property
    def current_mode(self) -> EnvironmentMode:
        """Returns the currently active EnvironmentMode."""
        return self._current_mode

    def get_mode(self) -> EnvironmentMode:
        """Accessor for current operating mode."""
        return self._current_mode

    def set_mode(self, new_mode: Union[EnvironmentMode, str]) -> EnvironmentMode:
        """
        Switches the operational mode and notifies registered listeners.

        Args:
            new_mode: Target operating mode.

        Returns:
            The newly set EnvironmentMode.

        Raises:
            InvalidModeError: If new_mode is unrecognized or invalid.
        """
        validated_mode = self._validate_mode(new_mode)
        old_mode = self._current_mode

        if validated_mode == old_mode:
            logger.debug("Mode unchanged: %s", old_mode.value)
            return self._current_mode

        self._current_mode = validated_mode
        logger.info("Mode changed: %s -> %s", old_mode.value, validated_mode.value)

        # Notify observers
        for callback in self._listeners:
            try:
                callback(old_mode, validated_mode)
            except Exception as exc:
                logger.warning("Error in mode change listener: %s", exc)

        return self._current_mode

    def reset(self) -> EnvironmentMode:
        """Resets the operating mode back to default (HOME)."""
        logger.info("Resetting mode to default: %s", self._default_mode.value)
        return self.set_mode(self._default_mode)

    def register_listener(self, callback: ModeChangeCallback) -> None:
        """Registers a callback function to be notified on mode transitions."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def unregister_listener(self, callback: ModeChangeCallback) -> None:
        """Unregisters a previously registered callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _validate_mode(self, mode: Union[EnvironmentMode, str]) -> EnvironmentMode:
        """Enforces that mode is a supported EnvironmentMode."""
        if isinstance(mode, EnvironmentMode):
            return mode
        if isinstance(mode, str):
            try:
                return EnvironmentMode.from_string(mode)
            except ValueError as exc:
                raise InvalidModeError(str(exc)) from exc
        raise InvalidModeError(
            f"Expected EnvironmentMode or string, got {type(mode).__name__}"
        )
