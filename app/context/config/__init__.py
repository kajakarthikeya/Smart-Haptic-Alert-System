"""
Context configuration and mode profiles initialization.

Provides:
1. ContextConfig & get_context_config helper.
2. ModeProfile, EnvironmentMode, SoundPriority, MODE_PROFILES (legacy compatibility).
"""

from typing import Optional
from config import Config, ContextConfig, settings
from app.context.config.mode_profiles import (
    EnvironmentMode,
    SoundPriority,
    ModeProfile,
    MODE_PROFILES,
)

__all__ = [
    "ContextConfig",
    "get_context_config",
    "EnvironmentMode",
    "SoundPriority",
    "ModeProfile",
    "MODE_PROFILES",
]


def get_context_config(override_config: Optional[ContextConfig] = None) -> ContextConfig:
    """Returns provided context configuration or system singleton."""
    return override_config or settings.context
