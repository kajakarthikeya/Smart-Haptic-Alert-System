"""
Context-Aware Decision Subsystem for the Smart Haptic Alert System.

Exports:
- Enums: EnvironmentMode, PriorityLevel, SoundPriority
- Models: SoundPrediction, DecisionResult, AlertPolicy
- Engines: PriorityEngine, ModeManager, ContextDecisionEngine, ContextManager
- Config & Rules: ContextConfig, DEFAULT_PRIORITY_MATRIX, TARGET_SOUNDS
- Exceptions: ContextError, InvalidModeError, UnknownSoundError,
              InvalidConfidenceError, PriorityRuleError, ConfigurationError
"""

from app.context.enums import EnvironmentMode, PriorityLevel, SoundPriority
from app.context.exceptions import (
    ContextError,
    InvalidModeError,
    UnknownSoundError,
    InvalidConfidenceError,
    PriorityRuleError,
    ConfigurationError,
)
from app.context.models import AlertPolicy, DecisionResult, SoundPrediction
from app.context.rules import DEFAULT_PRIORITY_MATRIX, TARGET_SOUNDS
from app.context.priority_engine import PriorityEngine
from app.context.mode_manager import ModeManager
from app.context.decision_engine import ContextDecisionEngine
from app.context.context_manager import ContextManager
from app.context.config import ContextConfig, get_context_config

__all__ = [
    # Enums
    "EnvironmentMode",
    "PriorityLevel",
    "SoundPriority",
    # Models
    "AlertPolicy",
    "DecisionResult",
    "SoundPrediction",
    # Engines
    "PriorityEngine",
    "ModeManager",
    "ContextDecisionEngine",
    "ContextManager",
    # Rules & Config
    "ContextConfig",
    "DEFAULT_PRIORITY_MATRIX",
    "TARGET_SOUNDS",
    "get_context_config",
    # Exceptions
    "ContextError",
    "InvalidModeError",
    "UnknownSoundError",
    "InvalidConfidenceError",
    "PriorityRuleError",
    "ConfigurationError",
]
