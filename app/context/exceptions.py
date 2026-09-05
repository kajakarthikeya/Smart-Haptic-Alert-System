"""
Domain Exceptions for Context-Aware Decision Module.

Hierarchy:
ContextError
├── InvalidModeError
├── UnknownSoundError
├── InvalidConfidenceError
├── PriorityRuleError
└── ConfigurationError
"""


class ContextError(Exception):
    """Base exception for all context and decision engine errors."""


class InvalidModeError(ContextError):
    """Raised when an unsupported or invalid operating mode is provided."""


class UnknownSoundError(ContextError):
    """Raised when a sound label is not recognized among target classes."""


class InvalidConfidenceError(ContextError):
    """Raised when confidence is NaN, infinite, or outside [0.0, 1.0]."""


class PriorityRuleError(ContextError):
    """Raised when a required priority rule is missing from the configuration matrix."""


class ConfigurationError(ContextError):
    """Raised when priority matrix or decision configuration is malformed."""
