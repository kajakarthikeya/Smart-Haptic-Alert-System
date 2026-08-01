"""Common Reusable Utility Functions."""

from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Union


def format_timestamp(dt: Union[datetime, None] = None) -> str:
    """Returns an ISO 8601 formatted UTC timestamp string.

    Args:
        dt: Optional datetime object. Defaults to current UTC time.

    Returns:
        Formatted UTC timestamp string.
    """
    target_dt = dt or datetime.now(timezone.utc)
    return target_dt.isoformat()


def generate_alert_id() -> str:
    """Generates a unique alert correlation identifier.

    Returns:
        Unique UUID4 string prefixed with 'alert-'.
    """
    return f"alert-{uuid.uuid4().hex[:8]}"


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Checks if a given path exists and represents a supported audio format.

    Args:
        file_path: Absolute or relative file path to check.

    Returns:
        True if file exists and has .wav or .flac extension, False otherwise.
    """
    path = Path(file_path)
    supported_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    return path.is_file() and path.suffix.lower() in supported_extensions
