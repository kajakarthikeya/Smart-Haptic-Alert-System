"""Centralized Logging Setup for Smart Haptic Alert System."""

import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from typing import Optional

from config import settings


def setup_logger(name: Optional[str] = None, log_file: Optional[str] = "app.log") -> logging.Logger:
    """Configures and returns a structured logger supporting console and rotating file output.

    Args:
        name: Logger identifier name. Defaults to system app name if None.
        log_file: Log file name created under settings.paths.log_dir.

    Returns:
        Configured logging.Logger instance.
    """
    logger_name = name or settings.system.app_name
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers if logger was already setup
    if logger.hasHandlers():
        return logger

    level = getattr(logging, settings.system.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler
    if log_file:
        log_path: Path = settings.paths.log_dir / log_file
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Helper method to fetch a child logger bound to a specific module name.

    Args:
        module_name: Name of the calling module (__name__).

    Returns:
        Child logger instance.
    """
    return logging.getLogger(f"{settings.system.app_name}.{module_name}")
