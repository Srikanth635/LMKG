"""
Logging utilities for SOMA Object Analysis System
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config import config


def setup_logging(
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
        include_timestamp: bool = True,
        include_module: bool = True
) -> logging.Logger:
    """
    Setup logging for the SOMA Object Analysis System

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        include_timestamp: Whether to include timestamps in log messages
        include_module: Whether to include module names in log messages

    Returns:
        Configured logger instance
    """

    # Use config defaults if not specified
    log_level = log_level or config.LOG_LEVEL

    # Create main logger
    logger = logging.getLogger("soma")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = create_formatter(include_timestamp, include_module)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    logger.info(f"SOMA logging initialized (level: {log_level})")
    return logger


def create_formatter(include_timestamp: bool = True, include_module: bool = True) -> logging.Formatter:
    """Create a log formatter with customizable components"""

    format_parts = []

    if include_timestamp:
        format_parts.append("%(asctime)s")

    format_parts.append("%(levelname)-8s")

    if include_module:
        format_parts.append("%(name)s")

    format_parts.append("%(message)s")

    format_string = " - ".join(format_parts)

    return logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name under the soma hierarchy"""
    return logging.getLogger(f"soma.{name}")