"""
utils/helper.py
================
Shared helper utilities: logging configuration, byte-size formatting,
and small pure functions reused across the application.

Author: Senior AI Engineering Team
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    Ensures handlers are not duplicated when Streamlit re-runs the script
    (which happens on every UI interaction).

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def format_file_size(size_bytes: int) -> str:
    """
    Convert a byte count into a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        str: Human readable size, e.g. "1.4 MB".
    """
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def truncate_text(text: str, max_len: int = 160) -> str:
    """
    Truncate text to a maximum length, appending an ellipsis if cut.

    Args:
        text: Input text.
        max_len: Maximum allowed length before truncation.

    Returns:
        str: Possibly truncated text.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Perform a lightweight sanity check on a Google API key string.

    This is NOT a call to the API - just a structural check to fail fast
    with a friendly error before attempting network calls.

    Args:
        api_key: The candidate API key.

    Returns:
        bool: True if the key looks structurally plausible.
    """
    if not api_key:
        return False
    if len(api_key.strip()) < 15:
        return False
    return True
