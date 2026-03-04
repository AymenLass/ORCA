"""Logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
) -> None:
    """Configure logging with optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file. If provided, logs are
            written to both stdout and the file.
        fmt: Log message format string.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )

    for noisy in ("transformers", "datasets", "urllib3", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
