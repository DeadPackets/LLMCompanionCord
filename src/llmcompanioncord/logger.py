"""Centralized logging module for the LLM Companion Bot.

Provides console and file logging with configurable levels.
"""

import logging
import sys
from pathlib import Path

_logger_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/bot.log",
    log_to_file: bool = True,
) -> None:
    """Initialize logging with console and optional file handlers.

    Args:
        level: Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
        log_to_file: Whether to enable file logging. If False, only console logging is used.
    """
    global _logger_initialized
    if _logger_initialized:
        return

    # Root logger config - capture all levels, filter at handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - respects configured level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - only if enabled
    if log_to_file:
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    Args:
        name: The module name (typically __name__).

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
