"""Application logging setup.

This module gives the project one shared logger configuration so logs are
consistent everywhere (API endpoints, services, background tasks, etc.).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Configure and return the application logger.

    What this function does in simple terms:
    1) Detect whether we are in development or production.
    2) Create a *console handler* so logs appear in the terminal.
    3) Create a *file handler* so logs are saved to disk.
    4) Use different log levels for dev/prod.

    Environment variables used (optional):
    - APP_ENV: set to "production" for stricter logging; anything else is dev.
    - LOG_DIR: folder where the log file will be written (default: "logs").
    - LOG_FILE: log filename (default: "app.log").
    """

    # Read mode from environment. Development is the safe default while building.
    app_env = os.getenv("APP_ENV", "development").lower()
    is_production = app_env == "production"

    # In development we keep verbose logs to help debugging.
    # In production we reduce noise to important operational events.
    logger_level = logging.INFO if is_production else logging.DEBUG
    console_level = logging.WARNING if is_production else logging.DEBUG

    # Use a named logger so the rest of the app can import and reuse it.
    logger = logging.getLogger("stroke_shield")
    logger.setLevel(logger_level)

    # Prevent duplicate handlers if setup_logging() is called more than once.
    if logger.handlers:
        return logger

    # ---------- Shared log format ----------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- Console handler (terminal output) ----------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # ---------- File handler (persistent logs on disk) ----------
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / os.getenv("LOG_FILE", "app.log")

    # RotatingFileHandler keeps logs manageable by rotating after size limit.
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=5,  # Keep last 5 log files
        encoding="utf-8",
    )
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(formatter)

    # Attach both handlers so we get logs in terminal and on disk.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Avoid sending these logs to ancestor loggers (prevents duplicates).
    logger.propagate = False

    logger.debug("Logging initialized in %s mode. Log file: %s", app_env, log_file)
    return logger


# Shared module-level logger instance.
# Import this in other modules: `from app.core.logging import logger`
logger = setup_logging()
