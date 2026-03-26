"""Application configuration helpers.

This module centralizes environment-driven settings so the rest of the backend
can import a single `settings` object.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _get_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean value from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Container for runtime settings."""

    app_name: str = os.getenv("APP_NAME", "Stroke Shield API")
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = _get_bool("DEBUG", default=True)
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
