"""Application configuration.

Step 3.1: Core Configuration

This module centralizes all environment-driven settings so the rest of the
application can import one consistent settings object.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables.

    Why use this class?
    - It validates values as soon as the app starts.
    - It gives auto-complete and type hints in editors.
    - It keeps all config in one easy place.

    Environment variables loaded from `.env` (and normal OS env):
    - DATABASE_URL
    - SECRET_KEY
    - CORS_ORIGINS
    - MODEL_PATH
    """

    # Tell Pydantic Settings where to read variables from.
    # `extra="ignore"` lets unrelated `.env` keys exist without breaking startup.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database connection string.
    # Example: postgresql+psycopg://user:password@localhost:5432/stroke_shield
    DATABASE_URL: str = Field(..., description="Database connection URL")

    # Secret used for signing tokens/cookies or other security-sensitive operations.
    # Keep this long and random in production.
    SECRET_KEY: str = Field(..., description="Secret key for cryptographic operations")

    # Allowed frontend origins for CORS.
    # Comma-separated values in `.env` are parsed into a list automatically.
    # Example: CORS_ORIGINS=http://localhost:3000,https://app.example.com
    CORS_ORIGINS: list[str] = Field(default_factory=list, description="Allowed CORS origins")

    # File path to a trained ML model or model artifact directory.
    MODEL_PATH: str = Field(..., description="Filesystem path to model assets")


@lru_cache
def get_settings() -> Settings:
    """Return a singleton Settings instance.

    `@lru_cache` makes this function run once per process, so every import
    receives the same validated config object. This avoids repeated `.env` reads
    and keeps behavior consistent.
    """

    return Settings()
