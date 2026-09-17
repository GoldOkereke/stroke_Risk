"""Application configuration.

Step 3.1: Core Configuration

This module centralizes all environment-driven settings so the rest of the
application can import one consistent settings object.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from pydantic import Field, field_validator
except ImportError:  # Fallback for older Pydantic versions.
    from pydantic import Field, validator as field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App/runtime settings.
    app_name: str = Field(default="Stroke Shield", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    # Security settings.
    secret_key: str = Field(
        default="change-me-in-production",
        alias="SECRET_KEY",
        description="Development-only default. Override in production.",
    )
    access_token_expire_minutes: int = Field(
        default=60,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    # Database (optional; future use).
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # CORS settings (accept both BACKEND_CORS_ORIGINS and legacy CORS_ORIGINS).
    # Keep these as raw strings so comma-separated env values don't require JSON.
    backend_cors_origins: str | None = Field(
        default=None,
        alias="BACKEND_CORS_ORIGINS",
    )
    cors_origins: str | None = Field(default=None, alias="CORS_ORIGINS")

    # Paths.
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[4],
        alias="PROJECT_ROOT",
    )
    datasets_root: Path | None = Field(default=None, alias="DATASETS_ROOT")
    upload_dir: Path | None = Field(default=None, alias="UPLOAD_DIR")
    model_dir: Path | None = Field(default=None, alias="MODEL_DIR")

    # Known dataset locations (optional).
    afdb_dir: Path | None = Field(default=None, alias="AFDB_DIR")
    afdb_records_file: Path | None = Field(default=None, alias="AFDB_RECORDS_FILE")
    parkinsons_csv: Path | None = Field(default=None, alias="PARKINSONS_CSV")

    @staticmethod
    def _split_cors(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator(
        "project_root",
        "datasets_root",
        "upload_dir",
        "model_dir",
        "afdb_dir",
        "afdb_records_file",
        "parkinsons_csv",
        mode="before",
    )
    @classmethod
    def parse_path(cls, value: str | Path | None) -> Path | None:
        if value is None or value == "":
            return None
        return Path(value).expanduser()

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "release", "prod", "production"}:
                return False
        return bool(value)

    @field_validator("project_root", mode="after")
    @classmethod
    def resolve_project_root(cls, value: Path) -> Path:
        return value.resolve()

    @field_validator("datasets_root", mode="after")
    @classmethod
    def default_datasets_root(cls, value: Path | None, info) -> Path:
        if value is None:
            return (info.data["project_root"] / "datasets").resolve()
        return cls._resolve_from_project_root(value, info.data["project_root"])

    @field_validator("upload_dir", mode="after")
    @classmethod
    def default_upload_dir(cls, value: Path | None, info) -> Path:
        project_root = info.data["project_root"]
        base_value = value or Path("storage/uploads")
        return cls._resolve_from_project_root(base_value, project_root)

    @field_validator("model_dir", mode="after")
    @classmethod
    def default_model_dir(cls, value: Path | None, info) -> Path:
        project_root = info.data["project_root"]
        base_value = value or Path("storage/models")
        return cls._resolve_from_project_root(base_value, project_root)

    @field_validator("afdb_dir", mode="after")
    @classmethod
    def default_afdb_dir(cls, value: Path | None, info) -> Path:
        datasets_root = info.data["datasets_root"]
        base_value = value or Path("cardiac/mitbih_af/files/afdb")
        return cls._resolve_from_base(base_value, datasets_root)

    @field_validator("afdb_records_file", mode="after")
    @classmethod
    def default_afdb_records_file(cls, value: Path | None, info) -> Path:
        datasets_root = info.data["datasets_root"]
        base_value = value or Path("cardiac/mitbih_af/RECORDS")
        return cls._resolve_from_base(base_value, datasets_root)

    @field_validator("parkinsons_csv", mode="after")
    @classmethod
    def default_parkinsons_csv(cls, value: Path | None, info) -> Path:
        datasets_root = info.data["datasets_root"]
        base_value = value or Path("speech/uci_parkinsons/parkinsons.csv")
        return cls._resolve_from_base(base_value, datasets_root)

    @staticmethod
    def _resolve_from_project_root(path_value: Path, project_root: Path) -> Path:
        if path_value.is_absolute():
            return path_value.resolve()
        return (project_root / path_value).resolve()

    @staticmethod
    def _resolve_from_base(path_value: Path, base_root: Path) -> Path:
        if path_value.is_absolute():
            return path_value.resolve()
        return (base_root / path_value).resolve()

    @property
    def effective_cors_origins(self) -> list[str]:
        if self.backend_cors_origins:
            return self._split_cors(self.backend_cors_origins)
        if self.cors_origins:
            return self._split_cors(self.cors_origins)
        return ["http://localhost:3000", "http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
