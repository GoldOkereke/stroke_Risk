"""Application configuration.

Step 3.1: Core Configuration

This module centralizes all environment-driven settings so the rest of the
application can import one consistent settings object.
"""
#to allow forward references in type hints, which is useful for self-referential types or when a type is defined later in the code.
from __future__ import annotations

#memoize the get_settings function to cache the Settings instance, ensuring that the settings are only loaded once and reused throughout the application.
from functools import lru_cache
from pathlib import Path


try:
    from pydantic import Field, field_validator
except ImportError:  # Fallback for older Pydantic versions.
    from pydantic import Field, validator as field_validator
# Pydantic is a library for data validation and settings management using Python type annotations. The code attempts to import `Field` and `field_validator` from Pydantic. If the import fails (likely due to an older version of Pydantic), it falls back to importing `validator` as `field_validator`. This ensures compatibility with different versions of Pydantic.
from pydantic_settings import BaseSettings, SettingsConfigDict
# The `BaseSettings` class from `pydantic_settings` is used to define application settings that can be loaded from environment variables. `SettingsConfigDict` is a configuration dictionary for customizing the behavior of the settings class.

class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Load settings from a `.env` file located two directories above the current file's location. This allows for centralized configuration management, making it easier to change settings without modifying the code.
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        # Ignore extra fields not defined in the Settings class, preventing errors if additional environment variables are present.
        extra="ignore",
        case_sensitive=False,
    )

    # App/runtime settings.
    app_name: str = Field(default="Stroke Shield", alias="APP_NAME")
    # The `app_name` field defines the name of the application, with a default value of "Stroke Shield". It can be overridden by setting the `APP_NAME` environment variable.
    app_env: str = Field(default="development", alias="APP_ENV")
    # The `app_env` field specifies the environment in which the application is running (e.g., development, production), with a default value of "development". It can be overridden by setting the `APP_ENV` environment variable.
    debug: bool = Field(default=True, alias="DEBUG")
    # The `debug` field controls whether the application is running in debug mode, with a default value of True. It can be overridden by setting the `DEBUG` environment variable.
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    # The `api_v1_prefix` field defines the prefix for the API v1 endpoints, with a default value of "/api/v1". It can be overridden by setting the `API_V1_PREFIX` environment variable.

    # Security settings.
    secret_key: str = Field(
        # sign jwt tokens
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
    #users might store datasets in different locations, so these fields allow specifying paths to specific datasets. If not provided, default paths relative to the datasets_root will be used.
    afdb_dir: Path | None = Field(default=None, alias="AFDB_DIR")
    afdb_records_file: Path | None = Field(default=None, alias="AFDB_RECORDS_FILE")
    parkinsons_csv: Path | None = Field(default=None, alias="PARKINSONS_CSV")
    # The `neuro_face_samples_dir` and `neuro_voice_samples_dir` fields specify directories for neurological face and voice samples, respectively. Default paths are provided relative to the datasets_root if not explicitly set.
    neuro_face_samples_dir: Path | None = Field(default=None, alias="NEURO_FACE_SAMPLES_DIR")
    neuro_voice_samples_dir: Path | None = Field(default=None, alias="NEURO_VOICE_SAMPLES_DIR")

## Private methods ##
    @staticmethod
    # Split a comma-separated string into a list of strings, trimming whitespace and ignoring empty items. This is useful for parsing CORS origins from environment variables.
    def _split_cors(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

## Field validators ##
    @field_validator(
        "project_root",
        "datasets_root",
        "upload_dir",
        "model_dir",
        "afdb_dir",
        "afdb_records_file",
        "parkinsons_csv",
        "neuro_face_samples_dir",
        "neuro_voice_samples_dir",
        mode="before",
    )#whats a field validator? A field validator is a method that is used to validate and transform the value of a specific field in a Pydantic model before it is assigned to the field. It allows you to enforce constraints, perform type conversions, or apply custom logic to ensure that the data meets certain criteria before being stored in the model.

    @classmethod
    # The `parse_path` method is a class method that takes a value (which can be a string, Path object, or None) and returns a Path object or None. If the value is None or an empty string, it returns None. Otherwise, it converts the value to a Path object and expands any user directory references (e.g., `~`). used for? The `parse_path` method is used to normalize and validate path-related fields in the settings. It ensures that any string representation of a path is converted into a proper Path object, making it easier to work with file system paths throughout the application. This method also handles cases where the input might be None or an empty string, returning None in those cases to avoid errors when accessing file paths.
    def parse_path(cls, value: str | Path | None) -> Path | None:
        if value is None or value == "":
            return None
        return Path(value).expanduser()

    @field_validator("debug", mode="before")
    # The `parse_debug` method is a class method that takes a value (which can be a string or boolean) and returns a boolean. It normalizes the input value to determine if it represents a truthy or falsy value for the debug setting. This allows for flexible input formats (e.g., "true", "yes", "1" for True and "false", "no", "0" for False). used for? The `parse_debug` method is used to interpret various representations of boolean values for the debug setting. It allows users to specify the debug mode in different formats (strings or booleans) and ensures that the application correctly interprets these values as either True or False. This flexibility is particularly useful when reading environment variables, which are typically strings.
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

    @field_validator("neuro_face_samples_dir", mode="after")
    @classmethod
    def default_neuro_face_samples_dir(cls, value: Path | None, info) -> Path:
        datasets_root = info.data["datasets_root"]
        base_value = value or Path("neurological/face_samples")
        return cls._resolve_from_base(base_value, datasets_root)

    @field_validator("neuro_voice_samples_dir", mode="after")
    @classmethod
    def default_neuro_voice_samples_dir(cls, value: Path | None, info) -> Path:
        datasets_root = info.data["datasets_root"]
        base_value = value or Path("neurological/voice_samples")
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
        # Development defaults: Vite (:5173), CRA (:3000), and common alt ports.
        # Override in production via BACKEND_CORS_ORIGINS env var, e.g.:
        #   BACKEND_CORS_ORIGINS=https://yourdomain.com
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:4173",  # Vite preview
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()