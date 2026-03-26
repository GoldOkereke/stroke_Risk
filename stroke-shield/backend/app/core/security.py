"""Security helpers for authentication and API protection.

This module provides three core pieces:
1) Password hashing and verification (bcrypt via Passlib).
2) JWT token creation and validation.
3) A reusable CORS middleware setup function.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

# Hashing context:
# - bcrypt is a strong adaptive hash for passwords.
# - deprecated="auto" lets Passlib migrate old hashes safely if needed.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT defaults:
# - HS256 is a common symmetric signing algorithm.
# - Token expiry can be overridden per token, default is 30 minutes.
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt.

    Store only the returned hash in your database, never the raw password.
    """

    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a stored bcrypt hash."""

    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token.

    Args:
        subject: Usually a user id or username saved into the `sub` claim.
        expires_delta: Optional custom duration. If omitted, uses default.

    Returns:
        Encoded JWT string.
    """

    settings = get_settings()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    payload: dict[str, Any] = {
        "sub": subject,
        "exp": expire,
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Validate and decode a JWT access token.

    Returns decoded payload when valid, or `None` when invalid/expired.
    """

    settings = get_settings()

    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def configure_cors(app: FastAPI, origins: list[str] | None = None) -> None:
    """Attach CORS middleware to a FastAPI app.

    Simple explanation:
    - Browsers block cross-site requests by default.
    - CORS tells the browser which frontend origins are allowed.
    - In development, this usually includes local frontend URLs.

    If `origins` is not provided, values are loaded from settings.CORS_ORIGINS.
    """

    settings = get_settings()
    allowed_origins = origins if origins is not None else settings.CORS_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
