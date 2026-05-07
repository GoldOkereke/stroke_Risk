"""Database models (SQLAlchemy ORM).

These models store:
- users (login + metadata)
- baseline samples (for personalization / Isolation Forest training)
- test results (fusion history)
- neuro test results (reaction time, cognitive, tap speed, etc.)

The design mirrors the spec image the project is following.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    """User account.

    Fields:
    - id: primary key
    - username: unique login name
    - hashed_password: bcrypt hash (never store plain passwords)
    - created_at: account creation timestamp
    - baseline_status: JSON summary of how many baseline samples exist per stream
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)
    baseline_status: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)

    baseline_samples: Mapped[list["BaselineSample"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    test_results: Mapped[list["TestResult"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    neuro_test_results: Mapped[list["NeuroTestResult"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class BaselineSample(Base):
    """A single baseline feature sample for a user and a stream."""

    __tablename__ = "baseline_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    stream: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    features: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="baseline_samples")


class TestResult(Base):
    """One fusion run stored for history/trends."""

    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    # Store full fusion result payload so UI can replay/inspect it later.
    fusion_result: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)
    alert_tier: Mapped[str] = mapped_column(String(16), index=True, nullable=False)

    user: Mapped[User] = relationship(back_populates="test_results")


class NeuroTestResult(Base):
    """Stores one neuro test outcome (reaction time, cognitive, tap speed, etc.)."""

    __tablename__ = "neuro_test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    test_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    details: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="neuro_test_results")
