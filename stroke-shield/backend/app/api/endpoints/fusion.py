"""Fusion endpoints.

Prefix: /api/fusion

These endpoints provide:
- fusion score calculation
- cardiac-only shortcut
- history management for trends
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_user_id
from app.db.models import TestResult, User
from app.db.session import get_db
from app.services.fusion_engine import FusionEngine


router = APIRouter()


def _get_or_create_user(db, user_id: str) -> User:
    user = db.query(User).filter(User.username == user_id).first()
    if user:
        return user
    user = User(username=user_id, hashed_password="")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class FusionScoreRequest(BaseModel):
    """Request shape accepted by /api/fusion/score.

    We accept two shapes for frontend compatibility:
    1) New shape: { "scores": { "afib_cnn": 0.7, ... } }
    2) Legacy shape: { "cardiacScore": 0.5, "faceScore": 0.2, ... }
    """

    scores: dict[str, float] = Field(default_factory=dict)
    cardiacScore: float | None = None
    faceScore: float | None = None
    voiceScore: float | None = None
    dysarthriaScore: float | None = None
    parkinsonsScore: float | None = None


def _to_engine_scores(payload: FusionScoreRequest) -> dict[str, float]:
    """Map request fields to FusionEngine expected stream keys."""
    mapped: dict[str, float] = dict(payload.scores or {})
    # Map legacy keys to engine keys.
    if payload.cardiacScore is not None:
        mapped.setdefault("cardiac_if", float(payload.cardiacScore))
    if payload.faceScore is not None:
        mapped.setdefault("facial_if", float(payload.faceScore))
    if payload.voiceScore is not None:
        # voiceScore is an overall stream score; treat it as dysarthria as a proxy if nothing else present.
        mapped.setdefault("dysarthria_if", float(payload.voiceScore))
    if payload.dysarthriaScore is not None:
        mapped.setdefault("dysarthria_if", float(payload.dysarthriaScore))
    if payload.parkinsonsScore is not None:
        mapped.setdefault("parkinsons_if", float(payload.parkinsonsScore))
    return mapped


@router.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/score")
def get_fusion_score(
    payload: FusionScoreRequest,
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    engine = FusionEngine()

    # Load last 10 risk scores for trend.
    user = _get_or_create_user(db, user_id)
    rows = (
        db.query(TestResult)
        .filter(TestResult.user_id == user.id)
        .order_by(TestResult.timestamp.desc())
        .limit(10)
        .all()
    )
    history = [float(r.fusion_result.get("risk_score", 0.0)) for r in reversed(rows)]

    engine_scores = _to_engine_scores(payload)
    result = engine.fuse(engine_scores, history=history)

    # Persist to history.
    db.add(
        TestResult(
            user_id=user.id,
            fusion_result=result.__dict__,
            alert_tier=result.alert_tier,
        )
    )
    db.commit()

    return result.__dict__


class CardiacOnlyRequest(BaseModel):
    # Accept both snake_case and camelCase used by the frontend.
    afib_prob: float | None = Field(None, description="AFib probability 0-1 from CNN (snake_case).")
    cardiac_if: float | None = Field(None, description="Cardiac IF anomaly score 0-1 (snake_case).")
    afibProbability: float | None = Field(None, description="AFib probability 0-1 from CNN (camelCase).")
    cardiacIf: float | None = Field(None, description="Cardiac IF anomaly score 0-1 (camelCase).")


@router.post("/cardiac-only")
def get_cardiac_only(
    payload: CardiacOnlyRequest,
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    # Map to fusion keys expected by FusionEngine.
    engine = FusionEngine()
    afib_prob = payload.afib_prob if payload.afib_prob is not None else payload.afibProbability
    cardiac_if = payload.cardiac_if if payload.cardiac_if is not None else payload.cardiacIf
    scores = {
        "afib_cnn": float(afib_prob or 0.0),
        "cardiac_if": float(cardiac_if or 0.0),
    }
    result = engine.fuse(scores, history=None)

    user = _get_or_create_user(db, user_id)
    db.add(
        TestResult(
            user_id=user.id,
            fusion_result=result.__dict__,
            alert_tier=result.alert_tier,
        )
    )
    db.commit()
    return result.__dict__


@router.get("/history")
def get_history(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> list[dict[str, Any]]:
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return []
    rows = (
        db.query(TestResult)
        .filter(TestResult.user_id == user.id)
        .order_by(TestResult.timestamp.asc())
        .limit(50)
        .all()
    )
    return [row.fusion_result for row in rows]


@router.post("/reset")
def reset_history(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    user = db.query(User).filter(User.username == user_id).first()
    if user:
        db.query(TestResult).filter(TestResult.user_id == user.id).delete()
        db.commit()
    return {"cleared": True}
