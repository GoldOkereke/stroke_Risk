"""Neuro tests endpoints.

Prefix: /api/neuro-tests

These endpoints accept small test payloads from the frontend and compute:
- reaction time grade
- cognitive score
- tap speed + consistency proxy
- combined score
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_user_id
from app.db.models import NeuroTestResult, User
from app.db.session import get_db


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


class ReactionTimePayload(BaseModel):
    trials: list[float] = Field(..., description="Reaction times in milliseconds.")


@router.post("/reaction-time")
def send_reaction_time(
    payload: ReactionTimePayload,
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    trials = [float(x) for x in payload.trials if x > 0]
    if not trials:
        return {"average": 0.0, "fastest": 0.0, "slowest": 0.0, "grade": "NORMAL", "score": 0.0}

    fastest = min(trials)
    slowest = max(trials)
    avg = float(np.mean(trials))

    # Simple grading thresholds (ms).
    if avg < 300:
        grade = "NORMAL"
        score = 0.2
    elif avg < 450:
        grade = "SLOW"
        score = 0.55
    else:
        grade = "CRITICAL"
        score = 0.85

    user = _get_or_create_user(db, user_id)
    db.add(
        NeuroTestResult(
            user_id=user.id,
            test_type="reaction_time",
            score=float(score),
            details={"trials": trials, "average": avg, "fastest": fastest, "slowest": slowest, "grade": grade},
        )
    )
    db.commit()

    return {"average": avg, "fastest": fastest, "slowest": slowest, "grade": grade, "score": score}


class CognitivePayload(BaseModel):
    # Frontend may send different key casing; we accept common variants.
    max_span: int | None = Field(None, description="Maximum digit span achieved.")
    maxSpan: int | None = Field(None, description="Maximum digit span achieved (camelCase).")
    accuracy: float | None = Field(None, description="Accuracy 0-1.")
    response_time_ms: float | None = Field(None, description="Average response time in ms.")
    responseTimeMs: float | None = Field(None, description="Average response time in ms (camelCase).")


@router.post("/cognitive")
def send_cognitive_test(
    payload: CognitivePayload,
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    max_span_val = payload.max_span if payload.max_span is not None else payload.maxSpan
    accuracy_val = payload.accuracy if payload.accuracy is not None else 0.0
    response_time_val = (
        payload.response_time_ms if payload.response_time_ms is not None else payload.responseTimeMs
    )

    max_span = int(max_span_val or 0)
    accuracy = float(np.clip(float(accuracy_val or 0.0), 0.0, 1.0))
    response_time = float(max(float(response_time_val or 0.0), 0.0))

    # Score: higher span + higher accuracy + lower response time.
    span_score = min(1.0, max_span / 9.0)
    time_penalty = min(1.0, response_time / 1500.0)
    score = float(np.clip(0.55 * span_score + 0.35 * accuracy + 0.10 * (1.0 - time_penalty), 0.0, 1.0))

    user = _get_or_create_user(db, user_id)
    db.add(
        NeuroTestResult(
            user_id=user.id,
            test_type="cognitive",
            score=score,
            details={"max_span": max_span, "accuracy": accuracy, "response_time_ms": response_time},
        )
    )
    db.commit()

    return {"score": score}


class TapSpeedPayload(BaseModel):
    taps: list[float] = Field(..., description="Tap timestamps (seconds) or inter-tap intervals (seconds).")


@router.post("/tap-speed")
def send_tap_speed(
    payload: TapSpeedPayload,
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    taps = [float(x) for x in payload.taps if x >= 0]
    if len(taps) < 3:
        return {"tapsPerSecond": 0.0, "cv": 0.0, "score": 0.0}

    # If the payload is timestamps, convert to intervals.
    diffs = np.diff(taps)
    if np.all(diffs > 0):
        intervals = diffs
    else:
        intervals = np.asarray(taps, dtype=float)

    mean_it = float(np.mean(intervals)) if intervals.size else 0.0
    taps_per_sec = float(1.0 / mean_it) if mean_it > 1e-6 else 0.0
    cv = float(np.std(intervals) / (mean_it + 1e-9)) if mean_it > 0 else 0.0

    # Higher CV suggests inconsistency (tremor proxy). Map to 0-1 risk score.
    inconsistency = float(np.clip(cv / 0.6, 0.0, 1.0))
    score = float(np.clip(0.35 * inconsistency + 0.65 * (1.0 - min(1.0, taps_per_sec / 8.0)), 0.0, 1.0))

    user = _get_or_create_user(db, user_id)
    db.add(
        NeuroTestResult(
            user_id=user.id,
            test_type="tap_speed",
            score=score,
            details={"tapsPerSecond": taps_per_sec, "cv": cv},
        )
    )
    db.commit()

    return {"tapsPerSecond": taps_per_sec, "cv": cv, "score": score}


@router.post("/finger-tracking")
def send_finger_tracking(
    payload: dict[str, float],
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Placeholder finger tracking endpoint.

    The frontend sends an object of numeric metrics. We turn it into a simple score:
    - larger error/variance metrics increase risk
    """
    values = [float(v) for v in payload.values()] if payload else []
    if not values:
        score = 0.0
    else:
        # Risk proxy: mean of clamped values.
        score = float(np.clip(np.mean([min(1.0, max(0.0, v)) for v in values]), 0.0, 1.0))

    user = _get_or_create_user(db, user_id)
    db.add(
        NeuroTestResult(
            user_id=user.id,
            test_type="finger_tracking",
            score=score,
            details={"metrics": payload},
        )
    )
    db.commit()
    return {"score": score}


@router.post("/combined")
def get_combined_score(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Aggregate last known scores into one number."""
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return {"score": 0.0}

    rows = (
        db.query(NeuroTestResult)
        .filter(NeuroTestResult.user_id == user.id)
        .order_by(NeuroTestResult.timestamp.desc())
        .all()
    )
    latest: dict[str, float] = {}
    for row in rows:
        if row.test_type not in latest:
            latest[row.test_type] = float(row.score)

    # Weighted average; reaction time matters slightly more.
    rt = latest.get("reaction_time", 0.0)
    cog = latest.get("cognitive", 0.0)
    tap = latest.get("tap_speed", 0.0)
    score = float(np.clip(0.45 * rt + 0.30 * cog + 0.25 * tap, 0.0, 1.0))
    return {"score": score, "streams": latest}


@router.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok"}
