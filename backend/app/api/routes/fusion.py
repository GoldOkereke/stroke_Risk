"""
app/api/routes/fusion.py
=========================
FastAPI router for FR5 Fusion Engine endpoints.

Endpoints:
  POST /api/fusion/score        — fuse all available stream inputs → risk score
  POST /api/fusion/cardiac-only — score from cardiac stream only (ECG/PPG)
  GET  /api/fusion/history      — retrieve temporal score history
  POST /api/fusion/reset        — reset score history (new session)
  GET  /api/fusion/status       — fusion engine health check

The fusion endpoint is the single call the React frontend makes to get:
  - risk_score      (0–100)
  - alert_tier      (NORMAL / ADVISORY / CRITICAL)
  - confidence      (0–1)
  - contributing_factors
  - recommendation
  - uncertainty_flag + failure_case
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.fusion_engine import FusionEngine, FusionResult
from app.services.isolation_forest import AnomalyResult, IFStream, IsolationForestService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared instances ───────────────────────────────────────────────────────

_fusion_engine = FusionEngine()
_if_service    = IsolationForestService(user_id="default")


# ── Request schemas ────────────────────────────────────────────────────────

class AnomalyResultInput(BaseModel):
    """Serialised AnomalyResult from a stream — sent by frontend or other services."""
    stream: str
    anomaly_score: float      = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    confidence: float         = Field(..., ge=0.0, le=1.0)
    raw_score: float
    n_samples_in_baseline: int
    baseline_ready: bool


class NeuroTestScores(BaseModel):
    """Active neurological test results from FR4."""
    reaction_time: Optional[float]  = Field(None, ge=0.0, le=1.0,
        description="Normalised reaction time score — higher = slower = more anomalous")
    tracking: Optional[float]       = Field(None, ge=0.0, le=1.0,
        description="Finger tracking accuracy — higher = worse")
    speech_prompt: Optional[float]  = Field(None, ge=0.0, le=1.0,
        description="Speech prompt slurring score — higher = more slurred")


class FusionRequest(BaseModel):
    """
    Full multimodal fusion request.
    All stream inputs are optional — missing streams are handled via
    weight redistribution in the fusion engine.
    """
    # FR2 — CNN output (required — cardiac signal is the primary stream)
    afib_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="AFib probability from 1D CNN (FR2). Required."
    )

    # FR3 — Isolation Forest stream outputs (all optional)
    cardiac_anomaly:    Optional[AnomalyResultInput] = Field(
        None, description="IF cardiac stream result"
    )
    facial_anomaly:     Optional[AnomalyResultInput] = Field(
        None, description="IF facial stream result"
    )
    dysarthria_anomaly: Optional[AnomalyResultInput] = Field(
        None, description="IF dysarthria stream result (TORGO)"
    )
    parkinsons_anomaly: Optional[AnomalyResultInput] = Field(
        None, description="IF Parkinson's stream result (UCI)"
    )

    # FR4 — Active neuro tests (optional)
    neuro_test_scores: Optional[NeuroTestScores] = Field(
        None, description="Active neurological test scores from FR4"
    )


class CardiacOnlyRequest(BaseModel):
    """Simplified request for cardiac-only scoring (no face/voice available)."""
    afib_probability: float = Field(..., ge=0.0, le=1.0)
    cardiac_anomaly_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    cardiac_baseline_ready: bool = Field(default=False)


# ── Response schema ────────────────────────────────────────────────────────

class FusionResponse(BaseModel):
    """Fusion result returned to the React frontend."""
    risk_score: float
    alert_tier: str
    confidence: float
    contributing_factors: list[str]
    corroboration_count: int
    corroboration_multiplier: float
    trend_direction: str
    trend_bonus: float
    uncertainty_flag: bool
    failure_case: Optional[str]
    recommendation: str
    afib_probability: float
    streams_available: list[str]
    streams_missing: list[str]
    timestamp: str
    stream_scores: list[dict]


# ── Helpers ────────────────────────────────────────────────────────────────

def _deserialise_anomaly(inp: Optional[AnomalyResultInput]) -> Optional[AnomalyResult]:
    """Convert request input back into an AnomalyResult for the fusion engine."""
    if inp is None:
        return None
    try:
        stream = IFStream(inp.stream)
    except ValueError:
        logger.warning("Unknown IF stream '%s' — skipping.", inp.stream)
        return None

    return AnomalyResult(
        stream                 = stream,
        anomaly_score          = inp.anomaly_score,
        is_anomaly             = inp.is_anomaly,
        confidence             = inp.confidence,
        raw_score              = inp.raw_score,
        n_samples_in_baseline  = inp.n_samples_in_baseline,
        baseline_ready         = inp.baseline_ready,
    )


def _fusion_result_to_response(result: FusionResult) -> FusionResponse:
    """Map FusionResult dataclass → FusionResponse Pydantic model."""
    return FusionResponse(
        risk_score              = result.risk_score,
        alert_tier              = result.alert_tier.value,
        confidence              = result.confidence,
        contributing_factors    = result.contributing_factors,
        corroboration_count     = result.corroboration_count,
        corroboration_multiplier= result.corroboration_multiplier,
        trend_direction         = result.trend_direction,
        trend_bonus             = result.trend_bonus,
        uncertainty_flag        = result.uncertainty_flag,
        failure_case            = result.failure_case,
        recommendation          = result.recommendation,
        afib_probability        = result.afib_probability,
        streams_available       = result.streams_available,
        streams_missing         = result.streams_missing,
        timestamp               = result.timestamp.isoformat(),
        stream_scores           = [
            {
                "name":      s.name,
                "raw_score": round(s.raw_score, 4),
                "weight":    round(s.weight, 4),
                "available": s.available,
                "degraded":  s.degraded,
            }
            for s in result.stream_scores
        ],
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/status")
async def fusion_status():
    """Health check for the fusion engine."""
    return {
        "status":        "ok",
        "streams":       [s.value for s in IFStream],
        "alert_tiers":   ["NORMAL", "ADVISORY", "CRITICAL"],
        "thresholds":    {
            "normal":    "< 35",
            "advisory":  "35 – 74",
            "critical":  ">= 75",
        },
        "score_history_length": len(_fusion_engine.get_history()),
    }


@router.post("/score", response_model=FusionResponse)
async def fuse_all_streams(request: FusionRequest):
    """
    Main fusion endpoint — combines all available stream inputs.

    Called by the React frontend after collecting:
      - CNN AFib probability    (from /api/signal/*)
      - IF stream results       (from face + voice analyzers)
      - Neuro test scores       (from /api/neurological/*)

    Returns a full risk assessment with alert tier and recommendation.
    """
    logger.info(
        "Fusion request | afib_prob=%.3f | cardiac=%s | facial=%s | "
        "dysarthria=%s | parkinsons=%s | neuro=%s",
        request.afib_probability,
        request.cardiac_anomaly    is not None,
        request.facial_anomaly     is not None,
        request.dysarthria_anomaly is not None,
        request.parkinsons_anomaly is not None,
        request.neuro_test_scores  is not None,
    )

    try:
        result = _fusion_engine.fuse(
            afib_probability   = request.afib_probability,
            cardiac_anomaly    = _deserialise_anomaly(request.cardiac_anomaly),
            facial_anomaly     = _deserialise_anomaly(request.facial_anomaly),
            dysarthria_anomaly = _deserialise_anomaly(request.dysarthria_anomaly),
            parkinsons_anomaly = _deserialise_anomaly(request.parkinsons_anomaly),
            neuro_test_scores  = (
                request.neuro_test_scores.dict(exclude_none=True)
                if request.neuro_test_scores else None
            ),
        )
    except Exception as e:
        logger.exception("Fusion engine error: %s", e)
        raise HTTPException(status_code=500, detail=f"Fusion engine error: {e}")

    return _fusion_result_to_response(result)


@router.post("/cardiac-only", response_model=FusionResponse)
async def fuse_cardiac_only(request: CardiacOnlyRequest):
    """
    Simplified fusion endpoint for cardiac-only mode.
    Used when webcam/mic are unavailable or user declines.
    Still runs full fusion engine — face/voice streams just absent.
    """
    logger.info(
        "Cardiac-only fusion | afib_prob=%.3f | cardiac_if_ready=%s",
        request.afib_probability, request.cardiac_baseline_ready,
    )

    cardiac_anomaly = None
    if request.cardiac_anomaly_score is not None:
        cardiac_anomaly = AnomalyResult(
            stream                = IFStream.CARDIAC,
            anomaly_score         = request.cardiac_anomaly_score,
            is_anomaly            = request.cardiac_anomaly_score >= 0.5,
            confidence            = abs(request.cardiac_anomaly_score - 0.5) * 2,
            raw_score             = request.cardiac_anomaly_score,
            n_samples_in_baseline = 0,
            baseline_ready        = request.cardiac_baseline_ready,
        )

    try:
        result = _fusion_engine.fuse(
            afib_probability = request.afib_probability,
            cardiac_anomaly  = cardiac_anomaly,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion engine error: {e}")

    return _fusion_result_to_response(result)


@router.get("/history")
async def get_score_history():
    """
    Return the temporal risk score history.
    Used by the React frontend to render the trend chart.
    """
    history = _fusion_engine.get_history()
    if not history:
        return {"history": [], "trend": "no_data", "count": 0}

    import numpy as np
    scores = list(history)
    slope  = 0.0
    if len(scores) >= 3:
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])

    trend = "stable"
    if slope > 1.5:
        trend = "worsening"
    elif slope < -1.5:
        trend = "improving"

    return {
        "history":   [round(s, 2) for s in scores],
        "count":     len(scores),
        "trend":     trend,
        "slope":     round(slope, 4),
        "latest":    round(scores[-1], 2),
        "max":       round(max(scores), 2),
        "min":       round(min(scores), 2),
    }


@router.post("/reset")
async def reset_session():
    """
    Reset the fusion engine's score history.
    Call at the start of each new assessment session.
    """
    _fusion_engine.reset_history()
    logger.info("Fusion session reset.")
    return {"status": "ok", "message": "Score history cleared. New session started."}