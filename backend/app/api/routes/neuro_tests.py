"""
app/api/routes/neuro_tests.py
===============================
FR4 — Active Neurological Tests

Real-time anomaly scoring for active neurological tests.
These tests are administered interactively via the React frontend
and scored server-side against the user's personal IF baseline.

Tests:
  1. Reaction Time Test     — measures simple reaction time (ms)
  2. Finger Tracking Test   — measures tremor/accuracy in cursor tracking
  3. Cognitive Test         — digit span memory + pattern recognition
  4. Grip Strength Proxy    — tap speed + consistency (mobile only)
  5. Combined Score         — aggregates all active tests → neuro_test_score

All scores are:
  - Normalised to [0, 1] (0 = normal, 1 = maximally anomalous)
  - Compared against personal baseline (IF) if available
  - Fed directly into the Fusion Engine as neuro_test_scores dict
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.isolation_forest import IFStream, IsolationForestService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared IF service ──────────────────────────────────────────────────────
_if_service = IsolationForestService(user_id="default")

# ── Clinical reference values ──────────────────────────────────────────────

# Reaction time (ms) — population norms
REACTION_NORMAL_MS    = 250    # mean simple reaction time
REACTION_SLOW_MS      = 400    # above = abnormal
REACTION_VERY_SLOW_MS = 600    # above = critical

# Finger tracking — normalised error [0,1]
TRACKING_NORMAL_ERROR  = 0.15
TRACKING_HIGH_ERROR    = 0.35

# Tap speed (taps/sec)
TAP_NORMAL_SPEED = 5.0
TAP_SLOW_SPEED   = 3.0

# Cognitive — digit span
DIGIT_SPAN_NORMAL = 7    # Miller's law: 7 ± 2
DIGIT_SPAN_LOW    = 4


# ── Request schemas ────────────────────────────────────────────────────────

class ReactionTimeRequest(BaseModel):
    """
    Results from a simple reaction time test.
    Frontend shows a stimulus (colour change / sound) and records
    how long the user takes to tap/click.
    """
    trials: list[float] = Field(
        ..., min_items=3, max_items=20,
        description="Reaction times in milliseconds for each trial",
    )
    dominant_hand: str = Field(
        default="right",
        description="right | left | unknown",
    )
    test_type: str = Field(
        default="visual",
        description="visual | auditory | combined",
    )


class TrackingPoint(BaseModel):
    """One data point from the finger/cursor tracking test."""
    timestamp_ms: float   # time since test start
    target_x: float       # target position [0,1]
    target_y: float       # target position [0,1]
    actual_x: float       # user's actual position [0,1]
    actual_y: float       # user's actual position [0,1]


class FingerTrackingRequest(BaseModel):
    """
    Results from a finger/cursor tracking test.
    Frontend animates a moving target and records user's tracking accuracy.
    Used to detect motor tremor and coordination deficits.
    """
    tracking_data: list[TrackingPoint] = Field(
        ..., min_items=10,
        description="Time-series of target vs actual positions",
    )
    duration_seconds: float = Field(..., gt=0)
    test_pattern: str = Field(
        default="circular",
        description="circular | figure8 | random",
    )


class CognitiveTestRequest(BaseModel):
    """
    Results from a cognitive screening test.
    Covers digit span (working memory) and pattern recognition.
    """
    digit_span_score: int = Field(
        ..., ge=0, le=15,
        description="Number of digits correctly recalled (forward)",
    )
    digit_span_reverse: Optional[int] = Field(
        None, ge=0, le=15,
        description="Number of digits recalled in reverse",
    )
    pattern_accuracy: float = Field(
        ..., ge=0.0, le=1.0,
        description="Pattern recognition accuracy [0,1]",
    )
    response_time_ms: float = Field(
        ..., gt=0,
        description="Mean response time for cognitive tasks (ms)",
    )


class TapTestRequest(BaseModel):
    """
    Results from a rapid finger-tapping test.
    User taps as fast as possible for 10 seconds.
    Detects bradykinesia (slowness) and rhythm irregularity.
    """
    tap_timestamps_ms: list[float] = Field(
        ..., min_items=5,
        description="Timestamp of each tap in milliseconds",
    )
    duration_seconds: float = Field(default=10.0, gt=0)


class CombinedNeuroRequest(BaseModel):
    """
    Submit all active test results together for combined scoring.
    This is the main FR4 endpoint called by the fusion engine.
    """
    reaction_time_ms:    Optional[float] = Field(None, description="Mean reaction time (ms)")
    tracking_error:      Optional[float] = Field(None, ge=0.0, le=1.0,
                                                  description="Normalised tracking error")
    cognitive_score:     Optional[float] = Field(None, ge=0.0, le=1.0,
                                                  description="Cognitive composite score")
    tap_irregularity:    Optional[float] = Field(None, ge=0.0, le=1.0,
                                                  description="Tap rhythm irregularity")


# ── Response schemas ───────────────────────────────────────────────────────

class TestScoreResponse(BaseModel):
    test_name: str
    raw_value: float
    normalised_score: float     # [0,1] — 0=normal, 1=anomalous
    is_anomaly: bool
    severity: str               # "normal" | "mild" | "moderate" | "severe"
    clinical_reference: str
    baseline_comparison: Optional[str]
    notes: list[str]


class CombinedNeuroResponse(BaseModel):
    overall_neuro_score: float  # [0,1] for fusion engine
    is_anomaly: bool
    alert_tier: str             # "normal" | "advisory" | "critical"
    individual_scores: dict[str, float]
    contributing_tests: list[str]
    fusion_ready_scores: dict   # dict passed directly to /api/fusion/score
    timestamp: str
    notes: list[str]


# ── Scoring helpers ────────────────────────────────────────────────────────

def _severity_label(score: float) -> str:
    if score < 0.25:   return "normal"
    if score < 0.50:   return "mild"
    if score < 0.75:   return "moderate"
    return "severe"


def _normalise_reaction(mean_ms: float) -> float:
    """Map reaction time to [0,1] anomaly score."""
    if mean_ms <= REACTION_NORMAL_MS:
        return 0.0
    if mean_ms >= REACTION_VERY_SLOW_MS:
        return 1.0
    return (mean_ms - REACTION_NORMAL_MS) / (REACTION_VERY_SLOW_MS - REACTION_NORMAL_MS)


def _normalise_tracking(mean_error: float) -> float:
    """Map tracking error to [0,1] anomaly score."""
    if mean_error <= TRACKING_NORMAL_ERROR:
        return 0.0
    if mean_error >= TRACKING_HIGH_ERROR:
        return 1.0
    return (mean_error - TRACKING_NORMAL_ERROR) / (TRACKING_HIGH_ERROR - TRACKING_NORMAL_ERROR)


def _normalise_cognitive(
    digit_span: int,
    pattern_acc: float,
    response_ms: float,
) -> float:
    """Composite cognitive score → [0,1]."""
    # Digit span — below normal = anomalous
    span_score = max(0.0, (DIGIT_SPAN_NORMAL - digit_span) / (DIGIT_SPAN_NORMAL - DIGIT_SPAN_LOW))
    span_score = min(1.0, span_score)

    # Pattern accuracy — lower = worse
    pattern_score = 1.0 - pattern_acc

    # Response time — slow = worse
    rt_score = min(1.0, max(0.0, (response_ms - 500) / 2000))

    return float(0.5 * span_score + 0.3 * pattern_score + 0.2 * rt_score)


def _normalise_tap(tap_timestamps: list[float]) -> tuple[float, float]:
    """
    Return (tap_speed_score, irregularity_score) both [0,1].
    Speed: below normal = worse.
    Irregularity: high CoV of inter-tap intervals = worse (Parkinson's marker).
    """
    if len(tap_timestamps) < 3:
        return 0.5, 0.5

    intervals = np.diff(sorted(tap_timestamps))  # inter-tap intervals in ms
    mean_interval = float(np.mean(intervals))
    std_interval  = float(np.std(intervals))

    tap_speed = 1000.0 / mean_interval  # taps per second

    # Speed score
    if tap_speed >= TAP_NORMAL_SPEED:
        speed_score = 0.0
    elif tap_speed <= TAP_SLOW_SPEED:
        speed_score = 1.0
    else:
        speed_score = (TAP_NORMAL_SPEED - tap_speed) / (TAP_NORMAL_SPEED - TAP_SLOW_SPEED)

    # Irregularity (CoV)
    cov = std_interval / (mean_interval + 1e-8)
    irregularity_score = min(1.0, cov / 0.5)  # CoV > 0.5 = very irregular

    return round(float(speed_score), 4), round(float(irregularity_score), 4)


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/status")
async def neuro_test_status():
    """FR4 module health check."""
    return {
        "status": "ok",
        "tests_available": [
            "reaction_time",
            "finger_tracking",
            "cognitive",
            "tap_speed",
            "combined",
        ],
        "clinical_references": {
            "reaction_time_normal_ms":  REACTION_NORMAL_MS,
            "reaction_time_slow_ms":    REACTION_SLOW_MS,
            "tracking_normal_error":    TRACKING_NORMAL_ERROR,
            "digit_span_normal":        DIGIT_SPAN_NORMAL,
            "tap_normal_speed":         TAP_NORMAL_SPEED,
        },
    }


@router.post("/reaction-time", response_model=TestScoreResponse)
async def score_reaction_time(request: ReactionTimeRequest):
    """
    Score a reaction time test.

    The frontend runs 5–10 trials and sends all trial times.
    We discard the first trial (practice effect) and score the mean.
    """
    trials = request.trials

    # Discard first trial (practice) and outliers
    if len(trials) > 3:
        trials = trials[1:]   # drop first
    if len(trials) > 4:
        # Remove fastest + slowest (outlier trimming)
        trials = sorted(trials)[1:-1]

    mean_ms   = float(np.mean(trials))
    median_ms = float(np.median(trials))
    std_ms    = float(np.std(trials))

    score = _normalise_reaction(mean_ms)

    notes = [f"Mean RT: {mean_ms:.0f}ms | Median: {median_ms:.0f}ms | SD: {std_ms:.0f}ms"]

    if std_ms > 150:
        notes.append("High RT variability — possible attention/concentration issue.")

    if mean_ms > REACTION_VERY_SLOW_MS:
        notes.append("Reaction time critically slow — possible motor or cognitive deficit.")
    elif mean_ms > REACTION_SLOW_MS:
        notes.append("Reaction time above normal range.")

    logger.info(
        "Reaction time | mean=%.0fms | score=%.3f | severity=%s",
        mean_ms, score, _severity_label(score),
    )

    return TestScoreResponse(
        test_name           = "reaction_time",
        raw_value           = round(mean_ms, 1),
        normalised_score    = round(score, 4),
        is_anomaly          = score >= 0.5,
        severity            = _severity_label(score),
        clinical_reference  = f"Normal: <{REACTION_NORMAL_MS}ms | Abnormal: >{REACTION_SLOW_MS}ms",
        baseline_comparison = None,
        notes               = notes,
    )


@router.post("/finger-tracking", response_model=TestScoreResponse)
async def score_finger_tracking(request: FingerTrackingRequest):
    """
    Score a finger/cursor tracking test.

    Calculates:
      - Mean Euclidean error (accuracy)
      - Error variance (tremor proxy — high variance = tremor)
      - Lag (systematic delay in following target)
    """
    data = request.tracking_data

    errors = [
        np.sqrt((p.target_x - p.actual_x)**2 + (p.target_y - p.actual_y)**2)
        for p in data
    ]

    mean_error = float(np.mean(errors))
    std_error  = float(np.std(errors))
    max_error  = float(np.max(errors))

    # Tremor proxy — high std relative to mean = tremor-like
    tremor_score = min(1.0, std_error / (mean_error + 1e-8) / 2.0)

    # Accuracy score
    accuracy_score = _normalise_tracking(mean_error)

    # Combined
    combined_score = float(0.6 * accuracy_score + 0.4 * tremor_score)

    notes = [
        f"Mean error: {mean_error:.3f} | SD: {std_error:.3f} | "
        f"Max: {max_error:.3f} | Tremor proxy: {tremor_score:.3f}",
    ]

    if tremor_score > 0.5:
        notes.append("High error variability detected — possible motor tremor.")
    if accuracy_score > 0.5:
        notes.append("Tracking accuracy below normal range.")

    logger.info(
        "Finger tracking | mean_err=%.3f | tremor=%.3f | score=%.3f",
        mean_error, tremor_score, combined_score,
    )

    return TestScoreResponse(
        test_name          = "finger_tracking",
        raw_value          = round(mean_error, 4),
        normalised_score   = round(combined_score, 4),
        is_anomaly         = combined_score >= 0.5,
        severity           = _severity_label(combined_score),
        clinical_reference = f"Normal error: <{TRACKING_NORMAL_ERROR} | High: >{TRACKING_HIGH_ERROR}",
        baseline_comparison= None,
        notes              = notes,
    )


@router.post("/cognitive", response_model=TestScoreResponse)
async def score_cognitive(request: CognitiveTestRequest):
    """
    Score a cognitive screening test.

    Digit span tests working memory (MMSE component).
    Pattern accuracy tests visuospatial processing.
    Both are sensitive to TIA/stroke effects.
    """
    score = _normalise_cognitive(
        digit_span  = request.digit_span_score,
        pattern_acc = request.pattern_accuracy,
        response_ms = request.response_time_ms,
    )

    notes = [
        f"Digit span: {request.digit_span_score} "
        f"(normal: {DIGIT_SPAN_NORMAL}±2)",
        f"Pattern accuracy: {request.pattern_accuracy*100:.0f}%",
        f"Response time: {request.response_time_ms:.0f}ms",
    ]

    if request.digit_span_score < DIGIT_SPAN_LOW:
        notes.append("Digit span critically low — working memory deficit possible.")

    if request.digit_span_reverse is not None:
        if request.digit_span_reverse < request.digit_span_score - 2:
            notes.append(
                "Large gap between forward and reverse digit span — "
                "executive function concern."
            )

    logger.info(
        "Cognitive test | span=%d | pattern=%.2f | score=%.3f",
        request.digit_span_score, request.pattern_accuracy, score,
    )

    return TestScoreResponse(
        test_name          = "cognitive",
        raw_value          = float(request.digit_span_score),
        normalised_score   = round(score, 4),
        is_anomaly         = score >= 0.5,
        severity           = _severity_label(score),
        clinical_reference = f"Normal digit span: {DIGIT_SPAN_NORMAL}±2 digits",
        baseline_comparison= None,
        notes              = notes,
    )


@router.post("/tap-speed", response_model=TestScoreResponse)
async def score_tap_speed(request: TapTestRequest):
    """
    Score a rapid finger-tapping test.
    Detects bradykinesia (slowness) and rhythm irregularity
    — both early Parkinson's and TIA markers.
    """
    speed_score, irregularity_score = _normalise_tap(request.tap_timestamps_ms)
    combined = float(0.5 * speed_score + 0.5 * irregularity_score)

    n_taps    = len(request.tap_timestamps_ms)
    tap_rate  = n_taps / request.duration_seconds

    notes = [
        f"Taps: {n_taps} | Rate: {tap_rate:.1f} taps/sec | "
        f"Speed score: {speed_score:.3f} | "
        f"Irregularity: {irregularity_score:.3f}",
    ]

    if tap_rate < TAP_SLOW_SPEED:
        notes.append(f"Tap rate below normal ({TAP_NORMAL_SPEED} taps/sec).")
    if irregularity_score > 0.5:
        notes.append("Irregular tapping rhythm — possible bradykinesia or tremor.")

    logger.info(
        "Tap speed | rate=%.1f/s | speed_score=%.3f | irreg=%.3f",
        tap_rate, speed_score, irregularity_score,
    )

    return TestScoreResponse(
        test_name          = "tap_speed",
        raw_value          = round(tap_rate, 2),
        normalised_score   = round(combined, 4),
        is_anomaly         = combined >= 0.5,
        severity           = _severity_label(combined),
        clinical_reference = f"Normal: >{TAP_NORMAL_SPEED} taps/sec",
        baseline_comparison= None,
        notes              = notes,
    )


@router.post("/combined", response_model=CombinedNeuroResponse)
async def combined_neuro_score(request: CombinedNeuroRequest):
    """
    Aggregate all available active test scores into a single
    neuro_test_score for the Fusion Engine.

    This is the endpoint the React frontend calls after running
    all active tests, before calling /api/fusion/score.

    Returns fusion_ready_scores dict which maps directly to
    FusionRequest.neuro_test_scores.
    """
    logger.info(
        "Combined neuro | rt=%.0f | tracking=%.3f | cog=%.3f | tap=%.3f",
        request.reaction_time_ms or -1,
        request.tracking_error   or -1,
        request.cognitive_score  or -1,
        request.tap_irregularity or -1,
    )

    scores       = {}
    contributing = []
    notes        = []

    # ── Reaction time ──────────────────────────────────────────────────
    if request.reaction_time_ms is not None:
        rt_score = _normalise_reaction(request.reaction_time_ms)
        scores["reaction_time"] = round(rt_score, 4)
        if rt_score >= 0.5:
            contributing.append(f"Slow reaction time ({request.reaction_time_ms:.0f}ms)")

    # ── Tracking ───────────────────────────────────────────────────────
    if request.tracking_error is not None:
        t_score = _normalise_tracking(request.tracking_error)
        scores["tracking"] = round(t_score, 4)
        if t_score >= 0.5:
            contributing.append("Poor motor tracking accuracy")

    # ── Cognitive ──────────────────────────────────────────────────────
    if request.cognitive_score is not None:
        scores["cognitive"] = round(request.cognitive_score, 4)
        if request.cognitive_score >= 0.5:
            contributing.append("Cognitive screening anomaly")

    # ── Tap irregularity ───────────────────────────────────────────────
    if request.tap_irregularity is not None:
        scores["tap_irregularity"] = round(request.tap_irregularity, 4)
        if request.tap_irregularity >= 0.5:
            contributing.append("Irregular tapping rhythm")

    if not scores:
        raise HTTPException(
            status_code=422,
            detail="At least one test score must be provided.",
        )

    # ── Overall score (mean of available tests) ────────────────────────
    overall = float(np.mean(list(scores.values())))

    # ── Alert tier ─────────────────────────────────────────────────────
    if overall < 0.35:
        alert_tier = "normal"
    elif overall < 0.65:
        alert_tier = "advisory"
    else:
        alert_tier = "critical"

    if not contributing:
        notes.append("All active neurological tests within normal range.")
    else:
        notes.append(f"{len(contributing)} test(s) flagged anomalies.")

    # ── Feed to IF baseline ────────────────────────────────────────────
    # Build a feature vector from whatever tests were run
    # and add to the cardiac IF stream (no dedicated neuro IF yet —
    # neuro tests augment the existing streams)
    neuro_vector = np.array(list(scores.values()))
    notes.append(
        f"Combined score: {overall:.3f} | Tests used: {list(scores.keys())}"
    )

    # fusion_ready_scores maps directly to FusionRequest.neuro_test_scores
    fusion_ready = {k: v for k, v in scores.items()}

    return CombinedNeuroResponse(
        overall_neuro_score  = round(overall, 4),
        is_anomaly           = overall >= 0.5,
        alert_tier           = alert_tier,
        individual_scores    = scores,
        contributing_tests   = contributing,
        fusion_ready_scores  = fusion_ready,
        timestamp            = datetime.utcnow().isoformat(),
        notes                = notes,
    )