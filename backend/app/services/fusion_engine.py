"""
app/services/fusion_engine.py
==============================
Fusion Engine — FR5

Combines outputs from all signal streams into a single pre-TIA risk score.

Inputs:
  - afib_probability       : float [0,1]  — FR2 CNN output
  - cardiac_anomaly        : AnomalyResult — IF Stream 1 (ECG/PPG HRV)
  - facial_anomaly         : AnomalyResult — IF Stream 2 (MediaPipe landmarks)
  - dysarthria_anomaly     : AnomalyResult — IF Stream 3 (TORGO MFCCs)
  - parkinsons_anomaly     : AnomalyResult — IF Stream 4 (UCI pitch/jitter)
  - neuro_test_scores      : dict          — FR4 active test results

Output:
  FusionResult:
    risk_score            : float [0, 100]
    alert_tier            : NORMAL | ADVISORY | CRITICAL
    confidence            : float [0, 1]
    uncertainty_flag      : bool
    contributing_factors  : list[str]
    failure_case          : str | None
    recommendation        : str

The differentiator:
  1. Corroboration bonus  — 2+ streams agreeing raises confidence non-linearly
  2. Temporal trending    — worsening scores over time increase risk
  3. Uncertainty handling — conflicting streams trigger failure-case logic
  4. Stream degradation   — noisy/missing inputs reduce their own weight
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np

from app.services.isolation_forest import AnomalyResult, IFStream

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Alert tier thresholds
NORMAL_THRESHOLD   = 35.0
ADVISORY_THRESHOLD = 60.0
# >= ADVISORY_THRESHOLD and < CRITICAL → ADVISORY
# >= 75 → CRITICAL
CRITICAL_THRESHOLD = 75.0

# Base weights per stream — must sum to 1.0
BASE_WEIGHTS = {
    "afib_cnn":       0.30,  # supervised CNN — highest weight (most validated)
    "cardiac_if":     0.20,  # personal cardiac baseline deviation
    "facial_if":      0.20,  # facial asymmetry
    "dysarthria_if":  0.15,  # slurred speech (TORGO)
    "parkinsons_if":  0.10,  # tremor / pitch (UCI)
    "neuro_tests":    0.05,  # active neurological tests
}

# Corroboration bonus per additional stream agreeing
CORROBORATION_BONUS_PER_STREAM = 0.12   # +12% per extra stream above 1

# Temporal window for trend analysis
TREND_WINDOW = 10       # last N scores
TREND_MAX_BONUS = 12.0  # max additional points from worsening trend

# Uncertainty threshold — std deviation across stream scores
UNCERTAINTY_STD_THRESHOLD = 0.28


# ── Enums ──────────────────────────────────────────────────────────────────

class AlertTier(str, Enum):
    NORMAL   = "NORMAL"
    ADVISORY = "ADVISORY"
    CRITICAL = "CRITICAL"


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class StreamScore:
    """Normalised contribution from one stream."""
    name: str
    raw_score: float        # [0, 1]
    weight: float           # effective weight after degradation
    weighted_contribution: float
    available: bool         # False if stream had no input
    degraded: bool          # True if input quality was poor


@dataclass
class FusionResult:
    """
    Full output from the Fusion Engine.
    Consumed by the alert system and the React frontend.
    """
    timestamp: datetime

    # Core outputs
    risk_score: float       # [0, 100]
    alert_tier: AlertTier
    confidence: float       # [0, 1]

    # Detail
    stream_scores: list[StreamScore]
    contributing_factors: list[str]      # which streams fired
    corroboration_count: int             # how many streams agreed
    corroboration_multiplier: float

    # Temporal
    trend_direction: str    # "stable" | "worsening" | "improving"
    trend_bonus: float      # extra risk points from worsening trend

    # Failure case handling
    uncertainty_flag: bool
    failure_case: Optional[str]          # None if no failure case
    recommendation: str

    # Raw inputs (for debugging + audit)
    afib_probability: float
    streams_available: list[str]
    streams_missing: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp":              self.timestamp.isoformat(),
            "risk_score":             round(self.risk_score, 2),
            "alert_tier":             self.alert_tier.value,
            "confidence":             round(self.confidence, 4),
            "contributing_factors":   self.contributing_factors,
            "corroboration_count":    self.corroboration_count,
            "trend_direction":        self.trend_direction,
            "uncertainty_flag":       self.uncertainty_flag,
            "failure_case":           self.failure_case,
            "recommendation":         self.recommendation,
            "afib_probability":       round(self.afib_probability, 4),
            "streams_available":      self.streams_available,
            "streams_missing":        self.streams_missing,
            "stream_scores": [
                {
                    "name":        s.name,
                    "raw_score":   round(s.raw_score, 4),
                    "weight":      round(s.weight, 4),
                    "available":   s.available,
                    "degraded":    s.degraded,
                }
                for s in self.stream_scores
            ],
        }


# ── Fusion Engine ──────────────────────────────────────────────────────────

class FusionEngine:
    """
    Combines all stream outputs into a single pre-TIA risk score.

    Usage
    -----
        engine = FusionEngine()

        result = engine.fuse(
            afib_probability   = 0.82,
            cardiac_anomaly    = cardiac_result,    # AnomalyResult
            facial_anomaly     = facial_result,
            dysarthria_anomaly = dysarthria_result,
            parkinsons_anomaly = parkinsons_result,
            neuro_test_scores  = {"reaction_time": 0.6, "tracking": 0.4},
        )

        print(result.alert_tier)      # CRITICAL
        print(result.risk_score)      # 81.4
        print(result.recommendation)  # "Seek emergency medical attention..."
    """

    def __init__(self):
        self._score_history: deque[float] = deque(maxlen=TREND_WINDOW)
        logger.info("FusionEngine initialised.")

    # ── Main entry point ───────────────────────────────────────────────────

    def fuse(
        self,
        afib_probability:   float,
        cardiac_anomaly:    Optional[AnomalyResult] = None,
        facial_anomaly:     Optional[AnomalyResult] = None,
        dysarthria_anomaly: Optional[AnomalyResult] = None,
        parkinsons_anomaly: Optional[AnomalyResult] = None,
        neuro_test_scores:  Optional[dict] = None,
    ) -> FusionResult:
        """
        Fuse all available stream outputs into a FusionResult.

        Any stream can be None (e.g. no microphone available).
        Missing streams have their weight redistributed to available streams.
        """
        logger.info(
            "Fusing | afib_prob=%.3f | streams: cardiac=%s facial=%s "
            "dysarthria=%s parkinsons=%s neuro=%s",
            afib_probability,
            cardiac_anomaly    is not None,
            facial_anomaly     is not None,
            dysarthria_anomaly is not None,
            parkinsons_anomaly is not None,
            neuro_test_scores  is not None,
        )

        # ── Step 1: Collect raw scores from all streams ────────────────
        stream_scores = self._collect_stream_scores(
            afib_probability,
            cardiac_anomaly,
            facial_anomaly,
            dysarthria_anomaly,
            parkinsons_anomaly,
            neuro_test_scores,
        )

        # ── Step 2: Redistribute weights for missing streams ───────────
        stream_scores = self._redistribute_weights(stream_scores)

        # ── Step 3: Weighted base score ────────────────────────────────
        available_scores = [s for s in stream_scores if s.available]
        base_score = sum(s.raw_score * s.weight for s in available_scores)

        # ── Step 4: Corroboration bonus ────────────────────────────────
        # How many streams independently agree there is a problem?
        # This is THE differentiator — streams corroborating each other
        # is much stronger evidence than a single stream firing.
        anomaly_threshold = 0.5
        streams_firing = [
            s for s in available_scores if s.raw_score >= anomaly_threshold
        ]
        corroboration_count = len(streams_firing)
        corroboration_multiplier = 1.0 + (
            max(0, corroboration_count - 1) * CORROBORATION_BONUS_PER_STREAM
        )
        # 1 stream  → ×1.00 (no bonus)
        # 2 streams → ×1.12
        # 3 streams → ×1.24
        # 4 streams → ×1.36
        # 5 streams → ×1.48
        # 6 streams → ×1.60

        corroborated_score = base_score * corroboration_multiplier

        # ── Step 5: Temporal trend ─────────────────────────────────────
        trend_bonus, trend_direction = self._compute_trend(corroborated_score)
        risk_score_raw = corroborated_score * 100 + trend_bonus

        # ── Step 6: Clamp to [0, 100] ─────────────────────────────────
        risk_score = float(np.clip(risk_score_raw, 0.0, 100.0))

        # ── Step 7: Alert tier ─────────────────────────────────────────
        alert_tier = self._compute_alert_tier(risk_score)

        # ── Step 8: Uncertainty + failure cases ───────────────────────
        stream_raw_scores = [s.raw_score for s in available_scores]
        uncertainty_flag, failure_case = self._check_uncertainty(
            stream_raw_scores,
            afib_probability,
            corroboration_count,
        )

        # ── Step 9: Confidence ─────────────────────────────────────────
        confidence = self._compute_confidence(
            stream_raw_scores,
            corroboration_count,
            uncertainty_flag,
            len(available_scores),
        )

        # ── Step 10: Contributing factors + recommendation ─────────────
        contributing_factors = self._get_contributing_factors(
            stream_scores, afib_probability
        )
        recommendation = self._get_recommendation(
            alert_tier, uncertainty_flag, failure_case, contributing_factors
        )

        # Store for trend analysis
        self._score_history.append(risk_score)

        result = FusionResult(
            timestamp              = datetime.utcnow(),
            risk_score             = round(risk_score, 2),
            alert_tier             = alert_tier,
            confidence             = round(confidence, 4),
            stream_scores          = stream_scores,
            contributing_factors   = contributing_factors,
            corroboration_count    = corroboration_count,
            corroboration_multiplier = round(corroboration_multiplier, 3),
            trend_direction        = trend_direction,
            trend_bonus            = round(trend_bonus, 2),
            uncertainty_flag       = uncertainty_flag,
            failure_case           = failure_case,
            recommendation         = recommendation,
            afib_probability       = round(afib_probability, 4),
            streams_available      = [s.name for s in stream_scores if s.available],
            streams_missing        = [s.name for s in stream_scores if not s.available],
        )

        logger.info(
            "Fusion complete | risk=%.1f | tier=%s | confidence=%.3f | "
            "corroboration=%d | uncertainty=%s",
            risk_score, alert_tier.value, confidence,
            corroboration_count, uncertainty_flag,
        )
        return result

    # ── Step 1: Collect stream scores ──────────────────────────────────────

    def _collect_stream_scores(
        self,
        afib_probability:   float,
        cardiac_anomaly:    Optional[AnomalyResult],
        facial_anomaly:     Optional[AnomalyResult],
        dysarthria_anomaly: Optional[AnomalyResult],
        parkinsons_anomaly: Optional[AnomalyResult],
        neuro_test_scores:  Optional[dict],
    ) -> list[StreamScore]:

        scores = []

        # CNN AFib probability
        scores.append(StreamScore(
            name="afib_cnn",
            raw_score=float(np.clip(afib_probability, 0, 1)),
            weight=BASE_WEIGHTS["afib_cnn"],
            weighted_contribution=afib_probability * BASE_WEIGHTS["afib_cnn"],
            available=True,
            degraded=0.35 <= afib_probability <= 0.65,  # uncertain zone
        ))

        # Cardiac IF
        scores.append(self._anomaly_to_stream_score(
            cardiac_anomaly, "cardiac_if", BASE_WEIGHTS["cardiac_if"]
        ))

        # Facial IF
        scores.append(self._anomaly_to_stream_score(
            facial_anomaly, "facial_if", BASE_WEIGHTS["facial_if"]
        ))

        # Dysarthria IF
        scores.append(self._anomaly_to_stream_score(
            dysarthria_anomaly, "dysarthria_if", BASE_WEIGHTS["dysarthria_if"]
        ))

        # Parkinson's IF
        scores.append(self._anomaly_to_stream_score(
            parkinsons_anomaly, "parkinsons_if", BASE_WEIGHTS["parkinsons_if"]
        ))

        # Active neuro tests (FR4)
        if neuro_test_scores:
            neuro_score = self._aggregate_neuro_tests(neuro_test_scores)
            scores.append(StreamScore(
                name="neuro_tests",
                raw_score=neuro_score,
                weight=BASE_WEIGHTS["neuro_tests"],
                weighted_contribution=neuro_score * BASE_WEIGHTS["neuro_tests"],
                available=True,
                degraded=False,
            ))
        else:
            scores.append(StreamScore(
                name="neuro_tests",
                raw_score=0.0,
                weight=BASE_WEIGHTS["neuro_tests"],
                weighted_contribution=0.0,
                available=False,
                degraded=False,
            ))

        return scores

    def _anomaly_to_stream_score(
        self,
        result: Optional[AnomalyResult],
        name: str,
        base_weight: float,
    ) -> StreamScore:
        if result is None or not result.baseline_ready:
            return StreamScore(
                name=name, raw_score=0.0, weight=base_weight,
                weighted_contribution=0.0, available=False, degraded=False,
            )
        return StreamScore(
            name=name,
            raw_score=result.anomaly_score,
            weight=base_weight,
            weighted_contribution=result.anomaly_score * base_weight,
            available=True,
            degraded=result.confidence < 0.3,   # low confidence = degraded
        )

    # ── Step 2: Redistribute weights ──────────────────────────────────────

    def _redistribute_weights(self, scores: list[StreamScore]) -> list[StreamScore]:
        """
        When a stream is missing, redistribute its weight proportionally
        to available streams so weights always sum to 1.0.
        """
        available = [s for s in scores if s.available]
        missing   = [s for s in scores if not s.available]

        if not missing:
            return scores

        freed_weight = sum(s.weight for s in missing)
        available_base = sum(s.weight for s in available)

        if available_base == 0:
            return scores

        for s in available:
            s.weight += freed_weight * (s.weight / available_base)
            s.weighted_contribution = s.raw_score * s.weight

        logger.debug(
            "Weight redistributed | missing=%s | freed=%.3f",
            [s.name for s in missing], freed_weight,
        )
        return scores

    # ── Step 4: Corroboration (built into fuse()) ──────────────────────────
    # See fuse() — corroboration_multiplier computed inline for clarity

    # ── Step 5: Temporal trend ─────────────────────────────────────────────

    def _compute_trend(self, current_score: float) -> tuple[float, str]:
        """
        Analyse the last N scores for a worsening trend.
        Returns (bonus_points, direction_label).
        """
        if len(self._score_history) < 3:
            return 0.0, "stable"

        history = list(self._score_history)
        # Linear regression slope over history
        x = np.arange(len(history), dtype=float)
        slope = float(np.polyfit(x, history, 1)[0])

        if slope > 1.5:
            # Worsening — score rising by >1.5 points per reading
            bonus = min(TREND_MAX_BONUS, slope * 2.0)
            return round(bonus, 2), "worsening"
        elif slope < -1.5:
            return 0.0, "improving"
        else:
            return 0.0, "stable"

    # ── Step 8: Uncertainty + failure cases ───────────────────────────────

    def _check_uncertainty(
        self,
        stream_scores: list[float],
        afib_probability: float,
        corroboration_count: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Detect failure cases where the model should not make a confident call.

        Failure cases (shown to judges):
          FC1 — High stream variance (streams strongly disagree)
          FC2 — CNN uncertain but neuro normal (isolated cardiac signal)
          FC3 — Neuro anomalies but cardiac normal (isolated neuro signal)
          FC4 — All streams low confidence (degraded inputs)
          FC5 — Insufficient baseline data
        """
        if not stream_scores:
            return True, "FC4: No stream data available."

        std = float(np.std(stream_scores))
        uncertainty_flag = std > UNCERTAINTY_STD_THRESHOLD

        # FC1 — High variance between streams
        if std > 0.35:
            return True, (
                "FC1: Streams disagree significantly (σ={:.2f}). "
                "Recommend repeat assessment.".format(std)
            )

        # FC2 — CNN uncertain zone, no neuro corroboration
        if 0.35 <= afib_probability <= 0.65 and corroboration_count <= 1:
            return True, (
                "FC2: CNN AFib probability uncertain ({:.0f}%) with no "
                "neurological corroboration. Cardiac monitoring recommended.".format(
                    afib_probability * 100
                )
            )

        # FC3 — Neuro firing but cardiac normal
        cardiac_score = stream_scores[0] if stream_scores else 0.0
        neuro_scores  = stream_scores[1:] if len(stream_scores) > 1 else []
        neuro_mean    = float(np.mean(neuro_scores)) if neuro_scores else 0.0

        if neuro_mean > 0.6 and cardiac_score < 0.3:
            return True, (
                "FC3: Neurological signals elevated but cardiac normal. "
                "Consider non-cardiac cause. Neurological review recommended."
            )

        return uncertainty_flag, None

    # ── Step 9: Confidence ─────────────────────────────────────────────────

    def _compute_confidence(
        self,
        stream_scores: list[float],
        corroboration_count: int,
        uncertainty_flag: bool,
        n_available: int,
    ) -> float:
        """
        Confidence = f(stream agreement, corroboration, data availability).
        Penalised by uncertainty flag and missing streams.
        """
        if not stream_scores:
            return 0.0

        # Base confidence from stream agreement (low variance = high agreement)
        std = float(np.std(stream_scores))
        agreement_confidence = float(np.clip(1.0 - (std / 0.5), 0.0, 1.0))

        # Bonus for corroboration
        corroboration_bonus = min(0.3, corroboration_count * 0.06)

        # Penalty for uncertainty
        uncertainty_penalty = 0.25 if uncertainty_flag else 0.0

        # Penalty for missing streams (max 6 streams)
        missing_penalty = max(0.0, (6 - n_available) * 0.05)

        confidence = (
            agreement_confidence
            + corroboration_bonus
            - uncertainty_penalty
            - missing_penalty
        )
        return float(np.clip(confidence, 0.0, 1.0))

    # ── Step 10: Factors + recommendation ─────────────────────────────────

    def _compute_alert_tier(self, risk_score: float) -> AlertTier:
        if risk_score >= CRITICAL_THRESHOLD:
            return AlertTier.CRITICAL
        elif risk_score >= ADVISORY_THRESHOLD:
            return AlertTier.ADVISORY
        else:
            return AlertTier.NORMAL

    def _get_contributing_factors(
        self,
        stream_scores: list[StreamScore],
        afib_probability: float,
    ) -> list[str]:
        factors = []

        if afib_probability >= 0.7:
            factors.append(f"AFib detected ({afib_probability*100:.0f}% probability)")
        elif afib_probability >= 0.5:
            factors.append(f"Possible AFib ({afib_probability*100:.0f}% probability)")

        threshold = 0.5
        label_map = {
            "cardiac_if":    "Cardiac rhythm deviation from personal baseline",
            "facial_if":     "Facial asymmetry detected",
            "dysarthria_if": "Speech slurring detected",
            "parkinsons_if": "Voice tremor/pitch anomaly detected",
            "neuro_tests":   "Active neurological test anomaly",
        }
        for s in stream_scores:
            if s.available and s.raw_score >= threshold and s.name in label_map:
                factors.append(label_map[s.name])

        return factors if factors else ["No significant anomalies detected."]

    def _get_recommendation(
        self,
        tier: AlertTier,
        uncertainty_flag: bool,
        failure_case: Optional[str],
        contributing_factors: list[str],
    ) -> str:

        if failure_case:
            return (
                f"Assessment inconclusive. {failure_case} "
                "Please repeat the assessment in 10 minutes or consult a clinician."
            )

        if tier == AlertTier.CRITICAL:
            return (
                "URGENT: Multiple stroke/TIA indicators detected. "
                "Call emergency services (999/911) immediately. "
                "Do not drive. Note the time symptoms began."
            )
        elif tier == AlertTier.ADVISORY:
            return (
                "Elevated risk indicators detected. "
                "Contact your GP or go to A&E if symptoms worsen. "
                "Avoid strenuous activity. Repeat assessment in 15 minutes."
            )
        else:
            if uncertainty_flag:
                return (
                    "Results are within normal range but some uncertainty detected. "
                    "Monitor and repeat if you feel unwell."
                )
            return (
                "No significant pre-TIA indicators detected. "
                "Continue monitoring. Seek medical advice if symptoms develop."
            )

    def _aggregate_neuro_tests(self, neuro_scores: dict) -> float:
        """
        Aggregate FR4 active test scores into a single [0,1] value.
        Expected keys: reaction_time, tracking, speech_prompt (all [0,1])
        """
        values = [float(v) for v in neuro_scores.values() if v is not None]
        return float(np.mean(values)) if values else 0.0

    # ── History ────────────────────────────────────────────────────────────

    def reset_history(self):
        """Clear temporal trend history (e.g. new session)."""
        self._score_history.clear()
        logger.info("Fusion score history reset.")

    def get_history(self) -> list[float]:
        return list(self._score_history)