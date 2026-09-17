"""Fusion engine.

This module combines multiple modality scores into one overall risk score.

Input assumption:
- each stream score is a float in [0, 1] where higher means higher risk
- streams may be missing (not every modality is always available)

Output:
- overall risk score in [0, 100]
- alert tier: NORMAL / ADVISORY / CRITICAL
- confidence in [0, 1]
- contributing factors (which streams pushed the score up)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    # Final overall risk score (0-100).
    risk_score: float

    # Tier label derived from the risk_score thresholds.
    alert_tier: str

    # How confident the engine is in the result (0-1).
    confidence: float

    # Per-stream contributions after weight redistribution.
    contributing_factors: dict[str, float]

    # Plain-language recommendation string.
    recommendation: str

    # Optional "failure case" string when results are contradictory/uncertain.
    failure_case: Optional[str] = None


class FusionEngine:
    """Combines multiple stream scores into a single risk score."""

    def __init__(self) -> None:
        # Base weights represent how much each stream should matter when present.
        # If a stream is missing, its weight is redistributed to the streams we do have.
        self.weights: dict[str, float] = {
            "afib_cnn": 0.30,
            "cardiac_if": 0.20,
            "facial_if": 0.20,
            "dysarthria_if": 0.15,
            "parkinsons_if": 0.10,
            "neuro_tests": 0.05,
        }

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def _redistribute_weights(self, present: set[str]) -> dict[str, float]:
        """Redistribute weights proportionally among streams that are present."""
        base = {k: v for k, v in self.weights.items() if k in present}
        total = float(sum(base.values()))
        if total <= 0:
            return {}
        return {k: v / total for k, v in base.items()}

    @staticmethod
    def _corroboration_bonus(scores: dict[str, float]) -> tuple[float, int]:
        """Compute a multiplicative bonus based on how many streams "fire"."""
        firing = sum(1 for v in scores.values() if v > 0.5)
        if firing <= 1:
            return 1.0, firing
        bonus = 1.0 + 0.12 * (firing - 1)
        return float(min(1.60, bonus)), firing

    @staticmethod
    def _trend_bonus(history: Optional[list[float]]) -> tuple[float, float]:
        """Compute an additive bonus (0-12 points) if risk is trending upward.

        We fit a straight line to the last up-to-10 risk scores and use its slope.
        """
        if not history:
            return 0.0, 0.0
        recent = np.asarray(history[-10:], dtype=float)
        if recent.size < 3:
            return 0.0, 0.0

        x = np.arange(recent.size, dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])  # points per test
        if slope <= 0:
            return 0.0, slope

        # Convert slope into a capped bonus.
        # Example: slope ~= 1.0 (one point increase per test) -> 6 bonus points.
        bonus = float(min(12.0, slope * 6.0))
        return bonus, slope

    @staticmethod
    def _tier(score: float) -> str:
        if score >= 75:
            return "CRITICAL"
        if score >= 35:
            return "ADVISORY"
        return "NORMAL"

    @staticmethod
    def _confidence(
        present_scores: dict[str, float],
        firing_count: int,
        missing_count: int,
        trend_slope: float,
    ) -> float:
        """Heuristic confidence in [0, 1].

        Confidence increases when:
        - more streams are present
        - multiple streams agree (firing_count)
        Confidence decreases when:
        - many streams are missing
        - the trend slope is high (rapid change can mean instability / transient artifacts)
        """
        present = len(present_scores)
        if present == 0:
            return 0.0

        # Coverage: more present streams -> more confidence.
        coverage = present / (present + missing_count)

        # Agreement: if many streams are above 0.5, we assume corroboration.
        agreement = min(1.0, firing_count / max(1, present))

        # Penalty for fast changes.
        trend_penalty = min(0.35, max(0.0, trend_slope / 10.0))

        conf = 0.15 + 0.55 * coverage + 0.30 * agreement - trend_penalty
        return float(np.clip(conf, 0.0, 1.0))

    @staticmethod
    def _recommendation(tier: str, factors: list[str]) -> str:
        if tier == "CRITICAL":
            return "High risk detected. Seek urgent medical evaluation. Key factors: " + ", ".join(factors) + "."
        if tier == "ADVISORY":
            return "Moderate risk detected. Consider follow-up screening and monitor symptoms. Key factors: " + ", ".join(factors) + "."
        return "Low risk detected. Maintain healthy habits and repeat checks periodically."

    @staticmethod
    def _failure_case(
        stream_scores: dict[str, float],
        confidence: float,
        trend_bonus: float,
    ) -> Optional[str]:
        """Identify specific situations that should be flagged for review."""
        cardiac = max(stream_scores.get("afib_cnn", 0.0), stream_scores.get("cardiac_if", 0.0))
        neuro = max(
            stream_scores.get("facial_if", 0.0),
            stream_scores.get("dysarthria_if", 0.0),
            stream_scores.get("parkinsons_if", 0.0),
            stream_scores.get("neuro_tests", 0.0),
        )

        if neuro > 0.75 and cardiac < 0.35:
            return "FC1: High neuro scores but normal cardiac."
        if cardiac > 0.75 and neuro < 0.35:
            return "FC2: High cardiac scores but normal neuro."
        if confidence < 0.35:
            return "FC3: Low confidence due to missing/unstable streams."
        if 0.35 <= stream_scores.get("afib_cnn", 0.0) <= 0.65:
            return "FC4: Uncertain CNN prediction."
        if trend_bonus >= 8.0:
            return "FC5: Rapidly increasing trend."
        return None

    def fuse(self, results: dict[str, float], history: Optional[list[float]] = None) -> FusionResult:
        """Fuse stream scores into an overall result.

        Steps (mirrors the spec in your screenshot):
        1) Collect available stream scores.
        2) Redistribute weights for missing streams.
        3) Compute weighted base score.
        4) Compute corroboration bonus based on firing streams.
        5) Apply bonus.
        6) Apply trend bonus based on score history.
        7) Clamp to [0, 100].
        8) Determine alert tier.
        9) Compute confidence.
        10) Identify contributing factors.
        11) Generate recommendation.
        12) Identify failure cases.
        """
        # 1) Collect and clamp scores to [0, 1].
        present_scores = {
            k: self._clamp(float(v), 0.0, 1.0)
            for k, v in (results or {}).items()
            if k in self.weights
        }
        present = set(present_scores.keys())
        missing_count = len(self.weights) - len(present)

        # 2) Redistribute weights among present streams.
        adj_weights = self._redistribute_weights(present)

        # 3) Weighted base score in [0, 1].
        base = 0.0
        contributions: dict[str, float] = {}
        for stream, w in adj_weights.items():
            s = present_scores.get(stream, 0.0)
            base += w * s
            contributions[stream] = float(w * s)

        # 4) Corroboration bonus based on streams above 0.5.
        bonus, firing_count = self._corroboration_bonus(present_scores)

        # 5) Apply bonus, convert to 0-100 scale.
        score = float(base * bonus * 100.0)

        # 6) Add trend bonus (0-12 points).
        trend_bonus, trend_slope = self._trend_bonus(history)
        score += float(trend_bonus)

        # 7) Clamp.
        score = self._clamp(score, 0.0, 100.0)

        # 8) Tier.
        tier = self._tier(score)

        # 9) Confidence.
        confidence = self._confidence(present_scores, firing_count, missing_count, trend_slope)

        # 10) Contributing factors: streams that are strongly positive.
        factors = [k for k, v in present_scores.items() if v > 0.6]
        if not factors:
            # Fall back to the highest scored stream so we always have an explanation.
            if present_scores:
                factors = [max(present_scores, key=present_scores.get)]

        # 11) Recommendation.
        recommendation = self._recommendation(tier, factors)

        # 12) Failure cases.
        failure_case = self._failure_case(present_scores, confidence, trend_bonus)

        return FusionResult(
            risk_score=float(score),
            alert_tier=tier,
            confidence=float(confidence),
            contributing_factors=contributions,
            recommendation=recommendation,
            failure_case=failure_case,
        )
