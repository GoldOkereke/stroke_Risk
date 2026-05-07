from __future__ import annotations

from app.services.fusion_engine import FusionEngine


def test_weight_redistribution_when_streams_missing():
    engine = FusionEngine()
    scores = {"afib_cnn": 1.0, "cardiac_if": 0.0}  # only 2 present
    result = engine.fuse(scores)

    # Risk should be between 0 and 100.
    assert 0.0 <= result.risk_score <= 100.0
    # Contributing factors should only include present streams.
    assert set(result.contributing_factors.keys()) <= set(scores.keys())


def test_corroboration_bonus_increases_score():
    engine = FusionEngine()
    one_stream = engine.fuse({"afib_cnn": 0.8})
    two_streams = engine.fuse({"afib_cnn": 0.8, "cardiac_if": 0.8})
    assert two_streams.risk_score >= one_stream.risk_score


def test_temporal_trend_adds_bonus():
    engine = FusionEngine()
    history = [10, 12, 15, 18, 22, 27, 33, 40, 48, 57]
    no_history = engine.fuse({"afib_cnn": 0.5, "cardiac_if": 0.5})
    with_history = engine.fuse({"afib_cnn": 0.5, "cardiac_if": 0.5}, history=history)
    assert with_history.risk_score >= no_history.risk_score


def test_alert_tier_boundaries():
    engine = FusionEngine()
    r1 = engine.fuse({"afib_cnn": 0.2})
    assert r1.alert_tier in {"NORMAL", "ADVISORY", "CRITICAL"}

    # Force high score.
    r2 = engine.fuse({"afib_cnn": 1.0, "cardiac_if": 1.0, "facial_if": 1.0})
    assert r2.alert_tier in {"ADVISORY", "CRITICAL"}


def test_failure_cases_trigger():
    engine = FusionEngine()

    # FC1: high neuro, low cardiac.
    fc1 = engine.fuse({"facial_if": 0.9, "dysarthria_if": 0.9, "afib_cnn": 0.1, "cardiac_if": 0.1})
    assert fc1.failure_case is None or fc1.failure_case.startswith("FC")

    # FC4: uncertain CNN prediction.
    fc4 = engine.fuse({"afib_cnn": 0.5, "cardiac_if": 0.5})
    assert fc4.failure_case is None or fc4.failure_case.startswith("FC")

