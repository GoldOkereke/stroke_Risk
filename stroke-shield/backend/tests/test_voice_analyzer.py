from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("librosa")

from app.services.voice_analyzer import VoiceAnalyzer  # noqa: E402


def test_feature_extraction_keys_and_types():
    sr = 16000
    t = np.arange(sr) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    analyzer = VoiceAnalyzer()
    dys = analyzer.extract_dysarthria_features(audio, sr)
    pk = analyzer.extract_parkinsons_features(audio, sr)

    assert len(dys) == 10
    assert len(pk) == 7
    assert all(isinstance(v, float) for v in dys.values())
    assert all(isinstance(v, float) for v in pk.values())


def test_analyze_returns_scores():
    sr = 16000
    t = np.arange(sr) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    analyzer = VoiceAnalyzer()
    result = analyzer.analyze(audio, sr)
    assert "overall_score" in result.scores
    assert 0.0 <= float(result.scores["overall_score"]) <= 1.0
