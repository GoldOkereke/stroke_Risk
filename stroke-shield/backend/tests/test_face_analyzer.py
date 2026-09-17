from __future__ import annotations

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from app.services.face_analyzer import FaceAnalyzer  # noqa: E402


def test_quality_scoring_range_on_blank_image():
    analyzer = FaceAnalyzer()
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    result = analyzer.analyze(img, use_if=False)

    assert result.quality is not None
    assert 0.0 <= result.quality <= 1.0
    assert result.face_detected in (True, False)


def test_batch_analysis_returns_aggregate():
    analyzer = FaceAnalyzer()
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(3)]
    result = analyzer.analyze_batch(imgs, use_if=False)
    assert 0.0 <= result.aggregated_score <= 1.0
    assert result.quality is not None
