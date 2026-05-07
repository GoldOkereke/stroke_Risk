from __future__ import annotations

import numpy as np

from app.services.isolation_forest import IFStream, IsolationForestService


def test_baseline_addition_and_auto_train(tmp_path):
    service = IsolationForestService(user_id="u1", baseline_threshold=5, weights_dir=tmp_path)

    # Add baseline samples; training should trigger at threshold.
    for _ in range(5):
        service.add_baseline(IFStream.FACIAL, np.random.normal(0, 1, size=(6,)))

    assert (tmp_path / "isolation_forest" / "facial_u1.pkl").exists()
    assert (tmp_path / "isolation_forest" / "facial_u1_scaler.pkl").exists()


def test_scoring_returns_none_when_no_models(tmp_path):
    service = IsolationForestService(user_id="u2", weights_dir=tmp_path)
    result = service.score(IFStream.FACIAL, np.random.normal(0, 1, size=(6,)))
    assert result is None


def test_training_and_scoring_produces_valid_range(tmp_path):
    service = IsolationForestService(user_id="u3", baseline_threshold=3, weights_dir=tmp_path)
    for _ in range(3):
        service.add_baseline(IFStream.DYSARTHRIA, np.random.normal(0, 1, size=(10,)))

    result = service.score(IFStream.DYSARTHRIA, np.random.normal(3, 1, size=(10,)))
    assert result is not None
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_save_and_load_user_model(tmp_path):
    service = IsolationForestService(user_id="u4", baseline_threshold=3, weights_dir=tmp_path)
    for _ in range(3):
        service.add_baseline(IFStream.PARKINSONS, np.random.normal(0, 1, size=(7,)))

    # New instance should load from disk.
    service2 = IsolationForestService(user_id="u4", weights_dir=tmp_path)
    res = service2.score(IFStream.PARKINSONS, np.random.normal(0, 1, size=(7,)))
    assert res is not None
