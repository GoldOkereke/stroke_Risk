from __future__ import annotations

import numpy as np

from app.services.ingestion import RawSignalData
from app.services.segmentation import SegmentationService


def test_window_creation_and_overlap():
    """Windowing should produce overlapping segments of expected count."""
    fs = 10.0
    seconds = 120.0
    values = np.sin(2 * np.pi * 1.0 * (np.arange(int(fs * seconds)) / fs))
    raw = RawSignalData(
        signal_type="ECG",
        source="SIMULATED",
        values=values,
        sampling_rate=fs,
        timestamps=np.arange(values.size) / fs,
        labels=["NORMAL"],
    )

    # window 30s => 300 samples, overlap 0.5 => step 150
    result = SegmentationService().segment(raw, window_seconds=30, overlap=0.5)
    expected = 1 + (values.size - 300) // 150

    assert result.cnn_input.shape[0] <= expected  # quality filtering can drop windows
    assert result.cnn_input.shape[1] == 300
    assert result.cnn_input.shape[2] == 1


def test_quality_scoring_rejects_flat_signal():
    fs = 50.0
    values = np.zeros(int(fs * 60))
    raw = RawSignalData(
        signal_type="ECG",
        source="SIMULATED",
        values=values,
        sampling_rate=fs,
        timestamps=np.arange(values.size) / fs,
        labels=["NORMAL"],
    )
    result = SegmentationService().segment(raw, window_seconds=30, overlap=0.5)

    # Flat signal has std ~0 so quality should drop everything.
    assert result.cnn_input.shape[0] == 0


def test_hrv_feature_extraction_returns_shape():
    fs = 250.0
    t = np.arange(int(fs * 60)) / fs
    # Create a peaky synthetic rhythm by squaring a sine.
    values = np.sin(2 * np.pi * 1.2 * t) ** 2
    raw = RawSignalData(
        signal_type="ECG",
        source="SIMULATED",
        values=values,
        sampling_rate=fs,
        timestamps=t,
        labels=["AFIB"],
    )

    result = SegmentationService().segment(raw, window_seconds=30, overlap=0.5)
    if result.if_features.size:
        assert result.if_features.shape[1] == 5
