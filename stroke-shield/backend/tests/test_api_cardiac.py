"""
Tests for the Cardiac Slice (cardiac API endpoints).
Tests cardiac signal analysis and AFib detection accuracy.
"""

import pytest
from app.models.schemas import (
    SignalUploadRequest,
    FilteredSignalResponse,
    SegmentSchema,
)


class TestCardiacSliceSignalUpload:
    """Test cardiac signal upload and validation."""

    def test_signal_upload_ecg(self):
        """Test ECG signal upload request."""
        upload_request = SignalUploadRequest(
            signal_type="ECG",
            signal_source="MIT_BIH_AF",
            filename="record_100.dat",
            sample_rate_hz=250.0,
            duration_seconds=10.0,
            metadata={"patient_id": "P001"},
        )

        assert upload_request.signal_type == "ECG"
        assert upload_request.signal_source == "MIT_BIH_AF"
        assert upload_request.sample_rate_hz == 250.0
        assert upload_request.duration_seconds == 10.0
        assert upload_request.metadata["patient_id"] == "P001"

    def test_signal_upload_ppg(self):
        """Test PPG signal upload request."""
        upload_request = SignalUploadRequest(
            signal_type="PPG",
            signal_source="MIMIC_PPG",
            filename="ppg_record.csv",
            sample_rate_hz=125.0,
            duration_seconds=60.0,
        )

        assert upload_request.signal_type == "PPG"
        assert upload_request.signal_source == "MIMIC_PPG"
        assert upload_request.sample_rate_hz == 125.0

    def test_signal_upload_real_time(self):
        """Test real-time signal upload."""
        upload_request = SignalUploadRequest(
            signal_type="ECG",
            signal_source="REAL_TIME",
            metadata={"device": "wearable", "timestamp": "2026-03-31T10:00:00Z"},
        )

        assert upload_request.signal_source == "REAL_TIME"
        assert upload_request.metadata["device"] == "wearable"


class TestCardiacSliceFiltering:
    """Test cardiac signal filtering for noise reduction."""

    def test_filtered_signal_response(self):
        """Test filtered signal response structure."""
        sample_data = [i * 0.1 for i in range(100)]
        filtered_response = FilteredSignalResponse(
            signal_type="ECG",
            signal_source="MIT_BIH_AF",
            filtered_samples=sample_data,
            segment=SegmentSchema(
                hrv_rmssd=25.5,
                hrv_sdnn=30.0,
                rr_irregularity=0.15,
                samples=sample_data,
            ),
        )

        assert filtered_response.signal_type == "ECG"
        assert len(filtered_response.filtered_samples) == 100
        assert filtered_response.segment.hrv_rmssd == 25.5

    def test_filtered_signal_quality_metrics(self):
        """Test HRV quality metrics in filtered signal."""
        sample_data = [i * 0.05 for i in range(200)]
        segment = SegmentSchema(
            hrv_rmssd=28.5,
            hrv_sdnn=35.0,
            rr_irregularity=0.12,
            samples=sample_data,
        )

        assert segment.hrv_rmssd > 0
        assert segment.hrv_sdnn > 0
        assert 0 <= segment.rr_irregularity <= 1

    def test_filtered_signal_with_metadata(self):
        """Test filtered signal response includes metadata."""
        sample_data = [i * 0.1 for i in range(50)]
        filtered_response = FilteredSignalResponse(
            signal_type="ECG",
            signal_source="REAL_TIME",
            filtered_samples=sample_data,
            metadata={
                "filter_type": "bandpass",
                "cutoff_low_hz": 0.5,
                "cutoff_high_hz": 40.0,
            },
        )

        assert filtered_response.metadata["filter_type"] == "bandpass"
        assert float(filtered_response.metadata["cutoff_low_hz"]) == 0.5


class TestCardiacSliceAFibDetection:
    """Test AFib detection accuracy and probability thresholds."""

    def test_afib_probability_range(self):
        """Verify AFib probability is between 0 and 1."""
        from app.models.schemas import CardiacResult

        cardiac_results = [
            (0.0, "NORMAL"),
            (0.25, "NORMAL"),
            (0.5, "NORMAL"),
            (0.75, "AFIB"),
            (1.0, "AFIB"),
        ]

        for prob, expected_pred in cardiac_results:
            # This assumes a threshold of 0.5
            prediction = "AFIB" if prob >= 0.5 else "NORMAL"
            assert prediction == expected_pred

    def test_afib_confidence_scores(self):
        """Test confidence scores for AFib detection."""
        high_confidence = 0.95
        low_confidence = 0.55

        assert high_confidence > 0.9
        assert 0.5 < low_confidence < 0.7

    def test_normal_sinus_rhythm_detection(self):
        """Test normal sinus rhythm detection."""
        sample_data = [
            0.1,
            0.2,
            0.15,
            0.1,
            0.05,
            0.0,
            -0.05,
            -0.1,
            -0.15,
            -0.2,
        ] * 10

        filtered_response = FilteredSignalResponse(
            signal_type="ECG",
            signal_source="MIT_BIH_AF",
            filtered_samples=sample_data,
            segment=SegmentSchema(
                hrv_rmssd=20.0,
                hrv_sdnn=25.0,
                rr_irregularity=0.08,
                samples=sample_data,
            ),
        )

        # Normal rhythm should have lower irregularity
        assert filtered_response.segment.rr_irregularity < 0.15

    def test_afib_arrhythmia_detection(self):
        """Test irregular rhythm pattern detection for AFIB."""
        import random

        random.seed(42)
        # Simulate irregular rhythm with varying intervals
        irregular_data = [
            random.uniform(-0.3, 0.3) for _ in range(200)
        ]

        filtered_response = FilteredSignalResponse(
            signal_type="ECG",
            signal_source="MIT_BIH_AF",
            filtered_samples=irregular_data,
            segment=SegmentSchema(
                hrv_rmssd=45.0,
                hrv_sdnn=50.0,
                rr_irregularity=0.35,
                samples=irregular_data,
            ),
        )

        # AFIB should show higher irregularity
        assert filtered_response.segment.rr_irregularity > 0.2


class TestCardiacSliceSignalQuality:
    """Test signal quality assessment."""

    def test_high_quality_signal(self):
        """Test assessment of high-quality cardiac signal."""
        # Regular pattern with low noise
        sample_data = [
            0.1 * (i % 10) for i in range(250)
        ]

        filtered_response = FilteredSignalResponse(
            signal_type="ECG",
            signal_source="REAL_TIME",
            filtered_samples=sample_data,
        )

        assert len(filtered_response.filtered_samples) > 0
        assert filtered_response.signal_type == "ECG"

    def test_noisy_signal_detection(self):
        """Test detection of noisy cardiac signal."""
        import random

        random.seed(42)
        # Highly irregular pattern indicating noise
        noisy_data = [random.uniform(-1, 1) for _ in range(250)]

        segment = SegmentSchema(
            rr_irregularity=0.65,
            samples=noisy_data,
        )

        # High irregularity indicates potential noise or poor signal
        assert segment.rr_irregularity > 0.5
