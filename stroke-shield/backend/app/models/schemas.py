from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# Soft enums (open-ended strings) with canonical values documented for clarity.
SignalType = str
"""Canonical values: ECG, PPG"""

SignalSource = str
"""Canonical values: MIT_BIH_AF, MIMIC_PPG, SIMULATED, REAL_TIME"""

RhythmLabel = str
"""Canonical values: NORMAL, AFIB"""

AlertTier = str
"""Canonical values: NORMAL, ADVISORY, CRITICAL"""


class SignalUploadRequest(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    signal_type: SignalType = Field(..., description="Type of signal (e.g., ECG, PPG).")
    signal_source: SignalSource = Field(
        ..., description="Source of the signal (e.g., MIT_BIH_AF, REAL_TIME)."
    )
    filename: Optional[str] = Field(None, description="Original filename of the upload.")
    sample_rate_hz: Optional[float] = Field(
        None, description="Sampling rate in Hz if known."
    )
    duration_seconds: Optional[float] = Field(
        None, description="Total duration of the recording in seconds."
    )
    metadata: Optional[dict[str, str]] = Field(
        None, description="Additional metadata supplied by the client."
    )


class SegmentSchema(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    hrv_rmssd: Optional[float] = Field(None, description="RMSSD heart rate variability.")
    hrv_sdnn: Optional[float] = Field(None, description="SDNN heart rate variability.")
    rr_irregularity: Optional[float] = Field(
        None, description="RR interval irregularity metric."
    )
    samples: list[float] = Field(..., description="Raw signal samples for the segment.")


class FilteredSignalResponse(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    signal_type: SignalType = Field(..., description="Type of signal (e.g., ECG, PPG).")
    signal_source: SignalSource = Field(
        ..., description="Source of the signal (e.g., MIT_BIH_AF, REAL_TIME)."
    )
    filtered_samples: list[float] = Field(
        ..., description="Filtered signal samples."
    )
    segment: Optional[SegmentSchema] = Field(
        None, description="Optional segment-level metrics."
    )
    metadata: Optional[dict[str, str]] = Field(
        None, description="Additional metadata for the response."
    )


class SignalStatusResponse(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    status: str = Field(..., description="System status indicator.")
    message: Optional[str] = Field(None, description="Human-readable status message.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Time of status report."
    )


class FaceFeatures(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    scores: dict[str, float] = Field(
        ..., description="Asymmetry scores keyed by feature name."
    )


class FaceAnalysisResult(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    features: FaceFeatures = Field(..., description="Face asymmetry features.")
    aggregated_score: float = Field(
        ..., description="Overall face asymmetry score."
    )
    anomaly_score: Optional[float] = Field(
        None,
        description="Optional anomaly score from an Isolation Forest model (higher means more anomalous).",
    )
    quality: Optional[float] = Field(
        None,
        description="Optional quality score (0-1) based on brightness and sharpness.",
    )
    face_detected: Optional[bool] = Field(
        None,
        description="Whether a face was detected in the image.",
    )


class DysarthriaFeatures(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    scores: dict[str, float] = Field(
        ..., description="Dysarthria feature scores (TORGO)."
    )


class ParkinsonsFeatures(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    scores: dict[str, float] = Field(
        ..., description="Parkinsons feature scores (UCI)."
    )


class VoiceFeatures(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    dysarthria: Optional[DysarthriaFeatures] = Field(
        None, description="Optional dysarthria feature set."
    )
    parkinsons: Optional[ParkinsonsFeatures] = Field(
        None, description="Optional Parkinsons feature set."
    )


class VoiceAnalysisResult(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    features: VoiceFeatures = Field(..., description="Voice feature groups.")
    scores: dict[str, float] = Field(
        ..., description="Aggregated voice analysis scores."
    )


class AnomalyResult(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    score: float = Field(..., description="Anomaly score.")
    confidence: float = Field(..., description="Confidence in anomaly score.")
    contributing_features: dict[str, float] = Field(
        ..., description="Feature contributions to anomaly score."
    )


class FusionResult(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True}

    risk_score: float = Field(..., description="Overall risk score (0-100).")
    alert_tier: AlertTier = Field(..., description="Alert tier label.")
    confidence: float = Field(..., description="Confidence in fusion result.")
    contributing_factors: dict[str, float] = Field(
        ..., description="Weighted contributions to risk score."
    )
    recommendation: str = Field(..., description="Recommendation text.")

# from __future__ import annotations
# This allows modern type hints (like list[float]) without issues across Python versions.

# from datetime import datetime
# Used for timestamps in status responses.

# from typing import Optional
# Used to say a field can be missing or None.

# from pydantic import BaseModel, Field
# BaseModel is the base class for all Pydantic models. Field adds descriptions and defaults.

# Soft enums:
# SignalType = str
# SignalSource = str
# RhythmLabel = str
# AlertTier = str
# These are intentionally just strings so any value is accepted. The docstrings under each one list the “canonical” values your system expects, but they don’t block new strings.

# class SignalUploadRequest(BaseModel)
# Represents metadata when a user uploads a signal file.
# Fields:

# signal_type: which kind of signal (ECG, PPG, or any custom string).
# signal_source: where it came from (MIT_BIH_AF, REAL_TIME, etc.).
# filename: optional original filename.
# sample_rate_hz: optional sampling rate.
# duration_seconds: optional duration.
# metadata: optional key/value info from client.
# class SegmentSchema(BaseModel)
# Represents one processed segment of signal data.
# Fields:

# hrv_rmssd: variability metric.
# hrv_sdnn: another variability metric.
# rr_irregularity: irregularity measure.
# samples: raw samples for the segment.
# class FilteredSignalResponse(BaseModel)
# Represents a filtered signal response returned by the backend.
# Fields:

# signal_type: which kind of signal.
# signal_source: where it came from.
# filtered_samples: list of filtered values.
# segment: optional SegmentSchema with metrics.
# metadata: optional extra info.
# class SignalStatusResponse(BaseModel)
# Represents system status.
# Fields:

# status: string like "ok" or "error".
# message: optional details.
# timestamp: time the status was generated.
# class FaceFeatures(BaseModel)
# Represents face asymmetry scores as a flexible map.
# Fields:

# scores: dict of feature name to score.
# class FaceAnalysisResult(BaseModel)
# Represents a face analysis outcome.
# Fields:

# features: a FaceFeatures object.
# aggregated_score: overall asymmetry score.
# class DysarthriaFeatures(BaseModel)
# Represents dysarthria features from TORGO.
# Fields:

# scores: dict of feature name to score.
# class ParkinsonsFeatures(BaseModel)
# Represents Parkinsons features from UCI.
# Fields:

# scores: dict of feature name to score.
# class VoiceFeatures(BaseModel)
# Represents a union of voice feature sets.
# Fields:

# dysarthria: optional DysarthriaFeatures.
# parkinsons: optional ParkinsonsFeatures.
# class VoiceAnalysisResult(BaseModel)
# Represents overall voice analysis.
# Fields:

# features: a VoiceFeatures bundle.
# scores: aggregate scores as a dict.
# class AnomalyResult(BaseModel)
# Represents anomaly detection output.
# Fields:

# score: anomaly score.
# confidence: confidence in the score.
# contributing_features: dict of feature weights.
# class FusionResult(BaseModel)
# Represents fused multi‑modal risk output.
# Fields:

# risk_score: numeric risk (0–100).
# alert_tier: string tier (open‑ended).
# confidence: confidence value.
# contributing_factors: dict of factor weights.
# recommendation: human‑readable recommendation.
# Tests added
# I added c:\Users\dell\Desktop\fyp\real_fyp\stroke-shield\backend\tests\test_schemas.py to validate:

# Arbitrary string values are accepted for “enums.”
# Segment metrics accept numeric values and samples.
# FusionResult serializes correctly.
# I didn’t run tests because dependencies are likely not installed here. If you want, I can run pytest after installing requirements.
