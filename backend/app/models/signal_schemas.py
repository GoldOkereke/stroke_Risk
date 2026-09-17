"""
app/models/signal_schemas.py
==============================
Pydantic schemas for the FR1 signal processing pipeline.

Covers:
  - ECG input  : MIT-BIH Atrial Fibrillation dataset (250 Hz, WFDB)
  - PPG input  : MIMIC PERform AF dataset (125 Hz, WFDB)
  - CSV upload : user-supplied files
  - Simulated  : FR6 demo mode synthetic signals
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, validator


# ── Enums ──────────────────────────────────────────────────────────────────

class SignalType(str, Enum):
    ECG = "ecg"   # MIT-BIH AF  — 250 Hz, passband 0.5–40 Hz
    PPG = "ppg"   # MIMIC PERform — 125 Hz, passband 0.5–8 Hz

class SignalSource(str, Enum):
    MITBIH    = "mitbih"     # MIT-BIH AF via wfdb
    MIMIC     = "mimic"      # MIMIC PERform AF via wfdb
    CSV       = "csv"        # user CSV upload
    SIMULATED = "simulated"  # FR6 demo mode

class RhythmLabel(str, Enum):
    AFIB    = "AFIB"
    NORMAL  = "NORMAL"
    FLUTTER = "AFL"
    UNKNOWN = "UNKNOWN"

class AlertTier(str, Enum):
    NORMAL   = "NORMAL"
    ADVISORY = "ADVISORY"
    CRITICAL = "CRITICAL"


# ── Request schemas ────────────────────────────────────────────────────────

class SignalUploadRequest(BaseModel):
    """
    Sent by client when submitting a signal for filtering.
    For CSV file uploads use multipart/form-data in the router.
    For MIT-BIH / MIMIC supply record_id.
    For demo mode supply raw_signal array.
    """
    signal_type: SignalType = Field(
        ...,
        description="ecg (MIT-BIH, 250 Hz) or ppg (MIMIC, 125 Hz)",
    )
    source: SignalSource = Field(
        ...,
        description="Where the signal comes from",
    )
    record_id: Optional[str] = Field(
        default=None,
        description="WFDB record ID — e.g. '04015' for MIT-BIH, 'mimic_af_001' for MIMIC",
    )
    lead: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Lead index (0 or 1). MIT-BIH AF has 2 ECG leads.",
    )
    raw_signal: Optional[list[float]] = Field(
        default=None,
        description="Raw samples as float array. Required for source=simulated.",
    )
    sample_rate: Optional[float] = Field(
        default=None,
        description="Override sample rate. Defaults: ECG=250, PPG=125.",
    )

    @validator("record_id", always=True)
    def record_required_for_wfdb(cls, v, values):
        source = values.get("source")
        if source in (SignalSource.MITBIH, SignalSource.MIMIC) and not v:
            raise ValueError(f"record_id is required when source is '{source}'.")
        return v

    @validator("raw_signal", always=True)
    def signal_required_for_simulated(cls, v, values):
        if values.get("source") == SignalSource.SIMULATED and not v:
            raise ValueError("raw_signal array is required when source is 'simulated'.")
        return v

    @property
    def resolved_sample_rate(self) -> float:
        if self.sample_rate:
            return self.sample_rate
        return 250.0 if self.signal_type == SignalType.ECG else 125.0


# ── Segment schema ─────────────────────────────────────────────────────────

class SegmentSchema(BaseModel):
    """
    One 30-second windowed segment of a filtered signal.
    Shaped and labelled ready for FR2 (1D CNN) and FR3 (Isolation Forest).
    """
    index: int              = Field(..., description="Segment index (0-based)")
    start_sample: int       = Field(..., description="Start sample in full signal")
    end_sample: int         = Field(..., description="End sample in full signal")
    duration_seconds: float = Field(..., description="Segment duration in seconds")
    signal_type: SignalType = Field(..., description="ECG or PPG")
    rhythm_label: RhythmLabel = Field(..., description="AFIB / NORMAL from .atr annotation")
    quality_score: float    = Field(..., ge=0.0, le=1.0, description="Signal quality 0–1")
    is_valid: bool          = Field(..., description="False if noisy, clipped, or flat")
    data: list[float]       = Field(..., description="Filtered samples for this segment")

    # Features extracted for Isolation Forest (FR3) and CNN (FR2)
    hrv_rmssd: Optional[float]  = Field(None, description="HRV RMSSD from RR intervals")
    hrv_sdnn: Optional[float]   = Field(None, description="HRV SDNN from RR intervals")
    rr_irregularity: Optional[float] = Field(None, description="RR interval irregularity score")


# ── Response schemas ───────────────────────────────────────────────────────

class FilteredSignalResponse(BaseModel):
    """
    Returned after ingestion + filtering.
    Contains full arrays + labelled segments ready for FR2 and FR3.
    """
    record_id: Optional[str]    = Field(None, description="WFDB record ID if applicable")
    signal_type: SignalType      = Field(..., description="ECG or PPG")
    source: SignalSource         = Field(..., description="Origin of the signal")
    sample_rate: float           = Field(..., description="Sampling frequency in Hz")
    total_samples: int           = Field(..., description="Total samples in signal")
    duration_seconds: float      = Field(..., description="Total duration in seconds")

    raw_signal: list[float]       = Field(..., description="Unfiltered samples")
    filtered_signal: list[float]  = Field(..., description="After bandpass + notch filter")

    segments: list[SegmentSchema] = Field(..., description="30-second labelled windows")
    total_segments: int           = Field(..., description="Total segment count")
    valid_segment_count: int      = Field(..., description="Usable segments")
    afib_segment_count: int       = Field(..., description="Segments labelled AFIB")
    normal_segment_count: int     = Field(..., description="Segments labelled NORMAL")

    # Passed directly to FR3 Isolation Forest
    isolation_forest_features: Optional[list[list[float]]] = Field(
        None,
        description="Feature matrix (N_segments × N_features) for Isolation Forest input",
    )

    processing_notes: list[str]  = Field(
        default_factory=list,
        description="Pipeline warnings or info messages",
    )


class SignalStatusResponse(BaseModel):
    """GET /api/signal/status — pipeline health check."""
    status: str         = Field(default="ok")
    ecg_sample_rate: float = Field(default=250.0)
    ppg_sample_rate: float = Field(default=125.0)
    ecg_filter_band: str   = Field(default="0.5–40.0 Hz")
    ppg_filter_band: str   = Field(default="0.5–8.0 Hz")
    notch_freq: str        = Field(default="60 Hz")
    segment_length_s: int  = Field(default=30)
    datasets: list[str]    = Field(
        default=["MIT-BIH AF (ECG, 250Hz)", "MIMIC PERform AF (PPG, 125Hz)"]
    )