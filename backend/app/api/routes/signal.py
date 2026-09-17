"""
app/api/routes/signal.py
=========================
FastAPI router for FR1 signal processing endpoints.

Endpoints:
  POST /api/signal/upload      — upload CSV file (ECG or PPG)
  POST /api/signal/mitbih      — load MIT-BIH AF record by ID
  POST /api/signal/mimic       — load MIMIC PERform AF record by ID
  POST /api/signal/simulate    — generate synthetic signal (FR6 demo)
  GET  /api/signal/status      — pipeline health check
  GET  /api/signal/records     — list available MIT-BIH AF record IDs

All endpoints run the full FR1 pipeline:
  Ingestion → Bandpass Filter → Segmentation → CNN predict → IF score
  and return a FilteredSignalResponse ready for the React frontend.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.ml.cnn_model import AFibCNN
from app.models.signal_schemas import (
    FilteredSignalResponse,
    RhythmLabel,
    SignalSource,
    SignalStatusResponse,
    SignalType,
)
from app.services.bandpass_filter import BandpassFilter
from app.services.ingestion import IngestionService
from app.services.isolation_forest import IFStream, IsolationForestService
from app.services.segmentation import SegmentationService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared service instances ───────────────────────────────────────────────
# Instantiated once at module load — shared across requests

_ingestion    = IngestionService()
_segmentation = SegmentationService(window_seconds=30)
_if_service   = IsolationForestService(user_id="default")

# CNN instances — one per signal type
_cnn_ecg: Optional[AFibCNN] = None
_cnn_ppg: Optional[AFibCNN] = None

# Bandpass filters — one per signal type
_filter_ecg = BandpassFilter(sample_rate=250.0, lowcut=0.5, highcut=40.0)
_filter_ppg = BandpassFilter(sample_rate=125.0, lowcut=0.5, highcut=8.0)


def _get_cnn(signal_type: SignalType) -> Optional[AFibCNN]:
    """Lazy-load CNN model — avoids loading TF at import time."""
    global _cnn_ecg, _cnn_ppg
    try:
        if signal_type == SignalType.ECG and _cnn_ecg is None:
            _cnn_ecg = AFibCNN(signal_type="ecg")
            try:
                _cnn_ecg.load()
                logger.info("ECG CNN loaded from weights.")
            except FileNotFoundError:
                logger.warning("No saved ECG CNN weights — using untrained model.")
        if signal_type == SignalType.PPG and _cnn_ppg is None:
            _cnn_ppg = AFibCNN(signal_type="ppg")
            try:
                _cnn_ppg.load()
                logger.info("PPG CNN loaded from weights.")
            except FileNotFoundError:
                logger.warning("No saved PPG CNN weights — using untrained model.")
        return _cnn_ecg if signal_type == SignalType.ECG else _cnn_ppg
    except Exception as e:
        logger.error("CNN load failed: %s", e)
        return None


def _get_filter(signal_type: SignalType) -> BandpassFilter:
    return _filter_ecg if signal_type == SignalType.ECG else _filter_ppg


# ── Shared pipeline runner ─────────────────────────────────────────────────

def _run_pipeline(raw_data, signal_type: SignalType) -> FilteredSignalResponse:
    """
    Run the full FR1 pipeline on a RawSignalData object.
    Ingestion already done — this runs filter → segment → CNN → IF.
    """
    notes = list(raw_data.notes)

    # ── Filter ─────────────────────────────────────────────────────────
    bp_filter = _get_filter(signal_type)
    try:
        filtered = bp_filter.apply(raw_data.signal)
        notes.append(f"Bandpass filter applied: {bp_filter.lowcut}–{bp_filter.highcut} Hz")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter failed: {e}")

    # ── Segment ────────────────────────────────────────────────────────
    try:
        seg_result = _segmentation.segment(filtered, raw_data)
        notes.extend(seg_result.notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # ── CNN prediction ─────────────────────────────────────────────────
    cnn_probs: Optional[np.ndarray] = None
    cnn = _get_cnn(signal_type)
    if cnn and seg_result.cnn_input.shape[0] > 0:
        try:
            cnn_probs = cnn.predict(seg_result.cnn_input)
            mean_prob = float(cnn_probs.mean())
            notes.append(f"CNN mean AFib probability: {mean_prob:.3f}")
        except Exception as e:
            logger.warning("CNN prediction failed: %s", e)
            notes.append(f"CNN prediction skipped: {e}")
    else:
        notes.append("CNN skipped — no valid segments or model not loaded.")

    # ── Isolation Forest scoring ───────────────────────────────────────
    if seg_result.isolation_forest_features.shape[0] > 0:
        try:
            # Add CNN prob as 6th feature if available
            if_features = seg_result.isolation_forest_features  # (N, 5)
            if cnn_probs is not None:
                cnn_col = cnn_probs.reshape(-1, 1)
                if_features = np.hstack([if_features, cnn_col])  # (N, 6)

            # Use mean feature vector across all valid segments as the observation
            mean_features = if_features.mean(axis=0)
            _if_service.add_baseline(IFStream.CARDIAC, mean_features)

            if _if_service.baseline_sample_count(IFStream.CARDIAC) >= 10:
                _if_service.train(IFStream.CARDIAC)

            notes.append(
                f"IF cardiac baseline: "
                f"{_if_service.baseline_sample_count(IFStream.CARDIAC)} samples"
            )
        except Exception as e:
            logger.warning("IF scoring failed: %s", e)
            notes.append(f"IF scoring skipped: {e}")

    # ── Build response ─────────────────────────────────────────────────
    segment_schemas = [s.to_schema() for s in seg_result.segments]

    # Attach CNN probs to segments if available
    if cnn_probs is not None:
        valid_segs = [s for s in seg_result.segments if s.is_valid]
        for i, (seg_schema, prob) in enumerate(
            zip([s for s in segment_schemas if s.is_valid], cnn_probs)
        ):
            seg_schema.metadata = {"afib_cnn_probability": round(float(prob), 4)} \
                if hasattr(seg_schema, "metadata") else {}

    afib_count   = sum(1 for s in segment_schemas if s.rhythm_label == RhythmLabel.AFIB and s.is_valid)
    normal_count = sum(1 for s in segment_schemas if s.rhythm_label == RhythmLabel.NORMAL and s.is_valid)
    valid_count  = sum(1 for s in segment_schemas if s.is_valid)

    return FilteredSignalResponse(
        record_id            = raw_data.record_id,
        signal_type          = signal_type,
        source               = raw_data.source,
        sample_rate          = raw_data.sample_rate,
        total_samples        = len(raw_data.signal),
        duration_seconds     = raw_data.duration_seconds,
        raw_signal           = raw_data.signal.tolist(),
        filtered_signal      = filtered.tolist(),
        segments             = segment_schemas,
        total_segments       = len(segment_schemas),
        valid_segment_count  = valid_count,
        afib_segment_count   = afib_count,
        normal_segment_count = normal_count,
        isolation_forest_features = (
            seg_result.isolation_forest_features.tolist()
            if seg_result.isolation_forest_features.shape[0] > 0 else None
        ),
        processing_notes     = notes,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/status", response_model=SignalStatusResponse)
async def signal_status():
    """Health check — confirms pipeline is ready."""
    return SignalStatusResponse()


@router.get("/records")
async def list_mitbih_records():
    """Return available MIT-BIH AF record IDs."""
    # Standard MIT-BIH AF records
    records = [
        "04015", "04043", "04048", "04126", "04746",
        "04908", "04936", "05091", "05121", "05261",
        "06426", "06453", "06995", "07162", "07859",
        "07879", "07910", "08215", "08219", "08378",
        "08405", "08434", "08455",
    ]
    return {
        "dataset": "MIT-BIH Atrial Fibrillation Database",
        "sample_rate_hz": 250,
        "record_count": len(records),
        "record_ids": records,
    }


@router.post("/mitbih", response_model=FilteredSignalResponse)
async def process_mitbih(
    record_id: str = Query(..., description="MIT-BIH record ID e.g. '04015'"),
    lead: int      = Query(default=0, ge=0, le=1, description="ECG lead index (0 or 1)"),
):
    """
    Load and process a MIT-BIH AF ECG record.
    Downloads from PhysioNet if not cached locally.
    """
    logger.info("MIT-BIH request | record=%s | lead=%d", record_id, lead)
    try:
        raw_data = _ingestion.load_mitbih(record_id=record_id, lead=lead)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return _run_pipeline(raw_data, SignalType.ECG)


@router.post("/mimic", response_model=FilteredSignalResponse)
async def process_mimic(
    record_id: str = Query(..., description="MIMIC PERform AF record ID"),
    lead: int      = Query(default=0, ge=0, le=1, description="PPG channel index"),
):
    """
    Load and process a MIMIC PERform AF PPG record.
    Downloads from PhysioNet if not cached locally.
    """
    logger.info("MIMIC request | record=%s | lead=%d", record_id, lead)
    try:
        raw_data = _ingestion.load_mimic(record_id=record_id, lead=lead)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return _run_pipeline(raw_data, SignalType.PPG)


@router.post("/upload", response_model=FilteredSignalResponse)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file containing ECG or PPG signal"),
    signal_type: SignalType = Form(..., description="ecg or ppg"),
    sample_rate: float      = Form(..., description="Sampling rate in Hz"),
    column: Optional[str]   = Form(default=None, description="Column name to extract"),
):
    """
    Upload a CSV file and run it through the full signal pipeline.
    The CSV must contain at least one numeric column with signal samples.
    """
    logger.info(
        "CSV upload | filename=%s | type=%s | fs=%.1f",
        file.filename, signal_type.value, sample_rate,
    )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    from app.core.config import settings
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE // 1024 // 1024} MB.",
        )

    try:
        raw_data = _ingestion.load_csv(
            file_bytes   = content,
            signal_type  = signal_type,
            sample_rate  = sample_rate,
            column       = column,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return _run_pipeline(raw_data, signal_type)


@router.post("/simulate", response_model=FilteredSignalResponse)
async def simulate_signal(
    signal_type:      SignalType = Query(default=SignalType.ECG),
    duration_seconds: float      = Query(default=60.0, ge=10.0, le=600.0),
    scenario:         str        = Query(
        default="pre_tia",
        description="normal | afib | pre_tia",
    ),
):
    """
    Generate a synthetic ECG/PPG signal for FR6 demo mode.

    Scenarios:
      normal  — clean sinus rhythm
      afib    — simulated atrial fibrillation
      pre_tia — normal baseline transitioning to AFib (the demo scenario)
    """
    logger.info(
        "Simulate request | type=%s | duration=%.0fs | scenario=%s",
        signal_type.value, duration_seconds, scenario,
    )

    valid_scenarios = ("normal", "afib", "pre_tia")
    if scenario not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario '{scenario}'. Choose: {valid_scenarios}",
        )

    try:
        raw_data = _ingestion.load_simulated(
            signal_type      = signal_type,
            duration_seconds = duration_seconds,
            scenario         = scenario,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _run_pipeline(raw_data, signal_type)