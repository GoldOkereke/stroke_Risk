"""Cardiac endpoints.

Prefix: /api/cardiac

These endpoints let the frontend:
- list available record IDs
- load MIT-BIH / MIMIC records and run the cardiac pipeline
- upload a CSV and run the pipeline
- run a simulated demo signal
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.cardiac_pipeline import run_cardiac_pipeline
from app.services.ingestion import IngestionService


router = APIRouter()


@router.get("/status")
def status() -> dict[str, str]:
    """Simple health check for the cardiac router."""
    return {"status": "ok"}


@router.get("/records")
def get_records() -> list[str]:
    """Return available MIT-BIH AF record IDs from the local RECORDS file."""
    settings = get_settings()
    records_file = settings.afdb_records_file
    if not records_file.exists():
        return []
    content = records_file.read_text(encoding="utf-8", errors="ignore").strip()
    return [line.strip() for line in content.splitlines() if line.strip()]


@router.post("/mitbih/{record_id}")
def load_mitbih(record_id: str) -> dict[str, Any]:
    """Load a MIT-BIH record by id and run the pipeline."""
    ingestion = IngestionService()
    raw = ingestion.load_mitbih(record_id)
    result = run_cardiac_pipeline(raw)
    return {
        "afibProbability": result.afib_probability,
        "prediction": "AFIB" if result.afib_probability >= 0.5 else "NORMAL",
        "confidence": result.confidence,
        "signalQuality": result.signal_quality,
        "waveform": result.filtered_samples,
        "cardiacIfScore": result.cardiac_if_score,
    }


@router.post("/mimic/{record_id}")
def load_mimic(record_id: str) -> dict[str, Any]:
    """Load a MIMIC record (WFDB) and run the pipeline."""
    ingestion = IngestionService()
    raw = ingestion.load_mimic(record_id)
    result = run_cardiac_pipeline(raw)
    return {
        "afibProbability": result.afib_probability,
        "prediction": "AFIB" if result.afib_probability >= 0.5 else "NORMAL",
        "confidence": result.confidence,
        "signalQuality": result.signal_quality,
        "waveform": result.filtered_samples,
        "cardiacIfScore": result.cardiac_if_score,
    }


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    signalType: str = Form("PPG"),
    sampleRate: float = Form(125.0),
    column: str = Form("value"),
) -> dict[str, Any]:
    """Accept a CSV upload and run the pipeline on the selected column."""
    ingestion = IngestionService()
    raw_bytes = await file.read()
    raw = ingestion.load_csv(raw_bytes, signal_type=signalType, sample_rate=sampleRate, column=column)
    result = run_cardiac_pipeline(raw)
    return {
        "afibProbability": result.afib_probability,
        "prediction": "AFIB" if result.afib_probability >= 0.5 else "NORMAL",
        "confidence": result.confidence,
        "signalQuality": result.signal_quality,
        "waveform": result.filtered_samples,
        "cardiacIfScore": result.cardiac_if_score,
    }


@router.post("/simulate")
def simulate_ppg(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate synthetic signal and run pipeline (demo use).

    Frontend sends JSON like: `{ "scenario": "normal" }`.
    We also accept optional `durationSeconds` and `signalType`.
    """
    scenario = str(payload.get("scenario", "normal"))
    duration_seconds = float(payload.get("durationSeconds", 30.0))
    signal_type = str(payload.get("signalType", "PPG"))

    ingestion = IngestionService()
    raw = ingestion.load_simulated(signal_type, duration_seconds=duration_seconds, scenario=scenario)
    result = run_cardiac_pipeline(raw)
    return {
        "afibProbability": result.afib_probability,
        "prediction": "AFIB" if result.afib_probability >= 0.5 else "NORMAL",
        "confidence": result.confidence,
        "signalQuality": result.signal_quality,
        "waveform": result.filtered_samples,
        "cardiacIfScore": result.cardiac_if_score,
    }
