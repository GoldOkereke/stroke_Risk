"""
app/api/routes/neurological.py
================================
FastAPI router for FR3 neurological endpoints.

Replaces the placeholder endpoints with real face + voice analysis.

Endpoints:
  POST /api/neurological/face/analyze       — analyze single frame for asymmetry
  POST /api/neurological/face/session       — analyze multi-frame webcam session
  POST /api/neurological/face/baseline      — add frame(s) to IF baseline
  POST /api/neurological/voice/analyze      — analyze audio clip
  POST /api/neurological/voice/baseline     — add audio to IF baseline
  POST /api/neurological/voice/speech-test  — guided speech prompt test
  GET  /api/neurological/baseline/status    — how ready each IF stream is
  GET  /api/neurological/status             — module health check
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.face_analyzer import FaceAnalyzer, FaceAnalysisResult
from app.services.voice_analyzer import VoiceAnalyzer, VoiceAnalysisResult
from app.services.isolation_forest import IFStream, IsolationForestService

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared service instances ───────────────────────────────────────────────

_face_analyzer  = FaceAnalyzer()
_voice_analyzer = VoiceAnalyzer(sample_rate=16000)
_if_service     = IsolationForestService(user_id="default")


# ── Response schemas ───────────────────────────────────────────────────────

class FaceAnalysisResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    landmarks_detected: bool
    alert_indicators: list[str]
    features: Optional[dict]
    baseline_sample_count: int
    baseline_ready: bool
    notes: list[str]


class VoiceAnalysisResponse(BaseModel):
    dysarthria_score: float
    parkinsons_score: float
    is_anomaly: bool
    confidence: float
    alert_indicators: list[str]
    features: Optional[dict]
    dysarthria_baseline_count: int
    parkinsons_baseline_count: int
    baseline_ready: bool
    notes: list[str]


class BaselineStatusResponse(BaseModel):
    streams: dict[str, dict]
    overall_ready: bool
    recommendation: str


class SpeechTestResponse(BaseModel):
    prompt: str
    dysarthria_score: float
    parkinsons_score: float
    is_anomaly: bool
    confidence: float
    alert_indicators: list[str]
    pass_fail: str      # "PASS" | "FAIL" | "UNCERTAIN"
    notes: list[str]


# ── Speech test prompts ────────────────────────────────────────────────────
# Designed to stress articulation (detect slurring) and sustained phonation
# (detect tremor/breathiness). Used in FR4 active neuro tests.

SPEECH_PROMPTS = [
    "Please say: 'The quick brown fox jumps over the lazy dog'",
    "Please say: 'British Constitution' three times",
    "Please say: 'Methodist Episcopal' slowly and clearly",
    "Please sustain the sound 'aaah' for five seconds",
    "Please count from 1 to 10 at a normal pace",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _face_result_to_response(
    result: FaceAnalysisResult,
    baseline_count: int,
) -> FaceAnalysisResponse:
    features_dict = None
    if result.features:
        features_dict = {
            "mouth_droop":       round(result.features.mouth_droop,       4),
            "eye_droop":         round(result.features.eye_droop,         4),
            "brow_asymmetry":    round(result.features.brow_asymmetry,    4),
            "nasolabial_fold":   round(result.features.nasolabial_fold,   4),
            "smile_symmetry":    round(result.features.smile_symmetry,    4),
            "overall_asymmetry": round(result.features.overall_asymmetry, 4),
            "frame_quality":     round(result.features.frame_quality,     4),
        }

    return FaceAnalysisResponse(
        anomaly_score         = result.anomaly_score,
        is_anomaly            = result.is_anomaly,
        confidence            = result.confidence,
        landmarks_detected    = result.landmarks_detected,
        alert_indicators      = result.alert_indicators,
        features              = features_dict,
        baseline_sample_count = baseline_count,
        baseline_ready        = baseline_count >= 10,
        notes                 = result.notes,
    )


def _voice_result_to_response(
    result: VoiceAnalysisResult,
    d_baseline: int,
    p_baseline: int,
) -> VoiceAnalysisResponse:
    features_dict = None
    if result.features:
        features_dict = {
            "dysarthria": {
                "speech_tempo":      result.features.dysarthria.speech_tempo,
                "pause_ratio":       result.features.dysarthria.pause_ratio,
                "articulation_rate": result.features.dysarthria.articulation_rate,
                "spectral_centroid": result.features.dysarthria.spectral_centroid,
                "mfcc_mean_1":       result.features.dysarthria.mfcc_mean_1,
                "mfcc_mean_2":       result.features.dysarthria.mfcc_mean_2,
                "mfcc_mean_3":       result.features.dysarthria.mfcc_mean_3,
            },
            "parkinsons": {
                "jitter_pct":  result.features.parkinsons.jitter_pct,
                "shimmer_db":  result.features.parkinsons.shimmer_db,
                "hnr":         result.features.parkinsons.hnr,
                "pitch_mean":  result.features.parkinsons.pitch_mean,
                "pitch_std":   result.features.parkinsons.pitch_std,
                "rpde":        result.features.parkinsons.rpde,
                "dfa":         result.features.parkinsons.dfa,
            },
        }

    return VoiceAnalysisResponse(
        dysarthria_score         = result.dysarthria_score,
        parkinsons_score         = result.parkinsons_score,
        is_anomaly               = result.is_anomaly,
        confidence               = result.confidence,
        alert_indicators         = result.alert_indicators,
        features                 = features_dict,
        dysarthria_baseline_count= d_baseline,
        parkinsons_baseline_count= p_baseline,
        baseline_ready           = d_baseline >= 10 and p_baseline >= 10,
        notes                    = result.notes,
    )


# ── Face endpoints ─────────────────────────────────────────────────────────

@router.post("/face/analyze", response_model=FaceAnalysisResponse)
async def analyze_face(
    file: UploadFile = File(..., description="Image frame (JPEG/PNG) from webcam"),
):
    """
    Analyze a single webcam frame for facial asymmetry.
    Returns anomaly score + feature values + alert indicators.
    """
    import cv2

    logger.info("Face analyze request | filename=%s", file.filename)

    content = await file.read()
    nparr   = np.frombuffer(content, np.uint8)
    frame   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    result = _face_analyzer.analyze(frame)
    baseline_count = _if_service.baseline_sample_count(IFStream.FACIAL)

    # If IF is trained, upgrade score with personal baseline
    if (
        result.features
        and result.features.is_valid
        and _if_service._models[IFStream.FACIAL].is_trained
    ):
        if_result = _if_service.score(
            IFStream.FACIAL, result.features.to_array()
        )
        result.anomaly_score = if_result.anomaly_score
        result.is_anomaly    = if_result.is_anomaly
        result.confidence    = if_result.confidence

    return _face_result_to_response(result, baseline_count)


@router.post("/face/session", response_model=FaceAnalysisResponse)
async def analyze_face_session(
    files: list[UploadFile] = File(..., description="Multiple frames from a webcam session"),
):
    """
    Analyze multiple frames and aggregate into a single stable result.
    Use for 5–10 second webcam clips (send N frames from frontend).
    Better than single-frame analysis — reduces noise.
    """
    import cv2

    logger.info("Face session request | %d frames", len(files))

    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="Send at least 3 frames for a reliable session analysis."
        )

    frames  = []
    results = []

    for f in files:
        content = await f.read()
        nparr   = np.frombuffer(content, np.uint8)
        frame   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
            results.append(_face_analyzer.analyze(frame))

    if not results:
        raise HTTPException(status_code=422, detail="No valid frames could be decoded.")

    # Aggregate features across frames
    aggregated = _face_analyzer.aggregate_features(results)

    # Use last individual result as base, override score with aggregated IF
    final_result = results[-1]
    baseline_count = _if_service.baseline_sample_count(IFStream.FACIAL)

    if aggregated is not None and _if_service._models[IFStream.FACIAL].is_trained:
        if_result = _if_service.score(IFStream.FACIAL, aggregated)
        final_result.anomaly_score = if_result.anomaly_score
        final_result.is_anomaly    = if_result.is_anomaly
        final_result.confidence    = if_result.confidence
        final_result.notes.append(
            f"Score aggregated from {len(results)} valid frames."
        )

    return _face_result_to_response(final_result, baseline_count)


@router.post("/face/baseline")
async def add_face_baseline(
    file: UploadFile = File(..., description="Baseline frame from user's normal session"),
):
    """
    Add a frame to the user's personal facial baseline.
    Call this during onboarding (10+ frames to train the IF model).
    """
    import cv2

    content = await file.read()
    nparr   = np.frombuffer(content, np.uint8)
    frame   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    result = _face_analyzer.analyze(frame)

    if not result.features or not result.features.is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract valid features: {result.notes}",
        )

    _if_service.add_baseline(IFStream.FACIAL, result.features.to_array())
    n = _if_service.baseline_sample_count(IFStream.FACIAL)

    # Auto-train once we have enough
    if n >= 10 and not _if_service._models[IFStream.FACIAL].is_trained:
        _if_service.train(IFStream.FACIAL)
        logger.info("Facial IF model trained on %d baseline samples.", n)

    return {
        "status":          "ok",
        "baseline_samples": n,
        "baseline_ready":   n >= 10,
        "message": (
            f"Baseline updated ({n} samples). "
            + ("IF model trained — personal baseline active." if n >= 10
               else f"{10 - n} more samples needed to activate personal baseline.")
        ),
    }


# ── Voice endpoints ────────────────────────────────────────────────────────

@router.post("/voice/analyze", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    file: UploadFile = File(..., description="Audio clip WAV/MP3 from microphone"),
):
    """
    Analyze an audio clip for dysarthria (slurring) and Parkinson's markers.
    Returns scores for both TORGO and UCI streams.
    """
    logger.info("Voice analyze request | filename=%s", file.filename)

    content = await file.read()
    result  = _voice_analyzer.analyze_bytes(content)

    if result.features is None:
        raise HTTPException(status_code=422, detail=str(result.notes))

    d_baseline = _if_service.baseline_sample_count(IFStream.DYSARTHRIA)
    p_baseline = _if_service.baseline_sample_count(IFStream.PARKINSONS)

    # Upgrade to IF personal baseline scores if models are trained
    if _if_service._models[IFStream.DYSARTHRIA].is_trained:
        d_if = _if_service.score(
            IFStream.DYSARTHRIA, result.features.dysarthria.to_array()
        )
        result.dysarthria_score = d_if.anomaly_score

    if _if_service._models[IFStream.PARKINSONS].is_trained:
        p_if = _if_service.score(
            IFStream.PARKINSONS, result.features.parkinsons.to_array()
        )
        result.parkinsons_score = p_if.anomaly_score

    return _voice_result_to_response(result, d_baseline, p_baseline)


@router.post("/voice/baseline")
async def add_voice_baseline(
    file: UploadFile = File(..., description="Baseline audio from user's normal session"),
):
    """
    Add an audio clip to the user's personal voice baseline.
    Call during onboarding — 10+ clips recommended.
    """
    content = await file.read()
    result  = _voice_analyzer.analyze_bytes(content)

    if result.features is None:
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {result.notes}",
        )

    _if_service.add_baseline(IFStream.DYSARTHRIA, result.features.dysarthria.to_array())
    _if_service.add_baseline(IFStream.PARKINSONS, result.features.parkinsons.to_array())

    d_n = _if_service.baseline_sample_count(IFStream.DYSARTHRIA)
    p_n = _if_service.baseline_sample_count(IFStream.PARKINSONS)

    # Auto-train when ready
    for stream, n in [(IFStream.DYSARTHRIA, d_n), (IFStream.PARKINSONS, p_n)]:
        if n >= 10 and not _if_service._models[stream].is_trained:
            _if_service.train(stream)
            logger.info("%s IF model trained on %d samples.", stream.value, n)

    return {
        "status":                   "ok",
        "dysarthria_baseline_count": d_n,
        "parkinsons_baseline_count": p_n,
        "baseline_ready":            d_n >= 10 and p_n >= 10,
        "message": (
            f"Voice baseline updated. "
            f"Dysarthria: {d_n}/10 | Parkinson's: {p_n}/10 samples."
        ),
    }


@router.post("/voice/speech-test", response_model=SpeechTestResponse)
async def speech_prompt_test(
    file:       UploadFile = File(..., description="Recording of user reading the prompt"),
    prompt_idx: int        = Form(default=0, ge=0, le=4,
                                  description="Prompt index 0–4"),
):
    """
    FR4 active neurological test — guided speech prompt.
    User reads a specific sentence; we score their articulation.

    Prompt 0: 'The quick brown fox...' (general articulation)
    Prompt 1: 'British Constitution'   (consonant clarity)
    Prompt 2: 'Methodist Episcopal'    (complex articulation)
    Prompt 3: Sustained 'aaah'         (vocal tremor / HNR)
    Prompt 4: Count 1–10              (tempo / rhythm)
    """
    logger.info("Speech test | prompt=%d | file=%s", prompt_idx, file.filename)

    content = await file.read()
    result  = _voice_analyzer.analyze_bytes(content)

    if result.features is None:
        raise HTTPException(status_code=422, detail=str(result.notes))

    # Determine pass/fail
    max_score = max(result.dysarthria_score, result.parkinsons_score)
    if max_score < 0.35:
        pass_fail = "PASS"
    elif max_score < 0.60:
        pass_fail = "UNCERTAIN"
    else:
        pass_fail = "FAIL"

    return SpeechTestResponse(
        prompt            = SPEECH_PROMPTS[prompt_idx],
        dysarthria_score  = result.dysarthria_score,
        parkinsons_score  = result.parkinsons_score,
        is_anomaly        = result.is_anomaly,
        confidence        = result.confidence,
        alert_indicators  = result.alert_indicators,
        pass_fail         = pass_fail,
        notes             = result.notes,
    )


# ── Baseline status ────────────────────────────────────────────────────────

@router.get("/baseline/status", response_model=BaselineStatusResponse)
async def baseline_status():
    """
    Check how ready each IF stream's personal baseline is.
    Used by the frontend onboarding wizard to guide the user.
    """
    streams = {}
    all_ready = True

    for stream in IFStream:
        n       = _if_service.baseline_sample_count(stream)
        trained = _if_service._models[stream].is_trained
        ready   = n >= 10

        if not ready:
            all_ready = False

        streams[stream.value] = {
            "sample_count":  n,
            "required":      10,
            "is_trained":    trained,
            "ready":         ready,
            "progress_pct":  min(100, int(n / 10 * 100)),
        }

    if all_ready:
        recommendation = "All baselines ready. Full personalised scoring active."
    else:
        missing = [k for k, v in streams.items() if not v["ready"]]
        recommendation = (
            f"Complete baseline setup for: {', '.join(missing)}. "
            "System using population priors until personal baseline is ready."
        )

    return BaselineStatusResponse(
        streams        = streams,
        overall_ready  = all_ready,
        recommendation = recommendation,
    )


# ── Health check ───────────────────────────────────────────────────────────

@router.get("/status")
async def neurological_status():
    return {
        "status":          "ok",
        "face_analyzer":   "MediaPipe FaceMesh (468 landmarks)",
        "voice_analyzer":  "Librosa (TORGO + UCI Parkinson's features)",
        "speech_prompts":  len(SPEECH_PROMPTS),
        "if_streams":      [s.value for s in IFStream],
        "datasets": {
            "face":        "Facial Palsy dataset",
            "dysarthria":  "TORGO database",
            "parkinsons":  "UCI Parkinson's Telemonitoring",
        },
    }