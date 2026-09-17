"""Neurological endpoints (face + voice baseline and analysis).

Prefix: /api/neurological

This router supports:
- analyzing face images (single frame or session batch)
- adding baseline samples for personalization (IsolationForestService)
- analyzing voice recordings and optionally running a speech test prompt
- baseline status/reset
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from app.api.deps import get_user_id
from app.core.logging import logger
from app.db.models import BaselineSample, User
from app.db.session import get_db
from app.services.face_analyzer import FaceAnalyzer
from app.services.isolation_forest import IFStream, IsolationForestService
from app.services.voice_analyzer import VoiceAnalyzer


router = APIRouter()


def _get_or_create_user(db, user_id: str) -> User:
    user = db.query(User).filter(User.username == user_id).first()
    if user:
        return user
    user = User(username=user_id, hashed_password="")  # placeholder; auth not wired yet
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/face/analyze")
async def analyze_face(
    image: UploadFile = File(..., alias="file"),
) -> dict[str, Any]:
    """Analyze one image frame and return face features + scores."""
    try:
        analyzer = FaceAnalyzer()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    img = Image.open(image.file)
    result = analyzer.analyze(img, use_if=True)
    return result.model_dump()


@router.post("/face/session")
async def analyze_face_session(
    images: list[UploadFile] = File(..., alias="files"),
) -> dict[str, Any]:
    """Analyze multiple frames and return an aggregated result."""
    try:
        analyzer = FaceAnalyzer()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    imgs = [Image.open(img.file) for img in images]
    result = analyzer.analyze_batch(imgs, use_if=True)
    return result.model_dump()


@router.post("/face/baseline")
async def add_face_baseline(
    image: UploadFile = File(..., alias="file"),
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Extract features and add to the user's baseline for FACIAL stream."""
    logger.info(
        "face_baseline_request_start user_id=%s filename=%s content_type=%s",
        user_id,
        image.filename,
        image.content_type,
    )

    try:
        analyzer = FaceAnalyzer()
    except RuntimeError as exc:
        logger.exception("face_baseline_analyzer_init_failed user_id=%s error=%s", user_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    img = Image.open(image.file)
    logger.info(
        "face_baseline_image_loaded user_id=%s mode=%s size=%sx%s",
        user_id,
        getattr(img, "mode", "unknown"),
        getattr(img, "width", -1),
        getattr(img, "height", -1),
    )

    analysis = analyzer.analyze(img, use_if=False)
    features = analysis.features.scores
    logger.info(
        "face_baseline_analysis_result user_id=%s face_detected=%s quality=%.4f feature_count=%d feature_keys=%s",
        user_id,
        bool(analysis.face_detected),
        float(analysis.quality or 0.0),
        len(features),
        ",".join(sorted(features.keys())) if features else "",
    )

    if not analysis.face_detected or not features:
        logger.warning(
            "face_baseline_empty_features user_id=%s face_detected=%s quality=%.4f",
            user_id,
            bool(analysis.face_detected),
            float(analysis.quality or 0.0),
        )
        raise HTTPException(
            status_code=400,
            detail="No face detected or face features could not be extracted. Please upload a clearer front-facing image.",
        )

    user = _get_or_create_user(db, user_id)
    db.add(BaselineSample(user_id=user.id, stream=IFStream.FACIAL.value, features=features))
    db.commit()
    logger.info(
        "face_baseline_db_saved user_id=%s db_user_id=%s stream=%s feature_count=%d",
        user_id,
        user.id,
        IFStream.FACIAL.value,
        len(features),
    )

    if_service = IsolationForestService(user_id=user_id)
    vector = np.array(list(features.values()), dtype=float)
    logger.info(
        "face_baseline_if_vector user_id=%s stream=%s vector_size=%d vector_values=%s",
        user_id,
        IFStream.FACIAL.value,
        int(vector.size),
        np.array2string(vector, precision=6, separator=","),
    )
    count = if_service.add_baseline(IFStream.FACIAL, vector)
    logger.info("face_baseline_success user_id=%s baseline_count=%d", user_id, count)

    return {"added": True, "baselineCount": count}


@router.post("/voice/analyze")
async def analyze_voice(
    audio: UploadFile = File(..., alias="file"),
    sr: int = Form(16000),
) -> dict[str, Any]:
    """Analyze a voice sample and return features + scores."""
    analyzer = VoiceAnalyzer()
    raw = await audio.read()
    # Assume WAV/PCM-like data is handled client-side; here we treat bytes as float32 if possible.
    # For real deployments, decode with soundfile. This endpoint keeps the interface stable.
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    result = analyzer.analyze(data, sr=sr)
    return result.model_dump()


@router.post("/voice/baseline")
async def add_voice_baseline(
    audio: UploadFile = File(..., alias="file"),
    sr: int = Form(16000),
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Add a voice baseline sample for both dysarthria and Parkinsons streams."""
    analyzer = VoiceAnalyzer()
    raw = await audio.read()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    analysis = analyzer.analyze(data, sr=sr)

    user = _get_or_create_user(db, user_id)

    dys = analysis.features.dysarthria.scores if analysis.features.dysarthria else {}
    pk = analysis.features.parkinsons.scores if analysis.features.parkinsons else {}
    db.add(BaselineSample(user_id=user.id, stream=IFStream.DYSARTHRIA.value, features=dys))
    db.add(BaselineSample(user_id=user.id, stream=IFStream.PARKINSONS.value, features=pk))
    db.commit()

    if_service = IsolationForestService(user_id=user_id)
    dys_count = if_service.add_baseline(IFStream.DYSARTHRIA, np.array(list(dys.values()), dtype=float))
    pk_count = if_service.add_baseline(IFStream.PARKINSONS, np.array(list(pk.values()), dtype=float))

    return {"added": True, "dysarthriaCount": dys_count, "parkinsonsCount": pk_count}


@router.get("/baseline/status")
def baseline_status(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Return baseline counts per stream for the current user."""
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return {"user": user_id, "counts": {}}
    rows = db.query(BaselineSample.stream).filter(BaselineSample.user_id == user.id).all()
    counts: dict[str, int] = {}
    for (stream,) in rows:
        counts[stream] = counts.get(stream, 0) + 1
    return {"user": user_id, "counts": counts}


@router.get("/face/baseline")
def face_baseline_status(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Return whether the current user has any facial baseline samples."""
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return {"user": user_id, "stream": IFStream.FACIAL.value, "has_baseline": False, "count": 0}

    count = (
        db.query(BaselineSample)
        .filter(BaselineSample.user_id == user.id, BaselineSample.stream == IFStream.FACIAL.value)
        .count()
    )
    return {
        "user": user_id,
        "stream": IFStream.FACIAL.value,
        "has_baseline": count > 0,
        "count": count,
    }


@router.get("/voice/baseline")
def voice_baseline_status(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Return whether the current user has any voice baseline samples."""
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return {
            "user": user_id,
            "stream": "voice",
            "has_baseline": False,
            "dysarthria_count": 0,
            "parkinsons_count": 0,
        }

    dysarthria_count = (
        db.query(BaselineSample)
        .filter(BaselineSample.user_id == user.id, BaselineSample.stream == IFStream.DYSARTHRIA.value)
        .count()
    )
    parkinsons_count = (
        db.query(BaselineSample)
        .filter(BaselineSample.user_id == user.id, BaselineSample.stream == IFStream.PARKINSONS.value)
        .count()
    )
    has_baseline = (dysarthria_count + parkinsons_count) > 0

    return {
        "user": user_id,
        "stream": "voice",
        "has_baseline": has_baseline,
        "dysarthria_count": dysarthria_count,
        "parkinsons_count": parkinsons_count,
    }


@router.post("/baseline/reset")
def baseline_reset(
    user_id: str = Depends(get_user_id),
    db=Depends(get_db),
) -> dict[str, Any]:
    """Clear baseline samples for the current user."""
    user = db.query(User).filter(User.username == user_id).first()
    if not user:
        return {"cleared": True}
    db.query(BaselineSample).filter(BaselineSample.user_id == user.id).delete()
    db.commit()
    return {"cleared": True}
SPEECH_TEST_PROMPTS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "We the people of the United States, in order to form a more perfect union...",
    "Methodist Episcopal",
    "Sustained vowel: aaaah",
]


@router.post("/voice/speech-test")
async def run_speech_test(
    prompt_index: int = Form(..., alias="prompt_index"),
    audio: UploadFile = File(..., alias="file"),
    sr: int = Form(16000),
) -> dict[str, Any]:
    """Guided speech test endpoint used by the frontend."""
    idx = int(prompt_index)
    prompt = SPEECH_TEST_PROMPTS[idx] if 0 <= idx < len(SPEECH_TEST_PROMPTS) else SPEECH_TEST_PROMPTS[0]

    analyzer = VoiceAnalyzer()
    raw = await audio.read()
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    result = analyzer.analyze(data, sr=sr)

    payload = result.model_dump()
    payload["prompt"] = prompt
    payload["promptIndex"] = idx
    return payload
