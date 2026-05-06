"""Face analysis service (MediaPipe Face Mesh).

This module implements the "facial stream" part of the pipeline:
- detect a face
- extract a small set of asymmetry features
- optionally score the features with an Isolation Forest

The outputs are designed to match `app.models.schemas.FaceAnalysisResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.logging import logger
from app.models.schemas import FaceAnalysisResult, FaceFeatures
from app.services.isolation_forest import IFStream, IsolationForestService

# Optional dependencies: we import lazily so app startup doesn't fail if the user
# hasn't installed the "computer vision" extras yet.
try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None

try:  # pragma: no cover
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore
except Exception:  # pragma: no cover
    mp_python = None
    mp_vision = None


@dataclass(frozen=True)
class FaceQuality:
    """Quality signals used to decide whether a frame is usable."""

    brightness: float
    """Mean brightness in [0, 1]."""

    sharpness: float
    """Sharpness proxy based on Laplacian variance (normalized to [0, 1])."""

    score: float
    """Combined quality score in [0, 1]."""


class FaceAnalyzer:
    """Extracts facial asymmetry features using MediaPipe Face Mesh."""

    def __init__(self) -> None:
        # MediaPipe legacy "solutions" API was removed in newer wheels.
        # Support both:
        # - legacy: mp.solutions.face_mesh.FaceMesh
        # - current: MediaPipe Tasks FaceLandmarker (.task model required)
        self._mp_face_mesh = None
        self._mp_face_landmarker = None
        self._mp_face_landmarker_relaxed = None
        self._mediapipe_init_error: Optional[str] = None

        if mp is not None and hasattr(mp, "solutions"):  # pragma: no cover
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        elif mp is not None and mp_python is not None and mp_vision is not None:  # pragma: no cover
            model_path = Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"
            if model_path.exists():
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._mp_face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

                    relaxed_options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.3,
                        min_face_presence_confidence=0.3,
                        min_tracking_confidence=0.3,
                    )
                    self._mp_face_landmarker_relaxed = mp_vision.FaceLandmarker.create_from_options(
                        relaxed_options
                    )
                except Exception as exc:
                    self._mediapipe_init_error = str(exc)
            else:
                self._mediapipe_init_error = f"Model file not found: {model_path}"

        # Optional anomaly scorer. For now we use a fixed "default" user_id since the
        # service layer doesn't yet manage authenticated users.
        self._if = IsolationForestService(user_id="default")
        self._require_mediapipe()

    @staticmethod
    def _require_cv() -> None:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is not installed. Install backend requirements to enable face analysis."
            )

    def _require_mediapipe(self) -> None:
        has_backend = self._mp_face_mesh is not None or self._mp_face_landmarker is not None
        if not has_backend:
            if self._mediapipe_init_error:
                raise RuntimeError(
                    f"MediaPipe backend failed to initialize: {self._mediapipe_init_error}. "
                    "For Windows, use Python 3.11/3.12 for reliable MediaPipe support."
                )
            raise RuntimeError(
                "mediapipe is not installed. Install backend requirements to enable face analysis."
            )

    @staticmethod
    def _to_rgb(image: Any) -> np.ndarray:
        """Convert an input image to RGB uint8 numpy array."""
        FaceAnalyzer._require_cv()

        # Accept either a numpy array (common with OpenCV) or a PIL.Image.
        if hasattr(image, "convert"):  # PIL.Image-like
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return rgb

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Expected an image array with shape (H, W, 3/4).")

        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # OpenCV images are usually BGR; convert to RGB for MediaPipe.
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def _compute_quality(rgb: np.ndarray) -> FaceQuality:
        """Compute a simple quality score based on brightness and sharpness."""
        FaceAnalyzer._require_cv()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Brightness: normalize mean intensity from [0..255] -> [0..1]
        brightness = float(np.clip(np.mean(gray) / 255.0, 0.0, 1.0))

        # Sharpness: variance of Laplacian. Higher variance = sharper image.
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(lap))

        # Normalize Laplacian variance into [0, 1] using a soft saturation curve.
        # Values around ~100+ are typically "sharp enough" for webcam frames.
        sharpness = float(lap_var / (lap_var + 100.0))

        # Combine brightness + sharpness (equal weight).
        score = float(np.clip(0.5 * brightness + 0.5 * sharpness, 0.0, 1.0))
        return FaceQuality(brightness=brightness, sharpness=sharpness, score=score)

    @staticmethod
    def _enhance_for_detection(rgb: np.ndarray) -> np.ndarray:
        """Apply lightweight enhancement to improve face landmark detection on weak frames."""
        FaceAnalyzer._require_cv()
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        enhanced = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Gentle unsharp mask to recover edge detail.
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharp = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _detect_landmarks(self, rgb: np.ndarray, relaxed: bool = False) -> Optional[list[Any]]:
        """Run MediaPipe detection and return first face landmarks if available."""
        if self._mp_face_mesh is not None:
            result = self._mp_face_mesh.process(rgb)
            if result.multi_face_landmarks:
                return result.multi_face_landmarks[0].landmark
            return None

        if self._mp_face_landmarker is None:
            return None

        detector = self._mp_face_landmarker_relaxed if relaxed and self._mp_face_landmarker_relaxed else self._mp_face_landmarker
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # type: ignore[union-attr]
        result = detector.detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    @staticmethod
    def _landmark_px(landmarks: list[Any], idx: int, width: int, height: int) -> np.ndarray:
        """Convert a normalized landmark to pixel coordinates."""
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=float)

    def extract_features(self, image: Any) -> tuple[FaceFeatures, float, FaceQuality, bool]:
        """Extract asymmetry features.

        Returns:
        - FaceFeatures: a dict of feature_name -> value (already normalized by face height)
        - aggregated_score: overall asymmetry score in [0, 1]
        - quality: FaceQuality object (0-1)
        - face_detected: whether MediaPipe detected a face
        """
        self._require_mediapipe()

        rgb = self._to_rgb(image)
        quality = self._compute_quality(rgb)

        height, width = rgb.shape[0], rgb.shape[1]
        backend = "legacy_face_mesh" if self._mp_face_mesh is not None else "tasks_face_landmarker"
        logger.info(
            "face_extract_start backend=%s width=%d height=%d quality=%.4f brightness=%.4f sharpness=%.4f",
            backend,
            width,
            height,
            quality.score,
            quality.brightness,
            quality.sharpness,
        )

        landmarks = self._detect_landmarks(rgb, relaxed=False)
        retry_used = False
        if not landmarks:
            enhanced = self._enhance_for_detection(rgb)
            landmarks = self._detect_landmarks(enhanced, relaxed=True)
            retry_used = True
            if landmarks:
                rgb = enhanced
                logger.info("face_extract_retry_success backend=%s", backend)

        if not landmarks:
            logger.warning(
                "face_extract_no_landmarks backend=%s width=%d height=%d quality=%.4f retry_used=%s",
                backend,
                width,
                height,
                quality.score,
                retry_used,
            )
            empty = FaceFeatures(scores={})
            return empty, 0.0, quality, False

        # Landmarks used for the features (MediaPipe Face Mesh indices).
        # These indices are stable within MediaPipe's topology.
        MOUTH_LEFT = 61
        MOUTH_RIGHT = 291
        EYE_LEFT_OUTER = 33
        EYE_RIGHT_OUTER = 263
        BROW_LEFT = 105
        BROW_RIGHT = 334
        NOSE_TIP = 1
        CHIN = 152
        FOREHEAD = 10
        UPPER_LIP = 13

        mouth_l = self._landmark_px(landmarks, MOUTH_LEFT, width, height)
        mouth_r = self._landmark_px(landmarks, MOUTH_RIGHT, width, height)
        eye_l = self._landmark_px(landmarks, EYE_LEFT_OUTER, width, height)
        eye_r = self._landmark_px(landmarks, EYE_RIGHT_OUTER, width, height)
        brow_l = self._landmark_px(landmarks, BROW_LEFT, width, height)
        brow_r = self._landmark_px(landmarks, BROW_RIGHT, width, height)
        nose = self._landmark_px(landmarks, NOSE_TIP, width, height)
        chin = self._landmark_px(landmarks, CHIN, width, height)
        forehead = self._landmark_px(landmarks, FOREHEAD, width, height)
        upper_lip = self._landmark_px(landmarks, UPPER_LIP, width, height)

        # Face height is used as a scale reference so values are comparable across distances.
        face_height = float(np.linalg.norm(chin - forehead))
        if face_height <= 1e-6:
            face_height = float(height)

        # 1) mouth_droop: y difference between mouth corners.
        mouth_droop = float(abs(mouth_l[1] - mouth_r[1]) / face_height)

        # 2) eye_droop: y difference between outer eye corners.
        eye_droop = float(abs(eye_l[1] - eye_r[1]) / face_height)

        # 3) brow_asymmetry: y difference between eyebrows.
        brow_asymmetry = float(abs(brow_l[1] - brow_r[1]) / face_height)

        # 4) nasolabial_fold proxy: difference in nose->mouth distances.
        nose_to_mouth_l = float(np.linalg.norm(nose - mouth_l) / face_height)
        nose_to_mouth_r = float(np.linalg.norm(nose - mouth_r) / face_height)
        nasolabial_fold = float(abs(nose_to_mouth_l - nose_to_mouth_r))

        # 5) smile_symmetry proxy: compare how far each mouth corner sits below the upper lip.
        # Larger left-right differences suggest asymmetry in the smile line.
        smile_l = float((mouth_l[1] - upper_lip[1]) / face_height)
        smile_r = float((mouth_r[1] - upper_lip[1]) / face_height)
        smile_symmetry = float(abs(smile_l - smile_r))

        # 6) overall_asymmetry: average of the above measures.
        overall_asymmetry = float(
            np.mean([mouth_droop, eye_droop, brow_asymmetry, nasolabial_fold, smile_symmetry])
        )

        # Clamp to a safe range; typical values are small (<0.2).
        def clamp01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        scores = {
            "mouth_droop": clamp01(mouth_droop),
            "eye_droop": clamp01(eye_droop),
            "brow_asymmetry": clamp01(brow_asymmetry),
            "nasolabial_fold": clamp01(nasolabial_fold),
            "smile_symmetry": clamp01(smile_symmetry),
            "overall_asymmetry": clamp01(overall_asymmetry),
        }

        features = FaceFeatures(scores=scores)
        aggregated_score = scores["overall_asymmetry"]
        logger.info(
            "face_extract_success backend=%s feature_count=%d feature_keys=%s aggregated_score=%.6f",
            backend,
            len(scores),
            ",".join(sorted(scores.keys())),
            aggregated_score,
        )
        return features, aggregated_score, quality, True

    def analyze(self, image: Any, use_if: bool = True) -> FaceAnalysisResult:
        """Analyze a single image and return typed results."""
        features, aggregated_score, quality, face_detected = self.extract_features(image)

        anomaly_score = None
        if use_if and face_detected and features.scores:
            # IsolationForest expects a numeric vector. We use the same order every time.
            vector = np.array(
                [
                    features.scores.get("mouth_droop", 0.0),
                    features.scores.get("eye_droop", 0.0),
                    features.scores.get("brow_asymmetry", 0.0),
                    features.scores.get("nasolabial_fold", 0.0),
                    features.scores.get("smile_symmetry", 0.0),
                ],
                dtype=float,
            )
            result = self._if.score(IFStream.FACIAL, vector)
            anomaly_score = float(result.score) if result is not None else None

        return FaceAnalysisResult(
            features=features,
            aggregated_score=float(aggregated_score),
            anomaly_score=anomaly_score,
            quality=float(quality.score),
            face_detected=bool(face_detected),
        )

    def analyze_batch(self, images: list[Any], use_if: bool = True) -> FaceAnalysisResult:
        """Analyze multiple images and aggregate a session result.

        Aggregation strategy:
        - average the per-image asymmetry scores
        - average IF anomaly scores (when available)
        - average quality score
        - face_detected is True if any image contained a face
        """
        if not images:
            return FaceAnalysisResult(
                features=FaceFeatures(scores={}),
                aggregated_score=0.0,
                anomaly_score=None,
                quality=0.0,
                face_detected=False,
            )

        per_image = [self.analyze(img, use_if=use_if) for img in images]
        detected = any(item.face_detected for item in per_image if item.face_detected is not None)

        agg = float(np.mean([item.aggregated_score for item in per_image]))
        qual = float(np.mean([item.quality or 0.0 for item in per_image]))

        anomaly_values = [item.anomaly_score for item in per_image if item.anomaly_score is not None]
        anomaly = float(np.mean(anomaly_values)) if anomaly_values else None

        # Aggregate features by averaging each key across frames.
        all_keys: set[str] = set()
        for item in per_image:
            all_keys.update(item.features.scores.keys())
        averaged: dict[str, float] = {}
        for key in sorted(all_keys):
            vals = [it.features.scores.get(key, 0.0) for it in per_image]
            averaged[key] = float(np.mean(vals))

        return FaceAnalysisResult(
            features=FaceFeatures(scores=averaged),
            aggregated_score=agg,
            anomaly_score=anomaly,
            quality=qual,
            face_detected=detected,
        )
