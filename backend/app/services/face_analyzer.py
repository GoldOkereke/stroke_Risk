"""
app/services/face_analyzer.py
==============================
Facial asymmetry analyzer for FR3.

Uses MediaPipe Face Mesh (468 landmarks) to detect facial asymmetry
as a neurological marker for stroke/TIA/facial palsy.

Pipeline:
  Webcam frame (BGR)
    → MediaPipe Face Mesh
    → 468 3D landmarks
    → 6 asymmetry features
    → Isolation Forest #2 (facial stream)
    → facial_anomaly_score [0, 1]

Features extracted (all measure L vs R deviation):
  1. mouth_droop          — mouth corner height delta (key stroke indicator)
  2. eye_droop            — eye openness delta L vs R
  3. brow_asymmetry       — brow height delta
  4. nasolabial_fold      — nasolabial fold depth proxy
  5. smile_symmetry       — lip corner horizontal movement symmetry
  6. overall_asymmetry    — mean Euclidean delta across all paired landmarks

Trained against:
  - Facial Palsy dataset (palsy vs normal)
  - Personal baseline built from user's first N sessions

Landmark indices reference:
  https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.obj
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe imported lazily to avoid crashing if not installed
_mp_face_mesh = None
_mp_drawing   = None


def _get_mediapipe():
    global _mp_face_mesh, _mp_drawing
    if _mp_face_mesh is None:
        try:
            import mediapipe as mp
            _mp_face_mesh = mp.solutions.face_mesh
            _mp_drawing   = mp.solutions.drawing_utils
            logger.info("MediaPipe Face Mesh loaded.")
        except ImportError:
            raise ImportError(
                "MediaPipe is not installed. Run: pip install mediapipe"
            )
    return _mp_face_mesh, _mp_drawing


# ── Landmark index pairs (left, right) ────────────────────────────────────
# Each tuple: (left_landmark_idx, right_landmark_idx)
# MediaPipe 468-point canonical face model

LANDMARK_PAIRS = {
    # Mouth corners
    "mouth_left":       61,
    "mouth_right":      291,
    "mouth_top":        13,
    "mouth_bottom":     14,

    # Eyes
    "left_eye_top":     159,
    "left_eye_bottom":  145,
    "right_eye_top":    386,
    "right_eye_bottom": 374,

    # Eyebrows
    "left_brow":        105,
    "right_brow":       334,

    # Nose tip (midline reference)
    "nose_tip":         4,

    # Cheeks (nasolabial fold proxy)
    "left_cheek":       117,
    "right_cheek":      346,

    # Chin
    "chin":             152,

    # Forehead midline
    "forehead":         10,
}

# Symmetric paired landmarks for overall asymmetry
SYMMETRIC_PAIRS: list[tuple[int, int]] = [
    (61,  291),   # mouth corners
    (159, 386),   # eye tops
    (145, 374),   # eye bottoms
    (105, 334),   # brows
    (117, 346),   # cheeks
    (33,  263),   # outer eye corners
    (133, 362),   # inner eye corners
    (70,  300),   # brow outer
    (107, 336),   # brow inner
    (187, 411),   # cheek lower
]


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class FaceFeatures:
    """
    Extracted facial asymmetry features from one frame.
    Shape: (6,) — matches IFStream.FACIAL feature definition.
    """
    mouth_droop:         float   # positive = left drooping, negative = right
    eye_droop:           float   # eye openness delta (L - R), normalised
    brow_asymmetry:      float   # brow height delta (L - R), normalised
    nasolabial_fold:     float   # cheek position delta proxy
    smile_symmetry:      float   # mouth corner horizontal symmetry
    overall_asymmetry:   float   # mean delta across all symmetric pairs

    landmarks_detected:  bool    = True
    n_landmarks:         int     = 468
    frame_quality:       float   = 1.0   # [0,1] — low = poor lighting/occlusion
    notes: list[str]             = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        """Return feature vector — matches FEATURE_NAMES[IFStream.FACIAL]."""
        return np.array([
            self.mouth_droop,
            self.eye_droop,
            self.brow_asymmetry,
            self.nasolabial_fold,
            self.smile_symmetry,
            self.overall_asymmetry,
        ], dtype=np.float64)

    @property
    def is_valid(self) -> bool:
        return self.landmarks_detected and self.frame_quality >= 0.3


@dataclass
class FaceAnalysisResult:
    """Full output from FaceAnalyzer.analyze() for one frame."""
    features: Optional[FaceFeatures]
    anomaly_score: float         # [0,1] — passed to IF stream
    is_anomaly: bool
    confidence: float
    landmarks_detected: bool
    alert_indicators: list[str]  # human-readable flags e.g. "Left mouth droop"
    notes: list[str]             = field(default_factory=list)


# ── Face Analyzer ──────────────────────────────────────────────────────────

class FaceAnalyzer:
    """
    Extracts facial asymmetry features from webcam frames.

    Usage
    -----
        analyzer = FaceAnalyzer()

        # Single frame (numpy BGR array from OpenCV)
        result = analyzer.analyze(frame)

        # Feed to Isolation Forest
        if result.features and result.features.is_valid:
            if_service.add_baseline(IFStream.FACIAL, result.features.to_array())

        # After enough baseline:
        anomaly = if_service.score(IFStream.FACIAL, result.features.to_array())
    """

    # Asymmetry thresholds for rule-based pre-screening
    # (before IF baseline is ready)
    MOUTH_DROOP_THRESHOLD    = 0.04   # normalised by face height
    EYE_DROOP_THRESHOLD      = 0.03
    BROW_ASYMMETRY_THRESHOLD = 0.03
    OVERALL_THRESHOLD        = 0.025

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._mesh = None
        logger.info("FaceAnalyzer initialised.")

    def _get_mesh(self):
        """Lazy-load MediaPipe Face Mesh."""
        if self._mesh is None:
            mp_face_mesh, _ = _get_mediapipe()
            self._mesh = mp_face_mesh.FaceMesh(
                static_image_mode        = False,
                max_num_faces            = 1,
                refine_landmarks         = True,
                min_detection_confidence = self.min_detection_confidence,
                min_tracking_confidence  = 0.5,
            )
            logger.info("MediaPipe FaceMesh session started.")
        return self._mesh

    # ── Main API ───────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> FaceAnalysisResult:
        """
        Analyze one video frame for facial asymmetry.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV (webcam capture).
            Shape: (H, W, 3)

        Returns
        -------
        FaceAnalysisResult
        """
        if frame is None or frame.size == 0:
            return self._empty_result("Empty frame received.")

        # ── MediaPipe landmark detection ───────────────────────────────
        try:
            landmarks = self._detect_landmarks(frame)
        except Exception as e:
            logger.warning("Landmark detection failed: %s", e)
            return self._empty_result(f"Detection error: {e}")

        if landmarks is None:
            return self._empty_result("No face detected in frame.")

        # ── Feature extraction ─────────────────────────────────────────
        h, w = frame.shape[:2]
        features = self._extract_features(landmarks, h, w)

        # ── Frame quality check ────────────────────────────────────────
        features.frame_quality = self._estimate_frame_quality(frame)
        if features.frame_quality < 0.3:
            features.notes.append(f"Low frame quality ({features.frame_quality:.2f}) — poor lighting?")

        # ── Rule-based pre-screening (before IF baseline ready) ────────
        alert_indicators = self._rule_based_flags(features)

        # ── Simple anomaly score (rule-based, used before IF trains) ───
        anomaly_score, confidence = self._simple_anomaly_score(features)

        result = FaceAnalysisResult(
            features           = features,
            anomaly_score      = anomaly_score,
            is_anomaly         = anomaly_score >= 0.5,
            confidence         = confidence,
            landmarks_detected = True,
            alert_indicators   = alert_indicators,
            notes              = features.notes,
        )

        if alert_indicators:
            logger.info("Face anomaly flags: %s", alert_indicators)

        return result

    def analyze_batch(self, frames: list[np.ndarray]) -> list[FaceAnalysisResult]:
        """Analyze multiple frames and return results list."""
        return [self.analyze(f) for f in frames]

    def aggregate_features(
        self, results: list[FaceAnalysisResult]
    ) -> Optional[np.ndarray]:
        """
        Aggregate features across multiple frames into one feature vector.
        Used to build a stable per-session observation for the IF baseline.

        Returns mean feature vector across valid frames, or None if no valid frames.
        """
        valid = [
            r.features.to_array()
            for r in results
            if r.features and r.features.is_valid
        ]
        if not valid:
            logger.warning("No valid frames to aggregate.")
            return None

        aggregated = np.mean(valid, axis=0)
        logger.debug("Aggregated %d valid frames into feature vector.", len(valid))
        return aggregated

    # ── Landmark detection ─────────────────────────────────────────────────

    def _detect_landmarks(self, frame: np.ndarray):
        """Run MediaPipe and return landmark list or None."""
        import cv2
        mesh = self._get_mesh()
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = mesh.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    # ── Feature extraction ─────────────────────────────────────────────────

    def _extract_features(self, landmarks, h: int, w: int) -> FaceFeatures:
        """
        Extract 6 asymmetry features from MediaPipe landmarks.
        All features are normalised by face height to be scale-invariant.
        """

        def lm(idx) -> tuple[float, float, float]:
            """Get (x, y, z) in pixel coordinates."""
            p = landmarks[idx]
            return p.x * w, p.y * h, p.z * w

        # Face height for normalisation
        nose_x, nose_y, _ = lm(LANDMARK_PAIRS["nose_tip"])
        chin_x, chin_y, _ = lm(LANDMARK_PAIRS["chin"])
        face_height = max(abs(chin_y - nose_y) * 2, 1.0)

        # ── 1. Mouth droop ─────────────────────────────────────────────
        ml_x, ml_y, _ = lm(LANDMARK_PAIRS["mouth_left"])
        mr_x, mr_y, _ = lm(LANDMARK_PAIRS["mouth_right"])
        mouth_droop = (ml_y - mr_y) / face_height   # +ve = left lower

        # ── 2. Eye droop (openness delta) ──────────────────────────────
        let_x, let_y, _ = lm(LANDMARK_PAIRS["left_eye_top"])
        leb_x, leb_y, _ = lm(LANDMARK_PAIRS["left_eye_bottom"])
        ret_x, ret_y, _ = lm(LANDMARK_PAIRS["right_eye_top"])
        reb_x, reb_y, _ = lm(LANDMARK_PAIRS["right_eye_bottom"])

        left_eye_open  = abs(leb_y - let_y) / face_height
        right_eye_open = abs(reb_y - ret_y) / face_height
        eye_droop = left_eye_open - right_eye_open

        # ── 3. Brow asymmetry ──────────────────────────────────────────
        lb_x, lb_y, _ = lm(LANDMARK_PAIRS["left_brow"])
        rb_x, rb_y, _ = lm(LANDMARK_PAIRS["right_brow"])
        # Brow height relative to nose
        left_brow_h  = (nose_y - lb_y) / face_height
        right_brow_h = (nose_y - rb_y) / face_height
        brow_asymmetry = left_brow_h - right_brow_h

        # ── 4. Nasolabial fold (cheek position proxy) ──────────────────
        lc_x, lc_y, _ = lm(LANDMARK_PAIRS["left_cheek"])
        rc_x, rc_y, _ = lm(LANDMARK_PAIRS["right_cheek"])
        nasolabial = (lc_y - rc_y) / face_height

        # ── 5. Smile symmetry (horizontal mouth corner movement) ───────
        midpoint_x = (ml_x + mr_x) / 2
        left_dist  = abs(ml_x - midpoint_x) / face_height
        right_dist = abs(mr_x - midpoint_x) / face_height
        smile_symmetry = abs(left_dist - right_dist)

        # ── 6. Overall asymmetry (mean delta across paired landmarks) ───
        deltas = []
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            lx, ly, _ = lm(left_idx)
            rx, ry, _ = lm(right_idx)

            # Mirror right side across vertical midline for comparison
            mid_x = w / 2
            lx_mirrored = 2 * mid_x - lx

            delta = np.sqrt((lx_mirrored - rx)**2 + (ly - ry)**2) / face_height
            deltas.append(delta)

        overall_asymmetry = float(np.mean(deltas))

        return FaceFeatures(
            mouth_droop       = float(mouth_droop),
            eye_droop         = float(eye_droop),
            brow_asymmetry    = float(brow_asymmetry),
            nasolabial_fold   = float(nasolabial),
            smile_symmetry    = float(smile_symmetry),
            overall_asymmetry = overall_asymmetry,
            n_landmarks       = len(landmarks),
        )

    # ── Frame quality ──────────────────────────────────────────────────────

    def _estimate_frame_quality(self, frame: np.ndarray) -> float:
        """
        Estimate frame quality from brightness and blur.
        Returns [0, 1] — 1 = excellent quality.
        """
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness check
        mean_brightness = float(gray.mean())
        brightness_score = np.clip(mean_brightness / 128.0, 0.0, 1.0)

        # Blur check (Laplacian variance — low = blurry)
        blur_score = float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0.0, 1.0))

        return float((brightness_score + blur_score) / 2)

    # ── Rule-based flags ───────────────────────────────────────────────────

    def _rule_based_flags(self, features: FaceFeatures) -> list[str]:
        """
        Flag specific asymmetries before IF baseline is ready.
        These are clinical stroke indicators (FAST: Face, Arms, Speech, Time).
        """
        flags = []

        if abs(features.mouth_droop) > self.MOUTH_DROOP_THRESHOLD:
            side = "left" if features.mouth_droop > 0 else "right"
            flags.append(f"Mouth droop detected ({side} side)")

        if abs(features.eye_droop) > self.EYE_DROOP_THRESHOLD:
            side = "left" if features.eye_droop < 0 else "right"
            flags.append(f"Eyelid droop detected ({side} eye)")

        if abs(features.brow_asymmetry) > self.BROW_ASYMMETRY_THRESHOLD:
            flags.append("Brow asymmetry detected")

        if features.overall_asymmetry > self.OVERALL_THRESHOLD:
            flags.append(
                f"Overall facial asymmetry elevated "
                f"({features.overall_asymmetry:.3f})"
            )

        return flags

    # ── Simple scoring (before IF trains) ─────────────────────────────────

    def _simple_anomaly_score(
        self, features: FaceFeatures
    ) -> tuple[float, float]:
        """
        Rule-based anomaly score used before the IF model has enough baseline.
        Maps feature magnitudes to [0, 1].
        """
        scores = [
            min(1.0, abs(features.mouth_droop)    / (self.MOUTH_DROOP_THRESHOLD    * 2)),
            min(1.0, abs(features.eye_droop)       / (self.EYE_DROOP_THRESHOLD      * 2)),
            min(1.0, abs(features.brow_asymmetry)  / (self.BROW_ASYMMETRY_THRESHOLD * 2)),
            min(1.0, features.overall_asymmetry    / (self.OVERALL_THRESHOLD        * 2)),
        ]

        # Mouth droop weighted higher — strongest clinical stroke indicator
        weighted = (
            scores[0] * 0.40 +
            scores[1] * 0.25 +
            scores[2] * 0.15 +
            scores[3] * 0.20
        )

        # Confidence based on frame quality
        confidence = features.frame_quality

        return round(float(weighted), 4), round(float(confidence), 4)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _empty_result(self, note: str) -> FaceAnalysisResult:
        return FaceAnalysisResult(
            features           = None,
            anomaly_score      = 0.0,
            is_anomaly         = False,
            confidence         = 0.0,
            landmarks_detected = False,
            alert_indicators   = [],
            notes              = [note],
        )

    def close(self):
        """Release MediaPipe resources."""
        if self._mesh:
            self._mesh.close()
            self._mesh = None
            logger.info("FaceAnalyzer MediaPipe session closed.")