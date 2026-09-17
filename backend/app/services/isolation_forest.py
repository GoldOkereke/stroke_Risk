"""
app/services/isolation_forest.py
==================================
Shared Isolation Forest service used across ALL four signal streams:

  Stream 1 — Cardiac      : ECG/PPG HRV features  (MIT-BIH AF + MIMIC PERform)
  Stream 2 — Facial        : MediaPipe landmarks   (Facial Palsy dataset)
  Stream 3 — Dysarthria    : Librosa MFCCs         (TORGO dataset)
  Stream 4 — Parkinson's   : Pitch/jitter features (UCI Parkinson's dataset)

Each stream gets its own trained IsolationForest instance via IsolationForestService.
The personal baseline is built from the user's own normal sessions — this is the
core differentiator of the system: anomaly scoring against YOUR normal, not a
population average.

Output per stream:
  anomaly_score   : float [0, 1]  — 0 = normal, 1 = maximally anomalous
  is_anomaly      : bool          — True if score > threshold
  confidence      : float [0, 1]  — how certain the IF is
  raw_score       : float         — raw IF decision_function output (for debugging)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

MODEL_DIR            = Path("app/ml/weights/isolation_forest")
CONTAMINATION        = 0.1     # expected fraction of anomalies in training data
N_ESTIMATORS         = 200     # more trees = more stable scores
MIN_SAMPLES_TO_TRAIN = 10      # minimum baseline sessions before IF is reliable
ANOMALY_THRESHOLD    = 0.5     # normalised score above this = anomaly flagged


# ── Stream identifiers ─────────────────────────────────────────────────────

class IFStream(str, Enum):
    CARDIAC     = "cardiac"      # ECG + PPG HRV features
    FACIAL      = "facial"       # MediaPipe landmark deltas
    DYSARTHRIA  = "dysarthria"   # TORGO MFCCs — slurred speech
    PARKINSONS  = "parkinsons"   # UCI pitch/jitter — tremor


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """
    Output from one Isolation Forest scoring call.
    Consumed directly by the Fusion Engine (FR5).
    """
    stream: IFStream
    anomaly_score: float        # normalised [0, 1] — higher = more anomalous
    is_anomaly: bool            # True if score > ANOMALY_THRESHOLD
    confidence: float           # how far from the decision boundary [0, 1]
    raw_score: float            # raw IF decision_function output
    n_samples_in_baseline: int  # how many sessions trained the baseline
    baseline_ready: bool        # False if not enough baseline data yet
    notes: list[str] = field(default_factory=list)

    @property
    def alert_contribution(self) -> str:
        """Human-readable contribution label for the fusion engine."""
        if not self.baseline_ready:
            return f"{self.stream.value}: baseline not ready"
        if self.is_anomaly:
            return f"{self.stream.value}: ANOMALY (score={self.anomaly_score:.2f})"
        return f"{self.stream.value}: normal (score={self.anomaly_score:.2f})"


# ── Per-stream model container ─────────────────────────────────────────────

@dataclass
class IFModel:
    """Holds a trained IsolationForest + its scaler for one stream."""
    stream: IFStream
    forest: Optional[IsolationForest] = None
    scaler: Optional[StandardScaler]  = None
    n_training_samples: int = 0
    is_trained: bool = False
    feature_names: list[str] = field(default_factory=list)


# ── Main service ───────────────────────────────────────────────────────────

class IsolationForestService:
    """
    Single service managing all four Isolation Forest streams.

    Each stream maintains its own model trained on the user's personal baseline.
    Population-level priors (from datasets) can be loaded to warm-start the model
    before personal data is collected.

    Usage
    -----
        svc = IsolationForestService()

        # Add baseline sessions (user's normal data)
        svc.add_baseline(IFStream.CARDIAC, features_array)

        # Train once enough baseline exists
        svc.train(IFStream.CARDIAC)

        # Score new observation
        result = svc.score(IFStream.CARDIAC, new_features)

        # Score all streams at once
        results = svc.score_all(cardiac_feat, facial_feat, dysarthria_feat, pk_feat)
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._models: dict[IFStream, IFModel] = {
            stream: IFModel(stream=stream) for stream in IFStream
        }
        # Baseline feature buffers — accumulate before training
        self._baseline_buffer: dict[IFStream, list[np.ndarray]] = {
            stream: [] for stream in IFStream
        }
        logger.info("IsolationForestService initialised | user=%s", user_id)

    # ── Feature definitions ────────────────────────────────────────────────

    FEATURE_NAMES: dict[IFStream, list[str]] = {
        IFStream.CARDIAC: [
            "hrv_rmssd",          # root mean square successive differences
            "hrv_sdnn",           # std of RR intervals
            "rr_irregularity",    # coefficient of variation of RR
            "signal_mean",        # mean amplitude
            "signal_std",         # std amplitude
            "afib_cnn_prob",      # CNN AFib probability (FR2 output fed into IF)
        ],
        IFStream.FACIAL: [
            "left_right_asymmetry",   # overall L/R landmark delta
            "mouth_droop",            # mouth corner height delta
            "eye_droop",              # eye openness delta L vs R
            "brow_asymmetry",         # brow position delta
            "nasolabial_fold_delta",  # nasolabial fold depth delta
            "smile_symmetry",         # lip corner movement symmetry
        ],
        IFStream.DYSARTHRIA: [
            "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3",  # first 3 MFCC means
            "mfcc_std_1",  "mfcc_std_2",  "mfcc_std_3",   # first 3 MFCC stds
            "speech_tempo",       # syllables per second
            "articulation_rate",  # rate excluding pauses
            "pause_ratio",        # fraction of time silent
            "spectral_centroid",  # centre of spectral mass
        ],
        IFStream.PARKINSONS: [
            "jitter_pct",         # cycle-to-cycle pitch variation %
            "shimmer_db",         # amplitude variation dB
            "hnr",                # harmonics-to-noise ratio
            "rpde",               # recurrence period density entropy
            "dfa",                # detrended fluctuation analysis
            "pitch_mean",         # mean fundamental frequency
            "pitch_std",          # std of fundamental frequency
        ],
    }

    # ── Baseline management ────────────────────────────────────────────────

    def add_baseline(self, stream: IFStream, features: np.ndarray):
        """
        Add one observation to the baseline buffer for a stream.

        Parameters
        ----------
        stream   : which IF stream this observation belongs to
        features : 1-D array of feature values (must match FEATURE_NAMES length)
        """
        features = np.asarray(features, dtype=np.float64).flatten()
        expected = len(self.FEATURE_NAMES[stream])

        if len(features) != expected:
            raise ValueError(
                f"Stream {stream.value} expects {expected} features, "
                f"got {len(features)}. Expected: {self.FEATURE_NAMES[stream]}"
            )

        self._baseline_buffer[stream].append(features)
        n = len(self._baseline_buffer[stream])
        logger.debug("Baseline buffer | stream=%s | n=%d samples", stream.value, n)

    def add_baseline_batch(self, stream: IFStream, features_matrix: np.ndarray):
        """
        Add multiple observations at once (e.g. from dataset pre-training).

        Parameters
        ----------
        features_matrix : shape (N, n_features)
        """
        for row in features_matrix:
            self.add_baseline(stream, row)
        logger.info(
            "Batch baseline added | stream=%s | %d samples | total=%d",
            stream.value, len(features_matrix),
            len(self._baseline_buffer[stream]),
        )

    def baseline_sample_count(self, stream: IFStream) -> int:
        return len(self._baseline_buffer[stream])

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, stream: IFStream, contamination: float = CONTAMINATION):
        """
        Train Isolation Forest for one stream on its baseline buffer.

        Parameters
        ----------
        stream        : which stream to train
        contamination : expected fraction of anomalies in training data
                        (lower = stricter — fewer things flagged as anomalies)
        """
        buffer = self._baseline_buffer[stream]
        n = len(buffer)

        if n < MIN_SAMPLES_TO_TRAIN:
            logger.warning(
                "Stream %s has only %d baseline samples (min=%d). "
                "Skipping training — using population prior if available.",
                stream.value, n, MIN_SAMPLES_TO_TRAIN,
            )
            return

        X = np.vstack(buffer)

        # Standardise features before fitting
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        forest = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        forest.fit(X_scaled)

        model = self._models[stream]
        model.forest  = forest
        model.scaler  = scaler
        model.n_training_samples = n
        model.is_trained = True
        model.feature_names = self.FEATURE_NAMES[stream]

        logger.info(
            "IF trained | stream=%s | samples=%d | contamination=%.2f",
            stream.value, n, contamination,
        )

    def train_all(self):
        """Train all four streams from their baseline buffers."""
        for stream in IFStream:
            self.train(stream)

    # ── Scoring ────────────────────────────────────────────────────────────

    def score(self, stream: IFStream, features: np.ndarray) -> AnomalyResult:
        """
        Score one observation against the trained baseline for a stream.

        Parameters
        ----------
        stream   : which stream to score
        features : 1-D feature array

        Returns
        -------
        AnomalyResult with normalised anomaly_score, is_anomaly, confidence
        """
        features = np.asarray(features, dtype=np.float64).flatten().reshape(1, -1)
        model    = self._models[stream]
        n_baseline = len(self._baseline_buffer[stream])

        if not model.is_trained:
            logger.warning("Stream %s not trained. Returning neutral score.", stream.value)
            return AnomalyResult(
                stream=stream,
                anomaly_score=0.0,
                is_anomaly=False,
                confidence=0.0,
                raw_score=0.0,
                n_samples_in_baseline=n_baseline,
                baseline_ready=False,
                notes=["Model not trained — insufficient baseline data."],
            )

        # Scale using the same scaler fitted during training
        X_scaled = model.scaler.transform(features)

        # decision_function: negative = anomaly, positive = normal
        raw_score = float(model.forest.decision_function(X_scaled)[0])

        # Normalise to [0, 1] where 1 = maximally anomalous
        anomaly_score = self._normalise_score(raw_score)
        is_anomaly    = anomaly_score >= ANOMALY_THRESHOLD

        # Confidence = how far from the 0.5 boundary
        confidence = float(abs(anomaly_score - 0.5) * 2)

        result = AnomalyResult(
            stream=stream,
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=is_anomaly,
            confidence=round(confidence, 4),
            raw_score=round(raw_score, 4),
            n_samples_in_baseline=n_baseline,
            baseline_ready=True,
        )

        logger.debug(
            "IF score | stream=%s | score=%.3f | anomaly=%s | confidence=%.3f",
            stream.value, anomaly_score, is_anomaly, confidence,
        )
        return result

    def score_all(
        self,
        cardiac_features:    Optional[np.ndarray] = None,
        facial_features:     Optional[np.ndarray] = None,
        dysarthria_features: Optional[np.ndarray] = None,
        parkinsons_features: Optional[np.ndarray] = None,
    ) -> dict[IFStream, AnomalyResult]:
        """
        Score all available streams in one call.
        Pass None for any stream not yet available (e.g. no mic input).

        Returns
        -------
        Dict mapping IFStream → AnomalyResult
        Consumed directly by FusionEngine.
        """
        results = {}
        stream_map = {
            IFStream.CARDIAC:    cardiac_features,
            IFStream.FACIAL:     facial_features,
            IFStream.DYSARTHRIA: dysarthria_features,
            IFStream.PARKINSONS: parkinsons_features,
        }

        for stream, features in stream_map.items():
            if features is not None:
                results[stream] = self.score(stream, features)
            else:
                logger.debug("Stream %s skipped — no features provided.", stream.value)

        return results

    # ── Population priors (dataset warm-start) ─────────────────────────────

    def load_population_prior(self, stream: IFStream, features_matrix: np.ndarray):
        """
        Warm-start a stream's baseline with population-level dataset features.
        Call this before collecting personal baseline data so the system is
        usable from session 1.

        Typical use:
          - CARDIAC    ← feature vectors extracted from MIT-BIH AF normals
          - FACIAL     ← features from Facial Palsy dataset normal class
          - DYSARTHRIA ← features from TORGO normal speakers
          - PARKINSONS ← features from UCI Parkinson's healthy subjects
        """
        logger.info(
            "Loading population prior | stream=%s | samples=%d",
            stream.value, len(features_matrix),
        )
        self.add_baseline_batch(stream, features_matrix)

    # ── Anomaly evaluation (for judges) ───────────────────────────────────

    def evaluate(
        self,
        stream: IFStream,
        X_test: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Evaluate IF anomaly detection performance on labelled test data.

        Parameters
        ----------
        X_test : shape (N, n_features)
        y_true : shape (N,) — 1=anomaly, 0=normal

        Returns accuracy, precision, recall, F1, and false positive rate.
        """
        model = self._models[stream]
        if not model.is_trained:
            raise RuntimeError(f"Stream {stream.value} not trained.")

        X_scaled  = model.scaler.transform(X_test)
        raw_scores = model.forest.decision_function(X_scaled)
        norm_scores = np.array([self._normalise_score(s) for s in raw_scores])
        y_pred = (norm_scores >= ANOMALY_THRESHOLD).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy  = (tp + tn) / len(y_true)

        metrics = {
            "stream":            stream.value,
            "accuracy":          round(accuracy,  4),
            "precision":         round(precision, 4),
            "recall":            round(recall,    4),
            "f1_score":          round(f1,        4),
            "false_positive_rate": round(fpr,     4),
            "confusion_matrix":  {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "n_training_samples": model.n_training_samples,
        }

        logger.info(
            "IF evaluation | stream=%s | acc=%.3f | rec=%.3f | fpr=%.3f",
            stream.value, accuracy, recall, fpr,
        )
        return metrics

    # ── Save / load ────────────────────────────────────────────────────────

    def save(self, stream: IFStream):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"if_{stream.value}_{self.user_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._models[stream], f)
        logger.info("IF model saved → %s", path)

    def load(self, stream: IFStream):
        path = MODEL_DIR / f"if_{stream.value}_{self.user_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved IF model at {path}.")
        with open(path, "rb") as f:
            self._models[stream] = pickle.load(f)
        logger.info("IF model loaded ← %s", path)

    def save_all(self):
        for stream in IFStream:
            if self._models[stream].is_trained:
                self.save(stream)

    def load_all(self):
        for stream in IFStream:
            try:
                self.load(stream)
            except FileNotFoundError:
                logger.debug("No saved model for stream %s — will train fresh.", stream.value)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_score(raw_score: float) -> float:
        """
        Convert IF decision_function output to [0, 1].
        decision_function returns negative for anomalies, positive for normals.
        Typical range: [-0.5, 0.5].
        We map so that:
          raw =  0.5 → normalised = 0.0  (very normal)
          raw =  0.0 → normalised = 0.5  (boundary)
          raw = -0.5 → normalised = 1.0  (very anomalous)
        """
        clipped = np.clip(raw_score, -0.5, 0.5)
        return float((-clipped + 0.5))