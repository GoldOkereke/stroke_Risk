"""Isolation Forest personalization and scoring.

This module implements a small "personal baseline" pipeline:
- collect baseline feature vectors per user per stream
- train an IsolationForest once enough baseline samples exist
- score new feature vectors and normalize to a 0-1 anomaly score

Design goals:
- optional by default: missing models never block app startup
- filesystem-backed models: stored under `<project_root>/weights/isolation_forest/`
- consistent typing: returns `app.models.schemas.AnomalyResult` for API use
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.models.schemas import AnomalyResult


class IFStream(str, Enum):
    """Supported Isolation Forest streams.

    Using an Enum keeps filenames predictable while still being easy to serialize.
    """

    CARDIAC = "cardiac"
    FACIAL = "facial"
    DYSARTHRIA = "dysarthria"
    PARKINSONS = "parkinsons"


@dataclass(frozen=True)
class _ModelBundle:
    """In-memory holder for a scaler + model pair."""

    scaler: StandardScaler
    model: IsolationForest


class IsolationForestService:
    """Per-user baseline training and scoring for multiple streams."""

    def __init__(
        self,
        user_id: str,
        baseline_threshold: int = 10,
        n_estimators: int = 200,
        contamination: float = 0.1,
        random_state: int = 42,
        weights_dir: Optional[Path] = None,
    ) -> None:
        # user_id scopes "personal models" to a single person.
        self.user_id = str(user_id)

        # How many baseline samples we require before training a personal model.
        self.baseline_threshold = int(baseline_threshold)

        # Training knobs for Isolation Forest.
        self.n_estimators = int(n_estimators)
        self.contamination = float(contamination)
        self.random_state = int(random_state)

        # Baselines are stored in-memory for now (you can persist later if needed).
        self._baselines: dict[IFStream, list[np.ndarray]] = {s: [] for s in IFStream}

        # We lazily load user models on demand.
        self._loaded_models: dict[IFStream, Optional[_ModelBundle]] = {s: None for s in IFStream}

        settings = get_settings()
        base = weights_dir or (settings.project_root / "weights")
        self._stream_dir = base / "isolation_forest"
        self._stream_dir.mkdir(parents=True, exist_ok=True)

    def _user_model_path(self, stream: IFStream) -> Path:
        return self._stream_dir / f"{stream.value}_{self.user_id}.pkl"

    def _user_scaler_path(self, stream: IFStream) -> Path:
        return self._stream_dir / f"{stream.value}_{self.user_id}_scaler.pkl"

    def _prior_model_path(self, stream: IFStream) -> Path:
        return self._stream_dir / f"{stream.value}_prior.pkl"

    def _prior_scaler_path(self, stream: IFStream) -> Path:
        return self._stream_dir / f"{stream.value}_scaler.pkl"

    @staticmethod
    def _as_vector(features: Any) -> np.ndarray:
        """Convert an input feature object to a 1D float vector."""
        arr = np.asarray(features, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("Feature vector is empty.")
        return arr

    def add_baseline(self, stream: IFStream, features: Any) -> int:
        """Add one baseline feature vector for a stream.

        Returns the new baseline count for that stream.
        """
        vec = self._as_vector(features)
        self._baselines[stream].append(vec)

        # Auto-train once we reach the threshold.
        if len(self._baselines[stream]) >= self.baseline_threshold:
            self.train(stream)

        return len(self._baselines[stream])

    def train(self, stream: IFStream) -> None:
        """Train and persist a personal model for the given stream."""
        baseline = self._baselines.get(stream, [])
        if len(baseline) < 2:
            # Isolation Forest cannot learn anything useful from a single point.
            return

        X = np.vstack(baseline)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(Xs)

        # Persist both scaler and model to disk.
        joblib.dump(model, self._user_model_path(stream))
        joblib.dump(scaler, self._user_scaler_path(stream))

        self._loaded_models[stream] = _ModelBundle(scaler=scaler, model=model)

    def load_user_model(self, stream: IFStream) -> Optional[_ModelBundle]:
        """Load a user's personal model if it exists."""
        cached = self._loaded_models.get(stream)
        if cached is not None:
            return cached

        model_path = self._user_model_path(stream)
        scaler_path = self._user_scaler_path(stream)
        if not model_path.exists() or not scaler_path.exists():
            self._loaded_models[stream] = None
            return None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        bundle = _ModelBundle(scaler=scaler, model=model)
        self._loaded_models[stream] = bundle
        return bundle

    def load_population_prior(self, stream: IFStream) -> Optional[_ModelBundle]:
        """Load a population prior model if it exists."""
        model_path = self._prior_model_path(stream)
        scaler_path = self._prior_scaler_path(stream)
        if not model_path.exists() or not scaler_path.exists():
            return None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return _ModelBundle(scaler=scaler, model=model)

    @staticmethod
    def _normalize_decision(raw: float) -> float:
        """Map sklearn decision_function score to [0, 1] anomaly score.

        - decision_function: higher => more normal (inlier), lower => more anomalous (outlier)
        - we want: higher => more anomalous

        We use a smooth, monotonic mapping so the score is stable across streams.
        """
        return float(1.0 / (1.0 + np.exp(raw)))

    @staticmethod
    def _confidence_from_score(score01: float) -> float:
        """Compute a simple confidence: distance from the decision boundary (0.5)."""
        return float(min(1.0, abs(score01 - 0.5) * 2.0))

    def score(self, stream: IFStream, features: Any) -> Optional[AnomalyResult]:
        """Score features for a stream using personal model or population prior.

        Returns None if neither model is available.
        """
        vec = self._as_vector(features)

        bundle = self.load_user_model(stream)
        if bundle is None:
            bundle = self.load_population_prior(stream)
        if bundle is None:
            return None

        x = bundle.scaler.transform(vec.reshape(1, -1))
        raw = float(bundle.model.decision_function(x)[0])
        score01 = self._normalize_decision(raw)

        return AnomalyResult(
            score=float(score01),
            confidence=self._confidence_from_score(score01),
            contributing_features={},  # IF doesn't expose per-feature contributions without extra tooling.
        )

    def score_all(self, features_by_stream: dict[IFStream, Any]) -> dict[IFStream, AnomalyResult]:
        """Score multiple streams and return only those with available models."""
        results: dict[IFStream, AnomalyResult] = {}
        for stream, features in features_by_stream.items():
            result = self.score(stream, features)
            if result is not None:
                results[stream] = result
        return results

    def load_population_prior_from_matrix(self, stream: IFStream, features_matrix: np.ndarray) -> None:
        """Train and persist a population prior model from a feature matrix."""
        X = np.asarray(features_matrix, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("features_matrix must be 2D with at least 2 rows.")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(Xs)

        joblib.dump(model, self._prior_model_path(stream))
        joblib.dump(scaler, self._prior_scaler_path(stream))

    def evaluate(
        self,
        stream: IFStream,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluate a model against labeled data.

        y_test convention:
        - 0 = normal
        - 1 = anomaly
        """
        bundle = self.load_user_model(stream) or self.load_population_prior(stream)
        if bundle is None:
            raise RuntimeError(f"No model available for stream '{stream.value}'.")

        X = np.asarray(X_test, dtype=float)
        y = np.asarray(y_test, dtype=int).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X_test must be a 2D matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_test rows must match y_test length.")

        Xs = bundle.scaler.transform(X)
        raw = bundle.model.decision_function(Xs)
        scores = np.array([self._normalize_decision(v) for v in raw], dtype=float)
        preds = (scores >= float(threshold)).astype(int)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
        }

