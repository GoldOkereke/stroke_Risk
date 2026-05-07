from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.ml.cnn_model import AFibCNN
from app.services.ingestion import IngestionService
from app.services.segmentation import SegmentationService


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    metrics: dict[str, Any]
    roc_curve: dict[str, list[float]]
    confusion: list[list[int]]
    failure_cases: list[dict[str, Any]]


def _ensure_reports_dir(project_root: Path) -> Path:
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_cardiac_test_set() -> tuple[np.ndarray, np.ndarray]:
    ingestion = IngestionService()
    segmenter = SegmentationService()

    record_ids = ["04015", "04043", "04048", "04126"]
    segments: list[np.ndarray] = []
    labels: list[int] = []

    for record_id in record_ids:
        raw = ingestion.load_mitbih(record_id)
        result = segmenter.segment(raw)
        if result.cnn_labels.size == 0:
            continue
        segments.append(result.cnn_input)
        record_labels = np.array(
            [1 if label.upper() == "AFIB" else 0 for label in result.cnn_labels]
        )
        labels.append(record_labels)

    if not segments:
        raise RuntimeError("No cardiac segments available for evaluation.")

    X = np.concatenate(segments, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _extract_audio_features(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if samples.size == 0:
        return np.zeros(10, dtype=float)

    # Basic time-domain features.
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples))) > 0))

    # Spectral features.
    fft = np.fft.rfft(samples)
    mag = np.abs(fft) + 1e-8
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sr)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    rolloff_threshold = 0.85 * np.sum(mag)
    cumulative = np.cumsum(mag)
    rolloff = float(freqs[np.searchsorted(cumulative, rolloff_threshold)])

    # Signal shape features.
    max_amp = float(np.max(samples))
    min_amp = float(np.min(samples))
    range_amp = max_amp - min_amp

    return np.array([mean, std, rms, zcr, centroid, spread, rolloff, max_amp, min_amp, range_amp], dtype=float)


def _collect_dysarthria_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    settings = get_settings()
    torgo_root = settings.datasets_root / "speech" / "torgo"
    if not torgo_root.exists():
        raise FileNotFoundError(f"TORGO dataset not found at {torgo_root}")

    import soundfile as sf

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for label_dir, label in [("controls", 0), ("dysarthric", 1)]:
        base = torgo_root / label_dir
        if not base.exists():
            continue
        for wav_path in base.rglob("*.wav"):
            try:
                samples, sr = sf.read(str(wav_path), always_2d=False)
                feat = _extract_audio_features(np.asarray(samples), int(sr))
                features.append(feat)
                labels.append(label)
                paths.append(str(wav_path))
            except Exception:
                continue

    if not features:
        raise RuntimeError("No TORGO audio files were processed.")

    return np.vstack(features), np.array(labels), paths


def _parse_pts_file(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
            points.append([x, y])
        except ValueError:
            continue
    return np.asarray(points, dtype=float)


def _extract_facial_features(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(8, dtype=float)
    xs = points[:, 0]
    ys = points[:, 1]
    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))
    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    aspect = float(width / (height + 1e-6))
    median_x = float(np.median(xs))
    left = xs[xs <= median_x]
    right = xs[xs > median_x]
    symmetry = float(abs(left.mean() - (2 * median_x - right.mean()))) if left.size and right.size else 0.0
    return np.array([mean_x, mean_y, std_x, std_y, width, height, aspect, symmetry], dtype=float)


def _collect_facial_dataset() -> tuple[np.ndarray, list[str]]:
    settings = get_settings()
    aflfp_root = settings.datasets_root / "facial" / "facial_palsy_db" / "AFLFP"
    if not aflfp_root.exists():
        raise FileNotFoundError(f"AFLFP dataset not found at {aflfp_root}")

    features: list[np.ndarray] = []
    paths: list[str] = []
    for pts_path in aflfp_root.rglob("*.pts"):
        points = _parse_pts_file(pts_path)
        feat = _extract_facial_features(points)
        features.append(feat)
        paths.append(str(pts_path))

    if not features:
        raise RuntimeError("No AFLFP .pts files were processed.")

    return np.vstack(features), paths


def _collect_parkinsons_dataset() -> tuple[np.ndarray, np.ndarray]:
    settings = get_settings()
    csv_path = settings.datasets_root / "speech" / "uci_parkinsons" / "parkinsons_updrs.data"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(f"Parkinsons CSV is missing or empty: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    if "total_UPDRS" not in df.columns:
        raise ValueError("Parkinsons UPDRS data must include 'total_UPDRS'.")

    threshold = float(df["total_UPDRS"].median())
    labels = (df["total_UPDRS"] >= threshold).astype(int).to_numpy()
    features = df.drop(columns=["subject#", "total_UPDRS"], errors="ignore").to_numpy(
        dtype=float
    )
    return features, labels


def _evaluate_isolation_forest(
    features: np.ndarray,
    labels: np.ndarray,
    name: str,
    reports_dir: Path,
    failure_context: list[str] | None = None,
) -> EvaluationResult:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Train on normal/control samples only.
    train_mask = labels == 0
    if train_mask.sum() == 0:
        raise RuntimeError(f"No control samples available for {name}.")

    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    model.fit(scaled[train_mask])

    scores = -model.decision_function(scaled)
    preds = (scores > np.percentile(scores, 90)).astype(int)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    conf = confusion_matrix(labels, preds).tolist()

    failure_cases: list[dict[str, Any]] = []
    for idx, (label, pred, score) in enumerate(zip(labels, preds, scores)):
        if label != pred:
            failure = {
                "index": int(idx),
                "label": int(label),
                "prediction": int(pred),
                "score": float(score),
            }
            if failure_context:
                failure["source"] = failure_context[idx]
            failure_cases.append(failure)

    metrics = {
        "roc_auc": float(roc_auc),
        "total_samples": int(len(labels)),
        "control_samples": int(train_mask.sum()),
    }

    result = EvaluationResult(
        name=name,
        metrics=metrics,
        roc_curve={
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) for v in thresholds],
        },
        confusion=conf,
        failure_cases=failure_cases,
    )

    _save_json(reports_dir / f"{name}_metrics.json", result.metrics)
    _save_json(reports_dir / f"{name}_confusion.json", result.confusion)
    _save_json(reports_dir / f"{name}_roc.json", result.roc_curve)
    _save_json(reports_dir / f"{name}_failures.json", result.failure_cases)

    return result


def _evaluate_unsupervised(
    features: np.ndarray,
    name: str,
    reports_dir: Path,
    paths: list[str],
) -> EvaluationResult:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    model.fit(scaled)

    scores = -model.decision_function(scaled)
    top_indices = np.argsort(scores)[::-1][:50]

    failure_cases = [
        {"index": int(idx), "score": float(scores[idx]), "source": paths[idx]}
        for idx in top_indices
    ]

    metrics = {
        "total_samples": int(features.shape[0]),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "top_k": int(len(top_indices)),
    }

    result = EvaluationResult(
        name=name,
        metrics=metrics,
        roc_curve={"fpr": [], "tpr": [], "thresholds": []},
        confusion=[],
        failure_cases=failure_cases,
    )

    _save_json(reports_dir / f"{name}_metrics.json", result.metrics)
    _save_json(reports_dir / f"{name}_scores.json", scores.tolist())
    _save_json(reports_dir / f"{name}_failures.json", result.failure_cases)

    return result


def evaluate_cardiac_model() -> EvaluationResult:
    settings = get_settings()
    weights_path = settings.project_root / "weights" / "afib_cnn_ecg.keras"

    X, y = _collect_cardiac_test_set()
    model = AFibCNN()
    model.load(weights_path, (X.shape[1], X.shape[2]))

    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)

    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    conf = confusion_matrix(y, preds).tolist()

    failure_cases: list[dict[str, Any]] = []
    for idx, (label, prob, pred) in enumerate(zip(y, probs, preds)):
        if label != pred:
            failure_cases.append(
                {
                    "index": int(idx),
                    "label": int(label),
                    "prediction": int(pred),
                    "probability": float(prob),
                }
            )

    metrics = {
        "accuracy": float(np.mean(preds == y)),
        "roc_auc": float(roc_auc),
        "total_samples": int(len(y)),
    }

    return EvaluationResult(
        name="cardiac_afib_cnn",
        metrics=metrics,
        roc_curve={
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "thresholds": [float(v) for v in thresholds],
        },
        confusion=conf,
        failure_cases=failure_cases,
    )


def evaluate() -> list[EvaluationResult]:
    settings = get_settings()
    reports_dir = _ensure_reports_dir(settings.project_root)

    results: list[EvaluationResult] = []

    try:
        cardiac_result = evaluate_cardiac_model()
        results.append(cardiac_result)
        _save_json(reports_dir / "cardiac_metrics.json", cardiac_result.metrics)
        _save_json(reports_dir / "cardiac_confusion.json", cardiac_result.confusion)
        _save_json(reports_dir / "cardiac_roc.json", cardiac_result.roc_curve)
        _save_json(reports_dir / "cardiac_failures.json", cardiac_result.failure_cases)
    except Exception as exc:
        _save_json(
            reports_dir / "cardiac_error.json",
            {"error": str(exc)},
        )

    try:
        dys_features, dys_labels, dys_paths = _collect_dysarthria_dataset()
        results.append(
            _evaluate_isolation_forest(
                dys_features,
                dys_labels,
                "dysarthria_if",
                reports_dir,
                failure_context=dys_paths,
            )
        )
    except Exception as exc:
        _save_json(
            reports_dir / "dysarthria_error.json",
            {"error": str(exc)},
        )

    try:
        parkinsons_features, parkinsons_labels = _collect_parkinsons_dataset()
        results.append(
            _evaluate_isolation_forest(
                parkinsons_features,
                parkinsons_labels,
                "parkinsons_if",
                reports_dir,
            )
        )
    except Exception as exc:
        _save_json(
            reports_dir / "parkinsons_error.json",
            {"error": str(exc)},
        )

    try:
        facial_features, facial_paths = _collect_facial_dataset()
        results.append(
            _evaluate_unsupervised(
                facial_features,
                "facial_aflfp_if",
                reports_dir,
                facial_paths,
            )
        )
    except Exception as exc:
        _save_json(
            reports_dir / "facial_error.json",
            {"error": str(exc)},
        )

    return results


if __name__ == "__main__":
    evaluation_results = evaluate()
    for result in evaluation_results:
        print(f"{result.name} metrics:", result.metrics)
        print(f"{result.name} confusion:", result.confusion)
