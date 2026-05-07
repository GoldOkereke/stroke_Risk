from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.services.isolation_forest import IFStream


def _fit_and_save(
    stream: str,
    features: np.ndarray,
    weights_dir: Path,
    train_mask: np.ndarray | None = None,
) -> None:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)

    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool).reshape(-1)
        if mask.size != scaled.shape[0]:
            raise ValueError("train_mask must match number of feature rows.")
        if int(mask.sum()) == 0:
            raise RuntimeError(f"No training samples selected for stream '{stream}'.")
        model.fit(scaled[mask])
    else:
        model.fit(scaled)

    stream_dir = weights_dir / "isolation_forest"
    stream_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, stream_dir / f"{stream}_prior.pkl")
    joblib.dump(scaler, stream_dir / f"{stream}_scaler.pkl")


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
    """Cheap geometric summary features from AFLFP `.pts` landmark files."""
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
    symmetry = (
        float(abs(left.mean() - (2 * median_x - right.mean())))
        if left.size and right.size
        else 0.0
    )
    return np.array([mean_x, mean_y, std_x, std_y, width, height, aspect, symmetry], dtype=float)


def _collect_facial_dataset() -> np.ndarray:
    settings = get_settings()
    aflfp_root = settings.datasets_root / "facial" / "facial_palsy_db" / "AFLFP"
    if not aflfp_root.exists():
        raise FileNotFoundError(f"AFLFP dataset not found at {aflfp_root}")

    features: list[np.ndarray] = []
    for pts_path in aflfp_root.rglob("*.pts"):
        feat = _extract_facial_features(_parse_pts_file(pts_path))
        features.append(feat)

    if not features:
        raise RuntimeError("No AFLFP .pts files were processed.")

    return np.vstack(features)


def _extract_audio_features(samples: np.ndarray, sr: int) -> np.ndarray:
    """Cheap audio features (no librosa) used for IsolationForest prior training."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if samples.size == 0 or sr <= 0:
        return np.zeros(10, dtype=float)

    mean = float(np.mean(samples))
    std = float(np.std(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples))) > 0))

    fft = np.fft.rfft(samples)
    mag = np.abs(fft) + 1e-8
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sr)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    rolloff_threshold = 0.85 * np.sum(mag)
    cumulative = np.cumsum(mag)
    rolloff = float(freqs[np.searchsorted(cumulative, rolloff_threshold)])

    max_amp = float(np.max(samples))
    min_amp = float(np.min(samples))
    range_amp = max_amp - min_amp

    return np.array([mean, std, rms, zcr, centroid, spread, rolloff, max_amp, min_amp, range_amp], dtype=float)


def _collect_dysarthria_dataset() -> tuple[np.ndarray, np.ndarray]:
    settings = get_settings()
    torgo_root = settings.datasets_root / "speech" / "torgo"
    if not torgo_root.exists():
        raise FileNotFoundError(f"TORGO dataset not found at {torgo_root}")

    import soundfile as sf

    features: list[np.ndarray] = []
    labels: list[int] = []

    for label_dir, label in [("controls", 0), ("dysarthric", 1)]:
        base = torgo_root / label_dir
        if not base.exists():
            continue
        for wav_path in base.rglob("*.wav"):
            try:
                samples, sr = sf.read(str(wav_path), always_2d=False)
                features.append(_extract_audio_features(np.asarray(samples), int(sr)))
                labels.append(label)
            except Exception:
                continue

    if not features:
        raise RuntimeError("No TORGO audio files were processed.")

    return np.vstack(features), np.asarray(labels, dtype=int)


def _collect_parkinsons_dataset() -> tuple[np.ndarray, np.ndarray]:
    settings = get_settings()
    csv_path = settings.datasets_root / "speech" / "uci_parkinsons" / "parkinsons_updrs.data"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(f"Parkinsons UPDRS data is missing or empty: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    if "total_UPDRS" not in df.columns:
        raise ValueError("Parkinsons UPDRS data must include 'total_UPDRS'.")

    threshold = float(df["total_UPDRS"].median())
    labels = (df["total_UPDRS"] >= threshold).astype(int).to_numpy()

    # Use all columns except subject id + label as features.
    features = df.drop(columns=["subject#", "total_UPDRS"], errors="ignore").to_numpy(dtype=float)
    return features, labels


def _collect_cardiac_dataset() -> np.ndarray:
    """Collect HRV-like feature vectors from the segmentation pipeline (5 features)."""
    from app.services.ingestion import IngestionService
    from app.services.segmentation import SegmentationService

    ingestion = IngestionService()
    segmenter = SegmentationService()
    record_ids = ["04015", "04043", "04048", "04126"]

    feats: list[np.ndarray] = []
    for rid in record_ids:
        raw = ingestion.load_mitbih(rid)
        result = segmenter.segment(raw)
        if result.if_features.size:
            feats.append(result.if_features)

    if not feats:
        raise RuntimeError("No cardiac IF features available (segmentation produced none).")

    return np.vstack(feats)


def train_isolation_forests() -> None:
    settings = get_settings()
    weights_dir = settings.project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # FACIAL: unsupervised prior on all AFLFP features.
    facial_features = _collect_facial_dataset()
    _fit_and_save(IFStream.FACIAL.value, facial_features, weights_dir)

    # DYSARTHRIA: train prior on controls only (labels==0).
    dys_features, dys_labels = _collect_dysarthria_dataset()
    _fit_and_save(IFStream.DYSARTHRIA.value, dys_features, weights_dir, train_mask=(dys_labels == 0))

    # PARKINSONS: train prior on "lower symptom" half only (labels==0).
    pk_features, pk_labels = _collect_parkinsons_dataset()
    _fit_and_save(IFStream.PARKINSONS.value, pk_features, weights_dir, train_mask=(pk_labels == 0))

    # CARDIAC: optional prior from segmentation HRV-like features (unlabeled).
    try:
        cardiac_features = _collect_cardiac_dataset()
        _fit_and_save(IFStream.CARDIAC.value, cardiac_features, weights_dir)
    except Exception:
        # Cardiac IF is optional; do not fail the overall training if unavailable.
        pass


if __name__ == "__main__":
    train_isolation_forests()
