from __future__ import annotations

from pathlib import Path
from typing import Tuple
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

from app.core.config import get_settings
from app.ml.cnn_model import AFibCNN
from app.services.ingestion import IngestionService
from app.services.segmentation import SegmentationService


def _split_data(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _standardize_dataset(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mean = float(np.mean(X))
    std = float(np.std(X))
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-6:
        return X
    return (X - mean) / (std + 1e-6)


def train_ecg_model() -> None:
    settings = get_settings()
    weights_dir = settings.project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    ingestion = IngestionService()
    segmenter = SegmentationService()

    record_ids = ["04015", "04043", "04048", "04126"]
    segments: list[np.ndarray] = []
    labels: list[int] = []

    for record_id in record_ids:
        raw = ingestion.load_mitbih(record_id)
        result = segmenter.segment(raw)
        print("Unique cnn_labels:", set(result.cnn_labels))
        if result.cnn_labels.size > 0:
            print("Raw labels sample:", result.cnn_labels[:20])
            unique_labels = set(label.upper() for label in result.cnn_labels)
            print("Unique raw labels:", unique_labels)
            segments.append(result.cnn_input)
            record_labels = np.array(
                [1 if "AF" in label.upper() else 0 for label in result.cnn_labels]
            )
            labels.append(record_labels)
    if not segments:
        raise RuntimeError("No ECG segments available for training.")

    X = np.concatenate(segments, axis=0)
    y = np.concatenate(labels, axis=0)

    # Check for NaNs
    print("NaNs before cleaning (ECG X):", np.isnan(X).sum())
    print("NaNs before cleaning (ECG y):", np.isnan(y).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(int)
    print("NaNs after cleaning (ECG X):", np.isnan(X).sum())
    print("NaNs after cleaning (ECG y):", np.isnan(y).sum())

    X = _standardize_dataset(X)
    print("ECG dataset mean after standardization:", float(np.mean(X)))
    print("ECG dataset std after standardization:", float(np.std(X)))

    print("Total samples:", len(y))
    print("Unique labels:", np.unique(y))
    print("Label counts:", np.unique(y, return_counts=True))

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y)

    # Oversample AFIB to balance classes
    if len(np.unique(y_train)) > 1:
        X_afib = X_train[y_train == 1]
        y_afib = y_train[y_train == 1]
        X_normal = X_train[y_train == 0]
        y_normal = y_train[y_train == 0]
        X_afib_up, y_afib_up = resample(
            X_afib, y_afib,
            replace=True,
            n_samples=len(y_normal),
            random_state=42
        )
        X_train = np.concatenate([X_normal, X_afib_up])
        y_train = np.concatenate([y_normal, y_afib_up])

    print("=" * 60)
    print("ECG LABEL DISTRIBUTION")
    print("=" * 60)
    print("FULL DATA:", np.unique(y, return_counts=True))
    print("TRAIN:", np.unique(y_train, return_counts=True))
    print("VAL:", np.unique(y_val, return_counts=True))
    print("TEST:", np.unique(y_test, return_counts=True))
    print("=" * 60)

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    print("ECG label distribution:", class_counts)
    if len(np.unique(y_train)) > 1:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight = dict(enumerate(class_weights))
        print("ECG class weights:", class_weight)
    else:
        class_weight = None
        print("Warning: training labels contain only one class. Model may not learn discriminative features.")

    model = AFibCNN()
    model.build((X.shape[1], X.shape[2]))
    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weight=class_weight,
        checkpoint_path=weights_dir / "afib_cnn_ecg_best.weights.h5",
    )
    model.save(weights_dir / "afib_cnn_ecg.weights.h5")

    metrics = model.evaluate(X_test, y_test)
    print("ECG test metrics:", metrics)
    print("Validation-calibrated threshold:", model.threshold)

    # Debug predictions
    probs = model.predict(X_test)
    preds = (probs >= model.threshold).astype(int)
    print("ECG predictions unique:", np.unique(preds))
    print("ECG prediction counts:", np.unique(preds, return_counts=True))
    from sklearn.metrics import confusion_matrix
    print("ECG confusion matrix:\n", confusion_matrix(y_test, preds))

    # Inspect probabilities
    print("Min prob:", probs.min())
    print("Max prob:", probs.max())
    print("Sample probs:", probs[:20])

    # Test-set threshold for diagnostics only
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Test-derived optimal threshold (diagnostic):", optimal_threshold)

    preds_opt = (probs > optimal_threshold).astype(int)
    print("Optimal confusion matrix (diagnostic):\n", confusion_matrix(y_test, preds_opt))


def train_ppg_transfer() -> None:
    settings = get_settings()
    weights_dir = settings.project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    ingestion = IngestionService()
    segmenter = SegmentationService()

    af_records = [
        "mimic_perform_af_001",
        "mimic_perform_af_002",
        "mimic_perform_af_003",
        "mimic_perform_af_004",
        "mimic_perform_af_005",
    ]
    non_af_records = [
        "mimic_perform_non_af_001",
        "mimic_perform_non_af_002",
        "mimic_perform_non_af_003",
        "mimic_perform_non_af_004",
        "mimic_perform_non_af_005",
    ]

    segments: list[np.ndarray] = []
    labels: list[int] = []

    for record_id in af_records:
        raw = ingestion.load_mimic(
            record_id,
            dataset="mimic_perform_af_wfdb",
            label="AFIB",
        )
        result = segmenter.segment(raw)
        print("Unique cnn_labels:", set(result.cnn_labels))
        if result.cnn_labels.size > 0:
            print("Raw labels sample:", result.cnn_labels[:20])
            unique_labels = set(label.upper() for label in result.cnn_labels)
            print("Unique raw labels:", unique_labels)
            segments.append(result.cnn_input)
            record_labels = np.array(
                [1 if "AF" in label.upper() else 0 for label in result.cnn_labels]
            )
            labels.append(record_labels)

    for record_id in non_af_records:
        raw = ingestion.load_mimic(
            record_id,
            dataset="mimic_perform_non_af_wfdb",
            label="NORMAL",
        )
        result = segmenter.segment(raw)
        print("Unique cnn_labels:", set(result.cnn_labels))
        if result.cnn_labels.size > 0:
            print("Raw labels sample:", result.cnn_labels[:20])
            unique_labels = set(label.upper() for label in result.cnn_labels)
            print("Unique raw labels:", unique_labels)
            segments.append(result.cnn_input)
            record_labels = np.array(
                [1 if "AF" in label.upper() else 0 for label in result.cnn_labels]
            )
            labels.append(record_labels)

    if not segments:
        raise RuntimeError("No PPG segments available for training.")

    X = np.concatenate(segments, axis=0)
    y = np.concatenate(labels, axis=0)

    # Check for NaNs
    print("NaNs before cleaning (PPG X):", np.isnan(X).sum())
    print("NaNs before cleaning (PPG y):", np.isnan(y).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(int)
    print("NaNs after cleaning (PPG X):", np.isnan(X).sum())
    print("NaNs after cleaning (PPG y):", np.isnan(y).sum())

    X = _standardize_dataset(X)
    print("PPG dataset mean after standardization:", float(np.mean(X)))
    print("PPG dataset std after standardization:", float(np.std(X)))

    print("PPG total samples:", len(y))
    print("PPG unique labels:", np.unique(y))
    print("PPG label counts:", np.unique(y, return_counts=True))

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(X, y)

    # Oversample AFIB to balance classes
    if len(np.unique(y_train)) > 1:
        X_afib = X_train[y_train == 1]
        y_afib = y_train[y_train == 1]
        X_normal = X_train[y_train == 0]
        y_normal = y_train[y_train == 0]
        X_afib_up, y_afib_up = resample(
            X_afib, y_afib,
            replace=True,
            n_samples=len(y_normal),
            random_state=42
        )
        X_train = np.concatenate([X_normal, X_afib_up])
        y_train = np.concatenate([y_normal, y_afib_up])

    print("=" * 60)
    print("PPG LABEL DISTRIBUTION")
    print("=" * 60)
    print("FULL DATA:", np.unique(y, return_counts=True))
    print("TRAIN:", np.unique(y_train, return_counts=True))
    print("VAL:", np.unique(y_val, return_counts=True))
    print("TEST:", np.unique(y_test, return_counts=True))
    print("=" * 60)

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    print("PPG label distribution:", class_counts)
    if len(np.unique(y_train)) > 1:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight = dict(enumerate(class_weights))
        print("PPG class weights:", class_weight)
    else:
        class_weight = None
        print("Warning: PPG training labels contain only one class. Transfer learning may not work properly.")

    model = AFibCNN()
    model.build((X.shape[1], X.shape[2]))

    # Check if ECG weights exist (for future transfer learning)
    ecg_weights_path = weights_dir / "afib_cnn_ecg.weights.h5"
    if not ecg_weights_path.exists():
        print("Warning: ECG weights not found. Training PPG from scratch.")

    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weight=class_weight,
        checkpoint_path=weights_dir / "afib_cnn_ppg_best.weights.h5",
    )
    model.save(weights_dir / "afib_cnn_ppg.weights.h5")

    metrics = model.evaluate(X_test, y_test)
    print("PPG test metrics:", metrics)
    print("Validation-calibrated threshold:", model.threshold)

    # Debug predictions
    probs = model.predict(X_test)
    preds = (probs >= model.threshold).astype(int)
    print("PPG predictions unique:", np.unique(preds))
    print("PPG prediction counts:", np.unique(preds, return_counts=True))
    from sklearn.metrics import confusion_matrix
    print("PPG confusion matrix:\n", confusion_matrix(y_test, preds))

    # Inspect probabilities
    print("Min prob:", probs.min())
    print("Max prob:", probs.max())
    print("Sample probs:", probs[:20])

    # Test-set threshold for diagnostics only
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Test-derived optimal threshold (diagnostic):", optimal_threshold)

    preds_opt = (probs > optimal_threshold).astype(int)
    print("Optimal confusion matrix (diagnostic):\n", confusion_matrix(y_test, preds_opt))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ecg":
        train_ecg_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "ppg":
        train_ppg_transfer()
    else:
        print("Usage: python -m app.ml.train_cnn [ecg|ppg]")
