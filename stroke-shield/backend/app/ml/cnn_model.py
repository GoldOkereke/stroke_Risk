from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers


@dataclass(frozen=True)
class PredictionResult:
    afib_probability: float
    prediction: str
    confidence: float
    uncertain: bool


class AFibCNN:
    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.threshold = 0.5

    def build(self, input_shape: tuple[int, int]) -> tf.keras.Model:
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(
            32,
            kernel_size=50,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            32,
            kernel_size=25,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            64,
            kernel_size=10,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            64,
            kernel_size=5,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="afib_cnn")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weight: dict[int, float] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> tf.keras.callbacks.History:
        self._ensure_built(X_train)

        callbacks_list: list[tf.keras.callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=5,
                mode="max",
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                mode="min",
                min_lr=1e-6,
            ),
        ]

        if checkpoint_path is not None:
            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_auc",
                    mode="max",
                    save_best_only=True,
                    save_weights_only=True,
                )
            )

        history = self.model.fit(
            self._ensure_3d(X_train),
            y_train,
            validation_data=(self._ensure_3d(X_val), y_val),
            class_weight=class_weight,
            callbacks=callbacks_list,
            epochs=50,
            batch_size=64,
            verbose=1,
        )

        if len(np.unique(y_val)) > 1:
            val_probs = self.predict(X_val)
            self.threshold = self._find_optimal_threshold(y_val, val_probs)
            print("Calibrated validation threshold:", self.threshold)
        else:
            print("Validation set contains only one class; keeping default threshold", self.threshold)

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        self._ensure_built(X_test)
        probs = self.predict(X_test)
        preds = (probs >= self.threshold).astype(int)

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
        cm = confusion_matrix(y_test, preds).tolist()

        uncertain_mask = (probs >= 0.35) & (probs <= 0.65)
        uncertain_indices = np.where(uncertain_mask)[0].tolist()

        false_negatives = int(np.sum((y_test == 1) & (preds == 0)))

        return {
            "accuracy": float(np.mean(preds == y_test)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "confusion_matrix": cm,
            "uncertain_indices": uncertain_indices,
            "uncertain_count": int(len(uncertain_indices)),
            "false_negatives": false_negatives,
        }

    @staticmethod
    def _find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, probs)
        optimal_idx = int(np.argmax(tpr - fpr))
        return float(thresholds[optimal_idx]) if thresholds.size > 0 else 0.5

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_built(X)
        probs = self.model.predict(self._ensure_3d(X), verbose=0).reshape(-1)
        return probs

    def predict_single(self, segment: np.ndarray) -> PredictionResult:
        probs = self.predict(np.asarray(segment))
        prob = float(probs[0])
        prediction = "AFIB" if prob >= self.threshold else "NORMAL"
        confidence = float(abs(prob - self.threshold))
        uncertain = 0.35 <= prob <= 0.65
        return PredictionResult(
            afib_probability=prob,
            prediction=prediction,
            confidence=confidence,
            uncertain=uncertain,
        )

    def save(self, path: str | Path) -> None:
        self._ensure_model()
        path = Path(path)
        if path.suffix != ".h5" or not path.name.endswith(".weights.h5"):
            path = path.with_suffix("")
            path = path.with_name(f"{path.name}.weights.h5")
        self.model.save_weights(str(path))

    def load(self, path: str | Path, input_shape: tuple[int, int]) -> None:
        self.build(input_shape)
        path = Path(path)
        if path.suffix != ".h5" or not path.name.endswith(".weights.h5"):
            path = path.with_suffix("")
            path = path.with_name(f"{path.name}.weights.h5")
        self.model.load_weights(str(path))

    def load_ecg_weights_for_ppg(
        self,
        ecg_weights_path: str | Path,
        input_shape: tuple[int, int],
    ) -> None:
        self.build(input_shape)
        self.model.load_weights(str(ecg_weights_path))

        for layer in self.model.layers:
            if isinstance(layer, layers.Conv1D):
                layer.trainable = False

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def _ensure_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been built. Call build() first.")

    def _ensure_built(self, X: np.ndarray) -> None:
        if self.model is None:
            X = self._ensure_3d(X)
            self.build((X.shape[1], X.shape[2]))

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :, None]
        elif X.ndim == 2:
            # If the input is a single 2D sample with shape (length, 1), keep it as one example.
            if X.shape[1] == 1:
                X = X[None, :, :]
            else:
                X = X[:, :, None]
        return X
