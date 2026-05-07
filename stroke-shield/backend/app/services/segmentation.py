from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.services.ingestion import RawSignalData


@dataclass(frozen=True)
class Segment:
    samples: np.ndarray
    start_time: float
    label: str


@dataclass(frozen=True)
class SegmentationResult:
    segments: list[Segment]
    cnn_input: np.ndarray
    cnn_labels: np.ndarray
    if_features: np.ndarray
    quality_scores: np.ndarray


class SegmentationService:
    def segment(
        self,
        raw_signal: RawSignalData,
        window_seconds: int = 30,
        overlap: float = 0.5,
    ) -> SegmentationResult:
        values = np.asarray(raw_signal.values, dtype=float)
        fs = float(raw_signal.sampling_rate)
        if values.size == 0 or fs <= 0:
            return SegmentationResult([], np.empty((0, 0, 1)), np.empty(0), np.empty((0, 5)), np.empty(0))

        label_array = self._annotations_to_array(
            raw_signal.annotation_samples,
            raw_signal.annotation_aux,
            values.size
        )
        window_samples = max(1, int(window_seconds * fs))
        step = max(1, int(window_samples * (1 - overlap)))

        segments: list[Segment] = []
        cnn_inputs: list[np.ndarray] = []
        cnn_labels: list[str] = []
        if_features: list[list[float]] = []
        quality_scores: list[float] = []

        for start in range(0, values.size - window_samples + 1, step):
            end = start + window_samples
            window = values[start:end]
            label = self._majority_label(label_array[start:end])

            quality = self._quality_score(window)
            if quality < 0.4:
                continue

            rmssd, sdnn, rr_irregularity, mean_rr, cv_rr = self._hrv_features(window, fs)
            if_features.append([rmssd, sdnn, rr_irregularity, mean_rr, cv_rr])

            normalized = self._normalize(window)
            cnn_inputs.append(normalized[:, None])
            cnn_labels.append(label)
            quality_scores.append(quality)
            segments.append(Segment(samples=window, start_time=start / fs, label=label))

        if not cnn_inputs:
            return SegmentationResult([], np.empty((0, window_samples, 1)), np.empty(0), np.empty((0, 5)), np.empty(0))

        return SegmentationResult(
            segments=segments,
            cnn_input=np.stack(cnn_inputs, axis=0),
            cnn_labels=np.asarray(cnn_labels),
            if_features=np.asarray(if_features, dtype=float),
            quality_scores=np.asarray(quality_scores, dtype=float),
        )

    @staticmethod
    def _annotations_to_array(samples, aux_notes, length):
        label_array = np.full(length, "UNKNOWN", dtype=object)

        if samples is None or aux_notes is None:
            return label_array

        current_label = "UNKNOWN"
        j = 0

        for i in range(length):
            if j < len(samples) and i >= samples[j]:
                note = str(aux_notes[j]).upper()

                if "AFIB" in note:
                    current_label = "AFIB"
                elif "N" in note:
                    current_label = "NORMAL"

                j += 1

            label_array[i] = current_label

        return label_array

    @staticmethod
    def _majority_label(window_labels: np.ndarray) -> str:
        if window_labels.size == 0:
            return "UNKNOWN"
        values, counts = np.unique(window_labels, return_counts=True)
        return str(values[np.argmax(counts)])

    @staticmethod
    def _quality_score(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        std = float(np.std(window))
        if std < 1e-6:
            return 0.0

        clip_eps = 1e-6
        min_val = float(np.min(window))
        max_val = float(np.max(window))
        clip_ratio = float(np.mean((np.abs(window - min_val) < clip_eps) | (np.abs(window - max_val) < clip_eps)))

        diff = np.diff(window)
        noise = float(np.mean(np.abs(diff))) + 1e-6
        snr_score = (std / noise) / ((std / noise) + 1.0)

        flat_penalty = 0.0 if std > 1e-3 else 1.0
        quality = 0.4 * snr_score + 0.3 * (1 - clip_ratio) + 0.3 * (1 - flat_penalty)
        return float(np.clip(quality, 0.0, 1.0))

    @staticmethod
    def _hrv_features(window: np.ndarray, fs: float) -> tuple[float, float, float, float, float]:
        peak_indices = SegmentationService._simple_peak_indices(window)
        if peak_indices.size < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        rr = np.diff(peak_indices) / fs
        mean_rr = float(np.mean(rr))
        sdnn = float(np.std(rr))
        rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr))))) if rr.size > 1 else 0.0
        rr_irregularity = float(np.mean(np.abs(np.diff(rr)))) if rr.size > 1 else 0.0
        cv_rr = float(sdnn / mean_rr) if mean_rr > 0 else 0.0
        return rmssd, sdnn, rr_irregularity, mean_rr, cv_rr

    @staticmethod
    def _simple_peak_indices(window: np.ndarray) -> np.ndarray:
        if window.size < 3:
            return np.empty(0, dtype=int)
        threshold = np.percentile(window, 90)
        peaks = (window[1:-1] > window[:-2]) & (window[1:-1] > window[2:]) & (window[1:-1] > threshold)
        return np.where(peaks)[0] + 1

    @staticmethod
    def _normalize(window: np.ndarray) -> np.ndarray:
        mean = float(np.mean(window))
        std = float(np.std(window))

        if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-6:
            return np.zeros_like(window)

        normalized = (window - mean) / (std + 1e-6)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized
