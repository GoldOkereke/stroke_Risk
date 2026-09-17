"""
app/services/segmentation.py
==============================
Splits a filtered ECG/PPG signal into labelled 30-second windows.

Two-stage process:
  1. Annotation-driven split  — uses MIT-BIH .atr labels to mark
                                rhythm boundaries (AFIB vs NORMAL)
  2. Fixed-window slicing     — cuts each rhythm block into 30-second
                                segments (7500 samples @ 250Hz ECG,
                                3750 samples @ 125Hz PPG)

Each segment is quality-scored and flagged invalid if:
  - Signal is clipped (hits ADC rail)
  - Signal is flat (no variation — lead-off or dropout)
  - SNR is below threshold
  - Segment is too short (< 50% of window size)

Output shape:  (N_valid, window_samples, 1)  — ready for FR2 1D CNN
Output labels: (N_valid,)                    — 1=AFIB, 0=NORMAL
Output features: (N_valid, N_features)       — ready for FR3 Isolation Forest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.models.signal_schemas import RhythmLabel, SignalType, SegmentSchema
from app.services.ingestion import RawSignalData

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

WINDOW_SECONDS   = 30       # seconds per segment
MIN_QUALITY      = 0.4      # below this → is_valid = False
CLIP_THRESHOLD   = 0.98     # fraction of max ADC value = clipped
FLAT_STD_THRESH  = 0.001    # std below this = flat/lead-off signal
SNR_THRESH_DB    = 5.0      # minimum acceptable SNR in dB


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class Segment:
    """A single windowed segment — internal representation."""
    index: int
    start_sample: int
    end_sample: int
    data: np.ndarray          # filtered samples, shape (W,)
    signal_type: SignalType
    rhythm_label: RhythmLabel
    sample_rate: float
    quality_score: float = 0.0
    is_valid: bool = True

    # HRV features (computed in _extract_features)
    hrv_rmssd: Optional[float] = None
    hrv_sdnn: Optional[float]  = None
    rr_irregularity: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / self.sample_rate

    def to_schema(self) -> SegmentSchema:
        return SegmentSchema(
            index            = self.index,
            start_sample     = self.start_sample,
            end_sample       = self.end_sample,
            duration_seconds = self.duration_seconds,
            signal_type      = self.signal_type,
            rhythm_label     = self.rhythm_label,
            quality_score    = self.quality_score,
            is_valid         = self.is_valid,
            data             = self.data.tolist(),
            hrv_rmssd        = self.hrv_rmssd,
            hrv_sdnn         = self.hrv_sdnn,
            rr_irregularity  = self.rr_irregularity,
        )


@dataclass
class SegmentationResult:
    """
    Full output from the segmentation pipeline.
    Consumed directly by FR2 (CNN) and FR3 (Isolation Forest).
    """
    segments: list[Segment]
    sample_rate: float
    signal_type: SignalType
    window_samples: int

    # Ready-to-use arrays for ML models
    cnn_input: np.ndarray          # shape (N_valid, W, 1)
    cnn_labels: np.ndarray         # shape (N_valid,) — 1=AFIB, 0=NORMAL
    isolation_forest_features: np.ndarray  # shape (N_valid, N_features)

    notes: list[str] = field(default_factory=list)

    @property
    def valid_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_valid]

    @property
    def afib_count(self) -> int:
        return sum(1 for s in self.valid_segments if s.rhythm_label == RhythmLabel.AFIB)

    @property
    def normal_count(self) -> int:
        return sum(1 for s in self.valid_segments if s.rhythm_label == RhythmLabel.NORMAL)


# ── Main segmentation service ──────────────────────────────────────────────

class SegmentationService:
    """
    Segments a filtered signal into labelled windows.

    Usage
    -----
        svc = SegmentationService()
        result = svc.segment(filtered_signal, raw_data)

        # Feed into CNN
        model.predict(result.cnn_input)

        # Feed into Isolation Forest
        iso_forest.fit(result.isolation_forest_features)
    """

    def __init__(self, window_seconds: float = WINDOW_SECONDS):
        self.window_seconds = window_seconds

    # ── Public API ─────────────────────────────────────────────────────────

    def segment(
        self,
        filtered_signal: np.ndarray,
        raw_data: RawSignalData,
    ) -> SegmentationResult:
        """
        Main entry point. Takes a filtered signal + original RawSignalData
        (for annotations and metadata) and returns a SegmentationResult.

        Parameters
        ----------
        filtered_signal : np.ndarray shape (N,)
            Output of BandpassFilter.apply()
        raw_data : RawSignalData
            Original ingestion output — needed for annotations + sample_rate
        """
        fs             = raw_data.sample_rate
        window_samples = int(self.window_seconds * fs)

        logger.info(
            "Segmenting | type=%s | samples=%d | window=%ds (%d samples)",
            raw_data.signal_type.value, len(filtered_signal),
            self.window_seconds, window_samples,
        )

        # Step 1 — build a per-sample rhythm label array
        label_array = self._build_label_array(
            n_samples          = len(filtered_signal),
            annotation_samples = raw_data.annotation_samples,
            annotation_labels  = raw_data.annotation_labels,
        )

        # Step 2 — slice into fixed windows
        raw_segments = self._slice_windows(
            signal         = filtered_signal,
            label_array    = label_array,
            window_samples = window_samples,
            fs             = fs,
            signal_type    = raw_data.signal_type,
        )

        # Step 3 — score quality + extract HRV features
        scored_segments = [self._score_and_extract(s) for s in raw_segments]

        # Step 4 — build ML-ready arrays from valid segments only
        valid = [s for s in scored_segments if s.is_valid]

        cnn_input, cnn_labels, if_features = self._build_ml_arrays(
            valid, window_samples
        )

        notes = self._summarise(scored_segments, valid)

        result = SegmentationResult(
            segments                  = scored_segments,
            sample_rate               = fs,
            signal_type               = raw_data.signal_type,
            window_samples            = window_samples,
            cnn_input                 = cnn_input,
            cnn_labels                = cnn_labels,
            isolation_forest_features = if_features,
            notes                     = notes,
        )

        logger.info(
            "Segmentation done | total=%d | valid=%d | AFIB=%d | NORMAL=%d",
            len(scored_segments), len(valid), result.afib_count, result.normal_count,
        )
        return result

    # ── Step 1 — per-sample label array ───────────────────────────────────

    def _build_label_array(
        self,
        n_samples: int,
        annotation_samples: np.ndarray,
        annotation_labels: list[str],
    ) -> np.ndarray:
        """
        Build a (N,) array where each sample has a RhythmLabel string.
        Uses annotation boundary samples to assign labels between boundaries.
        """
        label_array = np.full(n_samples, RhythmLabel.UNKNOWN.value, dtype=object)

        if len(annotation_samples) == 0:
            logger.warning("No annotations found — labelling entire signal as UNKNOWN.")
            return label_array

        for i, (start, label) in enumerate(zip(annotation_samples, annotation_labels)):
            end = annotation_samples[i + 1] if i + 1 < len(annotation_samples) else n_samples
            start = int(np.clip(start, 0, n_samples))
            end   = int(np.clip(end,   0, n_samples))
            label_array[start:end] = label

        return label_array

    # ── Step 2 — fixed-window slicing ─────────────────────────────────────

    def _slice_windows(
        self,
        signal: np.ndarray,
        label_array: np.ndarray,
        window_samples: int,
        fs: float,
        signal_type: SignalType,
    ) -> list[Segment]:
        """
        Slice signal into non-overlapping windows of `window_samples`.
        Label each window by majority vote on the label_array.
        """
        segments = []
        n_samples = len(signal)
        idx = 0
        seg_num = 0

        while idx + window_samples <= n_samples:
            end = idx + window_samples
            chunk = signal[idx:end]

            # Majority vote for rhythm label in this window
            window_labels = label_array[idx:end]
            label = self._majority_label(window_labels)

            segments.append(Segment(
                index        = seg_num,
                start_sample = idx,
                end_sample   = end,
                data         = chunk.copy(),
                signal_type  = signal_type,
                rhythm_label = label,
                sample_rate  = fs,
            ))

            idx += window_samples
            seg_num += 1

        # Handle tail (< full window)
        if idx < n_samples:
            tail = signal[idx:]
            tail_labels = label_array[idx:]
            label = self._majority_label(tail_labels)
            segments.append(Segment(
                index        = seg_num,
                start_sample = idx,
                end_sample   = n_samples,
                data         = tail.copy(),
                signal_type  = signal_type,
                rhythm_label = label,
                sample_rate  = fs,
            ))

        return segments

    def _majority_label(self, labels: np.ndarray) -> RhythmLabel:
        """Return the most common RhythmLabel in a window."""
        if len(labels) == 0:
            return RhythmLabel.UNKNOWN
        unique, counts = np.unique(labels, return_counts=True)
        winner = unique[np.argmax(counts)]
        try:
            return RhythmLabel(winner)
        except ValueError:
            return RhythmLabel.UNKNOWN

    # ── Step 3 — quality scoring + HRV feature extraction ─────────────────

    def _score_and_extract(self, seg: Segment) -> Segment:
        """
        Compute quality score and HRV features for one segment.
        Marks segment invalid if quality < MIN_QUALITY.
        """
        data = seg.data
        score = 1.0
        reasons = []

        # ── Quality checks ─────────────────────────────────────────────

        # 1. Too short (tail segment)
        expected = int(self.window_seconds * seg.sample_rate)
        if len(data) < expected * 0.5:
            seg.is_valid = False
            seg.quality_score = 0.0
            return seg

        # 2. Flat signal (lead-off / dropout)
        if np.std(data) < FLAT_STD_THRESH:
            score -= 0.6
            reasons.append("flat")

        # 3. Clipped signal
        abs_max = np.max(np.abs(data))
        if abs_max > 0:
            clip_ratio = np.sum(np.abs(data) >= abs_max * CLIP_THRESHOLD) / len(data)
            if clip_ratio > 0.01:
                score -= 0.4
                reasons.append("clipped")

        # 4. SNR estimate (signal power vs high-freq noise power)
        snr = self._estimate_snr(data, seg.sample_rate)
        if snr < SNR_THRESH_DB:
            score -= 0.3
            reasons.append(f"low_snr={snr:.1f}dB")

        seg.quality_score = float(np.clip(score, 0.0, 1.0))
        seg.is_valid      = seg.quality_score >= MIN_QUALITY

        if reasons:
            logger.debug("Segment %d quality issues: %s", seg.index, ", ".join(reasons))

        # ── HRV features (used by both CNN and Isolation Forest) ───────
        if seg.is_valid:
            seg.hrv_rmssd, seg.hrv_sdnn, seg.rr_irregularity = (
                self._extract_hrv_features(data, seg.sample_rate)
            )

        return seg

    def _estimate_snr(self, data: np.ndarray, fs: float) -> float:
        """
        Simple SNR estimate:
        signal band power (0.5–40 Hz for ECG, 0.5–8 Hz for PPG)
        vs noise band power (above signal band).
        """
        n = len(data)
        fft_mag = np.abs(np.fft.rfft(data)) ** 2
        freqs   = np.fft.rfftfreq(n, d=1.0 / fs)

        signal_mask = (freqs >= 0.5) & (freqs <= 40.0)
        noise_mask  = freqs > 40.0

        signal_power = fft_mag[signal_mask].mean() if signal_mask.any() else 1e-9
        noise_power  = fft_mag[noise_mask].mean()  if noise_mask.any() else 1e-9

        if noise_power < 1e-12:
            return 60.0  # essentially noiseless

        return float(10 * np.log10(signal_power / noise_power))

    def _extract_hrv_features(
        self, data: np.ndarray, fs: float
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Extract HRV features from R-peak intervals.
        Returns (rmssd, sdnn, rr_irregularity).

        Uses simple threshold-based peak detection.
        """
        try:
            # Detect peaks above 60% of signal range
            threshold = data.mean() + 0.6 * data.std()
            min_distance = int(0.3 * fs)  # 300 ms minimum RR interval

            peaks = self._find_peaks(data, threshold, min_distance)

            if len(peaks) < 3:
                return None, None, None

            rr_intervals = np.diff(peaks) / fs * 1000  # convert to ms

            rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
            sdnn  = float(np.std(rr_intervals))

            # Irregularity: coefficient of variation of RR intervals
            # Higher = more irregular = more likely AFIB
            rr_irregularity = float(sdnn / np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0.0

            return rmssd, sdnn, rr_irregularity

        except Exception as e:
            logger.debug("HRV extraction failed for segment: %s", e)
            return None, None, None

    def _find_peaks(
        self, signal: np.ndarray, threshold: float, min_distance: int
    ) -> np.ndarray:
        """Simple peak finder — avoids scipy dependency for speed."""
        peaks = []
        i = 1
        while i < len(signal) - 1:
            if signal[i] > threshold and signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
            i += 1
        return np.array(peaks)

    # ── Step 4 — build ML arrays ───────────────────────────────────────────

    def _build_ml_arrays(
        self,
        valid_segments: list[Segment],
        window_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build arrays ready for FR2 CNN and FR3 Isolation Forest.

        CNN input    : (N, W, 1)   — zero-padded to window_samples if short
        CNN labels   : (N,)        — 1=AFIB, 0=NORMAL
        IF features  : (N, 5)      — [rmssd, sdnn, rr_irregularity, mean, std]
        """
        if not valid_segments:
            logger.warning("No valid segments to build ML arrays from.")
            empty = np.empty((0,))
            return (
                np.empty((0, window_samples, 1)),
                empty,
                np.empty((0, 5)),
            )

        cnn_input  = []
        cnn_labels = []
        if_feats   = []

        for seg in valid_segments:
            # CNN — zero-pad if tail segment is short
            padded = np.zeros(window_samples)
            padded[:len(seg.data)] = seg.data
            cnn_input.append(padded.reshape(-1, 1))

            # Label
            label = 1 if seg.rhythm_label == RhythmLabel.AFIB else 0
            cnn_labels.append(label)

            # Isolation Forest features
            rmssd = seg.hrv_rmssd or 0.0
            sdnn  = seg.hrv_sdnn  or 0.0
            irreg = seg.rr_irregularity or 0.0
            mean  = float(np.mean(seg.data))
            std   = float(np.std(seg.data))
            if_feats.append([rmssd, sdnn, irreg, mean, std])

        return (
            np.array(cnn_input),
            np.array(cnn_labels),
            np.array(if_feats),
        )

    # ── Summary ────────────────────────────────────────────────────────────

    def _summarise(
        self, all_segments: list[Segment], valid: list[Segment]
    ) -> list[str]:
        notes = []
        total   = len(all_segments)
        n_valid = len(valid)
        n_invalid = total - n_valid

        notes.append(f"Total segments: {total} | Valid: {n_valid} | Dropped: {n_invalid}")

        if n_invalid > 0:
            pct = n_invalid / total * 100
            notes.append(f"Warning: {pct:.1f}% of segments failed quality check.")
            if pct > 50:
                notes.append("Over 50% of signal dropped — check input quality.")

        afib_pct = (
            sum(1 for s in valid if s.rhythm_label == RhythmLabel.AFIB) / n_valid * 100
            if n_valid else 0
        )
        notes.append(f"AFIB segments: {afib_pct:.1f}% of valid segments.")

        return notes