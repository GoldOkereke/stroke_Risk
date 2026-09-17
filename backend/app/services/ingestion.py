"""
app/services/ingestion.py
==========================
Loads signals into the pipeline from three sources:

  1. MIT-BIH Atrial Fibrillation (ECG, 250 Hz) — via wfdb
  2. MIMIC PERform AF             (PPG, 125 Hz) — via wfdb
  3. User CSV upload              (ECG or PPG)  — via pandas
  4. Simulated signal             (FR6 demo)    — synthetic generator

Output is always a normalised NumPy array + sample rate + rhythm annotations,
ready to be passed straight into BandpassFilter → Segmentation.
"""

from __future__ import annotations

import logging
import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from app.models.signal_schemas import RhythmLabel, SignalSource, SignalType

logger = logging.getLogger(__name__)


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class RawSignalData:
    """
    Standardised output from every ingestion path.
    Everything downstream (filter, segmentation, CNN, IF) consumes this.
    """
    signal: np.ndarray          # shape (N,) — raw samples, float64
    sample_rate: float          # Hz
    signal_type: SignalType     # ECG or PPG
    source: SignalSource        # where it came from
    record_id: Optional[str]    # WFDB record ID or filename

    # Rhythm annotations from .atr file (WFDB sources only)
    annotation_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    annotation_labels: list[str]   = field(default_factory=list)

    # Metadata
    duration_seconds: float = 0.0
    lead_index: int         = 0
    notes: list[str]        = field(default_factory=list)

    def __post_init__(self):
        self.duration_seconds = len(self.signal) / self.sample_rate


# ── Main ingestion service ─────────────────────────────────────────────────

class IngestionService:
    """
    Single entry point for all signal ingestion.

    Usage
    -----
        svc = IngestionService()

        # From MIT-BIH AF
        data = svc.load_mitbih(record_id="04015", lead=0)

        # From MIMIC PERform AF
        data = svc.load_mimic(record_id="mimic_af_001", lead=0)

        # From user CSV
        data = svc.load_csv(file_bytes=..., signal_type=SignalType.ECG, sample_rate=250)

        # Demo mode
        data = svc.load_simulated(signal_type=SignalType.ECG, duration_seconds=30)
    """

    # PhysioNet directory paths
    MITBIH_DIR  = "afdb/1.0.0"
    MIMIC_DIR   = "mimic-perform-af/1.0.0"

    # Default sample rates per dataset
    MITBIH_FS   = 250.0
    MIMIC_FS    = 125.0

    # ── MIT-BIH AF ─────────────────────────────────────────────────────────

    def load_mitbih(self, record_id: str, lead: int = 0) -> RawSignalData:
        """
        Load one ECG record from MIT-BIH Atrial Fibrillation database.

        Parameters
        ----------
        record_id : str
            Record name e.g. '04015', '04043', '04048' ...
        lead : int
            0 = Lead I, 1 = Lead II

        Returns
        -------
        RawSignalData with ECG signal + AFIB/NORMAL annotations
        """
        logger.info("Loading MIT-BIH AF record: %s (lead %d)", record_id, lead)

        try:
            record = wfdb.rdrecord(record_id, pn_dir=self.MITBIH_DIR)
            annotation = wfdb.rdann(record_id, "atr", pn_dir=self.MITBIH_DIR)
        except Exception as e:
            raise ValueError(f"Failed to load MIT-BIH record '{record_id}': {e}") from e

        signal = record.p_signal[:, lead].astype(np.float64)
        signal = self._remove_nan(signal, record_id)

        labels = self._parse_mitbih_annotations(annotation.aux_note)

        data = RawSignalData(
            signal             = signal,
            sample_rate        = float(record.fs),
            signal_type        = SignalType.ECG,
            source             = SignalSource.MITBIH,
            record_id          = record_id,
            annotation_samples = np.array(annotation.sample),
            annotation_labels  = labels,
            lead_index         = lead,
        )

        logger.info(
            "MIT-BIH loaded | record=%s | samples=%d | duration=%.1f min | "
            "annotations=%d",
            record_id, len(signal), data.duration_seconds / 60, len(labels),
        )
        return data

    # ── MIMIC PERform AF ───────────────────────────────────────────────────

    def load_mimic(self, record_id: str, lead: int = 0) -> RawSignalData:
        """
        Load one PPG record from MIMIC PERform AF dataset.

        Parameters
        ----------
        record_id : str
            Record name from the MIMIC PERform AF database.
        lead : int
            Signal channel index (0 = PPG by default in MIMIC PERform).

        Returns
        -------
        RawSignalData with PPG signal + AFIB/NORMAL annotations
        """
        logger.info("Loading MIMIC PERform record: %s (channel %d)", record_id, lead)

        try:
            record = wfdb.rdrecord(record_id, pn_dir=self.MIMIC_DIR)
            annotation = wfdb.rdann(record_id, "atr", pn_dir=self.MIMIC_DIR)
        except Exception as e:
            raise ValueError(f"Failed to load MIMIC record '{record_id}': {e}") from e

        signal = record.p_signal[:, lead].astype(np.float64)
        signal = self._remove_nan(signal, record_id)

        labels = self._parse_mitbih_annotations(annotation.aux_note)

        data = RawSignalData(
            signal             = signal,
            sample_rate        = float(record.fs),
            signal_type        = SignalType.PPG,
            source             = SignalSource.MIMIC,
            record_id          = record_id,
            annotation_samples = np.array(annotation.sample),
            annotation_labels  = labels,
            lead_index         = lead,
        )

        logger.info(
            "MIMIC PERform loaded | record=%s | samples=%d | duration=%.1f min",
            record_id, len(signal), data.duration_seconds / 60,
        )
        return data

    # ── CSV upload ─────────────────────────────────────────────────────────

    def load_csv(
        self,
        file_bytes: bytes,
        signal_type: SignalType,
        sample_rate: float,
        column: Optional[str] = None,
    ) -> RawSignalData:
        """
        Parse a user-uploaded CSV file into a RawSignalData.

        The CSV must have at least one numeric column containing the signal.
        If multiple columns exist, pass `column` to specify which one,
        otherwise the first numeric column is used.

        Parameters
        ----------
        file_bytes  : raw bytes from FastAPI UploadFile.read()
        signal_type : ECG or PPG
        sample_rate : Hz — must be supplied by the user
        column      : optional column name to extract
        """
        logger.info("Parsing CSV upload | signal_type=%s | fs=%.1f", signal_type, sample_rate)

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not parse CSV: {e}") from e

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("CSV contains no numeric columns.")

        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
            signal = df[column].to_numpy(dtype=np.float64)
        else:
            signal = df[numeric_cols[0]].to_numpy(dtype=np.float64)
            logger.info("No column specified — using first numeric column: '%s'", numeric_cols[0])

        signal = self._remove_nan(signal, "csv_upload")

        data = RawSignalData(
            signal      = signal,
            sample_rate = sample_rate,
            signal_type = signal_type,
            source      = SignalSource.CSV,
            record_id   = None,
            notes       = [f"Loaded from CSV, column='{column or numeric_cols[0]}'"],
        )

        logger.info("CSV loaded | samples=%d | duration=%.1f s", len(signal), data.duration_seconds)
        return data

    # ── Simulated (FR6 demo mode) ──────────────────────────────────────────

    def load_simulated(
        self,
        signal_type: SignalType = SignalType.ECG,
        duration_seconds: float = 60.0,
        scenario: str = "pre_tia",
    ) -> RawSignalData:
        """
        Generate a synthetic ECG or PPG signal for FR6 demo mode.

        Scenarios
        ---------
        'normal'   — clean sinus rhythm
        'afib'     — simulated AF with irregular RR intervals
        'pre_tia'  — normal baseline → gradual AF onset (the demo scenario)

        Parameters
        ----------
        signal_type      : ECG or PPG
        duration_seconds : length of synthetic signal
        scenario         : 'normal' | 'afib' | 'pre_tia'
        """
        fs = self.MITBIH_FS if signal_type == SignalType.ECG else self.MIMIC_FS
        n_samples = int(duration_seconds * fs)
        t = np.linspace(0, duration_seconds, n_samples)

        logger.info(
            "Generating simulated %s | scenario=%s | duration=%.1f s | fs=%.0f Hz",
            signal_type.value, scenario, duration_seconds, fs,
        )

        if scenario == "normal":
            signal = self._simulate_normal(t, fs)
            annotation_labels = ["(N"] * 1
            annotation_samples = np.array([0])

        elif scenario == "afib":
            signal = self._simulate_afib(t, fs)
            annotation_labels = ["(AFIB"]
            annotation_samples = np.array([0])

        elif scenario == "pre_tia":
            # First half normal, second half transitions to AFIB — the key demo
            mid = n_samples // 2
            t1, t2 = t[:mid], t[mid:]
            normal_part = self._simulate_normal(t1, fs)
            afib_part   = self._simulate_afib(t2, fs)
            # Smooth crossfade at boundary
            fade = np.linspace(0, 1, min(int(fs * 2), mid))
            normal_part[-len(fade):] = (
                normal_part[-len(fade):] * (1 - fade) + afib_part[:len(fade)] * fade
            )
            signal = np.concatenate([normal_part, afib_part[len(fade):]])
            annotation_samples = np.array([0, mid])
            annotation_labels  = ["(N", "(AFIB"]
        else:
            raise ValueError(f"Unknown scenario '{scenario}'. Use: normal | afib | pre_tia")

        # Add realistic noise
        signal += np.random.normal(0, 0.02, size=signal.shape)

        data = RawSignalData(
            signal             = signal,
            sample_rate        = fs,
            signal_type        = signal_type,
            source             = SignalSource.SIMULATED,
            record_id          = f"sim_{scenario}_{int(duration_seconds)}s",
            annotation_samples = annotation_samples,
            annotation_labels  = annotation_labels,
            notes              = [f"Simulated scenario: {scenario}"],
        )

        logger.info("Simulated signal ready | samples=%d", len(signal))
        return data

    # ── Private helpers ────────────────────────────────────────────────────

    def _parse_mitbih_annotations(self, aux_notes: list[str]) -> list[str]:
        """
        Normalise raw MIT-BIH .atr aux_note strings to clean labels.
        Raw values look like: '(AFIB\x00', '(N\x00', '(AFL\x00'
        """
        cleaned = []
        for note in aux_notes:
            note = note.strip().replace("\x00", "").upper()
            if "AFIB" in note:
                cleaned.append(RhythmLabel.AFIB.value)
            elif note in ("(N", "N"):
                cleaned.append(RhythmLabel.NORMAL.value)
            elif "AFL" in note:
                cleaned.append(RhythmLabel.FLUTTER.value)
            else:
                cleaned.append(RhythmLabel.UNKNOWN.value)
        return cleaned

    def _remove_nan(self, signal: np.ndarray, record_id: str) -> np.ndarray:
        """Replace NaN values with linear interpolation."""
        nan_mask = np.isnan(signal)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            logger.warning("Record %s has %d NaN samples — interpolating.", record_id, n_nan)
            indices = np.arange(len(signal))
            signal = np.interp(indices, indices[~nan_mask], signal[~nan_mask])
        return signal

    def _simulate_normal(self, t: np.ndarray, fs: float) -> np.ndarray:
        """Synthetic normal sinus rhythm — regular 1 Hz pulse with harmonics."""
        heart_rate_hz = 1.1  # ~66 bpm
        signal = (
            0.8 * np.sin(2 * np.pi * heart_rate_hz * t)
            + 0.3 * np.sin(2 * np.pi * 2 * heart_rate_hz * t)
            + 0.1 * np.sin(2 * np.pi * 3 * heart_rate_hz * t)
        )
        return signal

    def _simulate_afib(self, t: np.ndarray, fs: float) -> np.ndarray:
        """
        Synthetic AF — irregular RR intervals with baseline fibrillatory wander.
        """
        signal = np.zeros_like(t)

        # Irregular beat timing (AF hallmark)
        rng = np.random.default_rng(seed=42)
        beat_intervals = rng.uniform(0.5, 1.2, size=int(t[-1] * 2))
        beat_times = np.cumsum(beat_intervals)
        beat_times = beat_times[beat_times < t[-1]]

        for bt in beat_times:
            idx = int(bt * fs)
            if idx < len(signal):
                # QRS-like spike
                width = max(1, int(0.04 * fs))
                for j in range(-width, width):
                    if 0 <= idx + j < len(signal):
                        signal[idx + j] += np.exp(-0.5 * (j / (width / 2)) ** 2)

        # Fibrillatory baseline wander (350 Hz equivalent scaled)
        signal += 0.15 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
        return signal