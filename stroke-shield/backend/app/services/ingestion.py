from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from app.core.config import get_settings


@dataclass(frozen=True)
class RawSignalData:
    signal_type: str
    source: str
    values: np.ndarray
    sampling_rate: float
    timestamps: np.ndarray
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    annotation_samples: Optional[np.ndarray] = None
    annotation_aux: Optional[list[str]] = None


class IngestionService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def load_mitbih(self, record_id: str, lead: int = 0) -> RawSignalData:
        record_path = self._settings.afdb_dir / record_id
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal[:, lead]
        fs = float(record.fs)
        timestamps = np.arange(signal.size) / fs

        labels = None
        annotation_samples = None
        annotation_aux = None
        try:
            ann = wfdb.rdann(str(record_path), "atr")
            print("Symbols:", ann.symbol[:20])
            print("Aux notes:", ann.aux_note[:20])
            labels = [
                "AFIB" if label.upper() == "AFIB" else label for label in ann.symbol
            ]
            annotation_samples = ann.sample
            annotation_aux = ann.aux_note
        except Exception:
            labels = None
            annotation_samples = None
            annotation_aux = None

        return RawSignalData(
            signal_type="ECG",
            source="MIT_BIH_AF",
            values=signal,
            sampling_rate=fs,
            timestamps=timestamps,
            annotation_samples=annotation_samples,
            annotation_aux=annotation_aux,
        )

    def load_mimic(
        self,
        record_id: str,
        lead: int = 0,
        dataset: str = "mimic_perform_af_wfdb",
        label: Optional[str] = None,
    ) -> RawSignalData:
        base_dir = (
            self._settings.datasets_root
            / "cardiac/mimic_perform_af/files"
            / dataset
        )
        record_path = base_dir / record_id
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal[:, lead]
        fs = float(record.fs)
        timestamps = np.arange(signal.size) / fs

        labels = None
        annotation_samples = None
        annotation_aux = None
        if label is not None:
            labels = [label]
            annotation_samples = np.array([0])
            annotation_aux = [f"({label}"]
        else:
            try:
                ann = wfdb.rdann(str(record_path), "atr")
                print("Symbols:", ann.symbol[:20])
                print("Aux notes:", ann.aux_note[:20])
                labels = [
                    "AFIB" if label.upper() == "AFIB" else label for label in ann.symbol
                ]
                annotation_samples = ann.sample
                annotation_aux = ann.aux_note
            except Exception:
                labels = None
                annotation_samples = None
                annotation_aux = None

        return RawSignalData(
            signal_type="ECG",
            source="MIMIC_PPG",
            values=signal,
            sampling_rate=fs,
            timestamps=timestamps,
            annotation_samples=annotation_samples,
            annotation_aux=annotation_aux,
        )

    def load_csv(
        self,
        file_bytes: bytes,
        signal_type: str,
        sample_rate: float,
        column: str | int,
    ) -> RawSignalData:
        df = pd.read_csv(BytesIO(file_bytes))
        if isinstance(column, int):
            series = df.iloc[:, column]
        else:
            # If the requested column doesn't exist, fall back to the first numeric column.
            if column not in df.columns:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    column = numeric_cols[0]
                else:
                    # Last resort: use first column and attempt to coerce to float.
                    column = df.columns[0]
            series = df[column]
        series = series.astype(float).interpolate(method="linear", limit_direction="both")
        values = series.to_numpy()
        timestamps = np.arange(values.size) / float(sample_rate)

        return RawSignalData(
            signal_type=signal_type,
            source="SIMULATED",
            values=values,
            sampling_rate=float(sample_rate),
            timestamps=timestamps,
        )

    def load_simulated(
        self,
        signal_type: str,
        duration_seconds: float,
        scenario: str,
    ) -> RawSignalData:
        fs = 250.0 if signal_type.upper() == "ECG" else 125.0
        t = np.arange(0, duration_seconds, 1 / fs)

        if scenario.lower() == "afib":
            phase_jitter = np.cumsum(np.random.normal(0.0, 0.05, size=t.size))
            base = 1.2
            signal = np.sin(2 * np.pi * base * t + phase_jitter)
        else:
            signal = np.sin(2 * np.pi * 1.2 * t)

        noise = np.random.normal(0.0, 0.05, size=t.size)
        signal = signal + noise

        return RawSignalData(
            signal_type=signal_type,
            source="SIMULATED",
            values=signal,
            sampling_rate=fs,
            timestamps=t,
            annotation_samples=np.array([0]),
            annotation_aux=[scenario.upper()],
        )
