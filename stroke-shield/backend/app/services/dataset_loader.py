from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import wfdb
from PIL import Image

from app.core.config import get_settings
from app.services.ingestion import RawSignalData


@dataclass(frozen=True)
class ScenarioSamples:
    ppg: str
    face: str
    voice: str


class DatasetLoader:
    def __init__(self, datasets_root: Path | None = None) -> None:
        settings = get_settings()
        self._datasets_root = Path(datasets_root) if datasets_root else settings.datasets_root

        self.samples: dict[str, ScenarioSamples] = {
            "normal": ScenarioSamples(
                ppg="mit-bih/100",
                face="facial_palsy_db/normal_001.jpg",
                voice="torgo/controls/F1/audio.wav",
            ),
            "afib_only": ScenarioSamples(
                ppg="mit-bih/202",
                face="facial_palsy_db/normal_002.jpg",
                voice="torgo/controls/M2/audio.wav",
            ),
            "palsy_only": ScenarioSamples(
                ppg="mit-bih/100",
                face="facial_palsy_db/palsy_017.jpg",
                voice="torgo/controls/M3/audio.wav",
            ),
            "dysarthria_only": ScenarioSamples(
                ppg="mit-bih/100",
                face="facial_palsy_db/normal_003.jpg",
                voice="torgo/dysarthric/M4/audio.wav",
            ),
            "pre_tia": ScenarioSamples(
                ppg="mit-bih/203",
                face="facial_palsy_db/palsy_017.jpg",
                voice="torgo/dysarthric/M4/audio.wav",
            ),
        }

    def get_ppg(self, scenario: str, lead: int = 0) -> RawSignalData:
        sample = self._get_scenario(scenario)
        record_path = self._resolve(sample.ppg)
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal[:, lead]
        fs = float(record.fs)
        timestamps = np.arange(signal.size) / fs
        return RawSignalData(
            signal_type="PPG",
            source="MIT_BIH_AF",
            values=signal,
            sampling_rate=fs,
            timestamps=timestamps,
        )

    def get_face_image(self, scenario: str) -> np.ndarray:
        sample = self._get_scenario(scenario)
        image_path = self._resolve(sample.face)
        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"))

    def get_voice_audio(self, scenario: str) -> tuple[np.ndarray, int]:
        sample = self._get_scenario(scenario)
        audio_path = self._resolve(sample.voice)
        audio, sr = sf.read(str(audio_path), always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        return audio, int(sr)

    def get_all_for_scenario(self, scenario: str) -> dict[str, Any]:
        return {
            "ppg": self.get_ppg(scenario),
            "face": self.get_face_image(scenario),
            "voice": self.get_voice_audio(scenario),
        }

    def _get_scenario(self, scenario: str) -> ScenarioSamples:
        key = scenario.lower()
        if key not in self.samples:
            valid = ", ".join(sorted(self.samples.keys()))
            raise KeyError(f"Unknown scenario '{scenario}'. Valid scenarios: {valid}.")
        return self.samples[key]

    def _resolve(self, relative_path: str) -> Path:
        if self._datasets_root is None:
            raise ValueError("datasets_root is not configured.")
        path = self._datasets_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset path: {path}")
        return path
