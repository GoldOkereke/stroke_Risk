"""Voice analysis service.

This module extracts two groups of features from audio:
- Dysarthria-like features (tempo, pauses, MFCC summary stats, etc.)
- Parkinsons-like features (pitch stability, shimmer/jitter proxies, etc.)

It optionally scores each feature vector using an Isolation Forest model trained
offline and saved under `<project_root>/weights/isolation_forest/`.

The public outputs are designed to match `app.models.schemas.VoiceAnalysisResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.models.schemas import (
    DysarthriaFeatures,
    ParkinsonsFeatures,
    VoiceAnalysisResult,
    VoiceFeatures,
)
from app.services.isolation_forest import IFStream, IsolationForestService

# Optional dependency: librosa provides robust audio feature extraction.
try:  # pragma: no cover
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None


@dataclass(frozen=True)
class TempoStats:
    """Small container for tempo/pause features."""

    speech_tempo: float
    articulation_rate: float
    pause_ratio: float


class VoiceAnalyzer:
    """Extracts voice features and produces aggregated anomaly scores."""

    def __init__(self) -> None:
        # Optional anomaly scorers. For now we use a fixed "default" user_id since the
        # service layer doesn't yet manage authenticated users.
        self._if = IsolationForestService(user_id="default")

    @staticmethod
    def _require_librosa() -> None:
        if librosa is None:
            raise RuntimeError(
                "librosa is not installed. Install backend requirements to enable voice analysis."
            )

    @staticmethod
    def _to_mono_float(audio: np.ndarray) -> np.ndarray:
        """Ensure audio is mono float32 in [-1, 1]."""
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim == 2:
            y = np.mean(y, axis=0)
        if y.size == 0:
            return y
        peak = float(np.max(np.abs(y))) + 1e-9
        y = y / peak
        return y

    @staticmethod
    def _tempo_pause_stats(y: np.ndarray, sr: int) -> TempoStats:
        """Estimate tempo and pause ratio from energy envelope.

        These are proxy measures:
        - pause_ratio: percent of frames classified as silence
        - speech_tempo: syllable-like event rate based on onset detections
        - articulation_rate: onset rate excluding pauses
        """
        VoiceAnalyzer._require_librosa()

        # RMS energy per frame.
        rms = librosa.feature.rms(y=y)[0]
        if rms.size == 0:
            return TempoStats(speech_tempo=0.0, articulation_rate=0.0, pause_ratio=1.0)

        threshold = float(np.percentile(rms, 20))
        silent = rms < threshold
        pause_ratio = float(np.mean(silent))

        # Onset events are used as a rough "syllable-like" proxy.
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration = float(len(y) / sr) if sr > 0 else 0.0
        if duration <= 1e-6:
            return TempoStats(speech_tempo=0.0, articulation_rate=0.0, pause_ratio=pause_ratio)

        speech_tempo = float(len(onsets) / duration)

        # Articulation rate: ignore silent fraction.
        non_pause_time = max(1e-6, duration * (1.0 - pause_ratio))
        articulation_rate = float(len(onsets) / non_pause_time)

        return TempoStats(
            speech_tempo=speech_tempo,
            articulation_rate=articulation_rate,
            pause_ratio=pause_ratio,
        )

    def extract_dysarthria_features(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Extract a set of 10 dysarthria-related features.

        Returns a dict of feature_name -> float.
        """
        self._require_librosa()

        y = self._to_mono_float(audio)
        if y.size == 0 or sr <= 0:
            return {k: 0.0 for k in self._dysarthria_feature_keys()}

        # MFCCs: first 3 coefficients summarized by mean and std (3 * 2 = 6 features).
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        tempo = self._tempo_pause_stats(y, sr)

        # Spectral centroid: mean frequency of the spectrum (proxy for "brightness" of voice).
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = float(np.mean(centroid)) if centroid.size else 0.0

        return {
            "mfcc1_mean": float(mfcc_mean[0]),
            "mfcc1_std": float(mfcc_std[0]),
            "mfcc2_mean": float(mfcc_mean[1]),
            "mfcc2_std": float(mfcc_std[1]),
            "mfcc3_mean": float(mfcc_mean[2]),
            "mfcc3_std": float(mfcc_std[2]),
            "speech_tempo": float(tempo.speech_tempo),
            "articulation_rate": float(tempo.articulation_rate),
            "pause_ratio": float(tempo.pause_ratio),
            "spectral_centroid": float(spectral_centroid),
        }

    @staticmethod
    def _dfa(signal: np.ndarray) -> float:
        """Detrended Fluctuation Analysis (DFA) proxy.

        This estimates long-range correlation in the signal amplitude envelope.
        """
        x = np.asarray(signal, dtype=float)
        if x.size < 16:
            return 0.0

        x = x - np.mean(x)
        y = np.cumsum(x)

        # Use a small set of window sizes.
        sizes = np.unique(np.round(np.logspace(np.log10(8), np.log10(min(256, x.size // 2)), 6)).astype(int))
        flucts = []
        for n in sizes:
            if n < 4:
                continue
            shape = (y.size // n, n)
            if shape[0] < 2:
                continue
            segments = y[: shape[0] * n].reshape(shape)
            t = np.arange(n)
            # Detrend each segment with a line fit and compute RMS.
            rms = []
            for seg in segments:
                coeff = np.polyfit(t, seg, 1)
                trend = coeff[0] * t + coeff[1]
                rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
            flucts.append(np.mean(rms))

        if len(flucts) < 2:
            return 0.0

        # Slope in log-log space is the DFA alpha.
        alpha = np.polyfit(np.log(sizes[: len(flucts)]), np.log(flucts), 1)[0]
        return float(alpha)

    @staticmethod
    def _rpde_proxy(y: np.ndarray, sr: int) -> float:
        """RPDE proxy (Recurrence Period Density Entropy).

        True RPDE is non-trivial; this function computes a practical proxy:
        - take autocorrelation of the amplitude envelope
        - find peak distances (recurrence periods)
        - compute normalized entropy of the period histogram
        """
        if y.size == 0 or sr <= 0:
            return 0.0

        # Envelope for more stable autocorrelation.
        env = np.abs(y)
        env = env - np.mean(env)
        if np.allclose(env, 0.0):
            return 0.0

        ac = np.correlate(env, env, mode="full")
        ac = ac[ac.size // 2 :]
        if ac.size < 32:
            return 0.0

        # Find local maxima indices excluding the zero-lag peak.
        peaks = (ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:])
        peak_idx = np.where(peaks)[0] + 1
        if peak_idx.size < 3:
            return 0.0

        periods = np.diff(peak_idx).astype(float)
        if periods.size < 2:
            return 0.0

        hist, _ = np.histogram(periods, bins=10, density=True)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        entropy = -np.sum(hist * np.log(hist))
        entropy_norm = float(entropy / np.log(hist.size + 1e-9))
        return float(np.clip(entropy_norm, 0.0, 1.0))

    def extract_parkinsons_features(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Extract a set of 7 Parkinsons-related features.

        Notes:
        - jitter/shimmer are implemented as practical proxies based on f0 and amplitude envelope.
        - HNR is approximated via harmonic vs residual energy.
        - RPDE and DFA are computed as described in helper functions above (proxy implementations).
        """
        self._require_librosa()

        y = self._to_mono_float(audio)
        if y.size == 0 or sr <= 0:
            return {k: 0.0 for k in self._parkinsons_feature_keys()}

        # Fundamental frequency estimation (f0). Use pyin for robustness.
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
        if voiced_f0.size == 0:
            pitch_mean = 0.0
            pitch_std = 0.0
            jitter_pct = 0.0
        else:
            pitch_mean = float(np.mean(voiced_f0))
            pitch_std = float(np.std(voiced_f0))
            diffs = np.abs(np.diff(voiced_f0))
            jitter_pct = float(np.mean(diffs) / (pitch_mean + 1e-9))

        # Shimmer proxy: amplitude perturbation from short-time RMS.
        rms = librosa.feature.rms(y=y)[0]
        if rms.size < 2:
            shimmer_db = 0.0
        else:
            amp_diffs = np.abs(np.diff(rms))
            shimmer = float(np.mean(amp_diffs) / (np.mean(rms) + 1e-9))
            shimmer_db = float(20.0 * np.log10(shimmer + 1e-9))

        # HNR proxy: energy of harmonic component vs residual.
        y_harm = librosa.effects.harmonic(y)
        y_res = y - y_harm
        harm_energy = float(np.mean(y_harm**2))
        res_energy = float(np.mean(y_res**2)) + 1e-9
        hnr = float(10.0 * np.log10((harm_energy + 1e-9) / res_energy))

        rpde = float(self._rpde_proxy(y, sr))
        dfa = float(self._dfa(y))

        return {
            "jitter_pct": float(jitter_pct),
            "shimmer_db": float(shimmer_db),
            "hnr": float(hnr),
            "rpde": float(rpde),
            "dfa": float(dfa),
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
        }

    @staticmethod
    def _dysarthria_feature_keys() -> list[str]:
        return [
            "mfcc1_mean",
            "mfcc1_std",
            "mfcc2_mean",
            "mfcc2_std",
            "mfcc3_mean",
            "mfcc3_std",
            "speech_tempo",
            "articulation_rate",
            "pause_ratio",
            "spectral_centroid",
        ]

    @staticmethod
    def _parkinsons_feature_keys() -> list[str]:
        return [
            "jitter_pct",
            "shimmer_db",
            "hnr",
            "rpde",
            "dfa",
            "pitch_mean",
            "pitch_std",
        ]

    def analyze(self, audio: np.ndarray, sr: int) -> VoiceAnalysisResult:
        """Analyze audio and return voice features + scores."""
        dys = self.extract_dysarthria_features(audio, sr)
        pk = self.extract_parkinsons_features(audio, sr)

        # Score each stream with Isolation Forest if models exist.
        dys_vector = np.array([dys[k] for k in self._dysarthria_feature_keys()], dtype=float)
        pk_vector = np.array([pk[k] for k in self._parkinsons_feature_keys()], dtype=float)

        dys_res = self._if.score(IFStream.DYSARTHRIA, dys_vector)
        pk_res = self._if.score(IFStream.PARKINSONS, pk_vector)

        dys_score = float(dys_res.score) if dys_res is not None else 0.0
        pk_score = float(pk_res.score) if pk_res is not None else 0.0

        # Overall score: slightly favor dysarthria because it's often more sensitive to speech motor issues.
        overall = float(0.55 * dys_score + 0.45 * pk_score)

        return VoiceAnalysisResult(
            features=VoiceFeatures(
                dysarthria=DysarthriaFeatures(scores=dys),
                parkinsons=ParkinsonsFeatures(scores=pk),
            ),
            scores={
                "dysarthria_score": dys_score,
                "parkinsons_score": pk_score,
                "overall_score": overall,
            },
        )
