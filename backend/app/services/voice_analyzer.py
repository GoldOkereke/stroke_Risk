"""
app/services/voice_analyzer.py
================================
Voice/speech analyzer for FR3.

Extracts acoustic features from microphone audio to detect:
  1. Dysarthria (slurred speech)  — TORGO dataset model
  2. Parkinson's voice markers    — UCI Parkinson's Telemonitoring dataset

Pipeline:
  Raw audio (WAV bytes / numpy array)
    → Librosa feature extraction
    → Two feature vectors:
        dysarthria_features  (10,) → IF Stream DYSARTHRIA
        parkinsons_features  (7,)  → IF Stream PARKINSONS
    → Isolation Forest scoring
    → dysarthria_score, parkinsons_score → Fusion Engine

Dysarthria features (TORGO dataset):
  mfcc_mean_1-3, mfcc_std_1-3   — articulation clarity
  speech_tempo                   — syllables/sec (slowing = bad)
  articulation_rate              — rate excluding pauses
  pause_ratio                    — silence fraction
  spectral_centroid              — centre of spectral mass

Parkinson's features (UCI dataset):
  jitter_pct                     — cycle-to-cycle pitch variation
  shimmer_db                     — amplitude variation
  hnr                            — harmonics-to-noise ratio
  rpde                           — recurrence period density entropy
  dfa                            — detrended fluctuation analysis
  pitch_mean                     — mean F0
  pitch_std                      — F0 standard deviation
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Librosa imported lazily
_librosa = None


def _get_librosa():
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
            logger.info("Librosa loaded.")
        except ImportError:
            raise ImportError("Librosa not installed. Run: pip install librosa")
    return _librosa


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_SR        = 16000   # 16 kHz — standard for speech analysis
N_MFCC            = 13      # number of MFCC coefficients (use first 3 for IF)
HOP_LENGTH        = 512
N_FFT             = 2048
SILENCE_THRESHOLD = 0.01    # RMS below this = silence
MIN_AUDIO_SECONDS = 2.0     # minimum audio length for reliable features


# ── Output dataclasses ─────────────────────────────────────────────────────

@dataclass
class DysarthriaFeatures:
    """
    10 features for TORGO Isolation Forest stream.
    Detects slurred/unclear articulation.
    """
    mfcc_mean_1: float    # MFCC coefficient 1 mean
    mfcc_mean_2: float    # MFCC coefficient 2 mean
    mfcc_mean_3: float    # MFCC coefficient 3 mean
    mfcc_std_1: float     # MFCC coefficient 1 std
    mfcc_std_2: float     # MFCC coefficient 2 std
    mfcc_std_3: float     # MFCC coefficient 3 std
    speech_tempo: float         # estimated syllables per second
    articulation_rate: float    # speech rate excluding pauses
    pause_ratio: float          # fraction of audio that is silence
    spectral_centroid: float    # mean spectral centroid (Hz)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.mfcc_mean_1, self.mfcc_mean_2, self.mfcc_mean_3,
            self.mfcc_std_1,  self.mfcc_std_2,  self.mfcc_std_3,
            self.speech_tempo,
            self.articulation_rate,
            self.pause_ratio,
            self.spectral_centroid,
        ], dtype=np.float64)


@dataclass
class ParkinsonsFeatures:
    """
    7 features for UCI Parkinson's Isolation Forest stream.
    Detects tremor, pitch instability, breathiness.
    """
    jitter_pct: float     # pitch period perturbation (%)
    shimmer_db: float     # amplitude perturbation (dB)
    hnr: float            # harmonics-to-noise ratio (dB)
    rpde: float           # recurrence period density entropy
    dfa: float            # detrended fluctuation analysis alpha
    pitch_mean: float     # mean fundamental frequency F0 (Hz)
    pitch_std: float      # F0 standard deviation (Hz)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.jitter_pct,
            self.shimmer_db,
            self.hnr,
            self.rpde,
            self.dfa,
            self.pitch_mean,
            self.pitch_std,
        ], dtype=np.float64)


@dataclass
class VoiceFeatures:
    """Combined voice features from one audio clip."""
    dysarthria: DysarthriaFeatures
    parkinsons: ParkinsonsFeatures
    duration_seconds: float
    sample_rate: int
    is_valid: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class VoiceAnalysisResult:
    """Full output from VoiceAnalyzer.analyze()"""
    features: Optional[VoiceFeatures]

    # Per-stream scores (rule-based before IF trains)
    dysarthria_score: float     # [0,1] — slurring severity
    parkinsons_score: float     # [0,1] — tremor/pitch severity

    # Combined
    is_anomaly: bool
    confidence: float
    alert_indicators: list[str]
    notes: list[str] = field(default_factory=list)


# ── Voice Analyzer ─────────────────────────────────────────────────────────

class VoiceAnalyzer:
    """
    Extracts dysarthria and Parkinson's features from audio.

    Usage
    -----
        analyzer = VoiceAnalyzer()

        # From raw bytes (WAV file upload)
        result = analyzer.analyze_bytes(wav_bytes)

        # From numpy array
        result = analyzer.analyze_array(audio_array, sample_rate=16000)

        # Feed to Isolation Forest
        if result.features:
            if_service.add_baseline(
                IFStream.DYSARTHRIA, result.features.dysarthria.to_array()
            )
            if_service.add_baseline(
                IFStream.PARKINSONS, result.features.parkinsons.to_array()
            )
    """

    # Rule-based thresholds (clinical reference values)
    JITTER_THRESHOLD         = 1.04   # % — above = abnormal (normal < 1.04%)
    SHIMMER_THRESHOLD        = 3.81   # dB — above = abnormal
    HNR_THRESHOLD            = 20.0   # dB — below = abnormal (normal > 20 dB)
    PAUSE_RATIO_THRESHOLD    = 0.40   # above = too many pauses
    TEMPO_LOW_THRESHOLD      = 2.0    # syllables/sec — below = slowed speech

    def __init__(self, sample_rate: int = DEFAULT_SR):
        self.sample_rate = sample_rate
        logger.info("VoiceAnalyzer initialised | sr=%d Hz", sample_rate)

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze_bytes(self, audio_bytes: bytes) -> VoiceAnalysisResult:
        """
        Analyze audio from raw WAV bytes (e.g. from UploadFile.read()).
        """
        try:
            lr = _get_librosa()
            audio, sr = lr.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
            return self.analyze_array(audio, sr)
        except Exception as e:
            logger.error("Audio load failed: %s", e)
            return self._empty_result(f"Audio load error: {e}")

    def analyze_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> VoiceAnalysisResult:
        """
        Analyze audio from a numpy float32 array.

        Parameters
        ----------
        audio       : 1-D float32 array of audio samples
        sample_rate : Hz
        """
        audio = np.asarray(audio, dtype=np.float32)

        # ── Validation ─────────────────────────────────────────────────
        duration = len(audio) / sample_rate
        if duration < MIN_AUDIO_SECONDS:
            return self._empty_result(
                f"Audio too short ({duration:.1f}s). Min {MIN_AUDIO_SECONDS}s required."
            )

        if np.max(np.abs(audio)) < 0.001:
            return self._empty_result("Audio is silent — check microphone.")

        # ── Normalise ──────────────────────────────────────────────────
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # ── Extract features ───────────────────────────────────────────
        try:
            dysarthria_feat = self._extract_dysarthria(audio, sample_rate)
            parkinsons_feat = self._extract_parkinsons(audio, sample_rate)
        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            return self._empty_result(f"Feature extraction error: {e}")

        features = VoiceFeatures(
            dysarthria       = dysarthria_feat,
            parkinsons       = parkinsons_feat,
            duration_seconds = duration,
            sample_rate      = sample_rate,
        )

        # ── Rule-based scoring ─────────────────────────────────────────
        dysarthria_score = self._score_dysarthria(dysarthria_feat)
        parkinsons_score = self._score_parkinsons(parkinsons_feat)
        alert_indicators = self._get_alert_indicators(
            dysarthria_feat, parkinsons_feat
        )

        confidence = min(1.0, duration / 10.0)   # more audio = higher confidence

        result = VoiceAnalysisResult(
            features          = features,
            dysarthria_score  = dysarthria_score,
            parkinsons_score  = parkinsons_score,
            is_anomaly        = max(dysarthria_score, parkinsons_score) >= 0.5,
            confidence        = round(confidence, 4),
            alert_indicators  = alert_indicators,
        )

        logger.info(
            "Voice analysis | duration=%.1fs | dysarthria=%.3f | parkinsons=%.3f | flags=%s",
            duration, dysarthria_score, parkinsons_score, alert_indicators,
        )
        return result

    # ── Dysarthria feature extraction ──────────────────────────────────────

    def _extract_dysarthria(
        self, audio: np.ndarray, sr: int
    ) -> DysarthriaFeatures:
        """Extract 10 TORGO-aligned dysarthria features."""
        lr = _get_librosa()

        # ── MFCCs (first 3 coefficients) ───────────────────────────────
        mfccs = lr.feature.mfcc(
            y=audio, sr=sr, n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH, n_fft=N_FFT,
        )  # shape (N_MFCC, T)

        mfcc_means = mfccs.mean(axis=1)
        mfcc_stds  = mfccs.std(axis=1)

        # ── Spectral centroid ───────────────────────────────────────────
        centroid = lr.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=HOP_LENGTH
        ).mean()

        # ── Silence / pause detection ───────────────────────────────────
        rms = lr.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        silence_frames = (rms < SILENCE_THRESHOLD).sum()
        pause_ratio    = float(silence_frames / len(rms))

        # ── Speech tempo estimation ─────────────────────────────────────
        # Use onset strength envelope to estimate beat/syllable rate
        onset_env = lr.onset.onset_strength(y=audio, sr=sr)
        tempo, _  = lr.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # Convert BPM-equivalent to syllables/sec (rough approximation)
        speech_tempo = float(tempo) / 60.0 * 1.5   # empirical scaling

        # Articulation rate (tempo excluding pause frames)
        speech_fraction  = 1.0 - pause_ratio
        articulation_rate = speech_tempo / speech_fraction if speech_fraction > 0 else 0.0

        return DysarthriaFeatures(
            mfcc_mean_1       = float(mfcc_means[0]),
            mfcc_mean_2       = float(mfcc_means[1]),
            mfcc_mean_3       = float(mfcc_means[2]),
            mfcc_std_1        = float(mfcc_stds[0]),
            mfcc_std_2        = float(mfcc_stds[1]),
            mfcc_std_3        = float(mfcc_stds[2]),
            speech_tempo      = round(float(speech_tempo),      4),
            articulation_rate = round(float(articulation_rate), 4),
            pause_ratio       = round(float(pause_ratio),       4),
            spectral_centroid = round(float(centroid),          4),
        )

    # ── Parkinson's feature extraction ────────────────────────────────────

    def _extract_parkinsons(
        self, audio: np.ndarray, sr: int
    ) -> ParkinsonsFeatures:
        """Extract 7 UCI Parkinson's-aligned voice features."""
        lr = _get_librosa()

        # ── Fundamental frequency (F0) via pyin ────────────────────────
        f0, voiced_flag, _ = lr.pyin(
            audio,
            fmin=lr.note_to_hz("C2"),   # ~65 Hz
            fmax=lr.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )

        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) < 10:
            # Not enough voiced frames — use fallback
            voiced_f0 = np.array([120.0])   # average human F0

        pitch_mean = float(np.mean(voiced_f0))
        pitch_std  = float(np.std(voiced_f0))

        # ── Jitter (cycle-to-cycle pitch perturbation) ──────────────────
        # Approximate from F0 consecutive differences
        if len(voiced_f0) > 1:
            f0_diffs   = np.abs(np.diff(voiced_f0))
            jitter_pct = float((f0_diffs.mean() / pitch_mean) * 100) if pitch_mean > 0 else 0.0
        else:
            jitter_pct = 0.0

        # ── Shimmer (amplitude perturbation) ───────────────────────────
        rms_frames = lr.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        rms_nonzero = rms_frames[rms_frames > 1e-6]
        if len(rms_nonzero) > 1:
            rms_diffs   = np.abs(np.diff(rms_nonzero))
            shimmer_raw = float(rms_diffs.mean() / rms_nonzero.mean())
            shimmer_db  = float(-20 * np.log10(1 - shimmer_raw + 1e-8))
        else:
            shimmer_db = 0.0

        # ── HNR (Harmonics-to-Noise Ratio) ──────────────────────────────
        # Approximate via spectral flatness (lower = more tonal = higher HNR)
        flatness = lr.feature.spectral_flatness(y=audio, hop_length=HOP_LENGTH)[0]
        mean_flatness = float(flatness.mean())
        hnr = float(-10 * np.log10(mean_flatness + 1e-8))   # invert: lower flatness = higher HNR

        # ── RPDE (Recurrence Period Density Entropy) ────────────────────
        # Approximated via spectral entropy of F0 sequence
        rpde = self._compute_spectral_entropy(voiced_f0)

        # ── DFA (Detrended Fluctuation Analysis) ────────────────────────
        dfa = self._compute_dfa(voiced_f0)

        return ParkinsonsFeatures(
            jitter_pct = round(jitter_pct, 4),
            shimmer_db = round(shimmer_db, 4),
            hnr        = round(hnr,        4),
            rpde       = round(rpde,       4),
            dfa        = round(dfa,        4),
            pitch_mean = round(pitch_mean, 4),
            pitch_std  = round(pitch_std,  4),
        )

    # ── Rule-based scoring ─────────────────────────────────────────────────

    def _score_dysarthria(self, f: DysarthriaFeatures) -> float:
        """Map dysarthria features to [0,1] anomaly score."""
        scores = []

        # Slow speech (below threshold = abnormal)
        if f.speech_tempo < self.TEMPO_LOW_THRESHOLD:
            scores.append(
                1.0 - (f.speech_tempo / self.TEMPO_LOW_THRESHOLD)
            )
        else:
            scores.append(0.0)

        # Excessive pauses
        scores.append(
            min(1.0, f.pause_ratio / self.PAUSE_RATIO_THRESHOLD)
        )

        # Low spectral centroid (muffled / slurred speech tends to be lower)
        normal_centroid = 2000.0   # Hz reference
        centroid_score  = max(0.0, 1.0 - (f.spectral_centroid / normal_centroid))
        scores.append(min(1.0, centroid_score))

        return round(float(np.mean(scores)), 4)

    def _score_parkinsons(self, f: ParkinsonsFeatures) -> float:
        """Map Parkinson's features to [0,1] anomaly score."""
        scores = []

        # Jitter — above threshold = abnormal
        scores.append(min(1.0, f.jitter_pct / (self.JITTER_THRESHOLD * 3)))

        # Shimmer — above threshold = abnormal
        scores.append(min(1.0, f.shimmer_db / (self.SHIMMER_THRESHOLD * 3)))

        # HNR — below threshold = abnormal (more noise than harmonics)
        hnr_score = max(0.0, 1.0 - (f.hnr / self.HNR_THRESHOLD))
        scores.append(min(1.0, hnr_score))

        # Pitch instability (high std relative to mean)
        if f.pitch_mean > 0:
            cv = f.pitch_std / f.pitch_mean   # coefficient of variation
            scores.append(min(1.0, cv / 0.3))
        else:
            scores.append(0.0)

        return round(float(np.mean(scores)), 4)

    # ── Alert indicators ───────────────────────────────────────────────────

    def _get_alert_indicators(
        self,
        d: DysarthriaFeatures,
        p: ParkinsonsFeatures,
    ) -> list[str]:
        flags = []

        if d.speech_tempo < self.TEMPO_LOW_THRESHOLD:
            flags.append(f"Slowed speech detected ({d.speech_tempo:.1f} syl/sec)")

        if d.pause_ratio > self.PAUSE_RATIO_THRESHOLD:
            flags.append(f"Excessive pauses ({d.pause_ratio*100:.0f}% silence)")

        if p.jitter_pct > self.JITTER_THRESHOLD:
            flags.append(f"Voice jitter elevated ({p.jitter_pct:.2f}%)")

        if p.shimmer_db > self.SHIMMER_THRESHOLD:
            flags.append(f"Voice shimmer elevated ({p.shimmer_db:.2f} dB)")

        if p.hnr < self.HNR_THRESHOLD:
            flags.append(f"Low harmonics-to-noise ratio ({p.hnr:.1f} dB)")

        return flags

    # ── Signal processing helpers ──────────────────────────────────────────

    def _compute_spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Compute spectral entropy as RPDE approximation.
        Higher entropy = more irregular/complex = more Parkinson's-like.
        """
        if len(signal) < 4:
            return 0.5
        power = np.abs(np.fft.rfft(signal)) ** 2
        power_norm = power / (power.sum() + 1e-12)
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))
        # Normalise to [0, 1]
        max_entropy = np.log2(len(power_norm))
        return float(np.clip(entropy / (max_entropy + 1e-8), 0.0, 1.0))

    def _compute_dfa(self, signal: np.ndarray) -> float:
        """
        Simplified Detrended Fluctuation Analysis.
        Returns scaling exponent alpha in [0, 1].
        alpha ~ 0.5 = uncorrelated (normal)
        alpha > 0.5 = long-range correlations (Parkinson's-like)
        """
        if len(signal) < 16:
            return 0.5

        signal = signal - signal.mean()
        cumsum = np.cumsum(signal)

        scales    = [4, 8, 16, min(32, len(signal) // 2)]
        flucts    = []

        for scale in scales:
            n_windows = len(cumsum) // scale
            if n_windows < 2:
                continue
            f_sq = []
            for i in range(n_windows):
                segment = cumsum[i * scale: (i + 1) * scale]
                x       = np.arange(len(segment))
                coeffs  = np.polyfit(x, segment, 1)
                trend   = np.polyval(coeffs, x)
                f_sq.append(np.mean((segment - trend) ** 2))
            flucts.append(np.sqrt(np.mean(f_sq)))

        if len(flucts) < 2:
            return 0.5

        log_scales = np.log2(scales[:len(flucts)])
        log_flucts = np.log2(np.array(flucts) + 1e-12)
        alpha      = float(np.polyfit(log_scales, log_flucts, 1)[0])
        return float(np.clip(alpha, 0.0, 2.0) / 2.0)   # normalise to [0,1]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _empty_result(self, note: str) -> VoiceAnalysisResult:
        logger.warning("VoiceAnalyzer: %s", note)
        return VoiceAnalysisResult(
            features         = None,
            dysarthria_score = 0.0,
            parkinsons_score = 0.0,
            is_anomaly       = False,
            confidence       = 0.0,
            alert_indicators = [],
            notes            = [note],
        )