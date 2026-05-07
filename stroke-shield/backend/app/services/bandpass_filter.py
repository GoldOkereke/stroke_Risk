from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


@dataclass(frozen=True)
class BandpassFilterConfig:
    lowcut: float
    highcut: float
    fs: float
    order: int = 4
    notch_freq: float = 60.0


class BandpassFilter:
    def design_butterworth(
        self,
        order: int,
        fs: float,
        lowcut: float,
        highcut: float,
        filter_type: str = "band",
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype=filter_type, output="sos")

    def apply(
        self,
        signal: Iterable[float],
        fs: float,
        lowcut: float,
        highcut: float,
        notch_freq: float = 60.0,
    ) -> np.ndarray:
        samples = np.asarray(signal, dtype=float)
        if samples.size == 0:
            return samples

        samples = self._fill_nans(samples)

        sos = self.design_butterworth(order=4, fs=fs, lowcut=lowcut, highcut=highcut)
        samples = self._safe_sosfiltfilt(sos, samples)

        if 0 < notch_freq < fs / 2:
            b, a = iirnotch(w0=notch_freq, Q=30.0, fs=fs)
            samples = self._safe_filtfilt(b, a, samples)

        return samples

    def apply_ecg(self, signal: Iterable[float]) -> np.ndarray:
        return self.apply(signal=signal, fs=250.0, lowcut=0.5, highcut=40.0)

    def apply_ppg(self, signal: Iterable[float]) -> np.ndarray:
        return self.apply(signal=signal, fs=125.0, lowcut=0.5, highcut=8.0)

    @staticmethod
    def _safe_sosfiltfilt(sos: np.ndarray, samples: np.ndarray) -> np.ndarray:
        padlen = 3 * (2 * sos.shape[0] - 1)
        original_len = samples.size
        samples, pad_left, pad_right = BandpassFilter._pad_samples(samples, padlen)
        filtered = sosfiltfilt(sos, samples, padlen=padlen)
        if pad_left or pad_right:
            return filtered[pad_left:pad_left + original_len]
        return filtered

    @staticmethod
    def _safe_filtfilt(b: np.ndarray, a: np.ndarray, samples: np.ndarray) -> np.ndarray:
        padlen = 3 * (max(len(a), len(b)) - 1)
        original_len = samples.size
        samples, pad_left, pad_right = BandpassFilter._pad_samples(samples, padlen)
        filtered = filtfilt(b, a, samples, padlen=padlen)
        if pad_left or pad_right:
            return filtered[pad_left:pad_left + original_len]
        return filtered

    @staticmethod
    def _pad_samples(samples: np.ndarray, padlen: int) -> tuple[np.ndarray, int, int]:
        if samples.size > padlen:
            return samples, 0, 0
        missing = padlen + 1 - samples.size
        pad_left = missing // 2
        pad_right = missing - pad_left
        return np.pad(samples, (pad_left, pad_right), mode="reflect"), pad_left, pad_right

    @staticmethod
    def _fill_nans(samples: np.ndarray) -> np.ndarray:
        if not np.isnan(samples).any():
            return samples

        indices = np.arange(samples.size)
        valid = ~np.isnan(samples)
        if valid.sum() == 0:
            return np.zeros_like(samples)
        samples = samples.copy()
        samples[~valid] = np.interp(indices[~valid], indices[valid], samples[valid])
        return samples

# Imports
# dataclass and Iterable: used for a small config structure and to accept any list‑like signal input.
# numpy: turns the signal into a clean numeric array and handles NaNs.
# scipy.signal functions:
# butter: designs the Butterworth filter.
# sosfiltfilt: applies the filter in forward+reverse (zero‑phase).
# iirnotch + filtfilt: makes and applies the 60 Hz notch.
# BandpassFilterConfig
# A small immutable config holder for filter settings. It’s optional but useful if you want to pass settings as a single object later.

# BandpassFilter.design_butterworth(...)
# Converts the low/high cut into normalized frequencies based on Nyquist (half the sampling rate).
# Returns filter coefficients in SOS (second‑order sections), which are numerically stable.
# BandpassFilter.apply(...)
# This is the main entry point.

# Convert input to a float array.
# If empty, return immediately.
# Replace NaN values with interpolated values so filtering doesn’t break.
# Design Butterworth bandpass and apply sosfiltfilt (zero‑phase filtering).
# If notch_freq is valid (not above Nyquist), apply a 60 Hz notch.
# Return the filtered signal.
# apply_ecg(...)
# Convenience wrapper for ECG defaults:

# 0.5–40 Hz bandpass at 250 Hz.
# apply_ppg(...)
# Convenience wrapper for PPG defaults:

# 0.5–8 Hz bandpass at 125 Hz.
# _safe_sosfiltfilt(...) and _safe_filtfilt(...)
# filtfilt requires the signal length to be longer than a padding length. If it’s too short, we reflect‑pad the signal first so filtering still works.

# _reflect_pad(...)
# Reflect‑pads the signal to satisfy the minimum length requirement.

# _fill_nans(...)
# If NaNs exist:

# If all values are NaN, return zeros.
# Otherwise, linearly interpolate missing values.
# Why this matches your spec
# Bandpass + notch: exactly as specified.
# Zero‑phase filtering: uses sosfiltfilt.
# Edge cases: handles short signals and NaNs.
# ECG/PPG configs: hardcoded helpers for the exact ranges you gave.