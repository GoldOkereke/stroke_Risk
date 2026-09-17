"""
services/bandpass_filter.py
============================
Digital bandpass filter for the MIT-BIH Atrial Fibrillation dataset.

Tuned for:
  - Sample rate  : 250 Hz
  - Passband     : 0.5 – 40.0 Hz  (preserves QRS complex)
  - Notch filter : 60 Hz          (removes powerline interference)
  - Filter type  : Butterworth order-4, zero-phase (sosfiltfilt)
"""

import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

logger = logging.getLogger(__name__)


class BandpassFilter:
    """
    Zero-phase Butterworth bandpass filter + 60 Hz notch,
    configured for the MIT-BIH AF ECG dataset (250 Hz).

    Usage
    -----
        filt = BandpassFilter()
        clean = filt.apply(raw_ecg_array)
    """

    def __init__(
        self,
        sample_rate: float = 250.0,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        order: int = 4,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freq = notch_freq
        self.notch_q = notch_q

        self._validate_params()
        self._sos_bandpass = self._design_bandpass()
        self._b_notch, self._a_notch = self._design_notch()

        logger.info(
            "BandpassFilter ready | fs=%.1f Hz | band=[%.1f, %.1f] Hz | "
            "notch=%.1f Hz | order=%d",
            sample_rate, lowcut, highcut, notch_freq, order,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Filter a raw ECG signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw ECG samples at self.sample_rate Hz.

        Returns
        -------
        np.ndarray, shape (N,)
            Bandpass + notch filtered signal, same length as input.
        """
        signal = self._prepare(signal)

        # Stage 1 — bandpass (removes baseline wander + high-freq noise)
        after_bp = self._apply_bandpass(signal)

        # Stage 2 — notch (removes 60 Hz powerline interference)
        after_notch = self._apply_notch(after_bp)

        logger.debug("Filter applied to signal of length %d", len(signal))
        return after_notch

    def apply_bandpass_only(self, signal: np.ndarray) -> np.ndarray:
        """Apply only the bandpass stage (skip notch)."""
        signal = self._prepare(signal)
        return self._apply_bandpass(signal)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_params(self):
        nyq = 0.5 * self.sample_rate
        if not (0 < self.lowcut < self.highcut < nyq):
            raise ValueError(
                f"Invalid cutoffs: lowcut={self.lowcut}, highcut={self.highcut}, "
                f"Nyquist={nyq}. Must satisfy 0 < lowcut < highcut < Nyquist."
            )
        if self.notch_freq >= nyq:
            raise ValueError(
                f"Notch frequency {self.notch_freq} Hz must be below "
                f"Nyquist {nyq} Hz."
            )

    def _design_bandpass(self) -> np.ndarray:
        nyq = 0.5 * self.sample_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = butter(self.order, [low, high], btype="band", output="sos")
        return sos

    def _design_notch(self):
        b, a = iirnotch(w0=self.notch_freq, Q=self.notch_q, fs=self.sample_rate)
        return b, a

    def _prepare(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1-D, got shape {signal.shape}.")
        if len(signal) == 0:
            raise ValueError("Signal is empty.")

        # Minimum length check for sosfiltfilt padding
        min_len = 3 * (2 * self.order + 1)
        if len(signal) < min_len:
            logger.warning(
                "Signal length %d is shorter than recommended minimum %d. "
                "Reflect-padding before filtering.",
                len(signal), min_len,
            )
            pad = min_len
            signal = np.pad(signal, pad, mode="reflect")
            signal = signal  # trimmed after filtering below — handled per stage
            self._trim_pad = pad
        else:
            self._trim_pad = 0

        return signal

    def _apply_bandpass(self, signal: np.ndarray) -> np.ndarray:
        filtered = sosfiltfilt(self._sos_bandpass, signal)
        if self._trim_pad:
            filtered = filtered[self._trim_pad: -self._trim_pad]
        return filtered

    def _apply_notch(self, signal: np.ndarray) -> np.ndarray:
        return filtfilt(self._b_notch, self._a_notch, signal)