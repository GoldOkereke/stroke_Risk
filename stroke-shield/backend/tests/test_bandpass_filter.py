from __future__ import annotations

import numpy as np

from app.services.bandpass_filter import BandpassFilter


def _sine(fs: float, freq: float, seconds: float, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def test_known_signal_passband_ecg():
    """1 Hz sine is inside ECG band (0.5-40Hz) and should mostly pass."""
    fs = 250.0
    signal = _sine(fs, freq=1.0, seconds=10.0)
    filtered = BandpassFilter().apply_ecg(signal)

    # RMS should not collapse to ~0.
    assert _rms(filtered) > 0.3 * _rms(signal)


def test_frequency_response_notch_60hz():
    """60 Hz component should be strongly attenuated by the notch."""
    fs = 250.0
    base = _sine(fs, freq=5.0, seconds=10.0, amp=1.0)
    hum = _sine(fs, freq=60.0, seconds=10.0, amp=0.6)
    mixed = base + hum

    filtered = BandpassFilter().apply(mixed, fs=fs, lowcut=0.5, highcut=40.0, notch_freq=60.0)

    # After filtering, the high-frequency hum should be much smaller.
    # We approximate this by checking that the signal energy didn't increase and the
    # output stays close to the low-frequency component scale.
    assert _rms(filtered) < _rms(mixed)
    assert _rms(filtered) < 1.2 * _rms(base)


def test_edge_case_short_signal_does_not_crash():
    """Very short signals should still return same length without throwing."""
    fs = 250.0
    short = _sine(fs, freq=2.0, seconds=0.1)  # ~25 samples
    filtered = BandpassFilter().apply_ecg(short)
    assert filtered.shape == short.shape


def test_edge_case_nans_are_filled():
    fs = 125.0
    signal = _sine(fs, freq=2.0, seconds=2.0)
    signal[10:20] = np.nan
    filtered = BandpassFilter().apply_ppg(signal)
    assert not np.isnan(filtered).any()


def test_ecg_and_ppg_configs_return_same_length():
    ecg = _sine(250.0, freq=2.0, seconds=5.0)
    ppg = _sine(125.0, freq=2.0, seconds=5.0)

    bp = BandpassFilter()
    ecg_out = bp.apply_ecg(ecg)
    ppg_out = bp.apply_ppg(ppg)

    assert ecg_out.shape == ecg.shape
    assert ppg_out.shape == ppg.shape
