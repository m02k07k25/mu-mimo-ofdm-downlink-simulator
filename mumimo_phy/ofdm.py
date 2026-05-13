from __future__ import annotations

import math

import numpy as np


def apply_clipping(time_symbol: np.ndarray, clip_ratio: float) -> np.ndarray:
    time_symbol = np.asarray(time_symbol, dtype=np.complex64)
    rms = np.sqrt(np.mean(np.abs(time_symbol) ** 2, axis=-1, keepdims=True))
    threshold = float(clip_ratio) * np.maximum(rms, 1e-12)
    magnitude = np.abs(time_symbol)
    scale = np.ones_like(magnitude, dtype=np.float32)
    mask = magnitude > threshold
    scale[mask] = (threshold / np.maximum(magnitude, 1e-12))[mask]
    return (time_symbol * scale).astype(np.complex64)


def ofdm_modulate_freq(
    freq_symbol: np.ndarray,
    *,
    n_fft: int,
    n_cp: int,
    case: str = "linear",
    clip_ratio: float = 1.6,
) -> np.ndarray:
    freq_symbol = np.asarray(freq_symbol, dtype=np.complex64)
    time_no_cp = np.fft.ifft(freq_symbol, n=int(n_fft), axis=-1) * math.sqrt(int(n_fft))
    if case == "clipping":
        time_no_cp = apply_clipping(time_no_cp, clip_ratio)
    if case == "cp_removal":
        return time_no_cp.astype(np.complex64)
    cp = time_no_cp[..., -int(n_cp) :] if int(n_cp) > 0 else time_no_cp[..., :0]
    return np.concatenate([cp, time_no_cp], axis=-1).astype(np.complex64)


def precoded_tx_frequency(stream_freq: np.ndarray, w_precoder: np.ndarray) -> np.ndarray:
    return np.einsum("kts,sk->tk", w_precoder, stream_freq).astype(np.complex64)
