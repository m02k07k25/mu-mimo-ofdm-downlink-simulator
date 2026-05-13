from __future__ import annotations

import math

import numpy as np


def db_to_linear(db_value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db_value) / 10.0)


def linear_to_db(value: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value), floor))


def noise_power_from_snr(signal_power: float, snr_db: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    return float(signal_power) / max(float(db_to_linear(float(snr_db))), 1e-300)


def matlab_fixed_noise_power(snr_db: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    return float(10.0 ** (-float(snr_db) / 10.0))


def add_awgn(
    values: np.ndarray,
    noise_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_power <= 0.0:
        return np.asarray(values, dtype=np.complex64)
    noise = (
        rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    ) * math.sqrt(noise_power / 2.0)
    return (values + noise).astype(np.complex64)
