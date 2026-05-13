from __future__ import annotations

import math

import numpy as np


def rf_impairment_widely_linear_coefficients(
    *,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_error_deg: float = 0.0,
    common_phase_error_deg: float = 0.0,
) -> tuple[complex, complex]:
    gain_ratio = 10.0 ** (float(iq_gain_imbalance_db) / 20.0)
    i_gain = math.sqrt(gain_ratio)
    q_gain = 1.0 / math.sqrt(gain_ratio)
    phase_error = math.radians(float(iq_phase_error_deg))
    rotation = np.exp(1j * math.radians(float(common_phase_error_deg)))

    alpha_iq = 0.5 * (
        i_gain
        + q_gain * math.cos(phase_error)
        + 1j * q_gain * math.sin(phase_error)
    )
    beta_iq = 0.5 * (
        i_gain
        - q_gain * math.cos(phase_error)
        + 1j * q_gain * math.sin(phase_error)
    )
    return complex(rotation * alpha_iq), complex(rotation * beta_iq)


def rf_impairment_real_matrix(
    *,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_error_deg: float = 0.0,
    common_phase_error_deg: float = 0.0,
) -> np.ndarray:
    gain_ratio = 10.0 ** (float(iq_gain_imbalance_db) / 20.0)
    i_gain = math.sqrt(gain_ratio)
    q_gain = 1.0 / math.sqrt(gain_ratio)
    iq_phase = math.radians(float(iq_phase_error_deg))
    common_phase = math.radians(float(common_phase_error_deg))
    cos_iq = math.cos(iq_phase)
    sin_iq = math.sin(iq_phase)
    cos_common = math.cos(common_phase)
    sin_common = math.sin(common_phase)

    iq_matrix = np.array(
        [
            [i_gain, 0.0],
            [q_gain * sin_iq, q_gain * cos_iq],
        ],
        dtype=np.float32,
    )
    rotation_matrix = np.array(
        [
            [cos_common, -sin_common],
            [sin_common, cos_common],
        ],
        dtype=np.float32,
    )
    return (rotation_matrix @ iq_matrix).astype(np.float32)


def apply_common_phase_rotation(
    waveform: np.ndarray,
    phase_error_deg: float,
) -> np.ndarray:
    """Rotate a complex waveform by a common phase offset."""
    if phase_error_deg == 0.0:
        return waveform.astype(np.complex64, copy=True)
    phase = math.radians(float(phase_error_deg))
    return (waveform * np.complex64(np.exp(1j * phase))).astype(np.complex64)


def apply_iq_imbalance(
    waveform: np.ndarray,
    gain_imbalance_db: float,
    phase_error_deg: float,
) -> np.ndarray:
    """Apply a simple receiver I/Q gain and quadrature phase imbalance model.

    gain_imbalance_db is the I-to-Q amplitude gain difference in dB. Positive
    values make I larger than Q while keeping the average gain near one.
    phase_error_deg is the quadrature error from an ideal 90 degree I/Q split.
    """
    if gain_imbalance_db == 0.0 and phase_error_deg == 0.0:
        return waveform.astype(np.complex64, copy=True)

    gain_ratio = 10.0 ** (float(gain_imbalance_db) / 20.0)
    i_gain = math.sqrt(gain_ratio)
    q_gain = 1.0 / math.sqrt(gain_ratio)
    phase_error = math.radians(float(phase_error_deg))

    i_raw = np.real(waveform).astype(np.float32, copy=False)
    q_raw = np.imag(waveform).astype(np.float32, copy=False)
    i_path = i_gain * i_raw
    q_path = q_gain * (q_raw * math.cos(phase_error) + i_raw * math.sin(phase_error))
    return (i_path + 1j * q_path).astype(np.complex64)


def apply_rf_impairments(
    waveform: np.ndarray,
    *,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_error_deg: float = 0.0,
    common_phase_error_deg: float = 0.0,
) -> np.ndarray:
    """Apply the RF impairment stack used by the MU-MIMO simulator."""
    impaired = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=iq_gain_imbalance_db,
        phase_error_deg=iq_phase_error_deg,
    )
    return apply_common_phase_rotation(impaired, common_phase_error_deg)
