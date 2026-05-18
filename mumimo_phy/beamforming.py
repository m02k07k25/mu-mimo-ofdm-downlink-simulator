from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrayConfig:
    n_y: int
    n_z: int = 1
    spacing_y_lambda: float = 0.5
    spacing_z_lambda: float = 0.5

    @property
    def n_ant(self) -> int:
        return int(self.n_y * self.n_z)

    def validate(self) -> None:
        if self.n_y <= 0 or self.n_z <= 0:
            raise ValueError("array dimensions must be positive")
        if self.spacing_y_lambda <= 0.0 or self.spacing_z_lambda <= 0.0:
            raise ValueError("array spacing must be positive")


def array_positions_lambda(array: ArrayConfig) -> np.ndarray:
    array.validate()
    y_index = np.tile(np.arange(array.n_y, dtype=np.float64), array.n_z)
    z_index = np.repeat(np.arange(array.n_z, dtype=np.float64), array.n_y)
    return np.stack(
        [
            np.zeros(array.n_ant, dtype=np.float64),
            y_index * float(array.spacing_y_lambda),
            z_index * float(array.spacing_z_lambda),
        ],
        axis=1,
    )


def spherical_unit_vector(theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    return np.stack(
        [
            np.sin(theta_arr) * np.cos(phi_arr),
            np.sin(theta_arr) * np.sin(phi_arr),
            np.cos(theta_arr),
        ],
        axis=0,
    )


def steering_precoder(
    carrier_freq_hz: float,
    array: ArrayConfig,
    angles: np.ndarray,
    *,
    mode: int = 1,
) -> np.ndarray:
    del carrier_freq_hz  # Spacing is represented in wavelengths, so fc cancels out.
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles.reshape(2, 1)
    if angles.shape[0] != 2:
        raise ValueError("angles must have shape [2, n_rf] with theta and phi rows")

    positions = array_positions_lambda(array)
    n_rf = angles.shape[1]
    if mode == 1:
        beams = np.zeros((array.n_ant, n_rf), dtype=np.complex64)
        for rf_index in range(n_rf):
            direction = spherical_unit_vector(angles[0, rf_index], angles[1, rf_index])
            beams[:, rf_index] = np.exp(-2j * np.pi * (positions @ direction)).astype(np.complex64)
        return beams
    if mode == 2:
        beams = np.zeros((array.n_ant * n_rf, n_rf), dtype=np.complex64)
        for rf_index in range(n_rf):
            direction = spherical_unit_vector(angles[0, rf_index], angles[1, rf_index])
            start = rf_index * array.n_ant
            stop = start + array.n_ant
            beams[start:stop, rf_index] = np.exp(-2j * np.pi * (positions @ direction)).astype(
                np.complex64
            )
        return beams
    raise ValueError("mode must be 1 for all-connected or 2 for sub-connected")


def normalize_precoder(precoder: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    w = np.asarray(precoder, dtype=np.complex64)
    if mode == "none":
        return w
    if mode == "column":
        norms = np.linalg.norm(w, axis=0)
        safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
        return (w / safe_norms[None, :]).astype(np.complex64)
    if mode == "fro":
        norm = np.linalg.norm(w, ord="fro")
        if norm > 1e-12:
            return (w / norm).astype(np.complex64)
        return w
    raise ValueError("precoder normalization must be one of none, column, fro")


def zf_precoder(channel: np.ndarray, *, normalization: str = "none") -> tuple[np.ndarray, float]:
    w = np.linalg.pinv(np.asarray(channel, dtype=np.complex64)).astype(np.complex64)
    k = float(math.sqrt(max(np.trace(w @ w.conj().T).real, 0.0)))
    return normalize_precoder(w, normalization), k


def hybrid_steering_beams(
    *,
    carrier_freq_hz: float,
    tx_array: ArrayConfig,
    rx_array: ArrayConfig,
    selected_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_angles = np.asarray(selected_angles, dtype=np.float64)
    if selected_angles.ndim != 2 or selected_angles.shape[1] != 4:
        raise ValueError("selected_angles must have shape [n_users, 4]")
    n_users = selected_angles.shape[0]
    w_tx = np.zeros((tx_array.n_ant, n_users), dtype=np.complex64)
    w_rx = np.zeros((n_users, rx_array.n_ant), dtype=np.complex64)
    for user_id in range(n_users):
        w_tx[:, user_id] = steering_precoder(
            carrier_freq_hz,
            tx_array,
            selected_angles[user_id, 0:2],
            mode=1,
        )[:, 0]
        w_rx[user_id] = steering_precoder(
            carrier_freq_hz,
            rx_array,
            selected_angles[user_id, 2:4],
            mode=2,
        )[:, 0]
    return w_tx, w_rx


def hybrid_zf_precoder_context(
    h_tx_est: np.ndarray,
    tx_beams: np.ndarray,
    rx_beams: np.ndarray,
    *,
    normalization: str = "column",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_tx_est = np.asarray(h_tx_est, dtype=np.complex64)
    tx_beams = np.asarray(tx_beams, dtype=np.complex64)
    rx_beams = np.asarray(rx_beams, dtype=np.complex64)
    n_fft, n_users, n_rx, n_tx = h_tx_est.shape
    if tx_beams.shape != (n_tx, n_users):
        raise ValueError(f"tx_beams must have shape {(n_tx, n_users)}, got {tx_beams.shape}")
    if rx_beams.shape != (n_users, n_rx):
        raise ValueError(f"rx_beams must have shape {(n_users, n_rx)}, got {rx_beams.shape}")

    # MATLAB uses non-conjugating transpose (Wr(:,d).') in the reference script.
    g_tx_est = np.einsum("ur,kurt->kut", rx_beams, h_tx_est, optimize=True).astype(np.complex64)
    h_eff_est = np.einsum("kut,ts->kus", g_tx_est, tx_beams, optimize=True)
    w_digital = np.linalg.pinv(h_eff_est).astype(np.complex64)
    w_precoder = np.einsum("tu,kus->kts", tx_beams, w_digital, optimize=True).astype(
        np.complex64
    )

    normalization = str(normalization).lower()
    if normalization == "none":
        return g_tx_est, w_digital, w_precoder
    if normalization == "column":
        norms = np.linalg.norm(w_precoder, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
        return g_tx_est, w_digital, (w_precoder / safe_norms).astype(np.complex64)
    if normalization == "fro":
        norms = np.linalg.norm(w_precoder, axis=(1, 2), keepdims=True)
        safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
        return g_tx_est, w_digital, (w_precoder / safe_norms).astype(np.complex64)
    raise ValueError("precoder normalization must be one of none, column, fro")
