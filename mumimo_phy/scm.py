from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .beamforming import ArrayConfig, array_positions_lambda, spherical_unit_vector


@dataclass(frozen=True)
class ScmChannelConfig:
    n_path: int = 7
    n_rays_per_path: int = 15
    n_rx: int = 4
    n_tx: int = 8
    pdp_decay: float = 5.0
    carrier_freq_hz: float = 800e6
    tx_array: ArrayConfig | None = None
    rx_array: ArrayConfig | None = None
    asd_deg: float = 3.0
    zsd_deg: float = 3.0
    asa_deg: float = 3.0
    zsa_deg: float = 3.0

    def validate(self) -> None:
        if self.n_path <= 0:
            raise ValueError("n_path must be positive")
        if self.n_rays_per_path <= 0:
            raise ValueError("n_rays_per_path must be positive")
        if self.n_rx <= 0 or self.n_tx <= 0:
            raise ValueError("antenna counts must be positive")
        if self.pdp_decay <= 0.0:
            raise ValueError("pdp_decay must be positive")
        if self.carrier_freq_hz <= 0.0:
            raise ValueError("carrier_freq_hz must be positive")


@dataclass(frozen=True)
class ScmChannelSample:
    h_time: np.ndarray
    center_angles: np.ndarray
    selected_angles: np.ndarray


class ScmChannelGenerator:
    def __init__(self, cfg: ScmChannelConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.tx_array = cfg.tx_array or ArrayConfig(cfg.n_tx)
        self.rx_array = cfg.rx_array or ArrayConfig(cfg.n_rx)
        if self.tx_array.n_ant != cfg.n_tx:
            raise ValueError("tx_array antenna count does not match n_tx")
        if self.rx_array.n_ant != cfg.n_rx:
            raise ValueError("rx_array antenna count does not match n_rx")
        self.tx_positions = array_positions_lambda(self.tx_array)
        self.rx_positions = array_positions_lambda(self.rx_array)

    def generate_multiuser(
        self,
        n_users: int,
        rng: np.random.Generator,
    ) -> ScmChannelSample:
        h_time = np.zeros(
            (n_users, self.cfg.n_path, self.cfg.n_rx, self.cfg.n_tx),
            dtype=np.complex64,
        )
        center_angles = np.zeros((n_users, 4, self.cfg.n_path), dtype=np.float32)
        selected_angles = np.zeros((n_users, 4), dtype=np.float32)
        for user_id in range(n_users):
            h_user, angles = self._generate_user_channel(rng)
            h_time[user_id] = h_user
            center_angles[user_id] = angles.astype(np.float32)
            selected_index = int(np.argmax(np.abs(h_user[:, 0, 0])))
            selected_angles[user_id] = angles[:, selected_index].astype(np.float32)
        return ScmChannelSample(
            h_time=h_time,
            center_angles=center_angles,
            selected_angles=selected_angles,
        )

    def _generate_user_channel(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        center_angles = np.zeros((4, cfg.n_path), dtype=np.float64)
        center_angles[0] = rng.random(cfg.n_path) * np.pi
        center_angles[1] = -np.pi / 2.0 + rng.random(cfg.n_path) * np.pi
        center_angles[2] = rng.random(cfg.n_path) * np.pi
        center_angles[3] = -np.pi / 2.0 + rng.random(cfg.n_path) * np.pi

        path_index = np.arange(1, cfg.n_path + 1, dtype=np.float64)
        pdp = np.exp(-path_index / float(cfg.pdp_decay))
        pdp /= np.sum(pdp)

        h_user = np.zeros((cfg.n_path, cfg.n_rx, cfg.n_tx), dtype=np.complex64)
        spreads = np.deg2rad(
            np.array([cfg.zsd_deg, cfg.asd_deg, cfg.zsa_deg, cfg.asa_deg], dtype=np.float64)
        )
        ray_scale = 1.0 / math.sqrt(float(cfg.n_rays_per_path))
        for path_id in range(cfg.n_path):
            h_path = np.zeros((cfg.n_rx, cfg.n_tx), dtype=np.complex128)
            for _ in range(cfg.n_rays_per_path):
                ray_angles = center_angles[:, path_id] + rng.standard_normal(4) * spreads
                phase = np.exp(2j * np.pi * rng.random())
                rx_response = self._array_response(self.rx_positions, ray_angles[2], ray_angles[3])
                tx_response = self._array_response(self.tx_positions, ray_angles[0], ray_angles[1])
                h_path += phase * np.outer(rx_response, tx_response) * ray_scale
            h_user[path_id] = (h_path * math.sqrt(float(pdp[path_id]))).astype(np.complex64)
        return h_user, center_angles

    @staticmethod
    def _array_response(positions_lambda: np.ndarray, theta: float, phi: float) -> np.ndarray:
        direction = spherical_unit_vector(theta, phi)
        return np.exp(2j * np.pi * (positions_lambda @ direction)).astype(np.complex64)


def channel_frequency_response(h_time: np.ndarray, *, n_fft: int) -> np.ndarray:
    h_time = np.asarray(h_time, dtype=np.complex64)
    n_users, n_path, n_rx, n_tx = h_time.shape
    h_pad = np.zeros((n_users, int(n_fft), n_rx, n_tx), dtype=np.complex64)
    h_pad[:, :n_path, :, :] = h_time
    return np.transpose(np.fft.fft(h_pad, n=int(n_fft), axis=1), (1, 0, 2, 3)).astype(
        np.complex64
    )


def apply_multipath_mimo(tx_time: np.ndarray, h_time: np.ndarray) -> np.ndarray:
    tx_time = np.asarray(tx_time, dtype=np.complex64)
    h_time = np.asarray(h_time, dtype=np.complex64)
    n_tx, time_len = tx_time.shape
    n_users, n_taps, n_rx, h_tx = h_time.shape
    if h_tx != n_tx:
        raise ValueError(f"Channel n_tx={h_tx} does not match waveform n_tx={n_tx}")
    y_time = np.zeros((n_users, n_rx, time_len), dtype=np.complex64)
    for tap_index in range(min(n_taps, time_len)):
        usable = time_len - tap_index
        y_time[:, :, tap_index:] += np.einsum(
            "urt,tl->url",
            h_time[:, tap_index, :, :],
            tx_time[:, :usable],
            optimize=True,
        )
    return y_time
