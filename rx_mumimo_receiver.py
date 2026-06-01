from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mumimo_phy import (
    bits_per_symbol,
    ofdm_demodulate_freq,
    qam_demodulate,
    rf_impairment_widely_linear_coefficients,
)
from environment_config import (
    DEFAULT_ENVIRONMENT_CONFIG,
    dataset_dir as environment_dataset_dir,
    load_environment_config,
    result_dir as environment_result_dir,
    rx_training_defaults,
)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_ENVIRONMENT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    environment = load_environment_config(config_args.config)
    dataset_name = environment["dataset_name"]

    parser = argparse.ArgumentParser(
        description="Train and evaluate a raw end-to-end MU-MIMO ComNet OFDM receiver."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_ENVIRONMENT_CONFIG),
        help="Shared environment JSON config. Individual CLI options override its values.",
    )
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--result-dir", type=str)
    parser.add_argument(
        "--mode",
        type=str,
        default="train-all",
        choices=["train-all", "train-ce", "train-sd", "eval"],
    )
    parser.add_argument("--sd-type", type=str, default="bilstm", choices=["bilstm"])
    parser.add_argument("--sd-loss", type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument(
        "--sd-feature-set",
        type=str,
        default="wl-zf-reliability",
        choices=["wl-zf-reliability"],
    )
    parser.add_argument("--ce-type", type=str, default="linear", choices=["linear"])
    parser.add_argument("--ce-init", type=str, default="lmmse", choices=["lmmse"])
    parser.add_argument(
        "--ce-target",
        type=str,
        default="auto",
        choices=["auto", "pre-rf", "rf-linear", "wl-rf"],
        help=(
            "Channel target for LMMSE/ComNet CE. auto uses wl-rf. wl-rf trains CE "
            "on the augmented widely-linear (A,B) channel; when RF impairment is "
            "zero, the conjugate branch is simply zero."
        ),
    )
    parser.add_argument("--ce-checkpoint", type=str, default=None)
    parser.add_argument("--bilstm-checkpoint", type=str, default=None)
    parser.add_argument("--lmmse-checkpoint", type=str, default=None)
    parser.add_argument(
        "--lmmse-mode",
        type=str,
        default="snr-binned",
        choices=["global", "snr-binned"],
        help=(
            "Empirical LMMSE CE estimator mode. global fits one weight matrix "
            "over all training SNRs. snr-binned fits one weight per training SNR "
            "and uses the nearest bin for unseen test SNRs."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="auto")
    parser.set_defaults(
        dataset_dir=str(environment_dataset_dir(dataset_name)),
        result_dir=str(environment_result_dir(dataset_name)),
    )
    args = parser.parse_args()
    for key, value in rx_training_defaults(environment).items():
        setattr(args, key, value)
    return args


class MuMimoCELinearNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.base = nn.Linear(self.input_dim, self.input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def init_identity(self) -> None:
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(self.input_dim, device=self.base.weight.device))

    def init_base(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != (self.input_dim, self.input_dim):
            raise ValueError(f"Expected CE weight shape {(self.input_dim, self.input_dim)}, got {tuple(weight.shape)}")
        with torch.no_grad():
            self.base.weight.copy_(weight)


MuMimoCEModel = MuMimoCELinearNet


def build_ce_model(
    ce_type: str,
    input_dim: int,
    *,
    hidden_dim: int = 512,
    dropout: float = 0.05,
) -> MuMimoCEModel:
    ce_type = str(ce_type).lower()
    if ce_type == "linear":
        return MuMimoCELinearNet(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported CE type: {ce_type}")


class MuMimoBiLSTMSDNet(nn.Module):
    def __init__(
        self,
        n_fft: int,
        bits_per_symbol_value: int,
        group_size: int,
        hidden_dims: tuple[int, int, int] = (64, 32, 16),
        feature_dim: int = 9,
        sd_feature_set: str = "wl-zf-reliability",
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.bits_per_symbol = int(bits_per_symbol_value)
        self.group_size = int(group_size)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.feature_dim = int(feature_dim)
        self.sd_feature_set = str(sd_feature_set)
        if self.n_fft % self.group_size != 0:
            raise ValueError("n_fft must be divisible by group_size")
        if len(self.hidden_dims) != 3 or min(self.hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain three positive integers")
        self.n_groups = self.n_fft // self.group_size
        h1, h2, h3 = self.hidden_dims
        self.lstm1 = nn.LSTM(self.feature_dim, h1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * h1, h2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * h2, h3, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * h3 * self.group_size, self.group_size * self.bits_per_symbol)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x.reshape(x.shape[0], self.n_groups, self.group_size * 2 * self.hidden_dims[2])
        return self.output(x)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_config(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "config.json"
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["modulation"] = str(cfg["modulation"]).upper()
    return cfg


def find_one(dataset_dir: Path, pattern: str) -> Path:
    matches = sorted(dataset_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} in {dataset_dir}, got {len(matches)}")
    return matches[0]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def ofdm_demodulate(rx_time: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    return ofdm_demodulate_freq(
        rx_time,
        n_fft=int(cfg["n_fft"]),
        n_cp=int(cfg["n_cp"]),
        case=str(cfg.get("case", "linear")),
    )


def resolve_ce_target_mode(ce_target: str, cfg: dict[str, Any]) -> str:
    ce_target = str(ce_target).lower()
    if ce_target == "auto":
        return "wl-rf"
    if ce_target in {"pre-rf", "rf-linear", "wl-rf"}:
        return ce_target
    raise ValueError(f"Unsupported CE target: {ce_target}")


def is_wl_ce_target(ce_target: str, cfg: dict[str, Any]) -> bool:
    return resolve_ce_target_mode(ce_target, cfg) == "wl-rf"


def rf_wl_coefficients(cfg: dict[str, Any]) -> tuple[complex, complex]:
    return rf_impairment_widely_linear_coefficients(
        iq_gain_imbalance_db=float(cfg.get("rx_iq_gain_imbalance_db", 0.0)),
        iq_phase_error_deg=float(cfg.get("rx_iq_phase_error_deg", 0.0)),
        common_phase_error_deg=float(cfg.get("rx_common_phase_error_deg", 0.0)),
    )


def mirror_subcarrier_indices(n_fft: int) -> np.ndarray:
    return (-np.arange(int(n_fft), dtype=np.int64)) % int(n_fft)


def make_wl_channel_from_pre_rf(a_pre_rf: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    a_pre_rf = np.asarray(a_pre_rf, dtype=np.complex64)
    alpha, beta = rf_wl_coefficients(cfg)
    mirror = mirror_subcarrier_indices(a_pre_rf.shape[1])
    a_wl = (np.complex64(alpha) * a_pre_rf).astype(np.complex64)
    b_wl = (np.complex64(beta) * np.conj(a_pre_rf[:, mirror])).astype(np.complex64)
    return np.concatenate([a_wl, b_wl], axis=-1).astype(np.complex64)


def split_wl_channel(ab_wl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab_wl = np.asarray(ab_wl, dtype=np.complex64)
    if ab_wl.shape[-1] % 2 != 0:
        raise ValueError(f"Expected augmented WL channel with even last dim, got {ab_wl.shape}")
    half = ab_wl.shape[-1] // 2
    return ab_wl[..., :half], ab_wl[..., half:]


def is_augmented_wl_channel(values: np.ndarray, cfg: dict[str, Any]) -> bool:
    values = np.asarray(values)
    n_streams = int(cfg.get("n_streams", cfg["n_users"]))
    return values.ndim >= 1 and values.shape[-1] == 2 * n_streams


def as_wl_channel(values: np.ndarray, cfg: dict[str, Any], ce_target: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex64)
    if is_augmented_wl_channel(values, cfg):
        return values
    mode = resolve_ce_target_mode(ce_target, cfg)
    if mode == "rf-linear":
        alpha, _ = rf_wl_coefficients(cfg)
        if abs(alpha) > 1e-12:
            values = (values / np.complex64(alpha)).astype(np.complex64)
    return make_wl_channel_from_pre_rf(values, cfg)


def make_ce_target(a_true: np.ndarray, cfg: dict[str, Any], ce_target: str) -> np.ndarray:
    mode = resolve_ce_target_mode(ce_target, cfg)
    a_true = np.asarray(a_true, dtype=np.complex64)
    if mode == "pre-rf":
        return a_true
    if mode == "rf-linear":
        alpha, _ = rf_wl_coefficients(cfg)
        return (np.complex64(alpha) * a_true).astype(np.complex64)
    if mode == "wl-rf":
        return make_wl_channel_from_pre_rf(a_true, cfg)
    raise ValueError(f"Unsupported CE target: {ce_target}")


def wl_ls_from_pilots(
    y_p_slots: np.ndarray,
    x_p: np.ndarray,
    cfg: dict[str, Any],
    eps: float,
) -> np.ndarray:
    y_p_slots = np.asarray(y_p_slots, dtype=np.complex64)
    x_p = np.asarray(x_p, dtype=np.complex64)
    n_frames, n_streams, n_users, n_rx, n_fft = y_p_slots.shape
    alpha, beta = rf_wl_coefficients(cfg)
    det = (abs(alpha) ** 2) - (abs(beta) ** 2)
    if abs(det) < float(eps):
        raise ValueError("I/Q imbalance WL coefficient matrix is singular; cannot form WL-LS")

    a_pre = np.zeros((n_frames, n_fft, n_users, n_rx, n_streams), dtype=np.complex64)
    for stream_id in range(n_streams):
        visited: set[int] = set()
        pilots = x_p[:, stream_id, stream_id, :]
        for subcarrier in range(n_fft):
            if subcarrier in visited:
                continue
            mirror = (-subcarrier) % n_fft
            visited.add(subcarrier)
            visited.add(mirror)

            x_k = pilots[:, subcarrier]
            y_k = y_p_slots[:, stream_id, :, :, subcarrier]
            safe_x_k = np.where(np.abs(x_k) < eps, eps + 0j, x_k)
            if mirror == subcarrier:
                u_k = (np.conj(alpha) * y_k - beta * np.conj(y_k)) / det
                a_pre[:, subcarrier, :, :, stream_id] = u_k / safe_x_k[:, None, None]
                continue

            x_m = pilots[:, mirror]
            y_m = y_p_slots[:, stream_id, :, :, mirror]
            safe_x_m_conj = np.where(np.abs(x_m) < eps, eps + 0j, np.conj(x_m))
            u_k = (np.conj(alpha) * y_k - beta * np.conj(y_m)) / det
            v_m = (-np.conj(beta) * y_k + alpha * np.conj(y_m)) / det
            a_pre[:, subcarrier, :, :, stream_id] = u_k / safe_x_k[:, None, None]
            a_pre[:, mirror, :, :, stream_id] = np.conj(v_m / safe_x_m_conj[:, None, None])

    return make_wl_channel_from_pre_rf(a_pre, cfg)


def preprocess_split(
    raw: dict[str, np.ndarray],
    cfg: dict[str, Any],
    eps: float,
    ce_target: str = "pre-rf",
) -> dict[str, np.ndarray]:
    y_p_slots = ofdm_demodulate(raw["rx_p_time"], cfg)
    y_d_urk = ofdm_demodulate(raw["rx_d_time"], cfg)
    x_p = np.asarray(raw["x_p_freq"], dtype=np.complex64)
    n_frames, n_streams, n_users, n_rx, n_fft = y_p_slots.shape

    a_plain_ls = np.zeros((n_frames, n_fft, n_users, n_rx, n_streams), dtype=np.complex64)
    for stream_id in range(n_streams):
        denom = x_p[:, stream_id, stream_id, :]
        safe_den = np.where(np.abs(denom) < eps, eps + 0j, denom)
        y_slot = np.transpose(y_p_slots[:, stream_id], (0, 3, 1, 2))
        a_plain_ls[..., stream_id] = y_slot / safe_den[:, :, None, None]

    a_true = np.asarray(raw["A_eff_true"], dtype=np.complex64)
    mode = resolve_ce_target_mode(ce_target, cfg)
    a_ce_target = make_ce_target(a_true, cfg, ce_target)
    a_wl_true = make_wl_channel_from_pre_rf(a_true, cfg)
    a_wl_ls = wl_ls_from_pilots(y_p_slots, x_p, cfg, eps) if mode == "wl-rf" else None
    a_ls = a_wl_ls if a_wl_ls is not None else a_plain_ls

    return {
        "y_p": y_p_slots,
        "y_d": np.transpose(y_d_urk, (0, 3, 1, 2)).astype(np.complex64),
        "a_ls": a_ls,
        "a_plain_ls": a_plain_ls,
        "a_wl_ls": a_wl_ls if a_wl_ls is not None else make_wl_channel_from_pre_rf(a_plain_ls, cfg),
        "a_true": a_true,
        "a_ce_target": a_ce_target,
        "a_wl_true": a_wl_true,
        "a_linear_target": make_ce_target(a_true, cfg, "rf-linear"),
        "bits": np.asarray(raw["bits"], dtype=np.int8),
        "x_d_freq": np.asarray(raw["x_d_freq"], dtype=np.complex64),
        "snr_db": np.asarray(raw["snr_db"], dtype=np.float32),
        "noise_power": np.asarray(raw["noise_power"], dtype=np.float32),
        "desired_power": np.asarray(raw.get("desired_power", np.zeros((n_frames, n_users))), dtype=np.float32),
        "inter_stream_power": np.asarray(
            raw.get("inter_stream_power", np.zeros((n_frames, n_users))),
            dtype=np.float32,
        ),
        "effective_sinr_db": np.asarray(
            raw.get("effective_sinr_db", np.zeros((n_frames, n_users))),
            dtype=np.float32,
        ),
        "cond_A": np.asarray(raw.get("cond_A", np.zeros((n_frames, n_fft, n_users))), dtype=np.float32),
    }


def ce_feature_dim(cfg: dict[str, Any], ce_target: str = "pre-rf") -> int:
    n_fft = int(cfg["n_fft"])
    n_users = int(cfg["n_users"])
    n_streams = int(cfg.get("n_streams", n_users))
    n_rx = int(cfg["n_rx_per_ue"])
    if is_wl_ce_target(ce_target, cfg):
        n_streams *= 2
    return 2 * n_fft * n_rx * n_streams


def ce_complex_to_ri(a_eff: np.ndarray) -> np.ndarray:
    a_eff = np.asarray(a_eff, dtype=np.complex64)
    n_frames, n_fft, n_users, n_rx, n_streams = a_eff.shape
    per_user = np.transpose(a_eff, (0, 2, 1, 3, 4)).reshape(
        n_frames * n_users,
        n_fft * n_rx * n_streams,
    )
    return np.concatenate([per_user.real, per_user.imag], axis=1).astype(np.float32)


def ce_ri_to_complex(
    values: np.ndarray,
    *,
    n_frames: int,
    n_fft: int,
    n_users: int,
    n_rx: int,
    n_streams: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    half = values.shape[1] // 2
    complex_values = values[:, :half] + 1j * values[:, half:]
    per_user = complex_values.reshape(n_frames, n_users, n_fft, n_rx, n_streams)
    return np.transpose(per_user, (0, 2, 1, 3, 4)).astype(np.complex64)


def ce_ri_to_complex_like(values: np.ndarray, like: np.ndarray) -> np.ndarray:
    n_frames, n_fft, n_users, n_rx, n_streams = np.asarray(like).shape
    return ce_ri_to_complex(
        values,
        n_frames=n_frames,
        n_fft=n_fft,
        n_users=n_users,
        n_rx=n_rx,
        n_streams=n_streams,
    )


def bit_error_rate(pred_bits: np.ndarray, true_bits: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_bits, dtype=np.int8) != np.asarray(true_bits, dtype=np.int8)))


def hard_demod_stream_grid(symbols: np.ndarray, modulation: str) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex64)
    bps = bits_per_symbol(modulation)
    return qam_demodulate(symbols.reshape(-1), modulation).reshape(*symbols.shape, bps)


def channel_mse(a_hat: np.ndarray, a_true: np.ndarray) -> float:
    return float(np.mean(np.abs(a_hat - a_true) ** 2))


def channel_nmse(a_hat: np.ndarray, a_true: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(a_hat - a_true) ** 2))
    denominator = float(np.sum(np.abs(a_true) ** 2))
    return numerator / max(denominator, 1e-300)


def to_db(value: float) -> float:
    return float(10.0 * math.log10(max(float(value), 1e-300)))


def linear_detect(
    y_d: np.ndarray,
    a_eff: np.ndarray,
    noise_power: np.ndarray,
    *,
    method: str,
    eps: float,
) -> np.ndarray | None:
    n_frames, _, _, n_rx = y_d.shape
    n_streams = a_eff.shape[-1]
    if method == "zf" and n_streams > n_rx:
        return None

    ah = np.swapaxes(np.conj(a_eff), -1, -2)
    gram = np.matmul(ah, a_eff)
    matched = np.matmul(ah, y_d[..., None])[..., 0]
    eye = np.eye(n_streams, dtype=np.complex64)

    if method == "zf":
        system = gram + (float(eps) * eye)[None, None, None, :, :]
    elif method == "mmse":
        sigma2 = np.asarray(noise_power, dtype=np.float32).reshape(n_frames, 1, 1, 1, 1)
        system = gram + sigma2 * eye[None, None, None, :, :]
    else:
        raise ValueError(f"Unsupported detector: {method}")

    try:
        estimates = np.linalg.solve(system, matched[..., None])[..., 0]
        if method == "mmse":
            response = np.linalg.solve(system, gram)
            gain = np.diagonal(response, axis1=-2, axis2=-1)
            estimates = estimates / np.where(np.abs(gain) > eps, gain, 1.0 + 0.0j)
        return estimates.astype(np.complex64)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(system)
        estimates = np.matmul(pinv, matched[..., None])[..., 0]
        if method == "mmse":
            response = np.matmul(pinv, gram)
            gain = np.diagonal(response, axis1=-2, axis2=-1)
            estimates = estimates / np.where(np.abs(gain) > eps, gain, 1.0 + 0.0j)
        return estimates.astype(np.complex64)


def _complex_channel_real_matrix(a_eff: np.ndarray) -> np.ndarray:
    a_eff = np.asarray(a_eff, dtype=np.complex64)
    *leading, n_rx, n_streams = a_eff.shape
    out = np.zeros((*leading, 2 * n_rx, 2 * n_streams), dtype=np.float32)
    real = a_eff.real.astype(np.float32, copy=False)
    imag = a_eff.imag.astype(np.float32, copy=False)
    out[..., :n_rx, :n_streams] = real
    out[..., :n_rx, n_streams:] = -imag
    out[..., n_rx:, :n_streams] = imag
    out[..., n_rx:, n_streams:] = real
    return out


def _complex_conj_input_real_matrix(a_eff: np.ndarray) -> np.ndarray:
    a_eff = np.asarray(a_eff, dtype=np.complex64)
    *leading, n_rx, n_streams = a_eff.shape
    out = np.zeros((*leading, 2 * n_rx, 2 * n_streams), dtype=np.float32)
    real = a_eff.real.astype(np.float32, copy=False)
    imag = a_eff.imag.astype(np.float32, copy=False)
    out[..., :n_rx, :n_streams] = real
    out[..., :n_rx, n_streams:] = imag
    out[..., n_rx:, :n_streams] = imag
    out[..., n_rx:, n_streams:] = -real
    return out


def _complex_grid_to_real(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex64)
    return np.concatenate(
        [values.real.astype(np.float32, copy=False), values.imag.astype(np.float32, copy=False)],
        axis=-1,
    )


def _block_gain_compensate_real(
    estimates: np.ndarray,
    response: np.ndarray,
    n_symbols: int,
    eps: float,
) -> np.ndarray:
    corrected = np.array(estimates, copy=True)
    half = corrected.shape[-1] // 2
    eye2 = (float(eps) * np.eye(2, dtype=np.float32))[None, None, :, :]
    for symbol_id in range(n_symbols):
        idx_re = symbol_id
        idx_im = half + symbol_id
        block = np.empty((*response.shape[:-2], 2, 2), dtype=np.float32)
        block[..., 0, 0] = response[..., idx_re, idx_re]
        block[..., 0, 1] = response[..., idx_re, idx_im]
        block[..., 1, 0] = response[..., idx_im, idx_re]
        block[..., 1, 1] = response[..., idx_im, idx_im]
        pair = np.stack([corrected[..., idx_re], corrected[..., idx_im]], axis=-1)
        try:
            pair = np.linalg.solve(block + eye2, pair[..., None])[..., 0]
        except np.linalg.LinAlgError:
            pair = np.matmul(np.linalg.pinv(block + eye2), pair[..., None])[..., 0]
        corrected[..., idx_re] = pair[..., 0]
        corrected[..., idx_im] = pair[..., 1]
    return corrected


def _solve_real_mmse(
    b_matrix: np.ndarray,
    y_real: np.ndarray,
    noise_power: np.ndarray,
    *,
    method: str,
    eps: float,
    n_symbols: int,
) -> np.ndarray:
    bt = np.swapaxes(b_matrix, -1, -2)
    gram = np.matmul(bt, b_matrix)
    matched = np.matmul(bt, y_real[..., None])[..., 0]
    dim = gram.shape[-1]
    eye = np.eye(dim, dtype=np.float32)
    if method == "zf":
        system = gram + (float(eps) * eye)[None, None, :, :]
    elif method == "mmse":
        sigma2 = np.asarray(noise_power, dtype=np.float32).reshape(-1, 1, 1, 1)
        system = gram + sigma2 * eye[None, None, :, :]
    else:
        raise ValueError(f"Unsupported detector: {method}")

    try:
        estimates = np.linalg.solve(system, matched[..., None])[..., 0]
        response = np.linalg.solve(system, gram)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(system)
        estimates = np.matmul(pinv, matched[..., None])[..., 0]
        response = np.matmul(pinv, gram)
    if method == "mmse":
        estimates = _block_gain_compensate_real(estimates, response, n_symbols, eps)
    return estimates.astype(np.float32)


def wl_detect(
    y_d: np.ndarray,
    ab_wl: np.ndarray,
    noise_power: np.ndarray,
    *,
    method: str,
    eps: float,
) -> np.ndarray | None:
    y_d = np.asarray(y_d, dtype=np.complex64)
    a_wl, b_wl = split_wl_channel(ab_wl)
    n_frames, n_fft, n_users, n_rx = y_d.shape
    n_streams = a_wl.shape[-1]
    if method == "zf" and n_streams > n_rx:
        return None

    estimates = np.zeros((n_frames, n_fft, n_users, n_streams), dtype=np.complex64)
    frame_noise_power = np.asarray(noise_power, dtype=np.float32)
    visited: set[int] = set()

    for subcarrier in range(n_fft):
        if subcarrier in visited:
            continue
        mirror = (-subcarrier) % n_fft
        visited.add(subcarrier)
        visited.add(mirror)

        a_k = a_wl[:, subcarrier]
        b_k = b_wl[:, subcarrier]
        y_k = _complex_grid_to_real(y_d[:, subcarrier])
        if mirror == subcarrier:
            b_matrix = _complex_channel_real_matrix(a_k) + _complex_conj_input_real_matrix(b_k)
            solved = _solve_real_mmse(
                b_matrix,
                y_k,
                frame_noise_power,
                method=method,
                eps=eps,
                n_symbols=n_streams,
            )
            estimates[:, subcarrier] = (
                solved[..., :n_streams] + 1j * solved[..., n_streams:]
            ).astype(np.complex64)
            continue

        a_m = a_wl[:, mirror]
        b_m = b_wl[:, mirror]
        y_m = _complex_grid_to_real(y_d[:, mirror])
        y_pair = np.concatenate([y_k, y_m], axis=-1)

        top_k = _complex_channel_real_matrix(a_k)
        top_m = _complex_conj_input_real_matrix(b_k)
        bottom_k = _complex_conj_input_real_matrix(b_m)
        bottom_m = _complex_channel_real_matrix(a_m)

        b_matrix = np.zeros(
            (n_frames, n_users, 4 * n_rx, 4 * n_streams),
            dtype=np.float32,
        )
        b_matrix[..., : 2 * n_rx, :n_streams] = top_k[..., :n_streams]
        b_matrix[..., : 2 * n_rx, 2 * n_streams : 3 * n_streams] = top_k[..., n_streams:]
        b_matrix[..., : 2 * n_rx, n_streams : 2 * n_streams] = top_m[..., :n_streams]
        b_matrix[..., : 2 * n_rx, 3 * n_streams :] = top_m[..., n_streams:]
        b_matrix[..., 2 * n_rx :, :n_streams] = bottom_k[..., :n_streams]
        b_matrix[..., 2 * n_rx :, 2 * n_streams : 3 * n_streams] = bottom_k[..., n_streams:]
        b_matrix[..., 2 * n_rx :, n_streams : 2 * n_streams] = bottom_m[..., :n_streams]
        b_matrix[..., 2 * n_rx :, 3 * n_streams :] = bottom_m[..., n_streams:]

        solved = _solve_real_mmse(
            b_matrix,
            y_pair,
            frame_noise_power,
            method=method,
            eps=eps,
            n_symbols=2 * n_streams,
        )
        estimates[:, subcarrier] = (
            solved[..., :n_streams] + 1j * solved[..., 2 * n_streams : 3 * n_streams]
        ).astype(np.complex64)
        estimates[:, mirror] = (
            solved[..., n_streams : 2 * n_streams]
            + 1j * solved[..., 3 * n_streams :]
        ).astype(np.complex64)

    return estimates


def target_user_streams(full_stream_estimates: np.ndarray) -> np.ndarray:
    n_frames, n_fft, n_users, _ = full_stream_estimates.shape
    out = np.zeros((n_frames, n_users, n_fft), dtype=np.complex64)
    for user_id in range(n_users):
        out[:, user_id, :] = full_stream_estimates[:, :, user_id, user_id]
    return out


def ber_for_user_grid(symbols: np.ndarray, bits: np.ndarray, modulation: str) -> float:
    pred_bits = hard_demod_stream_grid(symbols, modulation)
    return bit_error_rate(pred_bits, bits)


def detector_ber(
    y_d: np.ndarray,
    a_eff: np.ndarray,
    noise_power: np.ndarray,
    bits: np.ndarray,
    modulation: str,
    *,
    method: str,
    eps: float,
) -> tuple[float | None, np.ndarray | None]:
    estimates = linear_detect(y_d, a_eff, noise_power, method=method, eps=eps)
    if estimates is None:
        return None, None
    target = target_user_streams(estimates)
    return ber_for_user_grid(target, bits, modulation), estimates


def wl_detector_ber(
    y_d: np.ndarray,
    ab_wl: np.ndarray,
    noise_power: np.ndarray,
    bits: np.ndarray,
    modulation: str,
    *,
    method: str,
    eps: float,
) -> tuple[float | None, np.ndarray | None]:
    estimates = wl_detect(y_d, ab_wl, noise_power, method=method, eps=eps)
    if estimates is None:
        return None, None
    target = target_user_streams(estimates)
    return ber_for_user_grid(target, bits, modulation), estimates


def wl_reconstruct_y(ab_wl: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    a_wl, b_wl = split_wl_channel(ab_wl)
    mirror = mirror_subcarrier_indices(a_wl.shape[1])
    x_mirror = estimates[:, mirror]
    return (
        np.matmul(a_wl, estimates[..., None])[..., 0]
        + np.matmul(b_wl, np.conj(x_mirror)[..., None])[..., 0]
    ).astype(np.complex64)


def should_log(epoch: int, epochs: int, log_every: int) -> bool:
    return epoch == 1 or epoch == epochs or (log_every > 0 and epoch % log_every == 0)


def write_history(path: Path, rows: list[dict[str, float]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {path}")


def save_training_plot(
    path: Path,
    rows: list[dict[str, float]],
    title: str,
    *,
    include_ber: bool,
) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, training plot skipped: {exc}")
        return

    epochs = [row["epoch"] for row in rows]
    if include_ber:
        fig, (ax_loss, ax_ber) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(8, 4))
        ax_ber = None
    ax_loss.plot(epochs, [row["train_loss"] for row in rows], label="train loss", linewidth=2.0)
    ax_loss.plot(epochs, [row["val_loss"] for row in rows], label="validation loss", linewidth=2.0)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":")
    ax_loss.legend()
    if ax_ber is not None:
        ax_ber.plot(epochs, [row["val_ber"] for row in rows], label="validation BER", linewidth=2.0)
        ax_ber.set_xlabel("Epoch")
        ax_ber.set_ylabel("BER")
        ax_ber.grid(True, linestyle=":")
        ax_ber.legend()
    else:
        ax_loss.set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[SAVE] {path}")


def validate_lmmse_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"global", "snr-binned"}:
        raise ValueError(f"Unsupported LMMSE mode: {mode}")
    return mode


def fit_lmmse_weight_from_arrays(
    a_ls: np.ndarray,
    a_target: np.ndarray,
    ridge: float,
    *,
    fallback_weight: np.ndarray | None = None,
) -> np.ndarray:
    x = ce_complex_to_ri(a_ls)
    y = ce_complex_to_ri(a_target)
    if x.shape[0] <= x.shape[1]:
        print(
            "[WARN] LMMSE fit has fewer samples than features; "
            "using fallback/identity initialization for numerical stability."
        )
        if fallback_weight is not None:
            return np.asarray(fallback_weight, dtype=np.float32)
        return np.eye(x.shape[1], dtype=np.float32)
    xtx = x.T @ x
    xtx += float(ridge) * np.eye(xtx.shape[0], dtype=np.float32)
    weight_t = np.linalg.solve(xtx, x.T @ y)
    return weight_t.T.astype(np.float32)


def fit_lmmse_weight(train_data: dict[str, np.ndarray], ridge: float) -> np.ndarray:
    return fit_lmmse_weight_from_arrays(train_data["a_ls"], train_data["a_ce_target"], ridge)


def fit_lmmse_estimator(
    train_data: dict[str, np.ndarray],
    ridge: float,
    mode: str,
) -> dict[str, Any]:
    mode = validate_lmmse_mode(mode)
    global_weight = fit_lmmse_weight(train_data, ridge)
    if mode == "global":
        return {
            "mode": "global",
            "weight_ri": global_weight,
        }

    snr_db = np.asarray(train_data["snr_db"], dtype=np.float32).reshape(-1)
    if snr_db.shape[0] != train_data["a_ls"].shape[0]:
        raise ValueError(
            f"Expected one SNR value per frame, got snr_db={snr_db.shape} "
            f"for a_ls={train_data['a_ls'].shape}"
        )

    snr_bins = np.unique(np.round(snr_db, decimals=6)).astype(np.float32)
    weights: list[np.ndarray] = []
    for snr in snr_bins:
        mask = np.isclose(snr_db, float(snr), atol=1e-4)
        weight = fit_lmmse_weight_from_arrays(
            train_data["a_ls"][mask],
            train_data["a_ce_target"][mask],
            ridge,
            fallback_weight=global_weight,
        )
        print(f"[FIT] SNR-binned LMMSE bin={float(snr):g}dB frames={int(np.sum(mask))}")
        weights.append(weight)

    return {
        "mode": "snr-binned",
        "snr_bins_db": snr_bins,
        "weights_ri": np.stack(weights, axis=0).astype(np.float32),
        "global_weight_ri": global_weight,
    }


def save_lmmse_weight(
    path: Path,
    estimator: dict[str, Any] | np.ndarray,
    cfg: dict[str, Any],
    ridge: float,
    ce_target: str,
) -> None:
    estimator_dict = normalize_lmmse_estimator(estimator)
    mode = lmmse_estimator_mode(estimator_dict)
    arrays: dict[str, Any] = {
        "lmmse_mode": mode,
        "n_fft": int(cfg["n_fft"]),
        "n_users": int(cfg["n_users"]),
        "n_rx_per_ue": int(cfg["n_rx_per_ue"]),
        "n_streams": int(cfg.get("n_streams", cfg["n_users"])),
        "ridge": float(ridge),
        "ce_target": str(ce_target),
        "ce_target_resolved": resolve_ce_target_mode(str(ce_target), cfg),
        "channel_representation": "augmented_wl_ab" if is_wl_ce_target(ce_target, cfg) else "linear_a",
        "wl_lmmse_fit_split": "train" if is_wl_ce_target(ce_target, cfg) else "",
    }
    if mode == "global":
        arrays["weight_ri"] = np.asarray(estimator_dict["weight_ri"], dtype=np.float32)
    else:
        arrays["snr_bins_db"] = np.asarray(estimator_dict["snr_bins_db"], dtype=np.float32)
        arrays["weights_ri"] = np.asarray(estimator_dict["weights_ri"], dtype=np.float32)
        arrays["global_weight_ri"] = np.asarray(estimator_dict["global_weight_ri"], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[SAVE] {path}")


def load_lmmse_weight(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        mode = str(data["lmmse_mode"].item()) if "lmmse_mode" in data.files else "global"
        if mode == "snr-binned":
            estimator = {
                "mode": "snr-binned",
                "snr_bins_db": np.asarray(data["snr_bins_db"], dtype=np.float32),
                "weights_ri": np.asarray(data["weights_ri"], dtype=np.float32),
                "global_weight_ri": np.asarray(data["global_weight_ri"], dtype=np.float32),
            }
        else:
            estimator = {
                "mode": "global",
                "weight_ri": np.asarray(data["weight_ri"], dtype=np.float32),
            }
    print(f"[LOAD] LMMSE estimator: {path} (mode={lmmse_estimator_mode(estimator)})")
    return estimator


def lmmse_checkpoint_matches(path: Path, cfg: dict[str, Any], ce_target: str, lmmse_mode: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            stored_target = str(data["ce_target"].item()) if "ce_target" in data.files else "pre-rf"
            stored_resolved = (
                str(data["ce_target_resolved"].item()) if "ce_target_resolved" in data.files else stored_target
            )
            stored_mode = str(data["lmmse_mode"].item()) if "lmmse_mode" in data.files else "global"
    except Exception:
        return False
    return (
        stored_resolved == resolve_ce_target_mode(ce_target, cfg)
        and validate_lmmse_mode(stored_mode) == validate_lmmse_mode(lmmse_mode)
    )


def get_lmmse_weight(
    *,
    dataset_dir: Path,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_path: Path,
    ce_target_override: str | None = None,
    label: str = "LMMSE",
) -> dict[str, Any]:
    lmmse_mode = validate_lmmse_mode(str(args.lmmse_mode))
    ce_target = str(args.ce_target) if ce_target_override is None else str(ce_target_override)
    if lmmse_checkpoint_matches(checkpoint_path, cfg, ce_target, lmmse_mode):
        return load_lmmse_weight(checkpoint_path)
    if checkpoint_path.exists():
        print(
            f"[WARN] LMMSE checkpoint target/mode does not match "
            f"--ce-target={ce_target}, --lmmse-mode={lmmse_mode}; "
            f"refitting {checkpoint_path}"
        )
    train_path = find_one(dataset_dir, "train_snr*.npz")
    train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps), ce_target)
    print(f"[FIT] empirical MU-MIMO {label} channel estimator (mode={lmmse_mode}, split=train)")
    estimator = fit_lmmse_estimator(train_data, float(args.lmmse_ridge), lmmse_mode)
    save_lmmse_weight(checkpoint_path, estimator, cfg, float(args.lmmse_ridge), ce_target)
    return estimator


def normalize_lmmse_estimator(estimator: dict[str, Any] | np.ndarray) -> dict[str, Any]:
    if isinstance(estimator, dict):
        return estimator
    return {
        "mode": "global",
        "weight_ri": np.asarray(estimator, dtype=np.float32),
    }


def lmmse_estimator_mode(estimator: dict[str, Any] | np.ndarray) -> str:
    return validate_lmmse_mode(str(normalize_lmmse_estimator(estimator).get("mode", "global")))


def lmmse_snr_bins(estimator: dict[str, Any] | np.ndarray) -> list[float]:
    estimator_dict = normalize_lmmse_estimator(estimator)
    if lmmse_estimator_mode(estimator_dict) != "snr-binned":
        return []
    return [float(x) for x in np.asarray(estimator_dict["snr_bins_db"], dtype=np.float32).reshape(-1)]


def apply_lmmse_weight(
    a_ls: np.ndarray,
    estimator: dict[str, Any] | np.ndarray,
    snr_db: np.ndarray | None = None,
) -> np.ndarray:
    estimator_dict = normalize_lmmse_estimator(estimator)
    if lmmse_estimator_mode(estimator_dict) == "global":
        x = ce_complex_to_ri(a_ls)
        pred = x @ np.asarray(estimator_dict["weight_ri"], dtype=np.float32).T
        return ce_ri_to_complex_like(pred, a_ls)

    if snr_db is None:
        raise ValueError("snr_db is required when applying an SNR-binned LMMSE estimator")

    a_ls_arr = np.asarray(a_ls, dtype=np.complex64)
    snr_values = np.asarray(snr_db, dtype=np.float32).reshape(-1)
    if snr_values.shape[0] != a_ls_arr.shape[0]:
        raise ValueError(f"Expected {a_ls_arr.shape[0]} SNR values, got {snr_values.shape[0]}")

    bins = np.asarray(estimator_dict["snr_bins_db"], dtype=np.float32).reshape(-1)
    weights = np.asarray(estimator_dict["weights_ri"], dtype=np.float32)
    nearest = np.argmin(np.abs(snr_values[:, None] - bins[None, :]), axis=1)
    out = np.empty_like(a_ls_arr, dtype=np.complex64)
    for bin_idx in np.unique(nearest):
        mask = nearest == int(bin_idx)
        x = ce_complex_to_ri(a_ls_arr[mask])
        pred = x @ weights[int(bin_idx)].T
        out[mask] = ce_ri_to_complex_like(pred, a_ls_arr[mask])
    return out.astype(np.complex64)


def lmmse_init_matrix(lmmse_weight: dict[str, Any] | np.ndarray, input_dim: int) -> np.ndarray:
    def candidate_matrix(value: Any) -> np.ndarray | None:
        arr = np.asarray(value)
        if arr.ndim == 2 and input_dim in arr.shape:
            mat = arr.astype(np.float32, copy=False)
            if mat.shape == (input_dim, input_dim):
                return mat
            if mat.T.shape == (input_dim, input_dim):
                return mat.T
        if arr.ndim == 3 and arr.shape[-2:] == (input_dim, input_dim):
            return np.mean(arr.astype(np.float32, copy=False), axis=0)
        if arr.ndim == 3 and arr.shape[-2:] == (input_dim, input_dim)[::-1]:
            return np.mean(arr.astype(np.float32, copy=False), axis=0).T
        return None

    direct = candidate_matrix(lmmse_weight)
    if direct is not None:
        return direct
    if isinstance(lmmse_weight, dict):
        for key in ("weight", "weights", "W", "w"):
            if key in lmmse_weight:
                mat = candidate_matrix(lmmse_weight[key])
                if mat is not None:
                    return mat
        matrices = [
            candidate
            for value in lmmse_weight.values()
            if (candidate := candidate_matrix(value)) is not None
        ]
        if matrices:
            return np.mean(np.stack(matrices, axis=0), axis=0).astype(np.float32)
    raise ValueError("Could not extract a square LMMSE initialization matrix for the CE linear layer")


def train_ce(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    lmmse_weight: dict[str, Any] | np.ndarray | None,
) -> MuMimoCEModel:
    input_dim = ce_feature_dim(cfg, str(args.ce_target))
    model = build_ce_model(
        str(args.ce_type),
        input_dim,
        hidden_dim=int(args.ce_hidden_dim),
        dropout=float(args.ce_dropout),
    ).to(device)
    ce_type = str(args.ce_type).lower()
    if ce_type != "linear":
        raise ValueError(f"Unsupported CE type after WL cleanup: {ce_type}")
    if lmmse_weight is None:
        raise ValueError("WL CE requires an LMMSE/WL-LMMSE estimator for linear-layer initialization")

    model.init_base(torch.from_numpy(lmmse_init_matrix(lmmse_weight, input_dim)).to(device))
    x_train = ce_complex_to_ri(train_data["a_ls"])
    y_train = ce_complex_to_ri(train_data["a_ce_target"])
    x_val = torch.from_numpy(ce_complex_to_ri(val_data["a_ls"])).to(device)
    y_val = torch.from_numpy(ce_complex_to_ri(val_data["a_ce_target"])).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.ce_lr))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.ce_lr_step)),
        gamma=float(args.ce_lr_gamma),
    )
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.ce_epochs) + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())

        train_loss = loss_sum / max(n_seen, 1)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best": float(is_best),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if should_log(epoch, int(args.ce_epochs), int(args.log_every)):
            print(
                f"[CE {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / "train_history_ce.csv",
        history,
        ["epoch", "train_loss", "val_loss", "best", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "ce_training_curve.png",
        history,
        title="MU-MIMO CE Subnet Training",
        include_ber=False,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": input_dim,
            "modulation": str(cfg["modulation"]),
            "n_fft": int(cfg["n_fft"]),
            "n_users": int(cfg["n_users"]),
            "n_rx_per_ue": int(cfg["n_rx_per_ue"]),
            "n_streams": int(cfg.get("n_streams", cfg["n_users"])),
            "ce_type": str(args.ce_type),
            "ce_init": str(args.ce_init),
            "ce_target": str(args.ce_target),
            "ce_target_resolved": resolve_ce_target_mode(str(args.ce_target), cfg),
            "channel_representation": "augmented_wl_ab" if is_wl_ce_target(str(args.ce_target), cfg) else "linear_a",
            "ce_init_resolved": "wl-lmmse" if is_wl_ce_target(str(args.ce_target), cfg) else str(args.ce_init),
            "wl_lmmse_fit_split": "train" if is_wl_ce_target(str(args.ce_target), cfg) else "",
            "ce_hidden_dim": int(args.ce_hidden_dim),
            "ce_dropout": float(args.ce_dropout),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.6e})")
    return model


def load_ce_model(path: Path, cfg: dict[str, Any], device: torch.device) -> MuMimoCEModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    input_dim = int(checkpoint.get("input_dim", ce_feature_dim(cfg)))
    state_dict = checkpoint["state_dict"]
    ce_type = str(checkpoint.get("ce_type", "")).lower()
    if ce_type != "linear":
        raise ValueError(f"Legacy CE checkpoint is no longer supported: ce_type={ce_type!r}")
    model = build_ce_model(
        ce_type,
        input_dim,
        hidden_dim=int(checkpoint.get("ce_hidden_dim", 0)),
        dropout=float(checkpoint.get("ce_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[LOAD] WL linear CE checkpoint: {path} (type={ce_type})")
    return model


def sd_loss_value(logits: torch.Tensor, target: torch.Tensor, sd_loss: str) -> torch.Tensor:
    if sd_loss == "mse":
        return nn.functional.mse_loss(torch.sigmoid(logits), target)
    if sd_loss == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(f"Unsupported SD loss: {sd_loss}")


def validate_sd_feature_set(feature_set: str) -> str:
    feature_set = str(feature_set).lower()
    if feature_set != "wl-zf-reliability":
        raise ValueError(f"Unsupported SD feature set: {feature_set}")
    return feature_set


def infer_sd_feature_set(feature_dim: int, sd_kind: str, fallback: str) -> str:
    del feature_dim, sd_kind, fallback
    return "wl-zf-reliability"


def normalized_log_feature(values: np.ndarray, floor: float, lo: float, hi: float, scale: float) -> np.ndarray:
    logged = np.log10(np.maximum(np.asarray(values, dtype=np.float32), float(floor)))
    return (np.clip(logged, float(lo), float(hi)) / float(scale)).astype(np.float32)


def frame_feature(values: np.ndarray, n_users: int, n_fft: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.broadcast_to(values[:, None, None], (values.size, n_users, n_fft)).astype(np.float32)


def condition_feature(cond_a: np.ndarray, n_frames: int, n_users: int, n_fft: int) -> np.ndarray:
    cond = np.asarray(cond_a, dtype=np.float32)
    if cond.shape == (n_frames, n_fft, n_users):
        cond = np.transpose(cond, (0, 2, 1))
    elif cond.shape != (n_frames, n_users, n_fft):
        cond = np.ones((n_frames, n_users, n_fft), dtype=np.float32)
    return normalized_log_feature(np.maximum(cond, 1.0), 1.0, 0.0, 4.0, 4.0)


def make_sd_features(
    *,
    cfg: dict[str, Any],
    y_d: np.ndarray,
    a_hat: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    cond_a: np.ndarray,
    feature_set: str,
    sd_kind: str,
    ce_target: str,
    eps: float,
) -> np.ndarray:
    feature_set = validate_sd_feature_set(feature_set)
    del sd_kind
    y_d = np.asarray(y_d, dtype=np.complex64)
    a_hat = np.asarray(a_hat, dtype=np.complex64)
    n_frames, n_fft, n_users, n_rx = y_d.shape

    ab_wl = as_wl_channel(a_hat, cfg, ce_target)
    a_wl, b_wl = split_wl_channel(ab_wl)
    n_streams = a_wl.shape[-1]
    if n_users > n_streams:
        raise RuntimeError("WL-ZF SD features assume one target stream per user")
    wl_zf_estimates = wl_detect(y_d, ab_wl, noise_power, method="zf", eps=eps)
    if wl_zf_estimates is None:
        raise RuntimeError("WL-ZF SD features require n_streams <= n_rx_per_ue")
    wl_zf_target = target_user_streams(wl_zf_estimates)
    reconstructed = wl_reconstruct_y(ab_wl, wl_zf_estimates)
    residual = y_d - reconstructed
    h_target = np.stack([a_wl[:, :, user_id, :, user_id] for user_id in range(n_users)], axis=2)
    b_target = np.stack([b_wl[:, :, user_id, :, user_id] for user_id in range(n_users)], axis=2)
    channel_power = np.sum(np.abs(h_target) ** 2 + np.abs(b_target) ** 2, axis=-1)
    residual_matched = np.sum(np.conj(h_target) * residual, axis=-1)
    residual_matched = residual_matched / np.maximum(channel_power, float(eps))
    residual_matched = np.transpose(residual_matched, (0, 2, 1))
    residual_power = np.transpose(np.mean(np.abs(residual) ** 2, axis=-1), (0, 2, 1))
    log_res_power = normalized_log_feature(residual_power, 1e-12, -12.0, 2.0, 12.0)
    log_gain_power = normalized_log_feature(np.transpose(channel_power, (0, 2, 1)), 1e-12, -12.0, 2.0, 12.0)
    snr_norm = (np.asarray(snr_db, dtype=np.float32) / 40.0).astype(np.float32)
    noise_log = normalized_log_feature(noise_power, 1e-12, -12.0, 2.0, 12.0)
    return np.stack(
        [
            wl_zf_target.real,
            wl_zf_target.imag,
            residual_matched.real,
            residual_matched.imag,
            log_res_power,
            log_gain_power,
            condition_feature(cond_a, n_frames, n_users, n_fft),
            frame_feature(noise_log, n_users, n_fft),
            frame_feature(snr_norm, n_users, n_fft),
        ],
        axis=-1,
    ).astype(np.float32)


def make_bilstm_sd_arrays(
    *,
    cfg: dict[str, Any],
    y_d: np.ndarray,
    a_hat: np.ndarray,
    bits: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    cond_a: np.ndarray,
    group_size: int,
    feature_set: str,
    ce_target: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_streams, n_fft, bps = bits.shape
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")
    features = make_sd_features(
        cfg=cfg,
        y_d=y_d,
        a_hat=a_hat,
        noise_power=noise_power,
        snr_db=snr_db,
        cond_a=cond_a,
        feature_set=feature_set,
        sd_kind="bilstm",
        ce_target=ce_target,
        eps=eps,
    )
    if features.shape[:3] != (n_frames, n_streams, n_fft):
        raise ValueError(f"Expected SD feature shape {(n_frames, n_streams, n_fft)}, got {features.shape[:3]}")
    n_groups = n_fft // group_size
    x = features.reshape(n_frames * n_streams, n_fft, features.shape[-1])
    y = bits.reshape(n_frames, n_streams, n_groups, group_size, bps).reshape(
        n_frames * n_streams,
        n_groups,
        group_size * bps,
    )
    return x.astype(np.float32), y.astype(np.float32)


def train_bilstm_sd(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    ce_model: MuMimoCEModel,
    lmmse_weight: dict[str, Any] | np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> MuMimoBiLSTMSDNet:
    group_size = int(args.group_size)
    bps = bits_per_symbol(str(cfg["modulation"]))
    feature_set = validate_sd_feature_set(str(args.sd_feature_set))
    a_train = predict_ce(
        ce_model,
        train_data["a_ls"],
        lmmse_weight=lmmse_weight,
        snr_db=train_data["snr_db"],
        device=device,
        batch_size=int(args.batch_size),
    )
    a_val = predict_ce(
        ce_model,
        val_data["a_ls"],
        lmmse_weight=lmmse_weight,
        snr_db=val_data["snr_db"],
        device=device,
        batch_size=int(args.batch_size),
    )
    x_train, y_train = make_bilstm_sd_arrays(
        cfg=cfg,
        y_d=train_data["y_d"],
        a_hat=a_train,
        bits=train_data["bits"],
        noise_power=train_data["noise_power"],
        snr_db=train_data["snr_db"],
        cond_a=train_data["cond_A"],
        group_size=group_size,
        feature_set=feature_set,
        ce_target=str(args.ce_target),
        eps=float(args.eps),
    )
    x_val_np, y_val_np = make_bilstm_sd_arrays(
        cfg=cfg,
        y_d=val_data["y_d"],
        a_hat=a_val,
        bits=val_data["bits"],
        noise_power=val_data["noise_power"],
        snr_db=val_data["snr_db"],
        cond_a=val_data["cond_A"],
        group_size=group_size,
        feature_set=feature_set,
        ce_target=str(args.ce_target),
        eps=float(args.eps),
    )

    feature_dim = x_train.shape[-1]
    hidden_dims = tuple(int(x) for x in args.bilstm_hidden_dims)
    model = MuMimoBiLSTMSDNet(
        int(cfg["n_fft"]),
        bps,
        group_size,
        hidden_dims=hidden_dims,
        feature_dim=feature_dim,
        sd_feature_set=feature_set,
    ).to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 29)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )
    x_val = torch.from_numpy(x_val_np).to(device)
    y_val = torch.from_numpy(y_val_np).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.sd_lr))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.bilstm_lr_step)),
        gamma=float(args.sd_lr_gamma),
    )

    history: list[dict[str, float]] = []
    best_val_ber = math.inf
    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.bilstm_epochs) + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = sd_loss_value(logits, yb, str(args.sd_loss))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(sd_loss_value(val_logits, y_val, str(args.sd_loss)).item())
            val_bits = (torch.sigmoid(val_logits) > 0.5).float()
            val_ber = float(torch.mean((val_bits != y_val).float()).item())
        train_loss = loss_sum / max(n_seen, 1)
        is_best = val_ber < best_val_ber
        if is_best:
            best_val_ber = val_ber
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_ber": float(val_ber),
            "best": float(is_best),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if should_log(epoch, int(args.bilstm_epochs), int(args.log_every)):
            print(
                f"[BiLSTM-SD {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, val_BER={val_ber:.4e}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / "train_history_bilstm_sd.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_ber", "best", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "bilstm_sd_training_curve.png",
        history,
        title="MU-MIMO BiLSTM-SD Training",
        include_ber=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "n_fft": int(cfg["n_fft"]),
            "group_size": group_size,
            "bits_per_symbol": bps,
            "hidden_dims": hidden_dims,
            "feature_dim": int(model.feature_dim),
            "sd_feature_set": str(model.sd_feature_set),
            "modulation": str(cfg["modulation"]),
            "sd_loss": str(args.sd_loss),
            "sd_reference": "wl-zf",
            "proposed_detector": "wl-zf-bilstm",
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_ber": float(best_val_ber),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path} (best_epoch={best_epoch}, best_val_BER={best_val_ber:.4e})")
    return model


def infer_bilstm_hidden_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    h1 = int(state_dict["lstm1.weight_ih_l0"].shape[0] // 4)
    h2 = int(state_dict["lstm2.weight_ih_l0"].shape[0] // 4)
    h3 = int(state_dict["lstm3.weight_ih_l0"].shape[0] // 4)
    return h1, h2, h3


def infer_bilstm_feature_dim(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["lstm1.weight_ih_l0"].shape[1])


def load_bilstm_sd_model(
    path: Path,
    cfg: dict[str, Any],
    group_size: int,
    device: torch.device,
    args_feature_set: str,
) -> MuMimoBiLSTMSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_group_size = int(checkpoint.get("group_size", group_size))
    n_fft = int(checkpoint.get("n_fft", cfg["n_fft"]))
    bps = int(checkpoint.get("bits_per_symbol", bits_per_symbol(str(cfg["modulation"]))))
    hidden_dims = tuple(int(x) for x in checkpoint.get("hidden_dims", infer_bilstm_hidden_dims(state_dict)))
    feature_dim = int(checkpoint.get("feature_dim", infer_bilstm_feature_dim(state_dict)))
    feature_set = str(checkpoint.get("sd_feature_set", infer_sd_feature_set(feature_dim, "bilstm", args_feature_set)))
    model = MuMimoBiLSTMSDNet(
        n_fft,
        bps,
        model_group_size,
        hidden_dims=hidden_dims,
        feature_dim=feature_dim,
        sd_feature_set=feature_set,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"[LOAD] BiLSTM-SD checkpoint: {path} "
        f"(feature_set={feature_set}, feature_dim={feature_dim}, hidden_dims={hidden_dims})"
    )
    return model


def predict_bilstm_sd_bits(
    model: MuMimoBiLSTMSDNet,
    *,
    cfg: dict[str, Any],
    y_d: np.ndarray,
    a_hat: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    cond_a: np.ndarray,
    true_bits_shape: tuple[int, int, int, int],
    group_size: int,
    ce_target: str,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dummy_bits = np.zeros(true_bits_shape, dtype=np.int8)
    x_np, _ = make_bilstm_sd_arrays(
        cfg=cfg,
        y_d=y_d,
        a_hat=a_hat,
        bits=dummy_bits,
        noise_power=noise_power,
        snr_db=snr_db,
        cond_a=cond_a,
        group_size=group_size,
        feature_set=str(model.sd_feature_set),
        ce_target=ce_target,
        eps=eps,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=int(batch_size), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            chunks.append((prob > 0.5).astype(np.int8))
    pred_groups = np.concatenate(chunks, axis=0)
    n_frames, n_streams, n_fft, bps = true_bits_shape
    n_groups = n_fft // group_size
    return pred_groups.reshape(n_frames, n_streams, n_groups, group_size, bps).reshape(
        n_frames,
        n_streams,
        n_fft,
        bps,
    )


def predict_ce(
    model: MuMimoCEModel,
    a_ls: np.ndarray,
    *,
    lmmse_weight: dict[str, Any] | np.ndarray | None,
    snr_db: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    del lmmse_weight, snr_db
    x_np = ce_complex_to_ri(a_ls)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=int(batch_size), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            chunks.append(model(xb.to(device)).cpu().numpy())
    pred_np = np.concatenate(chunks, axis=0)
    a_shape = np.asarray(a_ls).shape
    if len(a_shape) != 5:
        raise ValueError(f"Expected CE channel shape (frames, fft, users, rx, streams), got {a_shape}")
    n_frames, n_fft, n_users, n_rx, n_streams = a_shape
    expected_rows = n_frames * n_users
    expected_half = n_fft * n_rx * n_streams
    expected_dim = 2 * expected_half
    if pred_np.shape != (expected_rows, expected_dim):
        raise ValueError(f"CE output shape {pred_np.shape} does not match channel shape {a_shape}")

    real = pred_np[:, :expected_half].reshape(n_frames, n_users, n_fft, n_rx, n_streams)
    imag = pred_np[:, expected_half:].reshape(n_frames, n_users, n_fft, n_rx, n_streams)
    pred_complex = real + 1j * imag
    return np.transpose(pred_complex, (0, 2, 1, 3, 4)).astype(np.complex64)


def format_ber(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4e}"


def split_diagnostics(data: dict[str, np.ndarray]) -> dict[str, float]:
    desired_power = np.asarray(data.get("desired_power", []), dtype=np.float32)
    inter_stream_power = np.asarray(data.get("inter_stream_power", []), dtype=np.float32)
    effective_sinr_db = np.asarray(data.get("effective_sinr_db", []), dtype=np.float32)
    cond = np.asarray(data["cond_A"], dtype=np.float32)
    noise_power = np.asarray(data["noise_power"], dtype=np.float32)

    desired_mean = float(np.mean(desired_power)) if desired_power.size else 0.0
    inter_mean = float(np.mean(inter_stream_power)) if inter_stream_power.size else 0.0
    interference_to_desired_ratio = inter_mean / max(desired_mean, 1e-300)
    if effective_sinr_db.size:
        effective_sinr_db_mean = float(np.mean(effective_sinr_db))
        effective_sinr_db_p10 = float(np.percentile(effective_sinr_db, 10.0))
    else:
        effective_sinr_db_mean = float("nan")
        effective_sinr_db_p10 = float("nan")

    return {
        "desired_power_mean": desired_mean,
        "inter_stream_power_mean": inter_mean,
        "interference_to_desired_ratio": float(interference_to_desired_ratio),
        "interference_to_desired_ratio_db": to_db(interference_to_desired_ratio),
        "effective_sinr_db_mean": effective_sinr_db_mean,
        "effective_sinr_db_p10": effective_sinr_db_p10,
        "cond_A_mean": float(np.mean(cond)),
        "cond_A_p95": float(np.percentile(cond, 95.0)),
        "noise_power_mean": float(np.mean(noise_power)),
    }


def evaluate_one_wl(
    *,
    path: Path,
    cfg: dict[str, Any],
    data: dict[str, np.ndarray],
    a_lmmse: np.ndarray,
    a_plain_lmmse: np.ndarray,
    a_comnet: np.ndarray,
    bilstm_model: MuMimoBiLSTMSDNet | None,
    args: argparse.Namespace,
    device: torch.device,
    snr: float,
    modulation: str,
    bits: np.ndarray,
    lmmse_weight: dict[str, Any] | np.ndarray,
) -> dict[str, Any]:
    ls_mmse_ber, _ = detector_ber(
        data["y_d"],
        data["a_plain_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    plain_lmmse_mmse_ber, _ = detector_ber(
        data["y_d"],
        a_plain_lmmse,
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    wl_ls_zf_ber, _ = wl_detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    wl_ls_mmse_ber, _ = wl_detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    wl_lmmse_mmse_ber, _ = wl_detector_ber(
        data["y_d"],
        a_lmmse,
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    true_wl_mmse_ber, _ = wl_detector_ber(
        data["y_d"],
        data["a_wl_true"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )

    ber: dict[str, float | None] = {
        "LS-MMSE": ls_mmse_ber,
        "LMMSE-MMSE": plain_lmmse_mmse_ber,
        "WL-LS -> WL-ZF": wl_ls_zf_ber,
        "WL-LS -> WL-MMSE": wl_ls_mmse_ber,
        "WL-LMMSE -> WL-MMSE": wl_lmmse_mmse_ber,
        "True WL-H -> WL-MMSE": true_wl_mmse_ber,
    }

    if bilstm_model is not None:
        pred_bilstm = predict_bilstm_sd_bits(
            bilstm_model,
            cfg=cfg,
            y_d=data["y_d"],
            a_hat=a_comnet,
            noise_power=data["noise_power"],
            snr_db=data["snr_db"],
            cond_a=data["cond_A"],
            true_bits_shape=bits.shape,
            group_size=int(bilstm_model.group_size),
            ce_target=str(args.ce_target),
            eps=float(args.eps),
            device=device,
            batch_size=int(args.batch_size),
        )
        ber["WL-CE -> WL-ZF-BiLSTM"] = bit_error_rate(pred_bilstm, bits)

    total_bit_count = int(bits.size)
    bit_errors = {
        key: None if value is None else int(round(float(value) * total_bit_count))
        for key, value in ber.items()
    }
    total_bits = {
        key: None if value is None else total_bit_count
        for key, value in ber.items()
    }
    a_mse = {
        "WL-LS": channel_mse(data["a_ls"], data["a_ce_target"]),
        "WL-LMMSE": channel_mse(a_lmmse, data["a_ce_target"]),
        "WL-CE": channel_mse(a_comnet, data["a_ce_target"]),
    }
    a_nmse = {
        "WL-LS": channel_nmse(data["a_ls"], data["a_ce_target"]),
        "WL-LMMSE": channel_nmse(a_lmmse, data["a_ce_target"]),
        "WL-CE": channel_nmse(a_comnet, data["a_ce_target"]),
    }
    diagnostics = split_diagnostics(data)
    sd_parts = []
    if "WL-CE -> WL-ZF-BiLSTM" in ber:
        sd_parts.append(f"CE-BiLSTM={format_ber(ber['WL-CE -> WL-ZF-BiLSTM'])}")
    sd_piece = f"{', '.join(sd_parts)}, " if sd_parts else ""
    print(
        f"[EVAL] {path.name} SNR={snr:g} "
        f"LS-MMSE={format_ber(ber['LS-MMSE'])}, "
        f"LMMSE-MMSE={format_ber(ber['LMMSE-MMSE'])}, "
        f"WL-LS->WL-ZF={format_ber(ber['WL-LS -> WL-ZF'])}, "
        f"WL-LS->WL-MMSE={format_ber(ber['WL-LS -> WL-MMSE'])}, "
        f"WL-LMMSE->WL-MMSE={format_ber(ber['WL-LMMSE -> WL-MMSE'])}, "
        f"{sd_piece}"
        f"True-WL-H={format_ber(ber['True WL-H -> WL-MMSE'])}, "
        f"WL_NMSE_LS={to_db(a_nmse['WL-LS']):.2f}dB, "
        f"WL_NMSE_LMMSE={to_db(a_nmse['WL-LMMSE']):.2f}dB, "
        f"WL_NMSE_CE={to_db(a_nmse['WL-CE']):.2f}dB"
    )
    return {
        "snr": snr,
        "ce_target": str(args.ce_target),
        "ce_target_resolved": resolve_ce_target_mode(str(args.ce_target), cfg),
        "lmmse_mode": lmmse_estimator_mode(lmmse_weight),
        "lmmse_snr_bins_db": lmmse_snr_bins(lmmse_weight),
        "channel_representation": "augmented_wl_ab",
        "ce_init_resolved": "wl-lmmse",
        "wl_lmmse_fit_split": "train",
        "sd_reference": "wl-zf",
        "proposed_detector": "wl-zf-bilstm",
        "proposed_sd_models": ["bilstm"] if bilstm_model is not None else [],
        "baseline_detector": "wl-mmse",
        "a_mse": a_mse,
        "a_mse_db": {key: to_db(value) for key, value in a_mse.items()},
        "a_nmse": a_nmse,
        "a_nmse_db": {key: to_db(value) for key, value in a_nmse.items()},
        "ber": ber,
        "bit_errors": bit_errors,
        "total_bits": total_bits,
        "diagnostic": diagnostics,
        "condition": {
            "mean_cond_A": diagnostics["cond_A_mean"],
            "p95_cond_A": diagnostics["cond_A_p95"],
        },
    }


def evaluate_one(
    *,
    path: Path,
    cfg: dict[str, Any],
    ce_model: MuMimoCEModel,
    bilstm_model: MuMimoBiLSTMSDNet | None,
    lmmse_weight: dict[str, Any] | np.ndarray,
    plain_lmmse_weight: dict[str, Any] | np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    data = preprocess_split(load_npz(path), cfg, float(args.eps), str(args.ce_target))
    snr = float(np.mean(data["snr_db"]))
    modulation = str(cfg["modulation"])
    bits = data["bits"]
    a_lmmse = apply_lmmse_weight(data["a_ls"], lmmse_weight, data["snr_db"])
    a_plain_lmmse = (
        apply_lmmse_weight(data["a_plain_ls"], plain_lmmse_weight, data["snr_db"])
        if plain_lmmse_weight is not None
        else a_lmmse
    )
    a_comnet = predict_ce(
        ce_model,
        data["a_ls"],
        lmmse_weight=lmmse_weight,
        snr_db=data["snr_db"],
        device=device,
        batch_size=int(args.batch_size),
    )

    if not is_wl_ce_target(str(args.ce_target), cfg):
        raise ValueError("This receiver has been trimmed to the WL-RF ComNet path; use --ce-target wl-rf/auto with RF impairment.")
    return evaluate_one_wl(
        path=path,
        cfg=cfg,
        data=data,
        a_lmmse=a_lmmse,
        a_plain_lmmse=a_plain_lmmse,
        a_comnet=a_comnet,
        bilstm_model=bilstm_model,
        args=args,
        device=device,
        snr=snr,
        modulation=modulation,
        bits=bits,
        lmmse_weight=lmmse_weight,
    )


def save_eval_summary(result_dir: Path, cfg: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": cfg,
        "receiver": {
            "ce_target": results[0].get("ce_target") if results else None,
            "ce_target_resolved": results[0].get("ce_target_resolved") if results else None,
            "lmmse_mode": results[0].get("lmmse_mode") if results else None,
            "lmmse_snr_bins_db": results[0].get("lmmse_snr_bins_db") if results else [],
            "channel_mse_reference": "a_ce_target",
            "channel_representation": results[0].get("channel_representation") if results else None,
            "ce_init_resolved": results[0].get("ce_init_resolved") if results else None,
            "wl_lmmse_fit_split": results[0].get("wl_lmmse_fit_split") if results else None,
            "sd_reference": results[0].get("sd_reference") if results else None,
            "proposed_detector": results[0].get("proposed_detector") if results else None,
            "proposed_sd_models": results[0].get("proposed_sd_models") if results else [],
            "baseline_detector": results[0].get("baseline_detector") if results else None,
        },
        "a_mse_db": {},
        "a_nmse_db": {},
        "ber": {},
        "bit_errors": {},
        "total_bits": {},
        "diagnostic": {},
        "condition": {},
    }
    for item in sorted(results, key=lambda x: x["snr"]):
        snr_key = f"{item['snr']:g}"
        summary["a_mse_db"][snr_key] = item["a_mse_db"]
        summary["a_nmse_db"][snr_key] = item["a_nmse_db"]
        summary["ber"][snr_key] = item["ber"]
        summary["bit_errors"][snr_key] = item["bit_errors"]
        summary["total_bits"][snr_key] = item["total_bits"]
        summary["diagnostic"][snr_key] = item["diagnostic"]
        summary["condition"][snr_key] = item["condition"]

    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "eval_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[SAVE] {path}")
    return summary


def save_metric_csv(path: Path, summary: dict[str, Any], section: str) -> None:
    snrs = sorted(float(x) for x in summary[section].keys())
    metric_names = sorted({name for snr in summary[section].values() for name in snr.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", *metric_names])
        for snr in snrs:
            key = f"{snr:g}"
            writer.writerow([snr, *[summary[section][key].get(name, "") for name in metric_names]])
    print(f"[SAVE] {path}")


def ordered_metric_names(names: Iterable[str], preferred: list[str]) -> list[str]:
    remaining = sorted(set(names) - set(preferred))
    return [name for name in preferred if name in set(names)] + remaining


def save_eval_plots(result_dir: Path, summary: dict[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    save_metric_csv(result_dir / "ber_vs_snr.csv", summary, "ber")
    save_metric_csv(result_dir / "channel_mse_vs_snr.csv", summary, "a_mse_db")
    save_metric_csv(result_dir / "channel_nmse_vs_snr.csv", summary, "a_nmse_db")
    save_metric_csv(result_dir / "diagnostic_vs_snr.csv", summary, "diagnostic")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, PNG plots skipped; CSV files were saved: {exc}")
        return

    snrs = sorted(float(x) for x in summary["ber"].keys())
    all_ber_names = {key for item in summary["ber"].values() for key in item.keys()}
    ber_names = ordered_metric_names(
        all_ber_names,
        [
            "LS-MMSE",
            "LMMSE-MMSE",
            "WL-LS -> WL-ZF",
            "WL-LS -> WL-MMSE",
            "WL-LMMSE -> WL-MMSE",
            "WL-CE -> WL-ZF-BiLSTM",
            "True WL-H -> WL-MMSE",
        ],
    )
    plt.figure(figsize=(8, 5))
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
    linestyle_cycle = ["-", "--", "-.", ":"]
    for idx, name in enumerate(ber_names):
        values = []
        for snr in snrs:
            value = summary["ber"][f"{snr:g}"].get(name)
            values.append(np.nan if value is None else max(float(value), 1e-7))
        plt.semilogy(
            snrs,
            values,
            marker=marker_cycle[idx % len(marker_cycle)],
            linestyle=linestyle_cycle[(idx // len(marker_cycle)) % len(linestyle_cycle)],
            linewidth=2.0,
            markersize=6.0,
            label=name,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Raw MU-MIMO ComNet BER vs SNR")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    ber_path = result_dir / "ber_vs_snr.png"
    plt.savefig(ber_path, dpi=160)
    plt.close()
    print(f"[SAVE] {ber_path}")

    all_mse_names = {key for item in summary["a_mse_db"].values() for key in item.keys()}
    mse_names = ordered_metric_names(all_mse_names, ["WL-LS", "WL-LMMSE", "WL-CE"])
    plt.figure(figsize=(8, 5))
    for name in mse_names:
        values = [summary["a_mse_db"][f"{snr:g}"][name] for snr in snrs]
        plt.plot(snrs, values, marker="s", linewidth=2.0, label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("A_eff MSE (dB)")
    plt.title("Raw MU-MIMO Effective Channel MSE")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    mse_path = result_dir / "a_mse_vs_snr.png"
    plt.savefig(mse_path, dpi=160)
    plt.close()
    print(f"[SAVE] {mse_path}")


def evaluate_all(
    *,
    dataset_dir: Path,
    cfg: dict[str, Any],
    ce_model: MuMimoCEModel,
    bilstm_model: MuMimoBiLSTMSDNet | None,
    lmmse_weight: dict[str, Any] | np.ndarray,
    plain_lmmse_weight: dict[str, Any] | np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    test_paths = sorted(dataset_dir.glob("test_snr*.npz"))
    if not test_paths:
        raise FileNotFoundError(f"No test_snr*.npz files found in {dataset_dir}")
    results = [
        evaluate_one(
            path=path,
            cfg=cfg,
            ce_model=ce_model,
            bilstm_model=bilstm_model,
            lmmse_weight=lmmse_weight,
            plain_lmmse_weight=plain_lmmse_weight,
            args=args,
            device=device,
        )
        for path in test_paths
    ]
    summary = save_eval_summary(Path(args.result_dir), cfg, results)
    save_eval_plots(Path(args.result_dir), summary)
    return summary


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ce_checkpoint = Path(args.ce_checkpoint) if args.ce_checkpoint else result_dir / "mumimo_ce_linear.pt"
    legacy_ce_checkpoint = result_dir / "mumimo_ce_refinenet.pt"
    bilstm_checkpoint = Path(args.bilstm_checkpoint) if args.bilstm_checkpoint else result_dir / "mumimo_wl_zf_bilstm.pt"
    legacy_bilstm_checkpoint = result_dir / "mumimo_wl_zf_refinenet_bilstm.pt"
    lmmse_checkpoint = (
        Path(args.lmmse_checkpoint) if args.lmmse_checkpoint else result_dir / "mumimo_lmmse_estimator.npz"
    )
    plain_lmmse_checkpoint = result_dir / "mumimo_plain_lmmse_estimator.npz"
    cfg = load_config(dataset_dir)
    if str(cfg.get("waveform_type")) != "raw_mumimo_e2e":
        raise ValueError(
            f"Expected raw_mumimo_e2e dataset, got waveform_type={cfg.get('waveform_type')!r}"
        )
    if not is_wl_ce_target(str(args.ce_target), cfg):
        raise ValueError(
            "This receiver has been trimmed to the WL-RF ComNet path; use an RF-impaired dataset "
            "with --ce-target auto or pass --ce-target wl-rf explicitly."
        )
    device = resolve_device(str(args.device))
    n_users = int(cfg["n_users"])
    n_streams = int(cfg.get("n_streams", n_users))
    n_rx_per_ue = int(cfg["n_rx_per_ue"])
    print(f"[DEVICE] {device}")
    print(
        f"[CONFIG] modulation={cfg['modulation']}, n_fft={cfg['n_fft']}, "
        f"n_users={n_users}, n_streams={n_streams}, n_rx_per_ue={n_rx_per_ue}, "
        f"group_size={args.group_size}, ce_type={args.ce_type}, sd_type={args.sd_type}, "
        f"sd_feature_set={args.sd_feature_set}, ce_target_resolved={resolve_ce_target_mode(str(args.ce_target), cfg)}, "
        f"lmmse_mode={args.lmmse_mode}"
    )
    if n_streams > n_rx_per_ue:
        print("[WARN] n_streams > n_rx_per_ue, ZF baselines and SD ZF features will be disabled.")

    lmmse_weight = get_lmmse_weight(
        dataset_dir=dataset_dir,
        cfg=cfg,
        args=args,
        checkpoint_path=lmmse_checkpoint,
        label="WL-LMMSE" if is_wl_ce_target(str(args.ce_target), cfg) else "LMMSE",
    )
    plain_lmmse_weight = (
        get_lmmse_weight(
            dataset_dir=dataset_dir,
            cfg=cfg,
            args=args,
            checkpoint_path=plain_lmmse_checkpoint,
            ce_target_override="rf-linear",
            label="plain LMMSE",
        )
        if is_wl_ce_target(str(args.ce_target), cfg)
        else lmmse_weight
    )

    ce_model: MuMimoCEModel | None = None
    bilstm_model: MuMimoBiLSTMSDNet | None = None

    if args.mode in {"train-all", "train-ce"}:
        train_path = find_one(dataset_dir, "train_snr*.npz")
        val_path = find_one(dataset_dir, "val_snr*.npz")
        train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps), str(args.ce_target))
        val_data = preprocess_split(load_npz(val_path), cfg, float(args.eps), str(args.ce_target))
        ce_model = train_ce(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            args=args,
            device=device,
            checkpoint_path=ce_checkpoint,
            lmmse_weight=lmmse_weight,
        )

    if args.mode in {"train-sd", "eval"}:
        if args.ce_checkpoint is None and not ce_checkpoint.exists() and legacy_ce_checkpoint.exists():
            ce_checkpoint = legacy_ce_checkpoint
        if not ce_checkpoint.exists():
            raise FileNotFoundError(f"CE checkpoint not found: {ce_checkpoint}")
        ce_model = load_ce_model(ce_checkpoint, cfg, device)

    if args.mode in {"train-all", "train-sd"}:
        if ce_model is None:
            raise RuntimeError("CE model is required before SD training")
        train_path = find_one(dataset_dir, "train_snr*.npz")
        val_path = find_one(dataset_dir, "val_snr*.npz")
        train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps), str(args.ce_target))
        val_data = preprocess_split(load_npz(val_path), cfg, float(args.eps), str(args.ce_target))
        bilstm_model = train_bilstm_sd(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            ce_model=ce_model,
            lmmse_weight=lmmse_weight,
            args=args,
            device=device,
            checkpoint_path=bilstm_checkpoint,
        )

    if args.mode == "eval":
        if args.bilstm_checkpoint is None and not bilstm_checkpoint.exists() and legacy_bilstm_checkpoint.exists():
            bilstm_checkpoint = legacy_bilstm_checkpoint
        if bilstm_checkpoint.exists():
            bilstm_model = load_bilstm_sd_model(
                bilstm_checkpoint,
                cfg,
                int(args.group_size),
                device,
                str(args.sd_feature_set),
            )
        else:
            print(
                f"[WARN] WL-ZF BiLSTM checkpoint not found, BiLSTM-SD result will be skipped: "
                f"{bilstm_checkpoint}"
            )

    if ce_model is None:
        raise RuntimeError("CE model is required for evaluation")

    evaluate_all(
        dataset_dir=dataset_dir,
        cfg=cfg,
        ce_model=ce_model,
        bilstm_model=bilstm_model,
        lmmse_weight=lmmse_weight,
        plain_lmmse_weight=plain_lmmse_weight,
        args=args,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
