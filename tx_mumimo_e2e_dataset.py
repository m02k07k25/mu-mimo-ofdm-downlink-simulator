from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mumimo_phy import (
    ArrayConfig,
    ScmChannelConfig,
    ScmChannelGenerator,
    add_awgn,
    apply_multipath_mimo,
    apply_rf_impairments,
    bits_per_symbol,
    channel_frequency_response,
    hybrid_steering_beams,
    hybrid_zf_precoder_context,
    linear_to_db,
    noise_power_from_snr,
    ofdm_modulate_freq,
    precoded_tx_frequency,
    qam_modulate,
    rf_impairment_widely_linear_coefficients,
)


@dataclass
class MuMimoE2EConfig:
    n_fft: int = 64
    n_cp: int = 16
    modulation: str = "64QAM"
    n_users: int = 2
    n_tx: int = 8
    n_rx_per_ue: int = 4
    n_taps: int = 7
    n_rays_per_path: int = 15
    pdp_decay: float = 5.0
    carrier_freq_hz: float = 800e6
    antenna_spacing_lambda: float = 0.5
    scm_angle_spread_deg: float = 3.0
    snr_train_db: float = 40.0
    snr_train_db_list: tuple[float, ...] = (15, 20, 25, 30, 35, 40)
    snr_test_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40)
    n_train_frames: int = 50000
    n_val_frames: int = 10000
    n_test_frames_per_snr: int = 10000
    csit_error_var: float = 0.005
    precoder_norm: str = "column"
    case: str = "clipping"
    clip_ratio: float = 1.6
    pilot_kind: str = "qpsk"
    rx_iq_gain_imbalance_db: float = 0.5
    rx_iq_phase_error_deg: float = 3.0
    rx_common_phase_error_deg: float = 5.0
    seed: int = 7

    @property
    def n_streams(self) -> int:
        return int(self.n_users)

    def validate(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.n_cp < 0:
            raise ValueError("n_cp must be non-negative")
        if self.n_taps <= 0:
            raise ValueError("n_taps must be positive")
        if self.n_rays_per_path <= 0:
            raise ValueError("n_rays_per_path must be positive")
        if self.n_taps > self.n_cp and self.case in {"linear", "clipping"}:
            raise ValueError(f"{self.case} MU-MIMO OFDM requires n_taps <= n_cp")
        if self.pdp_decay <= 0:
            raise ValueError("pdp_decay must be positive")
        if self.carrier_freq_hz <= 0:
            raise ValueError("carrier_freq_hz must be positive")
        if self.antenna_spacing_lambda <= 0:
            raise ValueError("antenna_spacing_lambda must be positive")
        if self.scm_angle_spread_deg < 0:
            raise ValueError("scm_angle_spread_deg must be non-negative")
        if self.modulation.upper() not in {"16QAM", "64QAM"}:
            raise ValueError("modulation must be one of 16QAM, 64QAM")
        if not (1 <= self.n_users <= self.n_tx):
            raise ValueError("n_users must satisfy 1 <= n_users <= n_tx")
        if self.n_rx_per_ue <= 0:
            raise ValueError("n_rx_per_ue must be positive")
        if self.csit_error_var < 0.0:
            raise ValueError("csit_error_var must be non-negative")
        if self.precoder_norm not in {"none", "column", "fro"}:
            raise ValueError("precoder_norm must be one of none, column, fro")
        if self.case not in {"linear", "cp_removal", "clipping"}:
            raise ValueError("case must be one of linear, cp_removal, clipping")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive")
        if self.pilot_kind not in {"ones", "qpsk"}:
            raise ValueError("pilot_kind must be one of ones, qpsk")
        if not math.isfinite(self.rx_iq_gain_imbalance_db):
            raise ValueError("rx_iq_gain_imbalance_db must be finite")
        if not math.isfinite(self.rx_iq_phase_error_deg):
            raise ValueError("rx_iq_phase_error_deg must be finite")
        if not math.isfinite(self.rx_common_phase_error_deg):
            raise ValueError("rx_common_phase_error_deg must be finite")
        if len(self.snr_train_db_list) == 0:
            raise ValueError("snr_train_db_list must not be empty")
        if min(self.n_train_frames, self.n_val_frames, self.n_test_frames_per_snr) < 0:
            raise ValueError("frame counts must be non-negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw end-to-end downlink MU-MIMO OFDM datasets."
    )
    parser.add_argument("--out-dir", type=str, default="outputs_mumimo_e2e_64qam")
    parser.add_argument("--modulation", type=str, default="64QAM", choices=["16QAM", "64QAM"])
    parser.add_argument("--n-users", type=int, default=2)
    parser.add_argument("--n-tx", type=int, default=8)
    parser.add_argument("--n-rx-per-ue", type=int, default=4)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--n-cp", type=int, default=16)
    parser.add_argument("--n-taps", type=int, default=7, help="SCM path/tap count.")
    parser.add_argument("--n-rays-per-path", type=int, default=15)
    parser.add_argument("--pdp-decay", type=float, default=5.0)
    parser.add_argument("--carrier-freq-hz", type=float, default=800e6)
    parser.add_argument("--antenna-spacing-lambda", type=float, default=0.5)
    parser.add_argument("--scm-angle-spread-deg", type=float, default=3.0)
    parser.add_argument(
        "--snr-train-db",
        type=float,
        default=None,
        help="Legacy single train/val SNR override. By default mixed train SNR 15 20 25 30 35 40 is used.",
    )
    parser.add_argument(
        "--snr-train-db-list",
        nargs="+",
        type=float,
        default=None,
        help="Mixed SNR values for train/val frames. Defaults to 15 20 25 30 35 40.",
    )
    parser.add_argument(
        "--snr-test-db",
        nargs="+",
        type=float,
        default=None,
        help="SNR sweep for test datasets. Defaults to 0 5 ... 40.",
    )
    parser.add_argument("--n-train-frames", type=int, default=50000)
    parser.add_argument("--n-val-frames", type=int, default=10000)
    parser.add_argument("--n-test-frames-per-snr", type=int, default=10000)
    parser.add_argument(
        "--csit-error-var",
        type=float,
        default=0.001,
        help="Complex CSIT error variance E[|E|^2] for H_tx_est = H_true + E.",
    )
    parser.add_argument("--precoder-norm", type=str, default="column", choices=["none", "column", "fro"])
    parser.add_argument("--case", type=str, default="clipping", choices=["linear", "cp_removal", "clipping"])
    parser.add_argument("--clip-ratio", type=float, default=1.6)
    parser.add_argument(
        "--pilot-kind",
        type=str,
        default="qpsk",
        choices=["ones", "qpsk"],
        help="Pilot sequence on active stream slots. qpsk avoids the high-PAPR all-ones OFDM impulse.",
    )
    parser.add_argument(
        "--rx-iq-gain-imbalance-db",
        type=float,
        default=0.5,
        help="Receiver I/Q amplitude gain difference in dB. Positive means I gain > Q gain.",
    )
    parser.add_argument(
        "--rx-iq-phase-error-deg",
        type=float,
        default=3.0,
        help="Receiver I/Q quadrature phase error in degrees from ideal 90-degree separation.",
    )
    parser.add_argument(
        "--rx-common-phase-error-deg",
        type=float,
        default=5.0,
        help="Receiver common phase rotation in degrees applied to pilot and data waveforms.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def add_csit_error(
    H_true: np.ndarray,
    cfg: MuMimoE2EConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if cfg.csit_error_var == 0.0:
        return H_true.copy()
    error = (
        rng.standard_normal(H_true.shape) + 1j * rng.standard_normal(H_true.shape)
    ) * math.sqrt(cfg.csit_error_var / 2.0)
    return (H_true + error).astype(np.complex64)


def build_scm_generator(cfg: MuMimoE2EConfig) -> ScmChannelGenerator:
    tx_array = ArrayConfig(
        cfg.n_tx,
        1,
        cfg.antenna_spacing_lambda,
        cfg.antenna_spacing_lambda,
    )
    rx_array = ArrayConfig(
        cfg.n_rx_per_ue,
        1,
        cfg.antenna_spacing_lambda,
        cfg.antenna_spacing_lambda,
    )
    scm_cfg = ScmChannelConfig(
        n_path=cfg.n_taps,
        n_rays_per_path=cfg.n_rays_per_path,
        n_rx=cfg.n_rx_per_ue,
        n_tx=cfg.n_tx,
        pdp_decay=cfg.pdp_decay,
        carrier_freq_hz=cfg.carrier_freq_hz,
        tx_array=tx_array,
        rx_array=rx_array,
        asd_deg=cfg.scm_angle_spread_deg,
        zsd_deg=cfg.scm_angle_spread_deg,
        asa_deg=cfg.scm_angle_spread_deg,
        zsa_deg=cfg.scm_angle_spread_deg,
    )
    return ScmChannelGenerator(scm_cfg)


def modulate_ofdm(freq_symbol: np.ndarray, cfg: MuMimoE2EConfig) -> np.ndarray:
    return ofdm_modulate_freq(
        freq_symbol,
        n_fft=cfg.n_fft,
        n_cp=cfg.n_cp,
        case=cfg.case,
        clip_ratio=cfg.clip_ratio,
    )


def apply_rx_rf_impairments(waveform: np.ndarray, cfg: MuMimoE2EConfig) -> np.ndarray:
    return apply_rf_impairments(
        waveform,
        iq_gain_imbalance_db=cfg.rx_iq_gain_imbalance_db,
        iq_phase_error_deg=cfg.rx_iq_phase_error_deg,
        common_phase_error_deg=cfg.rx_common_phase_error_deg,
    )


def make_orthogonal_pilots(cfg: MuMimoE2EConfig, n_frames: int) -> np.ndarray:
    x_p = np.zeros((n_frames, cfg.n_streams, cfg.n_streams, cfg.n_fft), dtype=np.complex64)
    for stream_id in range(cfg.n_streams):
        if cfg.pilot_kind == "ones":
            pilot = np.ones(cfg.n_fft, dtype=np.complex64)
        else:
            rng = np.random.default_rng(int(cfg.seed) + 7919 * (stream_id + 1))
            phases = rng.integers(0, 4, size=cfg.n_fft)
            pilot = np.exp(1j * (np.pi / 2.0) * phases).astype(np.complex64)
        x_p[:, stream_id, stream_id, :] = pilot[None, :]
    return x_p


def _empty_split(cfg: MuMimoE2EConfig, n_frames: int) -> dict[str, np.ndarray]:
    rx_time_len = cfg.n_fft if cfg.case == "cp_removal" else cfg.n_fft + cfg.n_cp
    bps = bits_per_symbol(cfg.modulation)
    return {
        "rx_p_time": np.zeros(
            (n_frames, cfg.n_streams, cfg.n_users, cfg.n_rx_per_ue, rx_time_len),
            dtype=np.complex64,
        ),
        "rx_d_time": np.zeros(
            (n_frames, cfg.n_users, cfg.n_rx_per_ue, rx_time_len),
            dtype=np.complex64,
        ),
        "x_p_freq": make_orthogonal_pilots(cfg, n_frames),
        "x_d_freq": np.zeros((n_frames, cfg.n_streams, cfg.n_fft), dtype=np.complex64),
        "bits": np.zeros((n_frames, cfg.n_streams, cfg.n_fft, bps), dtype=np.int8),
        "H_true": np.zeros(
            (n_frames, cfg.n_fft, cfg.n_users, cfg.n_rx_per_ue, cfg.n_tx),
            dtype=np.complex64,
        ),
        "G_tx_est": np.zeros((n_frames, cfg.n_fft, cfg.n_streams, cfg.n_tx), dtype=np.complex64),
        "W_precoder": np.zeros((n_frames, cfg.n_fft, cfg.n_tx, cfg.n_streams), dtype=np.complex64),
        "W_digital": np.zeros(
            (n_frames, cfg.n_fft, cfg.n_streams, cfg.n_streams),
            dtype=np.complex64,
        ),
        "W_tx_analog": np.zeros((n_frames, cfg.n_tx, cfg.n_streams), dtype=np.complex64),
        "W_rx_analog": np.zeros((n_frames, cfg.n_users, cfg.n_rx_per_ue), dtype=np.complex64),
        "scm_selected_angles": np.zeros((n_frames, cfg.n_users, 4), dtype=np.float32),
        "scm_center_angles": np.zeros((n_frames, cfg.n_users, 4, cfg.n_taps), dtype=np.float32),
        "A_eff_true": np.zeros(
            (n_frames, cfg.n_fft, cfg.n_users, cfg.n_rx_per_ue, cfg.n_streams),
            dtype=np.complex64,
        ),
        "snr_db": np.zeros(n_frames, dtype=np.float32),
        "signal_power": np.zeros(n_frames, dtype=np.float32),
        "desired_power": np.zeros((n_frames, cfg.n_users), dtype=np.float32),
        "inter_stream_power": np.zeros((n_frames, cfg.n_users), dtype=np.float32),
        "noise_power": np.zeros(n_frames, dtype=np.float32),
        "effective_sinr_db": np.zeros((n_frames, cfg.n_users), dtype=np.float32),
        "cond_A": np.zeros((n_frames, cfg.n_fft, cfg.n_users), dtype=np.float32),
        "mean_cond_A": np.zeros(n_frames, dtype=np.float32),
        "p95_cond_A": np.zeros(n_frames, dtype=np.float32),
    }


def _condition_numbers(A_eff_true: np.ndarray) -> np.ndarray:
    cond = np.linalg.cond(A_eff_true).astype(np.float32)
    return np.where(np.isfinite(cond), cond, np.finfo(np.float32).max).astype(np.float32)


def _effective_sinr_db(
    desired_power: np.ndarray,
    inter_stream_power: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    denominator = np.maximum(inter_stream_power + float(noise_power), 1e-300)
    return linear_to_db(desired_power / denominator).astype(np.float32)


def _make_clean_frame(
    *,
    cfg: MuMimoE2EConfig,
    x_p_freq: np.ndarray,
    bps: int,
    scm_generator: ScmChannelGenerator,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    channel = scm_generator.generate_multiuser(cfg.n_users, rng)
    h_time = channel.h_time
    H_true = channel_frequency_response(h_time, n_fft=cfg.n_fft)
    H_tx_est = add_csit_error(H_true, cfg, rng)
    W_tx_analog, W_rx_analog = hybrid_steering_beams(
        carrier_freq_hz=cfg.carrier_freq_hz,
        tx_array=scm_generator.tx_array,
        rx_array=scm_generator.rx_array,
        selected_angles=channel.selected_angles,
    )
    G_tx_est, W_digital, W_precoder = hybrid_zf_precoder_context(
        H_tx_est,
        W_tx_analog,
        W_rx_analog,
        normalization=cfg.precoder_norm,
    )
    A_eff_true = np.einsum("kurt,kts->kurs", H_true, W_precoder).astype(np.complex64)

    frame_bits = rng.integers(
        0,
        2,
        size=(cfg.n_streams, cfg.n_fft, bps),
        dtype=np.int8,
    )
    x_data = qam_modulate(frame_bits.reshape(-1), cfg.modulation).reshape(
        cfg.n_streams,
        cfg.n_fft,
    )

    x_tx_data_freq = precoded_tx_frequency(x_data, W_precoder)
    x_tx_data_time = modulate_ofdm(x_tx_data_freq, cfg)
    y_d_clean = apply_rx_rf_impairments(apply_multipath_mimo(x_tx_data_time, h_time), cfg)
    signal_power = float(np.mean(np.abs(y_d_clean) ** 2))

    y_p_clean = np.zeros(
        (cfg.n_streams, cfg.n_users, cfg.n_rx_per_ue, x_tx_data_time.shape[-1]),
        dtype=np.complex64,
    )
    for pilot_slot in range(cfg.n_streams):
        x_pilot = x_p_freq[pilot_slot]
        x_tx_pilot_freq = precoded_tx_frequency(x_pilot, W_precoder)
        x_tx_pilot_time = modulate_ofdm(x_tx_pilot_freq, cfg)
        y_p_clean[pilot_slot] = apply_rx_rf_impairments(
            apply_multipath_mimo(x_tx_pilot_time, h_time),
            cfg,
        )

    desired_power = np.zeros(cfg.n_users, dtype=np.float32)
    inter_stream_power = np.zeros(cfg.n_users, dtype=np.float32)
    for user_id in range(cfg.n_users):
        desired = A_eff_true[:, user_id, :, user_id].T * x_data[user_id][None, :]
        inter = np.zeros_like(desired)
        for stream_id in range(cfg.n_streams):
            if stream_id == user_id:
                continue
            inter += A_eff_true[:, user_id, :, stream_id].T * x_data[stream_id][None, :]
        desired_power[user_id] = float(np.mean(np.abs(desired) ** 2))
        inter_stream_power[user_id] = float(np.mean(np.abs(inter) ** 2))

    return {
        "y_d_clean": y_d_clean,
        "y_p_clean": y_p_clean,
        "x_data": x_data,
        "frame_bits": frame_bits,
        "H_true": H_true,
        "G_tx_est": G_tx_est,
        "W_precoder": W_precoder,
        "W_digital": W_digital,
        "W_tx_analog": W_tx_analog,
        "W_rx_analog": W_rx_analog,
        "scm_selected_angles": channel.selected_angles,
        "scm_center_angles": channel.center_angles,
        "A_eff_true": A_eff_true,
        "signal_power": signal_power,
        "desired_power": desired_power,
        "inter_stream_power": inter_stream_power,
        "cond_A": _condition_numbers(A_eff_true),
    }


def _write_noisy_frame(
    *,
    data: dict[str, np.ndarray],
    frame_index: int,
    clean: dict[str, np.ndarray | float],
    y_d_time: np.ndarray,
    y_p_time: np.ndarray,
    snr_value: float,
    noise_power: float,
) -> None:
    cond_A = clean["cond_A"]
    desired_power = clean["desired_power"]
    inter_stream_power = clean["inter_stream_power"]
    if not isinstance(cond_A, np.ndarray):
        raise TypeError("cond_A must be an ndarray")
    if not isinstance(desired_power, np.ndarray) or not isinstance(inter_stream_power, np.ndarray):
        raise TypeError("power metrics must be ndarrays")

    data["rx_p_time"][frame_index] = y_p_time
    data["rx_d_time"][frame_index] = y_d_time
    data["x_d_freq"][frame_index] = clean["x_data"]
    data["bits"][frame_index] = clean["frame_bits"]
    data["H_true"][frame_index] = clean["H_true"]
    data["G_tx_est"][frame_index] = clean["G_tx_est"]
    data["W_precoder"][frame_index] = clean["W_precoder"]
    data["W_digital"][frame_index] = clean["W_digital"]
    data["W_tx_analog"][frame_index] = clean["W_tx_analog"]
    data["W_rx_analog"][frame_index] = clean["W_rx_analog"]
    data["scm_selected_angles"][frame_index] = clean["scm_selected_angles"]
    data["scm_center_angles"][frame_index] = clean["scm_center_angles"]
    data["A_eff_true"][frame_index] = clean["A_eff_true"]
    data["snr_db"][frame_index] = snr_value
    data["signal_power"][frame_index] = float(clean["signal_power"])
    data["desired_power"][frame_index] = desired_power
    data["inter_stream_power"][frame_index] = inter_stream_power
    data["noise_power"][frame_index] = noise_power
    data["effective_sinr_db"][frame_index] = _effective_sinr_db(
        desired_power,
        inter_stream_power,
        noise_power,
    )
    data["cond_A"][frame_index] = cond_A
    data["mean_cond_A"][frame_index] = float(np.mean(cond_A))
    data["p95_cond_A"][frame_index] = float(np.percentile(cond_A, 95.0))


def _apply_scaled_awgn(
    clean: np.ndarray,
    unit_noise: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    if noise_power <= 0.0:
        return clean.astype(np.complex64, copy=True)
    return (clean + unit_noise * math.sqrt(noise_power / 2.0)).astype(np.complex64)


def _make_split_dataset(
    *,
    cfg: MuMimoE2EConfig,
    n_frames: int,
    snr_db: float | Sequence[float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    data = _empty_split(cfg, n_frames)
    bps = bits_per_symbol(cfg.modulation)
    scm_generator = build_scm_generator(cfg)
    snr_values = np.asarray(snr_db, dtype=np.float32).reshape(-1)
    if snr_values.size == 0:
        raise ValueError("snr_db must contain at least one value")
    if snr_values.size == 1:
        frame_snr_db = np.full(n_frames, float(snr_values[0]), dtype=np.float32)
    else:
        frame_snr_db = rng.choice(snr_values, size=n_frames, replace=True).astype(np.float32)

    for frame_index in range(n_frames):
        snr_value = float(frame_snr_db[frame_index])
        clean = _make_clean_frame(
            cfg=cfg,
            x_p_freq=data["x_p_freq"][frame_index],
            bps=bps,
            scm_generator=scm_generator,
            rng=rng,
        )
        signal_power = float(clean["signal_power"])
        noise_power = noise_power_from_snr(signal_power, snr_value)
        y_d_time = add_awgn(clean["y_d_clean"], noise_power, rng)

        y_p_clean = clean["y_p_clean"]
        if not isinstance(y_p_clean, np.ndarray):
            raise TypeError("y_p_clean must be an ndarray")
        y_p_time = add_awgn(y_p_clean, noise_power, rng)

        _write_noisy_frame(
            data=data,
            frame_index=frame_index,
            clean=clean,
            y_d_time=y_d_time,
            y_p_time=y_p_time,
            snr_value=snr_value,
            noise_power=noise_power,
        )

    return data


def _make_paired_test_datasets(
    *,
    cfg: MuMimoE2EConfig,
    n_frames: int,
    snr_db_values: Sequence[float],
    rng: np.random.Generator,
) -> dict[float, dict[str, np.ndarray]]:
    snr_values = [float(value) for value in snr_db_values]
    if len(snr_values) == 0:
        raise ValueError("snr_db_values must contain at least one value")
    if len(set(snr_values)) != len(snr_values):
        raise ValueError("snr_db_values must not contain duplicates")

    datasets = {snr_value: _empty_split(cfg, n_frames) for snr_value in snr_values}
    reference_data = datasets[snr_values[0]]
    bps = bits_per_symbol(cfg.modulation)
    scm_generator = build_scm_generator(cfg)

    for frame_index in range(n_frames):
        clean = _make_clean_frame(
            cfg=cfg,
            x_p_freq=reference_data["x_p_freq"][frame_index],
            bps=bps,
            scm_generator=scm_generator,
            rng=rng,
        )

        y_d_clean = clean["y_d_clean"]
        y_p_clean = clean["y_p_clean"]
        if not isinstance(y_d_clean, np.ndarray) or not isinstance(y_p_clean, np.ndarray):
            raise TypeError("clean waveforms must be ndarrays")

        data_noise_unit = (
            rng.standard_normal(y_d_clean.shape) + 1j * rng.standard_normal(y_d_clean.shape)
        ).astype(np.complex64)
        pilot_noise_unit = (
            rng.standard_normal(y_p_clean.shape) + 1j * rng.standard_normal(y_p_clean.shape)
        ).astype(np.complex64)

        signal_power = float(clean["signal_power"])
        for snr_value in snr_values:
            noise_power = noise_power_from_snr(signal_power, snr_value)
            y_d_time = _apply_scaled_awgn(y_d_clean, data_noise_unit, noise_power)
            y_p_time = _apply_scaled_awgn(y_p_clean, pilot_noise_unit, noise_power)
            _write_noisy_frame(
                data=datasets[snr_value],
                frame_index=frame_index,
                clean=clean,
                y_d_time=y_d_time,
                y_p_time=y_p_time,
                snr_value=snr_value,
                noise_power=noise_power,
            )

    return datasets


def _save_npz(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(
        f"[SAVE] {path} frames={dataset['bits'].shape[0]}, "
        f"rx_p_time={dataset['rx_p_time'].shape}, rx_d_time={dataset['rx_d_time'].shape}, "
        f"A_eff_true={dataset['A_eff_true'].shape}"
    )


def _snr_name(snr_db: float) -> str:
    if math.isinf(float(snr_db)):
        return "inf"
    if float(snr_db).is_integer():
        return f"{int(snr_db):02d}"
    return str(snr_db).replace("-", "m").replace(".", "p")


def _snr_set_name(snr_db: float | Sequence[float]) -> str:
    values = np.asarray(snr_db, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("snr_db must contain at least one value")
    if values.size == 1:
        return _snr_name(float(values[0]))
    return f"{_snr_name(float(np.min(values)))}_{_snr_name(float(np.max(values)))}mixed"


def write_config(out_dir: Path, cfg: MuMimoE2EConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    data = asdict(cfg)
    data["snr_train_db_list"] = list(cfg.snr_train_db_list)
    data["snr_test_db"] = list(cfg.snr_test_db)
    data["n_streams"] = cfg.n_streams
    data["waveform_type"] = "raw_mumimo_e2e"
    data["channel_model"] = "SCM-style geometric clustered channel"
    data["pilot_design"] = f"orthogonal_stream_slots_{cfg.pilot_kind}"
    data["power_policy"] = f"hybrid steering + digital ZF, total precoder norm={cfg.precoder_norm}"
    data["nonlinear_processing"] = {
        "linear": "normal BS-antenna OFDM waveform with cyclic prefix",
        "cp_removal": "BS transmitter omits cyclic prefix; receiver FFT starts at sample 0",
        "clipping": "BS per-antenna time-domain OFDM symbol is clipped before cyclic prefix insertion",
    }[cfg.case]
    data["symbol_power"] = "E[|s_u|^2] = 1 for QAM data and active pilots"
    data["precoder"] = (
        "MATLAB-style analog steering beams from the strongest SCM path, followed by "
        "per-subcarrier digital ZF on Wr.T @ H_tx_est @ Wt"
    )
    data["extra_arrays"] = [
        "W_tx_analog",
        "W_rx_analog",
        "W_digital",
        "scm_selected_angles",
        "scm_center_angles",
    ]
    data["noise_power"] = (
        "complex per-antenna variance sigma2; generated as sqrt(sigma2/2)*(n_re+j*n_im)"
    )
    data["snr_definition"] = "sigma2 = mean(|raw received data waveform before AWGN|^2) / 10^(snr_db/10)"
    rf_alpha, rf_beta = rf_impairment_widely_linear_coefficients(
        iq_gain_imbalance_db=cfg.rx_iq_gain_imbalance_db,
        iq_phase_error_deg=cfg.rx_iq_phase_error_deg,
        common_phase_error_deg=cfg.rx_common_phase_error_deg,
    )
    data["rf_impairments"] = {
        "location": "receiver front-end, after multipath channel and before AWGN",
        "enabled": bool(
            cfg.rx_iq_gain_imbalance_db != 0.0
            or cfg.rx_iq_phase_error_deg != 0.0
            or cfg.rx_common_phase_error_deg != 0.0
        ),
        "rx_iq_gain_imbalance_db": cfg.rx_iq_gain_imbalance_db,
        "rx_iq_phase_error_deg": cfg.rx_iq_phase_error_deg,
        "rx_common_phase_error_deg": cfg.rx_common_phase_error_deg,
        "widely_linear_model": "y_rf = alpha * y_channel + beta * conj(y_channel)",
        "alpha": {"real": float(rf_alpha.real), "imag": float(rf_alpha.imag)},
        "beta": {"real": float(rf_beta.real), "imag": float(rf_beta.imag)},
        "default_policy": "enabled by default for RF stress testing; use the True-H WL-MMSE receiver baseline as the RF-aware oracle",
        "mild_stress_test_example": "0.5 dB I/Q gain imbalance, 3 degree I/Q phase error, 5 degree common phase rotation",
        "nonzero_warning": (
            "I/Q imbalance is a widely-linear RF impairment. With nonzero RF impairment, "
            "A_eff_true remains the pre-RF linear effective channel, so plain True-H and "
            "LMMSE channel-estimation baselines are no longer strict oracle references."
        ),
    }
    data["test_snr_policy"] = (
        "paired base frames: test_snr*.npz files share the same channel, bits, precoder, "
        "clean data waveform, and clean pilot waveform per frame; only AWGN scale changes by SNR"
    )
    data["receiver_compatibility"] = "raw UE antenna-domain waveform; use rx_mumimo_receiver.py"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[SAVE] {path}")
    return path


def generate_all(cfg: MuMimoE2EConfig, out_dir: Path) -> None:
    cfg.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(out_dir, cfg)

    rng = np.random.default_rng(cfg.seed)
    train_snr_db = tuple(float(x) for x in cfg.snr_train_db_list)
    train_snr_name = _snr_set_name(train_snr_db)
    train = _make_split_dataset(
        cfg=cfg,
        n_frames=cfg.n_train_frames,
        snr_db=train_snr_db,
        rng=rng,
    )
    _save_npz(out_dir / f"train_snr{train_snr_name}.npz", train)

    val = _make_split_dataset(
        cfg=cfg,
        n_frames=cfg.n_val_frames,
        snr_db=train_snr_db,
        rng=rng,
    )
    _save_npz(out_dir / f"val_snr{train_snr_name}.npz", val)

    test_sets = _make_paired_test_datasets(
        cfg=cfg,
        n_frames=cfg.n_test_frames_per_snr,
        snr_db_values=cfg.snr_test_db,
        rng=rng,
    )
    for snr_db in cfg.snr_test_db:
        test = test_sets[float(snr_db)]
        _save_npz(out_dir / f"test_snr{_snr_name(float(snr_db))}.npz", test)


def build_config(args: argparse.Namespace) -> MuMimoE2EConfig:
    if args.snr_train_db_list is None:
        if args.snr_train_db is None:
            snr_train_db_list: Sequence[float] = MuMimoE2EConfig.snr_train_db_list
        else:
            snr_train_db_list = (float(args.snr_train_db),)
    else:
        snr_train_db_list = tuple(float(x) for x in args.snr_train_db_list)
    snr_train_db = float(args.snr_train_db) if args.snr_train_db is not None else float(tuple(snr_train_db_list)[-1])

    if args.snr_test_db is None:
        snr_test_db: Sequence[float] = MuMimoE2EConfig.snr_test_db
    else:
        snr_test_db = tuple(float(x) for x in args.snr_test_db)

    return MuMimoE2EConfig(
        n_fft=int(args.n_fft),
        n_cp=int(args.n_cp),
        modulation=str(args.modulation).upper(),
        n_users=int(args.n_users),
        n_tx=int(args.n_tx),
        n_rx_per_ue=int(args.n_rx_per_ue),
        n_taps=int(args.n_taps),
        n_rays_per_path=int(args.n_rays_per_path),
        pdp_decay=float(args.pdp_decay),
        carrier_freq_hz=float(args.carrier_freq_hz),
        antenna_spacing_lambda=float(args.antenna_spacing_lambda),
        scm_angle_spread_deg=float(args.scm_angle_spread_deg),
        snr_train_db=snr_train_db,
        snr_train_db_list=tuple(snr_train_db_list),
        snr_test_db=tuple(snr_test_db),
        n_train_frames=int(args.n_train_frames),
        n_val_frames=int(args.n_val_frames),
        n_test_frames_per_snr=int(args.n_test_frames_per_snr),
        csit_error_var=float(args.csit_error_var),
        precoder_norm=str(args.precoder_norm),
        case=str(args.case),
        clip_ratio=float(args.clip_ratio),
        pilot_kind=str(args.pilot_kind),
        rx_iq_gain_imbalance_db=float(args.rx_iq_gain_imbalance_db),
        rx_iq_phase_error_deg=float(args.rx_iq_phase_error_deg),
        rx_common_phase_error_deg=float(args.rx_common_phase_error_deg),
        seed=int(args.seed),
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    generate_all(cfg, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
