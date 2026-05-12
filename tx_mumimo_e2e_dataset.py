from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MuMimoE2EConfig:
    n_fft: int = 64
    n_cp: int = 16
    modulation: str = "64QAM"
    n_users: int = 2
    n_tx: int = 8
    n_rx_per_ue: int = 4
    n_taps: int = 8
    pdp_decay: float = 2.0
    snr_train_db: float = 40.0
    snr_test_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40)
    n_train_frames: int = 50000
    n_val_frames: int = 10000
    n_test_frames_per_snr: int = 10000
    csit_error_var: float = 0.0
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
        if self.n_taps > self.n_cp:
            raise ValueError("MU-MIMO OFDM requires n_taps <= n_cp")
        if self.pdp_decay <= 0:
            raise ValueError("pdp_decay must be positive")
        if self.modulation.upper() not in {"16QAM", "64QAM"}:
            raise ValueError("modulation must be one of 16QAM, 64QAM")
        if not (1 <= self.n_users <= self.n_tx):
            raise ValueError("n_users must satisfy 1 <= n_users <= n_tx")
        if self.n_rx_per_ue <= 0:
            raise ValueError("n_rx_per_ue must be positive")
        if self.csit_error_var < 0.0:
            raise ValueError("csit_error_var must be non-negative")
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
    parser.add_argument("--n-taps", type=int, default=8)
    parser.add_argument("--pdp-decay", type=float, default=2.0)
    parser.add_argument("--snr-train-db", type=float, default=40.0)
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
        default=0.0,
        help="Complex CSIT error variance E[|E|^2] for H_tx_est = H_true + E.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def bits_per_symbol(modulation: str) -> int:
    table = {"16QAM": 4, "64QAM": 6}
    key = modulation.upper()
    if key not in table:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return table[key]


def _pam_levels(axis_bits: int) -> np.ndarray:
    if axis_bits == 2:
        return np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float64)
    if axis_bits == 3:
        return np.array([7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0], dtype=np.float64)
    raise ValueError(f"Unsupported PAM axis bit count: {axis_bits}")


def _gray_labels(axis_bits: int) -> np.ndarray:
    labels = np.arange(2**axis_bits, dtype=np.int64)
    return labels ^ (labels >> 1)


def _bits_to_ints(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    values = np.zeros(bits.shape[0], dtype=np.int64)
    for bit_pos in range(bits.shape[1]):
        values = (values << 1) | bits[:, bit_pos].astype(np.int64)
    return values


def _qam_normalization(modulation: str) -> float:
    bps = bits_per_symbol(modulation)
    if bps == 4:
        return math.sqrt(10.0)
    if bps == 6:
        return math.sqrt(42.0)
    raise ValueError(f"Unsupported modulation: {modulation}")


def qam_modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    modulation = modulation.upper()
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    bps = bits_per_symbol(modulation)
    if bits.size % bps != 0:
        raise ValueError("Number of bits must be a multiple of bits per symbol")

    axis_bits = bps // 2
    labels = _gray_labels(axis_bits)
    levels = _pam_levels(axis_bits)
    b = bits.reshape(-1, bps)
    re_label = _bits_to_ints(b[:, :axis_bits])
    im_label = _bits_to_ints(b[:, axis_bits:])
    re_index = np.empty_like(re_label)
    im_index = np.empty_like(im_label)
    for index, gray_label in enumerate(labels):
        re_index[re_label == gray_label] = index
        im_index[im_label == gray_label] = index
    symbols = levels[re_index] + 1j * levels[im_index]
    return (symbols / _qam_normalization(modulation)).astype(np.complex64)


def db_to_linear(db_value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db_value) / 10.0)


def linear_to_db(value: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value), floor))


def ofdm_modulate_freq(freq_symbol: np.ndarray, cfg: MuMimoE2EConfig) -> np.ndarray:
    freq_symbol = np.asarray(freq_symbol, dtype=np.complex64)
    time_no_cp = np.fft.ifft(freq_symbol, n=cfg.n_fft, axis=-1) * math.sqrt(cfg.n_fft)
    cp = time_no_cp[..., -cfg.n_cp :] if cfg.n_cp > 0 else time_no_cp[..., :0]
    return np.concatenate([cp, time_no_cp], axis=-1).astype(np.complex64)


def generate_multipath_channels(cfg: MuMimoE2EConfig, rng: np.random.Generator) -> np.ndarray:
    tap = np.arange(cfg.n_taps, dtype=np.float64)
    pdp = np.exp(-tap / cfg.pdp_decay)
    pdp /= np.sum(pdp)
    h = (
        rng.standard_normal((cfg.n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
        + 1j * rng.standard_normal((cfg.n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
    ) / math.sqrt(2.0)
    h *= np.sqrt(pdp)[None, :, None, None]
    return h.astype(np.complex64)


def channel_frequency_response(h_time: np.ndarray, cfg: MuMimoE2EConfig) -> np.ndarray:
    h_pad = np.zeros(
        (cfg.n_users, cfg.n_fft, cfg.n_rx_per_ue, cfg.n_tx),
        dtype=np.complex64,
    )
    h_pad[:, : cfg.n_taps, :, :] = h_time
    return np.transpose(np.fft.fft(h_pad, n=cfg.n_fft, axis=1), (1, 0, 2, 3)).astype(np.complex64)


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


def dominant_svd_receive_direction(H_user_k: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(H_user_k, full_matrices=False)
    direction = u[:, 0]
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
    return direction.astype(np.complex64)


def zf_precoder_column_normalized(G: np.ndarray) -> np.ndarray:
    W = np.linalg.pinv(G).astype(np.complex64)
    norms = np.linalg.norm(W, axis=0)
    safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
    return (W / safe_norms[None, :]).astype(np.complex64)


def make_precoder_context(
    H_true: np.ndarray,
    H_tx_est: np.ndarray,
    cfg: MuMimoE2EConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c_tx = np.zeros((cfg.n_fft, cfg.n_streams, cfg.n_rx_per_ue), dtype=np.complex64)
    G_tx_est = np.zeros((cfg.n_fft, cfg.n_streams, cfg.n_tx), dtype=np.complex64)
    W_precoder = np.zeros((cfg.n_fft, cfg.n_tx, cfg.n_streams), dtype=np.complex64)

    for subcarrier in range(cfg.n_fft):
        for user_id in range(cfg.n_streams):
            direction = dominant_svd_receive_direction(H_tx_est[subcarrier, user_id])
            c_tx[subcarrier, user_id] = direction
            G_tx_est[subcarrier, user_id] = direction.conj().T @ H_tx_est[subcarrier, user_id]
        W_precoder[subcarrier] = zf_precoder_column_normalized(G_tx_est[subcarrier])

    A_eff_true = np.einsum("kurt,kts->kurs", H_true, W_precoder).astype(np.complex64)
    return c_tx, G_tx_est, W_precoder, A_eff_true


def noise_power_from_snr(signal_power: float, snr_db: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    return float(signal_power) / max(float(db_to_linear(float(snr_db))), 1e-300)


def add_awgn(
    values: np.ndarray,
    noise_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_power <= 0.0:
        return values.astype(np.complex64)
    noise = (
        rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    ) * math.sqrt(noise_power / 2.0)
    return (values + noise).astype(np.complex64)


def make_orthogonal_pilots(cfg: MuMimoE2EConfig, n_frames: int) -> np.ndarray:
    x_p = np.zeros((n_frames, cfg.n_streams, cfg.n_streams, cfg.n_fft), dtype=np.complex64)
    for stream_id in range(cfg.n_streams):
        x_p[:, stream_id, stream_id, :] = 1.0 + 0.0j
    return x_p


def _empty_split(cfg: MuMimoE2EConfig, n_frames: int) -> dict[str, np.ndarray]:
    rx_time_len = cfg.n_fft + cfg.n_cp
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
    n_fft, n_users = A_eff_true.shape[0], A_eff_true.shape[1]
    cond = np.zeros((n_fft, n_users), dtype=np.float32)
    for subcarrier in range(n_fft):
        for user_id in range(n_users):
            value = np.linalg.cond(A_eff_true[subcarrier, user_id])
            cond[subcarrier, user_id] = np.float32(value if np.isfinite(value) else np.finfo(np.float32).max)
    return cond


def _make_split_dataset(
    *,
    cfg: MuMimoE2EConfig,
    n_frames: int,
    snr_db: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    data = _empty_split(cfg, n_frames)
    bps = bits_per_symbol(cfg.modulation)

    for frame_index in range(n_frames):
        h_time = generate_multipath_channels(cfg, rng)
        H_true = channel_frequency_response(h_time, cfg)
        H_tx_est = add_csit_error(H_true, cfg, rng)
        _, G_tx_est, W_precoder, A_eff_true = make_precoder_context(H_true, H_tx_est, cfg)

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

        y_d_clean = np.einsum("kurs,sk->urk", A_eff_true, x_data).astype(np.complex64)
        signal_power = float(np.mean(np.abs(y_d_clean) ** 2))
        noise_power = noise_power_from_snr(signal_power, snr_db)
        y_d_freq = add_awgn(y_d_clean, noise_power, rng)

        y_p_freq = np.zeros(
            (cfg.n_streams, cfg.n_users, cfg.n_rx_per_ue, cfg.n_fft),
            dtype=np.complex64,
        )
        for pilot_slot in range(cfg.n_streams):
            y_p_clean = np.transpose(A_eff_true[:, :, :, pilot_slot], (1, 2, 0))
            y_p_freq[pilot_slot] = add_awgn(y_p_clean, noise_power, rng)

        desired_power = np.zeros(cfg.n_users, dtype=np.float32)
        inter_stream_power = np.zeros(cfg.n_users, dtype=np.float32)
        effective_sinr_db = np.zeros(cfg.n_users, dtype=np.float32)
        for user_id in range(cfg.n_users):
            desired = A_eff_true[:, user_id, :, user_id].T * x_data[user_id][None, :]
            inter = np.zeros_like(desired)
            for stream_id in range(cfg.n_streams):
                if stream_id == user_id:
                    continue
                inter += A_eff_true[:, user_id, :, stream_id].T * x_data[stream_id][None, :]
            desired_power[user_id] = float(np.mean(np.abs(desired) ** 2))
            inter_stream_power[user_id] = float(np.mean(np.abs(inter) ** 2))
            sinr = desired_power[user_id] / max(inter_stream_power[user_id] + noise_power, 1e-300)
            effective_sinr_db[user_id] = float(linear_to_db(sinr))

        cond_A = _condition_numbers(A_eff_true)

        data["rx_p_time"][frame_index] = ofdm_modulate_freq(y_p_freq, cfg)
        data["rx_d_time"][frame_index] = ofdm_modulate_freq(y_d_freq, cfg)
        data["x_d_freq"][frame_index] = x_data
        data["bits"][frame_index] = frame_bits
        data["H_true"][frame_index] = H_true
        data["G_tx_est"][frame_index] = G_tx_est
        data["W_precoder"][frame_index] = W_precoder
        data["A_eff_true"][frame_index] = A_eff_true
        data["snr_db"][frame_index] = float(snr_db)
        data["signal_power"][frame_index] = signal_power
        data["desired_power"][frame_index] = desired_power
        data["inter_stream_power"][frame_index] = inter_stream_power
        data["noise_power"][frame_index] = noise_power
        data["effective_sinr_db"][frame_index] = effective_sinr_db
        data["cond_A"][frame_index] = cond_A
        data["mean_cond_A"][frame_index] = float(np.mean(cond_A))
        data["p95_cond_A"][frame_index] = float(np.percentile(cond_A, 95.0))

    return data


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


def write_config(out_dir: Path, cfg: MuMimoE2EConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    data = asdict(cfg)
    data["snr_test_db"] = list(cfg.snr_test_db)
    data["n_streams"] = cfg.n_streams
    data["case"] = "linear"
    data["waveform_type"] = "raw_mumimo_e2e"
    data["pilot_design"] = "orthogonal_stream_slots"
    data["power_policy"] = "per_stream_fixed_unit_precoder_column_norm"
    data["symbol_power"] = "E[|s_u|^2] = 1 for QAM data and active pilots"
    data["precoder"] = (
        "ZF on G_tx_est, where G_tx_est[u] = c_u^H H_tx_est[u] and c_u is the "
        "dominant-SVD receive direction used only for BS precoder design"
    )
    data["noise_power"] = (
        "complex per-antenna variance sigma2; generated as sqrt(sigma2/2)*(n_re+j*n_im)"
    )
    data["snr_definition"] = "sigma2 = mean(|A_eff_true @ x_d_freq|^2) / 10^(snr_db/10)"
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
    train = _make_split_dataset(
        cfg=cfg,
        n_frames=cfg.n_train_frames,
        snr_db=cfg.snr_train_db,
        rng=rng,
    )
    _save_npz(out_dir / f"train_snr{_snr_name(cfg.snr_train_db)}.npz", train)

    val = _make_split_dataset(
        cfg=cfg,
        n_frames=cfg.n_val_frames,
        snr_db=cfg.snr_train_db,
        rng=rng,
    )
    _save_npz(out_dir / f"val_snr{_snr_name(cfg.snr_train_db)}.npz", val)

    for snr_db in cfg.snr_test_db:
        test = _make_split_dataset(
            cfg=cfg,
            n_frames=cfg.n_test_frames_per_snr,
            snr_db=float(snr_db),
            rng=rng,
        )
        _save_npz(out_dir / f"test_snr{_snr_name(float(snr_db))}.npz", test)


def build_config(args: argparse.Namespace) -> MuMimoE2EConfig:
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
        pdp_decay=float(args.pdp_decay),
        snr_train_db=float(args.snr_train_db),
        snr_test_db=tuple(snr_test_db),
        n_train_frames=int(args.n_train_frames),
        n_val_frames=int(args.n_val_frames),
        n_test_frames_per_snr=int(args.n_test_frames_per_snr),
        csit_error_var=float(args.csit_error_var),
        seed=int(args.seed),
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    generate_all(cfg, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
