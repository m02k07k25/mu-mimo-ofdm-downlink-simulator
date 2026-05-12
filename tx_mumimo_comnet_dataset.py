from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MuMimoComNetConfig:
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
        if self.n_taps <= 0:
            raise ValueError("n_taps must be positive")
        if self.csit_error_var < 0.0:
            raise ValueError("csit_error_var must be non-negative")
        if min(self.n_train_frames, self.n_val_frames, self.n_test_frames_per_snr) < 0:
            raise ValueError("frame counts must be non-negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SISO-compatible effective-stream datasets from a "
            "downlink MU-MIMO OFDM system."
        )
    )
    parser.add_argument("--out-dir", type=str, default="outputs_mumimo_comnet_64qam")
    parser.add_argument("--modulation", type=str, default="64QAM", choices=["16QAM", "64QAM"])
    parser.add_argument("--n-users", type=int, default=2)
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


def ofdm_modulate_freq(freq_symbol: np.ndarray, cfg: MuMimoComNetConfig) -> np.ndarray:
    freq_symbol = np.asarray(freq_symbol, dtype=np.complex64)
    time_no_cp = np.fft.ifft(freq_symbol, n=cfg.n_fft, axis=-1) * math.sqrt(cfg.n_fft)
    cp = time_no_cp[..., -cfg.n_cp :] if cfg.n_cp > 0 else time_no_cp[..., :0]
    return np.concatenate([cp, time_no_cp], axis=-1).astype(np.complex64)


def generate_multipath_channels(cfg: MuMimoComNetConfig, rng: np.random.Generator) -> np.ndarray:
    tap = np.arange(cfg.n_taps, dtype=np.float64)
    pdp = np.exp(-tap / cfg.pdp_decay)
    pdp /= np.sum(pdp)
    h = (
        rng.standard_normal((cfg.n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
        + 1j * rng.standard_normal((cfg.n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
    ) / math.sqrt(2.0)
    h *= np.sqrt(pdp)[None, :, None, None]
    return h.astype(np.complex64)


def channel_frequency_response(h_time: np.ndarray, cfg: MuMimoComNetConfig) -> np.ndarray:
    h_pad = np.zeros(
        (cfg.n_users, cfg.n_fft, cfg.n_rx_per_ue, cfg.n_tx),
        dtype=np.complex64,
    )
    h_pad[:, : cfg.n_taps, :, :] = h_time
    return np.transpose(np.fft.fft(h_pad, n=cfg.n_fft, axis=1), (1, 0, 2, 3)).astype(np.complex64)


def add_csit_error(
    H_true: np.ndarray,
    cfg: MuMimoComNetConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if cfg.csit_error_var == 0.0:
        return H_true.copy()
    error = (
        rng.standard_normal(H_true.shape) + 1j * rng.standard_normal(H_true.shape)
    ) * math.sqrt(cfg.csit_error_var / 2.0)
    return (H_true + error).astype(np.complex64)


def dominant_rx_combiner(H_user_k: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(H_user_k, full_matrices=False)
    combiner = u[:, 0]
    norm = np.linalg.norm(combiner)
    if norm > 1e-12:
        combiner = combiner / norm
    return combiner.astype(np.complex64)


def zf_precoder_from_effective_channel(G_est: np.ndarray) -> np.ndarray:
    W_raw = np.linalg.pinv(G_est)
    norm = np.linalg.norm(W_raw, ord="fro")
    if norm > 1e-12:
        W_raw = W_raw / norm
    return W_raw.astype(np.complex64)


def make_mumimo_context(
    H_true: np.ndarray,
    H_tx_est: np.ndarray,
    cfg: MuMimoComNetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combiners = np.zeros((cfg.n_fft, cfg.n_users, cfg.n_rx_per_ue), dtype=np.complex64)
    W = np.zeros((cfg.n_fft, cfg.n_tx, cfg.n_users), dtype=np.complex64)
    h_eff_all = np.zeros((cfg.n_fft, cfg.n_users, cfg.n_users), dtype=np.complex64)

    for subcarrier in range(cfg.n_fft):
        G_est = np.zeros((cfg.n_users, cfg.n_tx), dtype=np.complex64)
        for user_id in range(cfg.n_users):
            combiner = dominant_rx_combiner(H_true[subcarrier, user_id])
            combiners[subcarrier, user_id] = combiner
            G_est[user_id] = combiner.conj().T @ H_tx_est[subcarrier, user_id]

        W[subcarrier] = zf_precoder_from_effective_channel(G_est)

        for rx_user in range(cfg.n_users):
            H_user_k = H_true[subcarrier, rx_user]
            c_user = combiners[subcarrier, rx_user]
            for tx_stream in range(cfg.n_users):
                h_eff_all[subcarrier, rx_user, tx_stream] = (
                    c_user.conj().T @ H_user_k @ W[subcarrier, :, tx_stream]
                )

    return combiners, W, h_eff_all


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


def noise_power_from_snr(signal_power: float, snr_db: float) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    return float(signal_power) / max(float(db_to_linear(float(snr_db))), 1e-300)


def _empty_split(cfg: MuMimoComNetConfig, n_frames: int) -> dict[str, np.ndarray]:
    n_eff = n_frames * cfg.n_users
    rx_time_len = cfg.n_fft + cfg.n_cp
    bps = bits_per_symbol(cfg.modulation)
    return {
        "rx_p_time": np.zeros((n_eff, rx_time_len), dtype=np.complex64),
        "rx_d_time": np.zeros((n_eff, rx_time_len), dtype=np.complex64),
        "x_p_freq": np.ones((n_eff, cfg.n_fft), dtype=np.complex64),
        "x_d_freq": np.zeros((n_eff, cfg.n_fft), dtype=np.complex64),
        "h_true": np.zeros((n_eff, cfg.n_fft), dtype=np.complex64),
        "bits": np.zeros((n_eff, cfg.n_fft, bps), dtype=np.int8),
        "snr_db": np.zeros(n_eff, dtype=np.float32),
        "frame_id": np.zeros(n_eff, dtype=np.int64),
        "user_id": np.zeros(n_eff, dtype=np.int64),
        "h_eff_all": np.zeros((n_eff, cfg.n_fft, cfg.n_users), dtype=np.complex64),
        "signal_power": np.zeros(n_eff, dtype=np.float32),
        "mui_power": np.zeros(n_eff, dtype=np.float32),
        "noise_power": np.zeros(n_eff, dtype=np.float32),
        "effective_sinr_db": np.zeros(n_eff, dtype=np.float32),
    }


def _make_split_dataset(
    *,
    cfg: MuMimoComNetConfig,
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
        _, _, h_eff_all = make_mumimo_context(H_true, H_tx_est, cfg)

        frame_bits = rng.integers(
            0,
            2,
            size=(cfg.n_users, cfg.n_fft, bps),
            dtype=np.int8,
        )
        x_data = qam_modulate(frame_bits.reshape(-1), cfg.modulation).reshape(
            cfg.n_users,
            cfg.n_fft,
        )

        for user_id in range(cfg.n_users):
            record_index = frame_index * cfg.n_users + user_id
            h_desired = h_eff_all[:, user_id, user_id]
            desired_data = h_desired * x_data[user_id]
            all_stream_data = np.sum(h_eff_all[:, user_id, :] * x_data.T, axis=1)
            mui_data = all_stream_data - desired_data
            signal_power = float(np.mean(np.abs(desired_data) ** 2))
            mui_power = float(np.mean(np.abs(mui_data) ** 2))
            noise_power = noise_power_from_snr(signal_power, snr_db)

            y_p_freq = add_awgn(h_desired, noise_power, rng)
            y_d_freq = add_awgn(desired_data + mui_data, noise_power, rng)
            effective_sinr = signal_power / max(mui_power + noise_power, 1e-300)

            data["rx_p_time"][record_index] = ofdm_modulate_freq(y_p_freq, cfg)
            data["rx_d_time"][record_index] = ofdm_modulate_freq(y_d_freq, cfg)
            data["x_d_freq"][record_index] = x_data[user_id]
            data["h_true"][record_index] = h_desired
            data["bits"][record_index] = frame_bits[user_id]
            data["snr_db"][record_index] = float(snr_db)
            data["frame_id"][record_index] = int(frame_index)
            data["user_id"][record_index] = int(user_id)
            data["h_eff_all"][record_index] = h_eff_all[:, user_id, :]
            data["signal_power"][record_index] = signal_power
            data["mui_power"][record_index] = mui_power
            data["noise_power"][record_index] = noise_power
            data["effective_sinr_db"][record_index] = float(linear_to_db(effective_sinr))

    return data


def _save_npz(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    n_frames = int(dataset["bits"].shape[0])
    print(
        f"[SAVE] {path} effective_records={n_frames}, "
        f"rx_time={dataset['rx_p_time'].shape}, h_true={dataset['h_true'].shape}, "
        f"bits={dataset['bits'].shape}"
    )


def _snr_name(snr_db: float) -> str:
    if math.isinf(float(snr_db)):
        return "inf"
    if float(snr_db).is_integer():
        return f"{int(snr_db):02d}"
    return str(snr_db).replace("-", "m").replace(".", "p")


def write_config(out_dir: Path, cfg: MuMimoComNetConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    data = asdict(cfg)
    data["snr_test_db"] = list(cfg.snr_test_db)
    data["case"] = "linear"
    data["waveform_type"] = "post_combining_effective_siso"
    data["power_norm_mode"] = "total_frobenius_fixed"
    data["precoding"] = "downlink ZF from combiner-projected estimated channel"
    data["combiner"] = "dominant SVD combiner from true channel"
    data["channel_estimate"] = "H_tx_est = H_true + complex Gaussian CSIT error"
    data["receiver_compatibility"] = (
        "rx_p_time and rx_d_time are synthetic SISO waveforms reconstructed "
        "from post-combining effective frequency-domain symbols."
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[SAVE] {path}")
    return path


def generate_all(cfg: MuMimoComNetConfig, out_dir: Path) -> None:
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


def build_config(args: argparse.Namespace) -> MuMimoComNetConfig:
    snr_test_db: Sequence[float]
    if args.snr_test_db is None:
        snr_test_db = MuMimoComNetConfig.snr_test_db
    else:
        snr_test_db = tuple(float(x) for x in args.snr_test_db)

    return MuMimoComNetConfig(
        n_fft=int(args.n_fft),
        n_cp=int(args.n_cp),
        modulation=str(args.modulation).upper(),
        n_users=int(args.n_users),
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
