from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class OfdmDatasetConfig:
    n_fft: int = 64
    n_cp: int = 16
    modulation: str = "64QAM"
    n_taps: int = 8
    pdp_decay: float = 2.0
    snr_train_db: float = 40.0
    snr_test_db: tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40)
    n_train_frames: int = 50000
    n_val_frames: int = 10000
    n_test_frames_per_snr: int = 10000
    case: str = "clipping"
    clip_ratio: float = 1.6
    seed: int = 7

    def validate(self) -> None:
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.n_cp < 0:
            raise ValueError("n_cp must be non-negative")
        if self.n_taps <= 0:
            raise ValueError("n_taps must be positive")
        if self.n_taps > self.n_cp and self.case in {"linear", "clipping"}:
            raise ValueError(f"{self.case} OFDM requires n_taps <= n_cp")
        if self.pdp_decay <= 0:
            raise ValueError("pdp_decay must be positive")
        if self.modulation.upper() not in {"QPSK", "16QAM", "64QAM"}:
            raise ValueError("modulation must be one of QPSK, 16QAM, 64QAM")
        if self.case not in {"linear", "cp_removal", "clipping"}:
            raise ValueError("case must be one of linear, cp_removal, clipping")
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if min(self.n_train_frames, self.n_val_frames, self.n_test_frames_per_snr) < 0:
            raise ValueError("frame counts must be non-negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SISO OFDM datasets for a ComNet-style receiver."
    )
    parser.add_argument("--out-dir", type=str, default="outputs_comnet_64qam_clipping")
    parser.add_argument("--modulation", type=str, default="64QAM", choices=["QPSK", "16QAM", "64QAM"])
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
    parser.add_argument("--case", type=str, default="clipping", choices=["linear", "cp_removal", "clipping"])
    parser.add_argument("--clip-ratio", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def bits_per_symbol(modulation: str) -> int:
    table = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
    key = modulation.upper()
    if key not in table:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return table[key]


def _pam_levels(axis_bits: int) -> np.ndarray:
    if axis_bits == 1:
        return np.array([1.0, -1.0], dtype=np.float64)
    if axis_bits == 2:
        return np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float64)
    if axis_bits == 3:
        return np.array([7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0], dtype=np.float64)
    raise ValueError(f"Unsupported PAM axis bit count: {axis_bits}")


def _gray_labels(axis_bits: int) -> np.ndarray:
    labels = np.arange(2**axis_bits, dtype=np.int64)
    return labels ^ (labels >> 1)


def _ints_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((values.size, width), dtype=np.int8)
    for bit_pos in range(width):
        shift = width - 1 - bit_pos
        out[:, bit_pos] = ((values >> shift) & 1).astype(np.int8)
    return out


def _bits_to_ints(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    values = np.zeros(bits.shape[0], dtype=np.int64)
    for bit_pos in range(bits.shape[1]):
        values = (values << 1) | bits[:, bit_pos].astype(np.int64)
    return values


def _qam_normalization(modulation: str) -> float:
    bps = bits_per_symbol(modulation)
    if bps == 2:
        return math.sqrt(2.0)
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

    if modulation == "QPSK":
        b = bits.reshape(-1, 2)
        symbols = (1.0 - 2.0 * b[:, 0]) + 1j * (1.0 - 2.0 * b[:, 1])
        return (symbols / _qam_normalization(modulation)).astype(np.complex64)

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


def qam_demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    modulation = modulation.upper()
    symbols = np.asarray(symbols).reshape(-1)
    bps = bits_per_symbol(modulation)

    if modulation == "QPSK":
        s = symbols * _qam_normalization(modulation)
        bits = np.zeros((symbols.size, 2), dtype=np.int8)
        bits[:, 0] = (s.real < 0).astype(np.int8)
        bits[:, 1] = (s.imag < 0).astype(np.int8)
        return bits

    axis_bits = bps // 2
    labels = _gray_labels(axis_bits)
    levels = _pam_levels(axis_bits)
    s = symbols * _qam_normalization(modulation)
    re_index = np.argmin(np.abs(s.real[:, None] - levels[None, :]), axis=1)
    im_index = np.argmin(np.abs(s.imag[:, None] - levels[None, :]), axis=1)
    re_bits = _ints_to_bits(labels[re_index], axis_bits)
    im_bits = _ints_to_bits(labels[im_index], axis_bits)
    return np.concatenate([re_bits, im_bits], axis=1).astype(np.int8)


def make_pilot_symbols(n_frames: int, cfg: OfdmDatasetConfig) -> np.ndarray:
    return np.ones((n_frames, cfg.n_fft), dtype=np.complex64)


def apply_clipping(time_symbol: np.ndarray, clip_ratio: float) -> np.ndarray:
    rms = math.sqrt(float(np.mean(np.abs(time_symbol) ** 2)))
    threshold = float(clip_ratio) * max(rms, 1e-12)
    magnitude = np.abs(time_symbol)
    scale = np.ones_like(magnitude, dtype=np.float32)
    mask = magnitude > threshold
    scale[mask] = threshold / np.maximum(magnitude[mask], 1e-12)
    return (time_symbol * scale).astype(np.complex64)


def ofdm_modulate(freq_symbol: np.ndarray, cfg: OfdmDatasetConfig) -> np.ndarray:
    freq_symbol = np.asarray(freq_symbol, dtype=np.complex64)
    time_no_cp = np.fft.ifft(freq_symbol, n=cfg.n_fft, axis=-1) * math.sqrt(cfg.n_fft)
    if cfg.case == "clipping":
        time_no_cp = apply_clipping(time_no_cp, cfg.clip_ratio)
    if cfg.case == "cp_removal":
        return time_no_cp.astype(np.complex64)
    cp = time_no_cp[..., -cfg.n_cp :] if cfg.n_cp > 0 else time_no_cp[..., :0]
    return np.concatenate([cp, time_no_cp], axis=-1).astype(np.complex64)


def generate_multipath_channel(cfg: OfdmDatasetConfig, rng: np.random.Generator) -> np.ndarray:
    tap = np.arange(cfg.n_taps, dtype=np.float64)
    pdp = np.exp(-tap / cfg.pdp_decay)
    pdp /= np.sum(pdp)
    h = (
        rng.standard_normal(cfg.n_taps) + 1j * rng.standard_normal(cfg.n_taps)
    ) / math.sqrt(2.0)
    h *= np.sqrt(pdp)
    return h.astype(np.complex64)


def channel_frequency_response(h_time: np.ndarray, cfg: OfdmDatasetConfig) -> np.ndarray:
    h_pad = np.zeros(cfg.n_fft, dtype=np.complex64)
    h_pad[: len(h_time)] = h_time
    return np.fft.fft(h_pad, n=cfg.n_fft).astype(np.complex64)


def apply_multipath_channel(tx_time: np.ndarray, h_time: np.ndarray) -> np.ndarray:
    tx_time = np.asarray(tx_time, dtype=np.complex64).reshape(-1)
    h_time = np.asarray(h_time, dtype=np.complex64).reshape(-1)
    y_time = np.zeros_like(tx_time, dtype=np.complex64)
    for tap_index, tap_value in enumerate(h_time):
        y_time[tap_index:] += tap_value * tx_time[: tx_time.size - tap_index]
    return y_time


def add_awgn_by_snr(
    rx_clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if math.isinf(float(snr_db)):
        return rx_clean.astype(np.complex64)
    signal_power = float(np.mean(np.abs(rx_clean) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-300)
    noise = (
        rng.standard_normal(rx_clean.shape) + 1j * rng.standard_normal(rx_clean.shape)
    ) * math.sqrt(noise_power / 2.0)
    return (rx_clean + noise).astype(np.complex64)


def _make_split_dataset(
    *,
    cfg: OfdmDatasetConfig,
    n_frames: int,
    snr_db: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    bps = bits_per_symbol(cfg.modulation)
    rx_time_len = cfg.n_fft if cfg.case == "cp_removal" else cfg.n_fft + cfg.n_cp
    rx_p_time = np.zeros((n_frames, rx_time_len), dtype=np.complex64)
    rx_d_time = np.zeros_like(rx_p_time)
    x_p_freq = make_pilot_symbols(n_frames, cfg)
    x_d_freq = np.zeros((n_frames, cfg.n_fft), dtype=np.complex64)
    h_true = np.zeros((n_frames, cfg.n_fft), dtype=np.complex64)
    bits = np.zeros((n_frames, cfg.n_fft, bps), dtype=np.int8)
    snr = np.full(n_frames, float(snr_db), dtype=np.float32)

    for frame_index in range(n_frames):
        frame_bits = rng.integers(0, 2, size=(cfg.n_fft, bps), dtype=np.int8)
        x_data = qam_modulate(frame_bits.reshape(-1), cfg.modulation).reshape(cfg.n_fft)
        h_time = generate_multipath_channel(cfg, rng)

        tx_p = ofdm_modulate(x_p_freq[frame_index], cfg)
        tx_d = ofdm_modulate(x_data, cfg)
        rx_p_clean = apply_multipath_channel(tx_p, h_time)
        rx_d_clean = apply_multipath_channel(tx_d, h_time)

        rx_p_time[frame_index] = add_awgn_by_snr(rx_p_clean, snr_db, rng)
        rx_d_time[frame_index] = add_awgn_by_snr(rx_d_clean, snr_db, rng)
        x_d_freq[frame_index] = x_data
        h_true[frame_index] = channel_frequency_response(h_time, cfg)
        bits[frame_index] = frame_bits

    return {
        "rx_p_time": rx_p_time,
        "rx_d_time": rx_d_time,
        "x_p_freq": x_p_freq,
        "x_d_freq": x_d_freq,
        "h_true": h_true,
        "bits": bits,
        "snr_db": snr,
    }


def _save_npz(path: Path, dataset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    n_frames = int(dataset["bits"].shape[0])
    print(
        f"[SAVE] {path} frames={n_frames}, "
        f"rx_time={dataset['rx_p_time'].shape}, h_true={dataset['h_true'].shape}, "
        f"bits={dataset['bits'].shape}"
    )


def _snr_name(snr_db: float) -> str:
    if float(snr_db).is_integer():
        return f"{int(snr_db):02d}"
    return str(snr_db).replace("-", "m").replace(".", "p")


def write_config(out_dir: Path, cfg: OfdmDatasetConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    data = asdict(cfg)
    data["snr_test_db"] = list(cfg.snr_test_db)
    data["nonlinear_processing"] = {
        "linear": "normal OFDM with cyclic prefix",
        "cp_removal": "transmitter omits cyclic prefix; receiver FFT starts at sample 0",
        "clipping": "time-domain OFDM symbol is clipped before cyclic prefix insertion",
    }[cfg.case]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[SAVE] {path}")
    return path


def generate_all(cfg: OfdmDatasetConfig, out_dir: Path) -> None:
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


def build_config(args: argparse.Namespace) -> OfdmDatasetConfig:
    snr_test_db: Sequence[float]
    if args.snr_test_db is None:
        snr_test_db = OfdmDatasetConfig.snr_test_db
    else:
        snr_test_db = tuple(float(x) for x in args.snr_test_db)

    return OfdmDatasetConfig(
        n_fft=int(args.n_fft),
        n_cp=int(args.n_cp),
        modulation=str(args.modulation).upper(),
        n_taps=int(args.n_taps),
        pdp_decay=float(args.pdp_decay),
        snr_train_db=float(args.snr_train_db),
        snr_test_db=tuple(snr_test_db),
        n_train_frames=int(args.n_train_frames),
        n_val_frames=int(args.n_val_frames),
        n_test_frames_per_snr=int(args.n_test_frames_per_snr),
        case=str(args.case),
        clip_ratio=float(args.clip_ratio),
        seed=int(args.seed),
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    generate_all(cfg, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
