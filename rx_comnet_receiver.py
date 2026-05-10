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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a SISO OFDM ComNet-style receiver."
    )
    parser.add_argument("--dataset-dir", type=str, default="outputs_comnet_64qam_clipping")
    parser.add_argument(
        "--mode",
        type=str,
        default="train-all",
        choices=["train-all", "train-ce", "train-sd", "eval"],
    )
    parser.add_argument("--result-dir", type=str, default="results_comnet_64qam_clipping_practical")
    parser.add_argument("--ce-checkpoint", type=str, default=None)
    parser.add_argument("--sd-checkpoint", type=str, default=None,
                        help="Backward-compatible alias for --fc-checkpoint")
    parser.add_argument("--fc-checkpoint", type=str, default=None)
    parser.add_argument("--bilstm-checkpoint", type=str, default=None)
    parser.add_argument("--lmmse-checkpoint", type=str, default=None)
    parser.add_argument("--ce-init", type=str, default="lmmse", choices=["identity", "lmmse"])
    parser.add_argument("--sd-type", type=str, default="both", choices=["fc", "bilstm", "both"])
    parser.add_argument("--sd-loss", type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument("--ce-epochs", type=int, default=200)
    parser.add_argument("--sd-epochs", type=int, default=500)
    parser.add_argument("--bilstm-epochs", type=int, default=None,
                        help="Defaults to --sd-epochs when omitted")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--ce-lr", type=float, default=1e-3)
    parser.add_argument("--sd-lr", type=float, default=1e-3)
    parser.add_argument("--bilstm-lr", type=float, default=None,
                        help="Defaults to --sd-lr when omitted")
    parser.add_argument("--ce-lr-step", type=int, default=100)
    parser.add_argument("--sd-lr-step", type=int, default=200)
    parser.add_argument("--ce-lr-gamma", type=float, default=0.1)
    parser.add_argument("--sd-lr-gamma", type=float, default=0.2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lmmse-ridge", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
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


def _qam_normalization(modulation: str) -> float:
    bps = bits_per_symbol(modulation)
    if bps == 2:
        return math.sqrt(2.0)
    if bps == 4:
        return math.sqrt(10.0)
    if bps == 6:
        return math.sqrt(42.0)
    raise ValueError(f"Unsupported modulation: {modulation}")


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


class LSRefineNet(nn.Module):
    def __init__(self, n_fft: int) -> None:
        super().__init__()
        dim = 2 * int(n_fft)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_identity(self) -> None:
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(self.linear.weight.shape[0]))


class FCSDNet(nn.Module):
    def __init__(self, group_size: int, bits_per_symbol_value: int) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.bits_per_symbol = int(bits_per_symbol_value)
        self.net = nn.Sequential(
            nn.Linear(2 * self.group_size, 120),
            nn.ReLU(),
            nn.Linear(120, self.group_size * self.bits_per_symbol),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiLSTMSDNet(nn.Module):
    def __init__(self, n_fft: int, bits_per_symbol_value: int, group_size: int) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.bits_per_symbol = int(bits_per_symbol_value)
        self.group_size = int(group_size)
        if self.n_fft % self.group_size != 0:
            raise ValueError("n_fft must be divisible by group_size")
        self.n_groups = self.n_fft // self.group_size
        self.lstm1 = nn.LSTM(
            input_size=6,
            hidden_size=20,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=40,
            hidden_size=10,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm3 = nn.LSTM(
            input_size=20,
            hidden_size=6,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(12 * self.group_size, self.group_size * self.bits_per_symbol)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x.reshape(x.shape[0], self.n_groups, self.group_size * 12)
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
    n_fft = int(cfg["n_fft"])
    n_cp = int(cfg["n_cp"])
    rx_time = np.asarray(rx_time, dtype=np.complex64)
    if str(cfg.get("case", "linear")) == "cp_removal":
        no_cp = rx_time[:, :n_fft]
    else:
        no_cp = rx_time[:, n_cp : n_cp + n_fft]
    if no_cp.shape[1] != n_fft:
        raise ValueError(f"Expected {n_fft} FFT samples, got {no_cp.shape[1]}")
    return (np.fft.fft(no_cp, n=n_fft, axis=1) / math.sqrt(n_fft)).astype(np.complex64)


def apply_time_clipping(time_symbol: np.ndarray, clip_ratio: float) -> np.ndarray:
    time_symbol = np.asarray(time_symbol, dtype=np.complex64)
    rms = np.sqrt(np.mean(np.abs(time_symbol) ** 2, axis=1, keepdims=True))
    threshold = float(clip_ratio) * np.maximum(rms, 1e-12)
    magnitude = np.abs(time_symbol)
    scale = np.minimum(1.0, threshold / np.maximum(magnitude, 1e-12)).astype(np.float32)
    return (time_symbol * scale).astype(np.complex64)


def clipping_reference_freq(x_freq: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    n_fft = int(cfg["n_fft"])
    time_no_cp = np.fft.ifft(x_freq, n=n_fft, axis=1) * math.sqrt(n_fft)
    clipped_time = apply_time_clipping(time_no_cp, float(cfg.get("clip_ratio", 1.0)))
    return (np.fft.fft(clipped_time, n=n_fft, axis=1) / math.sqrt(n_fft)).astype(np.complex64)


def no_clip_true_h_reference(
    *,
    y_d: np.ndarray,
    x_d_freq: np.ndarray,
    h_true: np.ndarray,
    cfg: dict[str, Any],
    eps: float,
) -> np.ndarray | None:
    if str(cfg.get("case", "linear")) != "clipping":
        return None
    x_clipped_freq = clipping_reference_freq(x_d_freq, cfg)
    clipped_clean_y = h_true * x_clipped_freq
    same_noise = y_d - clipped_clean_y
    no_clip_y = h_true * x_d_freq + same_noise
    return zf_equalize(no_clip_y, h_true, eps)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=np.complex64)
    safe_den = np.where(np.abs(denominator) < eps, eps + 0j, denominator)
    return (numerator / safe_den).astype(np.complex64)


def preprocess_split(data: dict[str, np.ndarray], cfg: dict[str, Any], eps: float) -> dict[str, np.ndarray]:
    y_p = ofdm_demodulate(data["rx_p_time"], cfg)
    y_d = ofdm_demodulate(data["rx_d_time"], cfg)
    h_ls = safe_divide(y_p, data["x_p_freq"], eps)
    return {
        "y_p": y_p,
        "y_d": y_d,
        "h_ls": h_ls,
        "h_true": np.asarray(data["h_true"], dtype=np.complex64),
        "bits": np.asarray(data["bits"], dtype=np.int8),
        "snr_db": np.asarray(data["snr_db"], dtype=np.float32),
    }


def complex_to_ri(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex64)
    return np.concatenate([values.real, values.imag], axis=-1).astype(np.float32)


def ri_to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    half = values.shape[-1] // 2
    return (values[..., :half] + 1j * values[..., half:]).astype(np.complex64)


def hard_demod_frame(symbols: np.ndarray, modulation: str) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex64)
    bps = bits_per_symbol(modulation)
    return qam_demodulate(symbols.reshape(-1), modulation).reshape(symbols.shape[0], symbols.shape[1], bps)


def bit_error_rate(pred_bits: np.ndarray, true_bits: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_bits, dtype=np.int8) != np.asarray(true_bits, dtype=np.int8)))


def channel_mse(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    return float(np.mean(np.abs(h_hat - h_true) ** 2))


def channel_nmse(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(h_hat - h_true) ** 2))
    denominator = float(np.sum(np.abs(h_true) ** 2))
    return numerator / max(denominator, 1e-300)


def to_db(value: float) -> float:
    return float(10.0 * math.log10(max(float(value), 1e-300)))


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
    *,
    title: str,
    include_ber: bool,
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, training plot skipped: {exc}")
        return

    epochs = [row["epoch"] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    if include_ber:
        fig, (ax_loss, ax_ber) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_loss.plot(epochs, [row["train_loss"] for row in rows], linewidth=2.0, label="train loss")
        ax_loss.plot(epochs, [row["val_loss"] for row in rows], linewidth=2.0, label="validation loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle=":")
        ax_loss.legend()

        ax_ber.plot(epochs, [row["val_ber"] for row in rows], linewidth=2.0, label="validation BER")
        ax_ber.set_xlabel("Epoch")
        ax_ber.set_ylabel("BER")
        ax_ber.grid(True, linestyle=":")
        ax_ber.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["train_loss"] for row in rows], linewidth=2.0, label="train loss")
        plt.plot(epochs, [row["val_loss"] for row in rows], linewidth=2.0, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    print(f"[SAVE] {path}")


def fit_lmmse_weight(train_data: dict[str, np.ndarray], ridge: float) -> np.ndarray:
    x = complex_to_ri(train_data["h_ls"]).astype(np.float64)
    y = complex_to_ri(train_data["h_true"]).astype(np.float64)
    dim = x.shape[1]
    xtx = x.T @ x
    xty = x.T @ y
    scale = float(np.trace(xtx) / max(dim, 1))
    reg = max(float(ridge), 0.0) * max(scale, 1e-12)
    system = xtx + reg * np.eye(dim, dtype=np.float64)
    try:
        beta = np.linalg.solve(system, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(system) @ xty
    return beta.T.astype(np.float32)


def save_lmmse_weight(path: Path, weight_ri: np.ndarray, cfg: dict[str, Any], ridge: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weight_ri=np.asarray(weight_ri, dtype=np.float32),
        n_fft=np.asarray(int(cfg["n_fft"]), dtype=np.int64),
        modulation=np.asarray(str(cfg["modulation"])),
        ridge=np.asarray(float(ridge), dtype=np.float64),
    )
    print(f"[SAVE] {path}")


def load_lmmse_weight(path: Path) -> np.ndarray:
    with np.load(path) as data:
        weight = np.asarray(data["weight_ri"], dtype=np.float32)
    print(f"[LOAD] LMMSE estimator: {path}")
    return weight


def get_lmmse_weight(
    *,
    dataset_dir: Path,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> np.ndarray:
    if checkpoint_path.exists() and str(args.mode) == "eval":
        return load_lmmse_weight(checkpoint_path)
    train_path = find_one(dataset_dir, "train_snr*.npz")
    train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps))
    weight = fit_lmmse_weight(train_data, float(args.lmmse_ridge))
    save_lmmse_weight(checkpoint_path, weight, cfg, float(args.lmmse_ridge))
    return weight


def apply_lmmse_weight(h_ls: np.ndarray, weight_ri: np.ndarray) -> np.ndarray:
    x = complex_to_ri(h_ls)
    y = x @ np.asarray(weight_ri, dtype=np.float32).T
    return ri_to_complex(y)


def snr_to_noise_variance(h_hat: np.ndarray, snr_db: float | np.ndarray) -> np.ndarray:
    h_hat = np.asarray(h_hat, dtype=np.complex64)
    snr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if snr.size == 1:
        snr = np.full(h_hat.shape[0], float(snr[0]), dtype=np.float64)
    snr_linear = np.power(10.0, snr / 10.0)
    channel_power = np.mean(np.abs(h_hat) ** 2, axis=1)
    noise_var = np.zeros(h_hat.shape[0], dtype=np.float64)
    finite = np.isfinite(snr_linear)
    noise_var[finite] = channel_power[finite] / np.maximum(snr_linear[finite], 1e-300)
    return noise_var.reshape(-1, 1).astype(np.float32)


def zf_equalize(y_d: np.ndarray, h_hat: np.ndarray, eps: float) -> np.ndarray:
    return safe_divide(y_d, h_hat, eps)


def mmse_equalize(y_d: np.ndarray, h_hat: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    h_hat = np.asarray(h_hat, dtype=np.complex64)
    denom = np.abs(h_hat) ** 2 + np.asarray(noise_var, dtype=np.float32)
    return (np.conj(h_hat) * y_d / np.maximum(denom, 1e-30)).astype(np.complex64)


def train_ce(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    lmmse_weight: np.ndarray | None = None,
) -> LSRefineNet:
    n_fft = int(cfg["n_fft"])
    model = LSRefineNet(n_fft).to(device)
    if args.ce_init == "identity":
        model.init_identity()
    elif args.ce_init == "lmmse":
        if lmmse_weight is None:
            raise ValueError("lmmse_weight is required for --ce-init lmmse")
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(np.asarray(lmmse_weight, dtype=np.float32)).to(device))

    x_train = complex_to_ri(train_data["h_ls"])
    y_train = complex_to_ri(train_data["h_true"])
    x_val = torch.from_numpy(complex_to_ri(val_data["h_ls"])).to(device)
    y_val = torch.from_numpy(complex_to_ri(val_data["h_true"])).to(device)

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
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
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
        ["epoch", "train_loss", "val_loss", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "ce_training_curve.png",
        history,
        title="CE Subnet Training",
        include_ber=False,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_fft": n_fft,
            "modulation": str(cfg["modulation"]),
            "ce_init": str(args.ce_init),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path}")
    return model


def load_ce_model(path: Path, cfg: dict[str, Any], device: torch.device) -> LSRefineNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    n_fft = int(checkpoint.get("n_fft", cfg["n_fft"]))
    model = LSRefineNet(n_fft).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] CE checkpoint: {path}")
    return model


def predict_ce(
    model: LSRefineNet,
    h_ls: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x = torch.from_numpy(complex_to_ri(h_ls))
    loader = DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).cpu().numpy()
            chunks.append(pred)
    return ri_to_complex(np.concatenate(chunks, axis=0))


def make_sd_arrays(
    *,
    y_d: np.ndarray,
    h_hat: np.ndarray,
    bits: np.ndarray,
    group_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_fft = y_d.shape
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")
    bps = bits.shape[-1]
    x_zf = safe_divide(y_d, h_hat, eps)
    n_groups = n_fft // group_size
    x_groups = x_zf.reshape(n_frames, n_groups, group_size)
    x_ri = np.concatenate([x_groups.real, x_groups.imag], axis=-1).astype(np.float32)
    bit_groups = bits.reshape(n_frames, n_groups, group_size, bps).reshape(
        n_frames,
        n_groups,
        group_size * bps,
    )
    return x_ri.reshape(n_frames * n_groups, 2 * group_size), bit_groups.reshape(
        n_frames * n_groups,
        group_size * bps,
    ).astype(np.float32)


def make_bilstm_arrays(
    *,
    y_d: np.ndarray,
    h_hat: np.ndarray,
    bits: np.ndarray,
    group_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_zf = zf_equalize(y_d, h_hat, eps)
    n_frames, n_fft = y_d.shape
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")
    bps = bits.shape[-1]
    n_groups = n_fft // group_size
    features = np.stack(
        [
            y_d.real,
            y_d.imag,
            h_hat.real,
            h_hat.imag,
            x_zf.real,
            x_zf.imag,
        ],
        axis=-1,
    ).astype(np.float32)
    bit_groups = bits.reshape(n_frames, n_groups, group_size, bps).reshape(
        n_frames,
        n_groups,
        group_size * bps,
    )
    return features, bit_groups.astype(np.float32)


def sd_loss_value(logits: torch.Tensor, target: torch.Tensor, sd_loss: str) -> torch.Tensor:
    if sd_loss == "mse":
        return nn.functional.mse_loss(torch.sigmoid(logits), target)
    if sd_loss == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(f"Unsupported SD loss: {sd_loss}")


def train_sd(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    ce_model: LSRefineNet,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> FCSDNet:
    group_size = int(args.group_size)
    bps = bits_per_symbol(str(cfg["modulation"]))
    h_train = predict_ce(ce_model, train_data["h_ls"], device=device, batch_size=int(args.batch_size))
    h_val = predict_ce(ce_model, val_data["h_ls"], device=device, batch_size=int(args.batch_size))
    x_train, y_train = make_sd_arrays(
        y_d=train_data["y_d"],
        h_hat=h_train,
        bits=train_data["bits"],
        group_size=group_size,
        eps=float(args.eps),
    )
    x_val_np, y_val_np = make_sd_arrays(
        y_d=val_data["y_d"],
        h_hat=h_val,
        bits=val_data["bits"],
        group_size=group_size,
        eps=float(args.eps),
    )

    model = FCSDNet(group_size, bps).to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 17)
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
        step_size=max(1, int(args.sd_lr_step)),
        gamma=float(args.sd_lr_gamma),
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.sd_epochs) + 1):
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
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_ber": float(val_ber),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if should_log(epoch, int(args.sd_epochs), int(args.log_every)):
            print(
                f"[SD {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, val_BER={val_ber:.4e}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / "train_history_sd.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_ber", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "fc_sd_training_curve.png",
        history,
        title="FC-SD Training",
        include_ber=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "group_size": group_size,
            "bits_per_symbol": bps,
            "modulation": str(cfg["modulation"]),
            "sd_loss": str(args.sd_loss),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path}")
    return model


def train_bilstm_sd(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    ce_model: LSRefineNet,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> BiLSTMSDNet:
    n_fft = int(cfg["n_fft"])
    bps = bits_per_symbol(str(cfg["modulation"]))
    group_size = int(args.group_size)
    h_train = predict_ce(ce_model, train_data["h_ls"], device=device, batch_size=int(args.batch_size))
    h_val = predict_ce(ce_model, val_data["h_ls"], device=device, batch_size=int(args.batch_size))
    x_train, y_train = make_bilstm_arrays(
        y_d=train_data["y_d"],
        h_hat=h_train,
        bits=train_data["bits"],
        group_size=group_size,
        eps=float(args.eps),
    )
    x_val_np, y_val_np = make_bilstm_arrays(
        y_d=val_data["y_d"],
        h_hat=h_val,
        bits=val_data["bits"],
        group_size=group_size,
        eps=float(args.eps),
    )

    model = BiLSTMSDNet(n_fft, bps, group_size).to(device)
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
    lr = float(args.bilstm_lr) if args.bilstm_lr is not None else float(args.sd_lr)
    n_epochs = int(args.bilstm_epochs) if args.bilstm_epochs is not None else int(args.sd_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(args.sd_lr_step)),
        gamma=float(args.sd_lr_gamma),
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, n_epochs + 1):
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
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_ber": float(val_ber),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if should_log(epoch, n_epochs, int(args.log_every)):
            print(
                f"[BiLSTM-SD {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, val_BER={val_ber:.4e}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / "train_history_bilstm_sd.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_ber", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "bilstm_sd_training_curve.png",
        history,
        title="BiLSTM-SD Training",
        include_ber=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_fft": n_fft,
            "group_size": group_size,
            "bits_per_symbol": bps,
            "modulation": str(cfg["modulation"]),
            "sd_loss": str(args.sd_loss),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path}")
    return model


def load_sd_model(path: Path, cfg: dict[str, Any], group_size: int, device: torch.device) -> FCSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_group_size = int(checkpoint.get("group_size", group_size))
    bps = int(checkpoint.get("bits_per_symbol", bits_per_symbol(str(cfg["modulation"]))))
    model = FCSDNet(model_group_size, bps).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] SD checkpoint: {path}")
    return model


def load_bilstm_sd_model(
    path: Path,
    cfg: dict[str, Any],
    group_size: int,
    device: torch.device,
) -> BiLSTMSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    n_fft = int(checkpoint.get("n_fft", cfg["n_fft"]))
    bps = int(checkpoint.get("bits_per_symbol", bits_per_symbol(str(cfg["modulation"]))))
    model_group_size = int(checkpoint.get("group_size", group_size))
    model = BiLSTMSDNet(n_fft, bps, model_group_size).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] BiLSTM-SD checkpoint: {path}")
    return model


def predict_sd_bits(
    model: FCSDNet,
    *,
    y_d: np.ndarray,
    h_hat: np.ndarray,
    true_bits_shape: tuple[int, int, int],
    group_size: int,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n_frames, n_fft, bps = true_bits_shape
    dummy_bits = np.zeros(true_bits_shape, dtype=np.int8)
    x_groups, _ = make_sd_arrays(
        y_d=y_d,
        h_hat=h_hat,
        bits=dummy_bits,
        group_size=group_size,
        eps=eps,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_groups)),
        batch_size=int(batch_size),
        shuffle=False,
    )
    pred_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            pred_chunks.append((prob > 0.5).astype(np.int8))
    pred_flat = np.concatenate(pred_chunks, axis=0)
    n_groups = n_fft // group_size
    return pred_flat.reshape(n_frames, n_groups, group_size, bps).reshape(n_frames, n_fft, bps)


def predict_bilstm_bits(
    model: BiLSTMSDNet,
    *,
    y_d: np.ndarray,
    h_hat: np.ndarray,
    true_bits_shape: tuple[int, int, int],
    group_size: int,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dummy_bits = np.zeros(true_bits_shape, dtype=np.int8)
    x_seq, _ = make_bilstm_arrays(
        y_d=y_d,
        h_hat=h_hat,
        bits=dummy_bits,
        group_size=group_size,
        eps=eps,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_seq)),
        batch_size=int(batch_size),
        shuffle=False,
    )
    pred_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            pred_chunks.append((prob > 0.5).astype(np.int8))
    pred_groups = np.concatenate(pred_chunks, axis=0)
    n_frames, n_fft, bps = true_bits_shape
    n_groups = n_fft // group_size
    return pred_groups.reshape(n_frames, n_groups, group_size, bps).reshape(n_frames, n_fft, bps)


def evaluate_one(
    *,
    path: Path,
    cfg: dict[str, Any],
    ce_model: LSRefineNet | None,
    fc_model: FCSDNet | None,
    bilstm_model: BiLSTMSDNet | None,
    lmmse_weight: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    raw = load_npz(path)
    data = preprocess_split(raw, cfg, float(args.eps))
    snr = float(np.mean(data["snr_db"]))
    h_ls = data["h_ls"]
    h_true = data["h_true"]
    y_d = data["y_d"]
    bits = data["bits"]
    modulation = str(cfg["modulation"])
    x_d_freq = np.asarray(raw["x_d_freq"], dtype=np.complex64)
    h_lmmse = apply_lmmse_weight(h_ls, lmmse_weight)

    if ce_model is None:
        h_comnet = h_ls
    else:
        h_comnet = predict_ce(
            ce_model,
            h_ls,
            device=device,
            batch_size=int(args.batch_size),
        )

    noise_var = snr_to_noise_variance(h_lmmse, data["snr_db"])
    x_ls = zf_equalize(y_d, h_ls, float(args.eps))
    x_lmmse_zf = zf_equalize(y_d, h_lmmse, float(args.eps))
    x_lmmse_mmse = mmse_equalize(y_d, h_lmmse, noise_var)
    x_comnet = zf_equalize(y_d, h_comnet, float(args.eps))
    x_true = zf_equalize(y_d, h_true, float(args.eps))
    x_no_clip_ref = no_clip_true_h_reference(
        y_d=y_d,
        x_d_freq=x_d_freq,
        h_true=h_true,
        cfg=cfg,
        eps=float(args.eps),
    )

    ber: dict[str, float] = {
        "LS-ZF": bit_error_rate(hard_demod_frame(x_ls, modulation), bits),
        "LMMSE-ZF": bit_error_rate(hard_demod_frame(x_lmmse_zf, modulation), bits),
        "LMMSE-MMSE": bit_error_rate(hard_demod_frame(x_lmmse_mmse, modulation), bits),
        "ComNet-CE-Hard": bit_error_rate(hard_demod_frame(x_comnet, modulation), bits),
    }
    if fc_model is not None:
        pred_sd = predict_sd_bits(
            fc_model,
            y_d=y_d,
            h_hat=h_comnet,
            true_bits_shape=bits.shape,
            group_size=int(args.group_size),
            eps=float(args.eps),
            device=device,
            batch_size=int(args.batch_size),
        )
        ber["ComNet-FC"] = bit_error_rate(pred_sd, bits)
    if bilstm_model is not None:
        pred_bilstm = predict_bilstm_bits(
            bilstm_model,
            y_d=y_d,
            h_hat=h_comnet,
            true_bits_shape=bits.shape,
            group_size=int(args.group_size),
            eps=float(args.eps),
            device=device,
            batch_size=int(args.batch_size),
        )
        ber["ComNet-BiLSTM"] = bit_error_rate(pred_bilstm, bits)
    ber["True-H ZF-Hard"] = bit_error_rate(hard_demod_frame(x_true, modulation), bits)
    if x_no_clip_ref is not None:
        ber["No-Clip True-H Ref"] = bit_error_rate(hard_demod_frame(x_no_clip_ref, modulation), bits)

    ce_mse = {
        "LS": channel_mse(h_ls, h_true),
        "LMMSE": channel_mse(h_lmmse, h_true),
        "ComNet-CE": channel_mse(h_comnet, h_true),
    }
    ce_nmse = {
        "LS": channel_nmse(h_ls, h_true),
        "LMMSE": channel_nmse(h_lmmse, h_true),
        "ComNet-CE": channel_nmse(h_comnet, h_true),
    }
    print(
        f"[EVAL] {path.name} SNR={snr:g} "
        f"LS-ZF={ber['LS-ZF']:.4e}, "
        f"LMMSE-ZF={ber['LMMSE-ZF']:.4e}, "
        f"LMMSE-MMSE={ber['LMMSE-MMSE']:.4e}, "
        f"ComNet-CE-Hard={ber['ComNet-CE-Hard']:.4e}, "
        + (f"ComNet-FC={ber['ComNet-FC']:.4e}, " if "ComNet-FC" in ber else "")
        + (f"ComNet-BiLSTM={ber['ComNet-BiLSTM']:.4e}, " if "ComNet-BiLSTM" in ber else "")
        + f"True-H ZF-Hard={ber['True-H ZF-Hard']:.4e}, "
        + (f"No-Clip True-H Ref={ber['No-Clip True-H Ref']:.4e}, " if "No-Clip True-H Ref" in ber else "")
        + f"CE_MSE_LS={to_db(ce_mse['LS']):.2f}dB, "
        f"CE_MSE_LMMSE={to_db(ce_mse['LMMSE']):.2f}dB, "
        f"CE_MSE_ComNet={to_db(ce_mse['ComNet-CE']):.2f}dB"
    )
    return {
        "snr": snr,
        "ce_mse": ce_mse,
        "ce_mse_db": {key: to_db(value) for key, value in ce_mse.items()},
        "ce_nmse": ce_nmse,
        "ce_nmse_db": {key: to_db(value) for key, value in ce_nmse.items()},
        "ber": ber,
    }


def save_eval_summary(result_dir: Path, cfg: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": cfg,
        "ce_mse_db": {},
        "ce_nmse_db": {},
        "ber": {},
    }
    for item in sorted(results, key=lambda x: x["snr"]):
        snr_key = f"{item['snr']:g}"
        summary["ce_mse_db"][snr_key] = item["ce_mse_db"]
        summary["ce_nmse_db"][snr_key] = item["ce_nmse_db"]
        summary["ber"][snr_key] = item["ber"]

    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "eval_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"[SAVE] {path}")
    return summary


def save_metric_csv(path: Path, summary: dict[str, Any], section: str) -> None:
    snrs = sorted((float(x) for x in summary[section].keys()))
    metric_names = sorted({name for snr in summary[section].values() for name in snr.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", *metric_names])
        for snr in snrs:
            key = f"{snr:g}"
            writer.writerow([snr, *[summary[section][key].get(name, "") for name in metric_names]])
    print(f"[SAVE] {path}")


def ordered_metric_names(names: Iterable[str], preferred: list[str]) -> list[str]:
    unique = list(dict.fromkeys(names))
    ordered = [name for name in preferred if name in unique]
    ordered.extend(sorted(name for name in unique if name not in ordered))
    return ordered


def normalize_legacy_summary_names(summary: dict[str, Any]) -> dict[str, Any]:
    """Migrate older result files that used an over-strong oracle label."""
    for ber_by_snr in summary.get("ber", {}).values():
        if "True-H Oracle" in ber_by_snr and "True-H ZF-Hard" not in ber_by_snr:
            ber_by_snr["True-H ZF-Hard"] = ber_by_snr.pop("True-H Oracle")
        ber_by_snr.pop("Genie-XD Bound", None)
    return summary


def save_eval_plots(result_dir: Path, summary: dict[str, Any]) -> None:
    summary = normalize_legacy_summary_names(summary)
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, saving CSV plot data instead: {exc}")
        save_metric_csv(result_dir / "ce_mse_vs_snr.csv", summary, "ce_mse_db")
        save_metric_csv(result_dir / "ber_vs_snr.csv", summary, "ber")
        return

    snrs = sorted(float(x) for x in summary["ce_mse_db"].keys())
    ce_names = ordered_metric_names(
        {name for item in summary["ce_mse_db"].values() for name in item.keys()},
        ["LS", "LMMSE", "ComNet-CE"],
    )
    plt.figure(figsize=(8, 5))
    for name in ce_names:
        values = [summary["ce_mse_db"][f"{snr:g}"][name] for snr in snrs]
        ce_style = {
            "LS": {"marker": "o", "linestyle": "-"},
            "LMMSE": {"marker": "s", "linestyle": "--"},
            "ComNet-CE": {"marker": "^", "linestyle": "-."},
        }.get(name, {"marker": "o", "linestyle": "-"})
        plt.plot(snrs, values, linewidth=2.0, label=name, **ce_style)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Channel MSE (dB)")
    plt.title("ComNet CE MSE vs SNR")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    ce_path = result_dir / "ce_mse_vs_snr.png"
    plt.savefig(ce_path, dpi=160)
    plt.close()
    print(f"[SAVE] {ce_path}")

    ber_names = ordered_metric_names(
        {name for item in summary["ber"].values() for name in item.keys()},
        [
            "LS-ZF",
            "LMMSE-ZF",
            "LMMSE-MMSE",
            "ComNet-CE-Hard",
            "ComNet-FC",
            "ComNet-BiLSTM",
            "True-H ZF-Hard",
            "No-Clip True-H Ref",
        ],
    )
    plt.figure(figsize=(8, 5))
    ber_styles = {
        "LS-ZF": {"marker": "o", "linestyle": "-"},
        "LMMSE-ZF": {"marker": "s", "linestyle": "--"},
        "LMMSE-MMSE": {"marker": "D", "linestyle": ":"},
        "ComNet-CE-Hard": {"marker": "^", "linestyle": "-."},
        "ComNet-FC": {"marker": "v", "linestyle": "-"},
        "ComNet-BiLSTM": {"marker": "P", "linestyle": "--"},
        "True-H ZF-Hard": {"marker": "*", "linestyle": "-", "linewidth": 2.6},
        "No-Clip True-H Ref": {"marker": "X", "linestyle": "-.", "linewidth": 2.2},
    }
    for name in ber_names:
        values = [max(float(summary["ber"][f"{snr:g}"].get(name, np.nan)), 1e-7) for snr in snrs]
        style = dict(ber_styles.get(name, {"marker": "o", "linestyle": "-"}))
        linewidth = style.pop("linewidth", 2.0)
        plt.semilogy(snrs, values, linewidth=linewidth, label=name, **style)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("ComNet BER vs SNR")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    ber_path = result_dir / "ber_vs_snr.png"
    plt.savefig(ber_path, dpi=160)
    plt.close()
    print(f"[SAVE] {ber_path}")


def evaluate_all(
    *,
    dataset_dir: Path,
    cfg: dict[str, Any],
    ce_model: LSRefineNet | None,
    fc_model: FCSDNet | None,
    bilstm_model: BiLSTMSDNet | None,
    lmmse_weight: np.ndarray,
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
            fc_model=fc_model,
            bilstm_model=bilstm_model,
            lmmse_weight=lmmse_weight,
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
    ce_checkpoint = Path(args.ce_checkpoint) if args.ce_checkpoint else result_dir / "ce_refinenet.pt"
    fc_checkpoint = (
        Path(args.fc_checkpoint)
        if args.fc_checkpoint
        else Path(args.sd_checkpoint)
        if args.sd_checkpoint
        else result_dir / "zf_refinenet_fc.pt"
    )
    bilstm_checkpoint = (
        Path(args.bilstm_checkpoint)
        if args.bilstm_checkpoint
        else result_dir / "zf_refinenet_bilstm.pt"
    )
    lmmse_checkpoint = (
        Path(args.lmmse_checkpoint)
        if args.lmmse_checkpoint
        else result_dir / "lmmse_estimator.npz"
    )
    cfg = load_config(dataset_dir)
    device = resolve_device(str(args.device))
    print(f"[DEVICE] {device}")
    print(
        f"[CONFIG] modulation={cfg['modulation']}, n_fft={cfg['n_fft']}, "
        f"n_cp={cfg['n_cp']}, group_size={args.group_size}, sd_type={args.sd_type}"
    )

    ce_model: LSRefineNet | None = None
    fc_model: FCSDNet | None = None
    bilstm_model: BiLSTMSDNet | None = None
    lmmse_weight = get_lmmse_weight(
        dataset_dir=dataset_dir,
        cfg=cfg,
        args=args,
        checkpoint_path=lmmse_checkpoint,
    )

    if args.mode in {"train-all", "train-ce"}:
        train_path = find_one(dataset_dir, "train_snr*.npz")
        val_path = find_one(dataset_dir, "val_snr*.npz")
        train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps))
        val_data = preprocess_split(load_npz(val_path), cfg, float(args.eps))
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
        if not ce_checkpoint.exists():
            raise FileNotFoundError(f"CE checkpoint not found: {ce_checkpoint}")
        ce_model = load_ce_model(ce_checkpoint, cfg, device)

    if args.mode in {"train-all", "train-sd"}:
        if ce_model is None:
            raise RuntimeError("CE model is required before SD training")
        train_path = find_one(dataset_dir, "train_snr*.npz")
        val_path = find_one(dataset_dir, "val_snr*.npz")
        train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps))
        val_data = preprocess_split(load_npz(val_path), cfg, float(args.eps))
        if args.sd_type in {"fc", "both"}:
            fc_model = train_sd(
                cfg=cfg,
                train_data=train_data,
                val_data=val_data,
                ce_model=ce_model,
                args=args,
                device=device,
                checkpoint_path=fc_checkpoint,
            )
        if args.sd_type in {"bilstm", "both"}:
            bilstm_model = train_bilstm_sd(
                cfg=cfg,
                train_data=train_data,
                val_data=val_data,
                ce_model=ce_model,
                args=args,
                device=device,
                checkpoint_path=bilstm_checkpoint,
            )

    if args.mode == "eval":
        if args.sd_type in {"fc", "both"} and fc_checkpoint.exists():
            fc_model = load_sd_model(fc_checkpoint, cfg, int(args.group_size), device)
        elif args.sd_type in {"fc", "both"}:
            print(f"[WARN] FC-SD checkpoint not found, ComNet-FC will be skipped: {fc_checkpoint}")
        if args.sd_type in {"bilstm", "both"} and bilstm_checkpoint.exists():
            bilstm_model = load_bilstm_sd_model(bilstm_checkpoint, cfg, int(args.group_size), device)
        elif args.sd_type in {"bilstm", "both"}:
            print(f"[WARN] BiLSTM-SD checkpoint not found, ComNet-BiLSTM will be skipped: {bilstm_checkpoint}")

    evaluate_all(
        dataset_dir=dataset_dir,
        cfg=cfg,
        ce_model=ce_model,
        fc_model=fc_model,
        bilstm_model=bilstm_model,
        lmmse_weight=lmmse_weight,
        args=args,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
