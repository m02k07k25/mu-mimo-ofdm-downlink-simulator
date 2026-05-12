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
        description="Train and evaluate a raw end-to-end MU-MIMO ComNet OFDM receiver."
    )
    parser.add_argument("--dataset-dir", type=str, default="outputs_mumimo_e2e_16qam_smoke")
    parser.add_argument("--result-dir", type=str, default="results_mumimo_e2e_16qam_smoke")
    parser.add_argument(
        "--mode",
        type=str,
        default="train-all",
        choices=["train-all", "train-ce", "train-sd", "eval"],
    )
    parser.add_argument("--sd-type", type=str, default="both", choices=["fc", "bilstm", "both"])
    parser.add_argument("--sd-loss", type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument("--ce-init", type=str, default="lmmse", choices=["identity", "lmmse"])
    parser.add_argument("--ce-checkpoint", type=str, default=None)
    parser.add_argument("--fc-checkpoint", type=str, default=None)
    parser.add_argument("--bilstm-checkpoint", type=str, default=None)
    parser.add_argument("--lmmse-checkpoint", type=str, default=None)
    parser.add_argument("--ce-epochs", type=int, default=50)
    parser.add_argument("--sd-epochs", type=int, default=50)
    parser.add_argument("--bilstm-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ce-lr", type=float, default=1e-3)
    parser.add_argument("--sd-lr", type=float, default=1e-3)
    parser.add_argument("--bilstm-lr", type=float, default=None)
    parser.add_argument("--ce-lr-step", type=int, default=25)
    parser.add_argument("--sd-lr-step", type=int, default=25)
    parser.add_argument("--ce-lr-gamma", type=float, default=0.5)
    parser.add_argument("--sd-lr-gamma", type=float, default=0.5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
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


class MuMimoCERefineNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.linear = nn.Linear(self.input_dim, self.input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_identity(self) -> None:
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(self.input_dim, device=self.linear.weight.device))


class MuMimoFCSDNet(nn.Module):
    def __init__(self, group_size: int, bits_per_symbol_value: int, hidden_dim: int) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.bits_per_symbol = int(bits_per_symbol_value)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * self.group_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.group_size * self.bits_per_symbol),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MuMimoBiLSTMSDNet(nn.Module):
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
        no_cp = rx_time[..., :n_fft]
    else:
        no_cp = rx_time[..., n_cp : n_cp + n_fft]
    if no_cp.shape[-1] != n_fft:
        raise ValueError(f"Expected {n_fft} FFT samples, got {no_cp.shape[-1]}")
    return (np.fft.fft(no_cp, n=n_fft, axis=-1) / math.sqrt(n_fft)).astype(np.complex64)


def preprocess_split(raw: dict[str, np.ndarray], cfg: dict[str, Any], eps: float) -> dict[str, np.ndarray]:
    y_p_slots = ofdm_demodulate(raw["rx_p_time"], cfg)
    y_d_urk = ofdm_demodulate(raw["rx_d_time"], cfg)
    x_p = np.asarray(raw["x_p_freq"], dtype=np.complex64)
    n_frames, n_streams, n_users, n_rx, n_fft = y_p_slots.shape

    a_ls = np.zeros((n_frames, n_fft, n_users, n_rx, n_streams), dtype=np.complex64)
    for stream_id in range(n_streams):
        denom = x_p[:, stream_id, stream_id, :]
        safe_den = np.where(np.abs(denom) < eps, eps + 0j, denom)
        y_slot = np.transpose(y_p_slots[:, stream_id], (0, 3, 1, 2))
        a_ls[..., stream_id] = y_slot / safe_den[:, :, None, None]

    return {
        "y_p": y_p_slots,
        "y_d": np.transpose(y_d_urk, (0, 3, 1, 2)).astype(np.complex64),
        "a_ls": a_ls,
        "a_true": np.asarray(raw["A_eff_true"], dtype=np.complex64),
        "bits": np.asarray(raw["bits"], dtype=np.int8),
        "x_d_freq": np.asarray(raw["x_d_freq"], dtype=np.complex64),
        "snr_db": np.asarray(raw["snr_db"], dtype=np.float32),
        "noise_power": np.asarray(raw["noise_power"], dtype=np.float32),
        "cond_A": np.asarray(raw.get("cond_A", np.zeros((n_frames, n_fft, n_users))), dtype=np.float32),
    }


def ce_feature_dim(cfg: dict[str, Any]) -> int:
    n_fft = int(cfg["n_fft"])
    n_users = int(cfg["n_users"])
    n_streams = int(cfg.get("n_streams", n_users))
    n_rx = int(cfg["n_rx_per_ue"])
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


def target_user_streams(full_stream_estimates: np.ndarray) -> np.ndarray:
    n_frames, n_fft, n_users, _ = full_stream_estimates.shape
    out = np.zeros((n_frames, n_users, n_fft), dtype=np.complex64)
    for user_id in range(n_users):
        out[:, user_id, :] = full_stream_estimates[:, :, user_id, user_id]
    return out


def desired_only_mrc(y_d: np.ndarray, a_eff: np.ndarray, eps: float) -> np.ndarray:
    n_frames, n_fft, n_users, _ = y_d.shape
    out = np.zeros((n_frames, n_users, n_fft), dtype=np.complex64)
    for user_id in range(n_users):
        a_desired = a_eff[:, :, user_id, :, user_id]
        y_user = y_d[:, :, user_id, :]
        numerator = np.sum(np.conj(a_desired) * y_user, axis=-1)
        denominator = np.sum(np.abs(a_desired) ** 2, axis=-1)
        out[:, user_id, :] = numerator / np.maximum(denominator, eps)
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


def fit_lmmse_weight(train_data: dict[str, np.ndarray], ridge: float) -> np.ndarray:
    x = ce_complex_to_ri(train_data["a_ls"])
    y = ce_complex_to_ri(train_data["a_true"])
    if x.shape[0] <= x.shape[1]:
        print(
            "[WARN] LMMSE fit has fewer samples than features; "
            "using identity initialization for numerical stability."
        )
        return np.eye(x.shape[1], dtype=np.float32)
    xtx = x.T @ x
    xtx += float(ridge) * np.eye(xtx.shape[0], dtype=np.float32)
    weight_t = np.linalg.solve(xtx, x.T @ y)
    return weight_t.T.astype(np.float32)


def save_lmmse_weight(path: Path, weight_ri: np.ndarray, cfg: dict[str, Any], ridge: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weight_ri=np.asarray(weight_ri, dtype=np.float32),
        n_fft=int(cfg["n_fft"]),
        n_users=int(cfg["n_users"]),
        n_rx_per_ue=int(cfg["n_rx_per_ue"]),
        n_streams=int(cfg.get("n_streams", cfg["n_users"])),
        ridge=float(ridge),
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
    if checkpoint_path.exists():
        return load_lmmse_weight(checkpoint_path)
    train_path = find_one(dataset_dir, "train_snr*.npz")
    train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps))
    print("[FIT] empirical MU-MIMO LMMSE channel estimator")
    weight = fit_lmmse_weight(train_data, float(args.lmmse_ridge))
    save_lmmse_weight(checkpoint_path, weight, cfg, float(args.lmmse_ridge))
    return weight


def apply_lmmse_weight(a_ls: np.ndarray, weight_ri: np.ndarray) -> np.ndarray:
    x = ce_complex_to_ri(a_ls)
    pred = x @ np.asarray(weight_ri, dtype=np.float32).T
    return ce_ri_to_complex_like(pred, a_ls)


def train_ce(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    lmmse_weight: np.ndarray | None,
) -> MuMimoCERefineNet:
    input_dim = ce_feature_dim(cfg)
    model = MuMimoCERefineNet(input_dim).to(device)
    if args.ce_init == "identity":
        model.init_identity()
    elif args.ce_init == "lmmse":
        if lmmse_weight is None:
            raise ValueError("lmmse_weight is required for --ce-init lmmse")
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(np.asarray(lmmse_weight, dtype=np.float32)).to(device))

    x_train = ce_complex_to_ri(train_data["a_ls"])
    y_train = ce_complex_to_ri(train_data["a_true"])
    x_val = torch.from_numpy(ce_complex_to_ri(val_data["a_ls"])).to(device)
    y_val = torch.from_numpy(ce_complex_to_ri(val_data["a_true"])).to(device)

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
        title="MU-MIMO CE Subnet Training",
        include_ber=False,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "modulation": str(cfg["modulation"]),
            "n_fft": int(cfg["n_fft"]),
            "n_users": int(cfg["n_users"]),
            "n_rx_per_ue": int(cfg["n_rx_per_ue"]),
            "n_streams": int(cfg.get("n_streams", cfg["n_users"])),
            "ce_init": str(args.ce_init),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path}")
    return model


def load_ce_model(path: Path, cfg: dict[str, Any], device: torch.device) -> MuMimoCERefineNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    input_dim = int(checkpoint.get("input_dim", ce_feature_dim(cfg)))
    model = MuMimoCERefineNet(input_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] CE checkpoint: {path}")
    return model


def predict_ce(
    model: MuMimoCERefineNet,
    a_ls: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    x = torch.from_numpy(ce_complex_to_ri(a_ls))
    loader = DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).cpu().numpy()
            chunks.append(pred)
    return ce_ri_to_complex_like(np.concatenate(chunks, axis=0), a_ls)


def sd_loss_value(logits: torch.Tensor, target: torch.Tensor, sd_loss: str) -> torch.Tensor:
    if sd_loss == "mse":
        return nn.functional.mse_loss(torch.sigmoid(logits), target)
    if sd_loss == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(f"Unsupported SD loss: {sd_loss}")


def make_fc_sd_arrays(
    *,
    y_d: np.ndarray,
    a_hat: np.ndarray,
    bits: np.ndarray,
    noise_power: np.ndarray,
    group_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    del noise_power
    n_frames, n_streams, n_fft, bps = bits.shape
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")
    estimates = linear_detect(y_d, a_hat, np.ones(n_frames, dtype=np.float32), method="zf", eps=eps)
    if estimates is None:
        raise RuntimeError("FC-SD requires n_streams <= n_rx_per_ue for ZF features")
    target = target_user_streams(estimates)
    n_groups = n_fft // group_size
    x_groups = target.reshape(n_frames, n_streams, n_groups, group_size)
    x_ri = np.concatenate([x_groups.real, x_groups.imag], axis=-1).astype(np.float32)
    y = bits.reshape(n_frames, n_streams, n_groups, group_size, bps).reshape(
        n_frames,
        n_streams,
        n_groups,
        group_size * bps,
    )
    return x_ri.reshape(n_frames * n_streams * n_groups, 2 * group_size), y.reshape(
        n_frames * n_streams * n_groups,
        group_size * bps,
    ).astype(np.float32)


def make_bilstm_sd_arrays(
    *,
    y_d: np.ndarray,
    a_hat: np.ndarray,
    bits: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    group_size: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_streams, n_fft, bps = bits.shape
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")
    zf_estimates = linear_detect(y_d, a_hat, noise_power, method="zf", eps=eps)
    if zf_estimates is None:
        raise RuntimeError("BiLSTM-SD requires n_streams <= n_rx_per_ue for ZF features")
    mmse_estimates = linear_detect(y_d, a_hat, noise_power, method="mmse", eps=eps)
    if mmse_estimates is None:
        raise RuntimeError("MMSE estimates are required for BiLSTM-SD features")
    zf_target = target_user_streams(zf_estimates)
    mmse_target = target_user_streams(mmse_estimates)
    noise_log = np.log10(np.maximum(noise_power, 1e-30)).astype(np.float32)
    snr_norm = (snr_db / 40.0).astype(np.float32)
    features = np.stack(
        [
            zf_target.real,
            zf_target.imag,
            mmse_target.real,
            mmse_target.imag,
            np.broadcast_to(noise_log[:, None, None], (n_frames, n_streams, n_fft)),
            np.broadcast_to(snr_norm[:, None, None], (n_frames, n_streams, n_fft)),
        ],
        axis=-1,
    ).astype(np.float32)
    n_groups = n_fft // group_size
    y = bits.reshape(n_frames, n_streams, n_groups, group_size, bps).reshape(
        n_frames,
        n_streams,
        n_groups,
        group_size * bps,
    )
    return features.reshape(n_frames * n_streams, n_fft, 6), y.reshape(
        n_frames * n_streams,
        n_groups,
        group_size * bps,
    ).astype(np.float32)


def train_fc_sd(
    *,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    ce_model: MuMimoCERefineNet,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> MuMimoFCSDNet:
    group_size = int(args.group_size)
    bps = bits_per_symbol(str(cfg["modulation"]))
    a_train = predict_ce(ce_model, train_data["a_ls"], device=device, batch_size=int(args.batch_size))
    a_val = predict_ce(ce_model, val_data["a_ls"], device=device, batch_size=int(args.batch_size))
    x_train, y_train = make_fc_sd_arrays(
        y_d=train_data["y_d"],
        a_hat=a_train,
        bits=train_data["bits"],
        noise_power=train_data["noise_power"],
        group_size=group_size,
        eps=float(args.eps),
    )
    x_val_np, y_val_np = make_fc_sd_arrays(
        y_d=val_data["y_d"],
        a_hat=a_val,
        bits=val_data["bits"],
        noise_power=val_data["noise_power"],
        group_size=group_size,
        eps=float(args.eps),
    )

    model = MuMimoFCSDNet(group_size, bps, int(args.hidden_dim)).to(device)
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
                f"[FC-SD {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, val_BER={val_ber:.4e}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / "train_history_fc_sd.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_ber", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / "fc_sd_training_curve.png",
        history,
        title="MU-MIMO FC-SD Training",
        include_ber=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "group_size": group_size,
            "bits_per_symbol": bps,
            "hidden_dim": int(args.hidden_dim),
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
    ce_model: MuMimoCERefineNet,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> MuMimoBiLSTMSDNet:
    n_fft = int(cfg["n_fft"])
    bps = bits_per_symbol(str(cfg["modulation"]))
    group_size = int(args.group_size)
    a_train = predict_ce(ce_model, train_data["a_ls"], device=device, batch_size=int(args.batch_size))
    a_val = predict_ce(ce_model, val_data["a_ls"], device=device, batch_size=int(args.batch_size))
    x_train, y_train = make_bilstm_sd_arrays(
        y_d=train_data["y_d"],
        a_hat=a_train,
        bits=train_data["bits"],
        noise_power=train_data["noise_power"],
        snr_db=train_data["snr_db"],
        group_size=group_size,
        eps=float(args.eps),
    )
    x_val_np, y_val_np = make_bilstm_sd_arrays(
        y_d=val_data["y_d"],
        a_hat=a_val,
        bits=val_data["bits"],
        noise_power=val_data["noise_power"],
        snr_db=val_data["snr_db"],
        group_size=group_size,
        eps=float(args.eps),
    )

    model = MuMimoBiLSTMSDNet(n_fft, bps, group_size).to(device)
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
        title="MU-MIMO BiLSTM-SD Training",
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


def load_fc_sd_model(path: Path, cfg: dict[str, Any], group_size: int, device: torch.device) -> MuMimoFCSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_group_size = int(checkpoint.get("group_size", group_size))
    bps = int(checkpoint.get("bits_per_symbol", bits_per_symbol(str(cfg["modulation"]))))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    model = MuMimoFCSDNet(model_group_size, bps, hidden_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] FC-SD checkpoint: {path}")
    return model


def load_bilstm_sd_model(
    path: Path,
    cfg: dict[str, Any],
    group_size: int,
    device: torch.device,
) -> MuMimoBiLSTMSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    n_fft = int(checkpoint.get("n_fft", cfg["n_fft"]))
    bps = int(checkpoint.get("bits_per_symbol", bits_per_symbol(str(cfg["modulation"]))))
    model_group_size = int(checkpoint.get("group_size", group_size))
    model = MuMimoBiLSTMSDNet(n_fft, bps, model_group_size).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] BiLSTM-SD checkpoint: {path}")
    return model


def predict_fc_sd_bits(
    model: MuMimoFCSDNet,
    *,
    y_d: np.ndarray,
    a_hat: np.ndarray,
    noise_power: np.ndarray,
    true_bits_shape: tuple[int, int, int, int],
    group_size: int,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dummy_bits = np.zeros(true_bits_shape, dtype=np.int8)
    x_np, _ = make_fc_sd_arrays(
        y_d=y_d,
        a_hat=a_hat,
        bits=dummy_bits,
        noise_power=noise_power,
        group_size=group_size,
        eps=eps,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(x_np)), batch_size=int(batch_size), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            chunks.append((prob > 0.5).astype(np.int8))
    pred_flat = np.concatenate(chunks, axis=0)
    n_frames, n_streams, n_fft, bps = true_bits_shape
    n_groups = n_fft // group_size
    return pred_flat.reshape(n_frames, n_streams, n_groups, group_size, bps).reshape(
        n_frames,
        n_streams,
        n_fft,
        bps,
    )


def predict_bilstm_sd_bits(
    model: MuMimoBiLSTMSDNet,
    *,
    y_d: np.ndarray,
    a_hat: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    true_bits_shape: tuple[int, int, int, int],
    group_size: int,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dummy_bits = np.zeros(true_bits_shape, dtype=np.int8)
    x_np, _ = make_bilstm_sd_arrays(
        y_d=y_d,
        a_hat=a_hat,
        bits=dummy_bits,
        noise_power=noise_power,
        snr_db=snr_db,
        group_size=group_size,
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


def format_ber(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4e}"


def evaluate_one(
    *,
    path: Path,
    cfg: dict[str, Any],
    ce_model: MuMimoCERefineNet,
    fc_model: MuMimoFCSDNet | None,
    bilstm_model: MuMimoBiLSTMSDNet | None,
    lmmse_weight: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    data = preprocess_split(load_npz(path), cfg, float(args.eps))
    snr = float(np.mean(data["snr_db"]))
    modulation = str(cfg["modulation"])
    bits = data["bits"]
    a_lmmse = apply_lmmse_weight(data["a_ls"], lmmse_weight)
    a_comnet = predict_ce(ce_model, data["a_ls"], device=device, batch_size=int(args.batch_size))

    ls_zf_ber, _ = detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    ls_mmse_ber, _ = detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    lmmse_zf_ber, _ = detector_ber(
        data["y_d"],
        a_lmmse,
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    lmmse_mmse_ber, _ = detector_ber(
        data["y_d"],
        a_lmmse,
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    comnet_zf_ber, _ = detector_ber(
        data["y_d"],
        a_comnet,
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    true_zf_ber, _ = detector_ber(
        data["y_d"],
        data["a_true"],
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    true_mmse_ber, _ = detector_ber(
        data["y_d"],
        data["a_true"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
        eps=float(args.eps),
    )
    mrc_symbols = desired_only_mrc(data["y_d"], data["a_ls"], float(args.eps))
    mrc_ber = ber_for_user_grid(mrc_symbols, bits, modulation)

    ber: dict[str, float | None] = {
        "LS-ZF": ls_zf_ber,
        "LS-MMSE": ls_mmse_ber,
        "LMMSE-ZF": lmmse_zf_ber,
        "LMMSE-MMSE": lmmse_mmse_ber,
        "ComNet-CE-ZF-Hard": comnet_zf_ber,
        "True-H ZF": true_zf_ber,
        "True-H MMSE": true_mmse_ber,
        "Desired-only MRC": mrc_ber,
    }

    if fc_model is not None:
        pred_fc = predict_fc_sd_bits(
            fc_model,
            y_d=data["y_d"],
            a_hat=a_comnet,
            noise_power=data["noise_power"],
            true_bits_shape=bits.shape,
            group_size=int(args.group_size),
            eps=float(args.eps),
            device=device,
            batch_size=int(args.batch_size),
        )
        ber["ComNet-FC"] = bit_error_rate(pred_fc, bits)
    if bilstm_model is not None:
        pred_bilstm = predict_bilstm_sd_bits(
            bilstm_model,
            y_d=data["y_d"],
            a_hat=a_comnet,
            noise_power=data["noise_power"],
            snr_db=data["snr_db"],
            true_bits_shape=bits.shape,
            group_size=int(args.group_size),
            eps=float(args.eps),
            device=device,
            batch_size=int(args.batch_size),
        )
        ber["ComNet-BiLSTM"] = bit_error_rate(pred_bilstm, bits)

    a_mse = {
        "LS": channel_mse(data["a_ls"], data["a_true"]),
        "LMMSE": channel_mse(a_lmmse, data["a_true"]),
        "ComNet-CE": channel_mse(a_comnet, data["a_true"]),
    }
    a_nmse = {
        "LS": channel_nmse(data["a_ls"], data["a_true"]),
        "LMMSE": channel_nmse(a_lmmse, data["a_true"]),
        "ComNet-CE": channel_nmse(a_comnet, data["a_true"]),
    }
    cond = np.asarray(data["cond_A"], dtype=np.float32)
    print(
        f"[EVAL] {path.name} SNR={snr:g} "
        f"LS-ZF={format_ber(ber['LS-ZF'])}, "
        f"LS-MMSE={format_ber(ber['LS-MMSE'])}, "
        f"LMMSE-ZF={format_ber(ber['LMMSE-ZF'])}, "
        f"LMMSE-MMSE={format_ber(ber['LMMSE-MMSE'])}, "
        f"ComNet-CE-ZF-Hard={format_ber(ber['ComNet-CE-ZF-Hard'])}, "
        + (f"ComNet-FC={format_ber(ber['ComNet-FC'])}, " if "ComNet-FC" in ber else "")
        + (f"ComNet-BiLSTM={format_ber(ber['ComNet-BiLSTM'])}, " if "ComNet-BiLSTM" in ber else "")
        + f"True-H ZF={format_ber(ber['True-H ZF'])}, "
        f"True-H MMSE={format_ber(ber['True-H MMSE'])}, "
        f"Desired-MRC={format_ber(ber['Desired-only MRC'])}, "
        f"A_MSE_LS={to_db(a_mse['LS']):.2f}dB, "
        f"A_MSE_LMMSE={to_db(a_mse['LMMSE']):.2f}dB, "
        f"A_MSE_ComNet={to_db(a_mse['ComNet-CE']):.2f}dB, "
        f"mean_cond={float(np.mean(cond)):.2f}, p95_cond={float(np.percentile(cond, 95.0)):.2f}"
    )
    return {
        "snr": snr,
        "a_mse": a_mse,
        "a_mse_db": {key: to_db(value) for key, value in a_mse.items()},
        "a_nmse": a_nmse,
        "a_nmse_db": {key: to_db(value) for key, value in a_nmse.items()},
        "ber": ber,
        "condition": {
            "mean_cond_A": float(np.mean(cond)),
            "p95_cond_A": float(np.percentile(cond, 95.0)),
        },
    }


def save_eval_summary(result_dir: Path, cfg: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": cfg,
        "a_mse_db": {},
        "a_nmse_db": {},
        "ber": {},
        "condition": {},
    }
    for item in sorted(results, key=lambda x: x["snr"]):
        snr_key = f"{item['snr']:g}"
        summary["a_mse_db"][snr_key] = item["a_mse_db"]
        summary["a_nmse_db"][snr_key] = item["a_nmse_db"]
        summary["ber"][snr_key] = item["ber"]
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
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, saving CSV plot data instead: {exc}")
        save_metric_csv(result_dir / "a_mse_vs_snr.csv", summary, "a_mse_db")
        save_metric_csv(result_dir / "ber_vs_snr.csv", summary, "ber")
        return

    snrs = sorted(float(x) for x in summary["ber"].keys())
    all_ber_names = {key for item in summary["ber"].values() for key in item.keys()}
    ber_names = ordered_metric_names(
        all_ber_names,
        [
            "LS-ZF",
            "LS-MMSE",
            "LMMSE-ZF",
            "LMMSE-MMSE",
            "ComNet-CE-ZF-Hard",
            "ComNet-FC",
            "ComNet-BiLSTM",
            "True-H ZF",
            "True-H MMSE",
            "Desired-only MRC",
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
    mse_names = ordered_metric_names(all_mse_names, ["LS", "LMMSE", "ComNet-CE"])
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
    ce_model: MuMimoCERefineNet,
    fc_model: MuMimoFCSDNet | None,
    bilstm_model: MuMimoBiLSTMSDNet | None,
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
    ce_checkpoint = Path(args.ce_checkpoint) if args.ce_checkpoint else result_dir / "mumimo_ce_refinenet.pt"
    fc_checkpoint = Path(args.fc_checkpoint) if args.fc_checkpoint else result_dir / "mumimo_refinenet_fc.pt"
    bilstm_checkpoint = (
        Path(args.bilstm_checkpoint) if args.bilstm_checkpoint else result_dir / "mumimo_refinenet_bilstm.pt"
    )
    lmmse_checkpoint = (
        Path(args.lmmse_checkpoint) if args.lmmse_checkpoint else result_dir / "mumimo_lmmse_estimator.npz"
    )
    cfg = load_config(dataset_dir)
    if str(cfg.get("waveform_type")) != "raw_mumimo_e2e":
        raise ValueError(
            f"Expected raw_mumimo_e2e dataset, got waveform_type={cfg.get('waveform_type')!r}"
        )
    device = resolve_device(str(args.device))
    n_users = int(cfg["n_users"])
    n_streams = int(cfg.get("n_streams", n_users))
    n_rx_per_ue = int(cfg["n_rx_per_ue"])
    print(f"[DEVICE] {device}")
    print(
        f"[CONFIG] modulation={cfg['modulation']}, n_fft={cfg['n_fft']}, "
        f"n_users={n_users}, n_streams={n_streams}, n_rx_per_ue={n_rx_per_ue}, "
        f"group_size={args.group_size}, sd_type={args.sd_type}"
    )
    if n_streams > n_rx_per_ue:
        print("[WARN] n_streams > n_rx_per_ue, ZF baselines and SD ZF features will be disabled.")

    lmmse_weight = get_lmmse_weight(
        dataset_dir=dataset_dir,
        cfg=cfg,
        args=args,
        checkpoint_path=lmmse_checkpoint,
    )

    ce_model: MuMimoCERefineNet | None = None
    fc_model: MuMimoFCSDNet | None = None
    bilstm_model: MuMimoBiLSTMSDNet | None = None

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
            fc_model = train_fc_sd(
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
            fc_model = load_fc_sd_model(fc_checkpoint, cfg, int(args.group_size), device)
        elif args.sd_type in {"fc", "both"}:
            print(f"[WARN] FC-SD checkpoint not found, ComNet-FC will be skipped: {fc_checkpoint}")
        if args.sd_type in {"bilstm", "both"} and bilstm_checkpoint.exists():
            bilstm_model = load_bilstm_sd_model(bilstm_checkpoint, cfg, int(args.group_size), device)
        elif args.sd_type in {"bilstm", "both"}:
            print(f"[WARN] BiLSTM-SD checkpoint not found, ComNet-BiLSTM will be skipped: {bilstm_checkpoint}")

    if ce_model is None:
        raise RuntimeError("CE model is required for evaluation")

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
