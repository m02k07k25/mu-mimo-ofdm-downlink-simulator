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
        description="Train and evaluate a raw end-to-end MU-MIMO OFDM receiver."
    )
    parser.add_argument("--dataset-dir", type=str, default="outputs_mumimo_e2e_16qam_smoke")
    parser.add_argument("--result-dir", type=str, default="results_mumimo_e2e_16qam_smoke")
    parser.add_argument("--mode", type=str, default="train-all", choices=["train-all", "train-sd", "eval"])
    parser.add_argument("--sd-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sd-lr", type=float, default=1e-3)
    parser.add_argument("--sd-lr-step", type=int, default=25)
    parser.add_argument("--sd-lr-gamma", type=float, default=0.5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
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


class MuMimoSDNet(nn.Module):
    def __init__(self, input_dim: int, group_size: int, bits_per_symbol_value: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.group_size = int(group_size)
        self.bits_per_symbol = int(bits_per_symbol_value)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.group_size * self.bits_per_symbol),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def bit_error_rate(pred_bits: np.ndarray, true_bits: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_bits, dtype=np.int8) != np.asarray(true_bits, dtype=np.int8)))


def hard_demod_user_symbols(symbols: np.ndarray, modulation: str) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex64)
    bps = bits_per_symbol(modulation)
    return qam_demodulate(symbols.reshape(-1), modulation).reshape(symbols.shape[0], symbols.shape[1], bps)


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
        return np.linalg.solve(system, matched[..., None])[..., 0].astype(np.complex64)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(system)
        return np.matmul(pinv, matched[..., None])[..., 0].astype(np.complex64)


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


def save_training_plot(path: Path, rows: list[dict[str, float]], title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed, training plot skipped: {exc}")
        return

    epochs = [row["epoch"] for row in rows]
    fig, (ax_loss, ax_ber) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_loss.plot(epochs, [row["train_loss"] for row in rows], label="train loss", linewidth=2.0)
    ax_loss.plot(epochs, [row["val_loss"] for row in rows], label="validation loss", linewidth=2.0)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":")
    ax_loss.legend()
    ax_ber.plot(epochs, [row["val_ber"] for row in rows], label="validation BER", linewidth=2.0)
    ax_ber.set_xlabel("Epoch")
    ax_ber.set_ylabel("BER")
    ax_ber.grid(True, linestyle=":")
    ax_ber.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[SAVE] {path}")


def make_sd_features(
    *,
    y_d: np.ndarray,
    a_ls: np.ndarray,
    mmse_estimates: np.ndarray,
    bits: np.ndarray,
    noise_power: np.ndarray,
    snr_db: np.ndarray,
    user_id: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_frames, n_fft, _, n_rx = y_d.shape
    n_streams = a_ls.shape[-1]
    bps = bits.shape[-1]
    if n_fft % group_size != 0:
        raise ValueError("n_fft must be divisible by group_size")

    y_user = y_d[:, :, user_id, :]
    a_user = a_ls[:, :, user_id, :, :]
    mmse_user = mmse_estimates[:, :, user_id, :]
    noise_log = np.log10(np.maximum(noise_power, 1e-30)).astype(np.float32)
    snr_norm = (snr_db / 40.0).astype(np.float32)

    features = np.concatenate(
        [
            y_user.real,
            y_user.imag,
            a_user.real.reshape(n_frames, n_fft, n_rx * n_streams),
            a_user.imag.reshape(n_frames, n_fft, n_rx * n_streams),
            mmse_user.real,
            mmse_user.imag,
            np.broadcast_to(noise_log[:, None, None], (n_frames, n_fft, 1)),
            np.broadcast_to(snr_norm[:, None, None], (n_frames, n_fft, 1)),
        ],
        axis=-1,
    ).astype(np.float32)
    feature_dim = int(features.shape[-1])
    n_groups = n_fft // group_size
    x = features.reshape(n_frames, n_groups, group_size, feature_dim).reshape(
        n_frames * n_groups,
        group_size * feature_dim,
    )
    y = bits[:, user_id].reshape(n_frames, n_groups, group_size, bps).reshape(
        n_frames * n_groups,
        group_size * bps,
    )
    return x.astype(np.float32), y.astype(np.float32), feature_dim


def train_user_sdnet(
    *,
    user_id: int,
    cfg: dict[str, Any],
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> MuMimoSDNet:
    group_size = int(args.group_size)
    bps = bits_per_symbol(str(cfg["modulation"]))
    train_mmse = linear_detect(
        train_data["y_d"],
        train_data["a_ls"],
        train_data["noise_power"],
        method="mmse",
        eps=float(args.eps),
    )
    val_mmse = linear_detect(
        val_data["y_d"],
        val_data["a_ls"],
        val_data["noise_power"],
        method="mmse",
        eps=float(args.eps),
    )
    if train_mmse is None or val_mmse is None:
        raise RuntimeError("MMSE estimates are required for SDNet features")

    x_train, y_train, feature_dim = make_sd_features(
        y_d=train_data["y_d"],
        a_ls=train_data["a_ls"],
        mmse_estimates=train_mmse,
        bits=train_data["bits"],
        noise_power=train_data["noise_power"],
        snr_db=train_data["snr_db"],
        user_id=user_id,
        group_size=group_size,
    )
    x_val_np, y_val_np, _ = make_sd_features(
        y_d=val_data["y_d"],
        a_ls=val_data["a_ls"],
        mmse_estimates=val_mmse,
        bits=val_data["bits"],
        noise_power=val_data["noise_power"],
        snr_db=val_data["snr_db"],
        user_id=user_id,
        group_size=group_size,
    )

    model = MuMimoSDNet(
        input_dim=group_size * feature_dim,
        group_size=group_size,
        bits_per_symbol_value=bps,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 100 + int(user_id))
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
    loss_fn = nn.BCEWithLogitsLoss()

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
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(loss_fn(val_logits, y_val).item())
            val_pred = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(np.int8)
        val_ber = bit_error_rate(val_pred, y_val_np.astype(np.int8))
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
                f"[SD u{user_id} {epoch:04d}] train_loss={train_loss:.6e}, "
                f"val_loss={val_loss:.6e}, val_BER={val_ber:.4e}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    write_history(
        Path(args.result_dir) / f"train_history_sd_user{user_id}.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_ber", "lr"],
    )
    save_training_plot(
        Path(args.result_dir) / f"sd_training_curve_user{user_id}.png",
        history,
        title=f"MU-MIMO SDNet User {user_id}",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "group_size": group_size,
            "bits_per_symbol": bps,
            "hidden_dim": int(args.hidden_dim),
            "feature_dim": feature_dim,
            "user_id": int(user_id),
            "modulation": str(cfg["modulation"]),
        },
        checkpoint_path,
    )
    print(f"[SAVE] {checkpoint_path}")
    return model


def load_user_sdnet(path: Path, device: torch.device) -> MuMimoSDNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MuMimoSDNet(
        input_dim=int(checkpoint["input_dim"]),
        group_size=int(checkpoint["group_size"]),
        bits_per_symbol_value=int(checkpoint["bits_per_symbol"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[LOAD] SDNet user {checkpoint.get('user_id', '?')}: {path}")
    return model


def predict_user_sdnet(
    model: MuMimoSDNet,
    *,
    user_id: int,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    mmse_estimates = linear_detect(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        method="mmse",
        eps=float(args.eps),
    )
    if mmse_estimates is None:
        raise RuntimeError("MMSE estimates are required for SDNet features")
    x_np, _, _ = make_sd_features(
        y_d=data["y_d"],
        a_ls=data["a_ls"],
        mmse_estimates=mmse_estimates,
        bits=data["bits"],
        noise_power=data["noise_power"],
        snr_db=data["snr_db"],
        user_id=user_id,
        group_size=int(args.group_size),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_np)),
        batch_size=int(args.batch_size),
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            chunks.append((prob > 0.5).astype(np.int8))
    pred_flat = np.concatenate(chunks, axis=0)
    n_frames, n_streams, n_fft, bps = data["bits"].shape
    n_groups = n_fft // int(args.group_size)
    return pred_flat.reshape(n_frames, n_groups, int(args.group_size), bps).reshape(n_frames, n_fft, bps)


def evaluate_one(
    *,
    path: Path,
    cfg: dict[str, Any],
    sd_models: dict[int, MuMimoSDNet],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    data = preprocess_split(load_npz(path), cfg, float(args.eps))
    snr = float(np.mean(data["snr_db"]))
    modulation = str(cfg["modulation"])
    bits = data["bits"]

    ls_zf_ber, _ = detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="zf",
        eps=float(args.eps),
    )
    ls_mmse_ber, ls_mmse_est = detector_ber(
        data["y_d"],
        data["a_ls"],
        data["noise_power"],
        bits,
        modulation,
        method="mmse",
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
        "True-H ZF": true_zf_ber,
        "True-H MMSE": true_mmse_ber,
        "Desired-only MRC": mrc_ber,
    }

    if sd_models:
        pred = np.zeros_like(bits)
        for user_id, model in sd_models.items():
            pred[:, user_id] = predict_user_sdnet(
                model,
                user_id=user_id,
                data=data,
                args=args,
                device=device,
            )
        ber["MU-MIMO SDNet"] = bit_error_rate(pred, bits)

    a_mse = {
        "LS": channel_mse(data["a_ls"], data["a_true"]),
    }
    a_nmse = {
        "LS": channel_nmse(data["a_ls"], data["a_true"]),
    }
    cond = np.asarray(data["cond_A"], dtype=np.float32)
    print(
        f"[EVAL] {path.name} SNR={snr:g} "
        f"LS-ZF={ber['LS-ZF'] if ber['LS-ZF'] is not None else 'n/a'}, "
        f"LS-MMSE={ber['LS-MMSE']:.4e}, "
        f"True-H ZF={ber['True-H ZF'] if ber['True-H ZF'] is not None else 'n/a'}, "
        f"True-H MMSE={ber['True-H MMSE']:.4e}, "
        f"Desired-MRC={ber['Desired-only MRC']:.4e}, "
        + (f"SDNet={ber['MU-MIMO SDNet']:.4e}, " if "MU-MIMO SDNet" in ber else "")
        + f"A_LS_MSE={to_db(a_mse['LS']):.2f}dB, "
        f"mean_cond={float(np.mean(cond)):.2f}, p95_cond={float(np.percentile(cond, 95.0)):.2f}"
    )
    if ls_mmse_est is None:
        raise RuntimeError("LS-MMSE estimates unexpectedly unavailable")
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
    ber_names = [
        "LS-ZF",
        "LS-MMSE",
        "True-H ZF",
        "True-H MMSE",
        "Desired-only MRC",
        "MU-MIMO SDNet",
    ]
    plt.figure(figsize=(8, 5))
    for name in ber_names:
        if name not in {key for item in summary["ber"].values() for key in item.keys()}:
            continue
        values = []
        for snr in snrs:
            value = summary["ber"][f"{snr:g}"].get(name)
            values.append(np.nan if value is None else max(float(value), 1e-7))
        plt.semilogy(snrs, values, marker="o", linewidth=2.0, label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Raw MU-MIMO BER vs SNR")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    ber_path = result_dir / "ber_vs_snr.png"
    plt.savefig(ber_path, dpi=160)
    plt.close()
    print(f"[SAVE] {ber_path}")

    plt.figure(figsize=(8, 5))
    values = [summary["a_mse_db"][f"{snr:g}"]["LS"] for snr in snrs]
    plt.plot(snrs, values, marker="s", linewidth=2.0, label="LS")
    plt.xlabel("SNR (dB)")
    plt.ylabel("A_eff MSE (dB)")
    plt.title("Pilot LS Effective Channel MSE")
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
    sd_models: dict[int, MuMimoSDNet],
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
            sd_models=sd_models,
            args=args,
            device=device,
        )
        for path in test_paths
    ]
    summary = save_eval_summary(Path(args.result_dir), cfg, results)
    save_eval_plots(Path(args.result_dir), summary)
    return summary


def model_path(result_dir: Path, user_id: int) -> Path:
    return result_dir / f"mumimo_sdnet_user{user_id}.pt"


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset_dir = Path(args.dataset_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
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
        f"group_size={args.group_size}"
    )
    if n_streams > n_rx_per_ue:
        print("[WARN] n_streams > n_rx_per_ue, ZF baselines will be disabled.")

    sd_models: dict[int, MuMimoSDNet] = {}
    if args.mode in {"train-all", "train-sd"}:
        train_path = find_one(dataset_dir, "train_snr*.npz")
        val_path = find_one(dataset_dir, "val_snr*.npz")
        train_data = preprocess_split(load_npz(train_path), cfg, float(args.eps))
        val_data = preprocess_split(load_npz(val_path), cfg, float(args.eps))
        for user_id in range(n_users):
            sd_models[user_id] = train_user_sdnet(
                user_id=user_id,
                cfg=cfg,
                train_data=train_data,
                val_data=val_data,
                args=args,
                device=device,
                checkpoint_path=model_path(result_dir, user_id),
            )

    if args.mode == "eval":
        for user_id in range(n_users):
            path = model_path(result_dir, user_id)
            if path.exists():
                sd_models[user_id] = load_user_sdnet(path, device)
            else:
                print(f"[WARN] SDNet checkpoint not found, user {user_id} will be skipped: {path}")

    evaluate_all(
        dataset_dir=dataset_dir,
        cfg=cfg,
        sd_models=sd_models,
        args=args,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
