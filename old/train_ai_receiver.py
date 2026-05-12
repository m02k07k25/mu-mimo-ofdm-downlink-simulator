from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP AI receiver from dl_equalizer_dataset.jsonl"
    )
    parser.add_argument("--dataset", type=str, default="outputs_mu_mimo_ofdm/dl_equalizer_dataset.jsonl",
                        help="JSONL dataset path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--model-out", type=str, default="outputs_mu_mimo_ofdm/ai_receiver.pt")
    parser.add_argument("--result-dir", type=str, default="result_model",
                        help="Directory for training curves and auxiliary results")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', 'cuda', or another torch device string")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_jsonl_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    features: List[List[float]] = []
    labels: List[int] = []
    tx_bits: List[List[int]] = []
    distances: List[float] = []
    bps_values = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                features.append(rec["input"]["feature_vector"])
                labels.append(int(rec["label"]["symbol_class"]))
                bits = [int(x) for x in rec["label"]["tx_bits"]]
                tx_bits.append(bits)
                bps_values.add(len(bits))
                distances.append(float(rec.get("meta", {}).get("distance_m", np.nan)))
            except KeyError as exc:
                raise ValueError(f"Missing key {exc} in {path} line {line_no}") from exc

    if not features:
        raise ValueError(f"No JSONL records found in {path}")

    if len(bps_values) != 1:
        raise ValueError(f"Expected one bits-per-symbol value, got {sorted(bps_values)}")

    feature_dim = len(features[0])
    for i, feat in enumerate(features, start=1):
        if len(feat) != feature_dim:
            raise ValueError(f"Inconsistent feature_dim at record {i}: {len(feat)} != {feature_dim}")

    bps = next(iter(bps_values))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(tx_bits, dtype=np.int8),
        np.asarray(distances, dtype=np.float32),
        int(bps),
    )


def class_indices_to_bits(class_indices: np.ndarray, bps: int) -> np.ndarray:
    class_indices = np.asarray(class_indices, dtype=np.int64).reshape(-1)
    bits = np.zeros((class_indices.size, bps), dtype=np.int8)
    for bit_pos in range(bps):
        shift = bps - 1 - bit_pos
        bits[:, bit_pos] = ((class_indices >> shift) & 1).astype(np.int8)
    return bits


def make_split(n_samples: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_val = max(1, int(round(n_samples * val_ratio)))
    n_val = min(n_val, n_samples - 1)
    return indices[n_val:], indices[:n_val]


def print_distance_ber(distances: np.ndarray, pred_bits: np.ndarray, true_bits: np.ndarray) -> None:
    valid_mask = np.isfinite(distances)
    if not np.any(valid_mask):
        return
    rounded = np.round(distances[valid_mask], decimals=6)
    for distance in sorted(np.unique(rounded)):
        mask = valid_mask & (np.round(distances, decimals=6) == distance)
        bit_errors = int(np.sum(pred_bits[mask] != true_bits[mask]))
        bit_total = int(np.prod(true_bits[mask].shape))
        ber = bit_errors / max(bit_total, 1)
        print(f"[VAL-DIST] distance={distance:g} m, BER={ber:.4e}, bits={bit_total}")


def epoch_tick_positions(n_epochs: int) -> List[int]:
    if n_epochs <= 0:
        return []
    return [1] + list(range(5, n_epochs + 1, 5))


def configure_epoch_axis(plt, n_epochs: int) -> None:
    plt.xticks(epoch_tick_positions(n_epochs))
    if n_epochs <= 1:
        plt.xlim(0.5, 1.5)
    else:
        plt.xlim(1, n_epochs)


def save_loss_plot(
    result_dir: Path,
    train_loss_history: Sequence[float],
    val_loss_history: Sequence[float],
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "loss_curve.png"
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        csv_path = result_dir / "loss_history.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for idx, (train_loss, val_loss) in enumerate(
                zip(train_loss_history, val_loss_history),
                start=1,
            ):
                f.write(f"{idx},{train_loss},{val_loss}\n")
        print(f"[WARN] matplotlib import failed, saved CSV instead: {csv_path} ({exc})")
        return csv_path

    epochs = np.arange(1, len(train_loss_history) + 1, dtype=int)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_history, linewidth=2.0, label="train loss")
    plt.plot(epochs, val_loss_history, linewidth=2.0, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AI Receiver Training Loss")
    plt.grid(True, linestyle=":")
    plt.legend()
    configure_epoch_axis(plt, len(epochs))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def save_accuracy_plot(
    result_dir: Path,
    train_acc_history: Sequence[float],
    val_acc_history: Sequence[float],
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / "accuracy_curve.png"
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        csv_path = result_dir / "accuracy_history.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_acc,val_acc\n")
            for idx, (train_acc, val_acc) in enumerate(
                zip(train_acc_history, val_acc_history),
                start=1,
            ):
                f.write(f"{idx},{train_acc},{val_acc}\n")
        print(f"[WARN] matplotlib import failed, saved CSV instead: {csv_path} ({exc})")
        return csv_path

    epochs = np.arange(1, len(train_acc_history) + 1, dtype=int)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc_history, linewidth=2.0, label="train accuracy")
    plt.plot(epochs, val_acc_history, linewidth=2.0, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("AI Receiver Training Accuracy")
    plt.grid(True, linestyle=":")
    plt.legend()
    configure_epoch_axis(plt, len(epochs))
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def main() -> int:
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError:
        print("PyTorch is not installed in this Python environment.")
        print("Install it first, for example:")
        print("  pip install torch")
        print("or use conda:")
        print("  conda install pytorch -c pytorch")
        return 2

    dataset_path = Path(args.dataset)
    X, y, bits, distances, bps = load_jsonl_dataset(dataset_path)
    n_samples, input_dim = X.shape
    num_classes = max(int(np.max(y)) + 1, 2 ** bps)

    train_idx, val_idx = make_split(n_samples, args.val_ratio, args.seed)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in args.hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, num_classes))
    model = nn.Sequential(*layers).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_norm),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[DATA] samples={n_samples}, input_dim={input_dim}, classes={num_classes}, bps={bps}")
    print(f"[DATA] train={len(train_idx)}, val={len(val_idx)}, device={device}")
    print(f"[MODEL] MLP dims: {[input_dim] + [int(x) for x in args.hidden_dims] + [num_classes]}")

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    train_acc_history: List[float] = []
    val_acc_history: List[float] = []

    for epoch in range(1, args.epochs + 1):
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

        model.eval()
        with torch.no_grad():
            train_logits = model(torch.from_numpy(X_train_norm).to(device))
            train_pred = torch.argmax(train_logits, dim=1).cpu().numpy()
            val_logits = model(torch.from_numpy(X_val_norm).to(device))
            val_loss = loss_fn(val_logits, torch.from_numpy(y_val).to(device))
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()

        train_epoch_loss = loss_sum / max(n_seen, 1)
        val_epoch_loss = float(val_loss.item())
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_acc = float(np.mean(train_pred == y_train))
        val_acc = float(np.mean(val_pred == y_val))
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        val_ser = 1.0 - val_acc
        val_pred_bits = class_indices_to_bits(val_pred, bps)
        val_true_bits = bits[val_idx]
        val_ber = float(np.mean(val_pred_bits != val_true_bits))
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_epoch_loss:.6f}, "
            f"val_loss={val_epoch_loss:.6f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, val_SER={val_ser:.4e}, val_BER={val_ber:.4e}"
        )

    print_distance_ber(distances[val_idx], val_pred_bits, val_true_bits)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "bits_per_symbol": bps,
            "hidden_dims": [int(x) for x in args.hidden_dims],
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "dataset": str(dataset_path),
            "train_loss_history": [float(x) for x in train_loss_history],
            "val_loss_history": [float(x) for x in val_loss_history],
            "train_acc_history": [float(x) for x in train_acc_history],
            "val_acc_history": [float(x) for x in val_acc_history],
        },
        model_out,
    )
    print(f"[SAVE] model: {model_out}")
    result_dir = Path(args.result_dir)
    loss_plot_path = save_loss_plot(result_dir, train_loss_history, val_loss_history)
    acc_plot_path = save_accuracy_plot(result_dir, train_acc_history, val_acc_history)
    print(f"[SAVE] loss plot: {loss_plot_path}")
    print(f"[SAVE] accuracy plot: {acc_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
