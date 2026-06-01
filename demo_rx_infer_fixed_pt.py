"""Demo RX wrapper: run fixed-condition inference with trained checkpoints."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from environment_config import (
    DEFAULT_ENVIRONMENT_CONFIG,
    dataset_dir,
    load_environment_config,
    result_dir,
)

ROOT = Path(__file__).resolve().parent
RX_SCRIPT = ROOT / "rx_mumimo_receiver.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run demo inference using the model selected in the environment JSON config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_ENVIRONMENT_CONFIG),
        help="Shared environment JSON config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = load_environment_config(args.config)
    demo_dataset_dir = dataset_dir(environment["dataset_name"], demo=True)
    demo_result_dir = result_dir(environment["dataset_name"], demo=True)
    model_result_dir = result_dir(environment["demo"]["model_name"])
    ce_checkpoint = model_result_dir / "mumimo_ce_linear.pt"
    bilstm_checkpoint = model_result_dir / "mumimo_wl_zf_bilstm.pt"
    lmmse_checkpoint = model_result_dir / "mumimo_lmmse_estimator.npz"
    plain_lmmse_checkpoint = model_result_dir / "mumimo_plain_lmmse_estimator.npz"
    required_paths = (
        demo_dataset_dir,
        ce_checkpoint,
        bilstm_checkpoint,
        lmmse_checkpoint,
        plain_lmmse_checkpoint,
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise SystemExit(f"Missing selected demo model/data path:\n{missing}")

    demo_result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plain_lmmse_checkpoint, demo_result_dir / plain_lmmse_checkpoint.name)

    cmd = [
        sys.executable,
        str(RX_SCRIPT),
        "--config",
        str(args.config),
        "--dataset-dir",
        str(demo_dataset_dir),
        "--result-dir",
        str(demo_result_dir),
        "--mode",
        "eval",
        "--ce-checkpoint",
        str(ce_checkpoint),
        "--bilstm-checkpoint",
        str(bilstm_checkpoint),
        "--lmmse-checkpoint",
        str(lmmse_checkpoint),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Demo inference complete. Results: {demo_result_dir}")


if __name__ == "__main__":
    main()
