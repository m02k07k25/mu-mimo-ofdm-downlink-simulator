"""Demo RX wrapper: run fixed-condition inference with trained checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RX_SCRIPT = ROOT / "rx_mumimo_receiver.py"

FIXED_MODEL_DIR = ROOT / "datasets" / "clip17_iq05_p2_cpe3"
FIXED_RESULT_DIR = ROOT / "results" / "clip17_iq05_p2_cpe3"
DEFAULT_DATASET_DIR = ROOT / "datasets" / "demo_clip17_iq05_p2_cpe3"
DEFAULT_RESULT_DIR = ROOT / "results" / "demo_clip17_iq05_p2_cpe3"
CE_CHECKPOINT = FIXED_RESULT_DIR / "mumimo_ce_linear.pt"
BILSTM_CHECKPOINT = FIXED_RESULT_DIR / "mumimo_wl_zf_bilstm.pt"
LMMSE_CHECKPOINT = FIXED_RESULT_DIR / "mumimo_lmmse_estimator.npz"
PLAIN_LMMSE_CHECKPOINT = FIXED_RESULT_DIR / "mumimo_plain_lmmse_estimator.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-condition demo inference with trained checkpoints."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to rx_mumimo_receiver.py. Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_paths = (
        FIXED_MODEL_DIR,
        CE_CHECKPOINT,
        BILSTM_CHECKPOINT,
        LMMSE_CHECKPOINT,
        PLAIN_LMMSE_CHECKPOINT,
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise SystemExit(f"Missing fixed demo model/data path:\n{missing}")

    DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PLAIN_LMMSE_CHECKPOINT, DEFAULT_RESULT_DIR / PLAIN_LMMSE_CHECKPOINT.name)

    cmd = [
        sys.executable,
        str(RX_SCRIPT),
        "--dataset-dir",
        str(DEFAULT_DATASET_DIR),
        "--result-dir",
        str(DEFAULT_RESULT_DIR),
        "--mode",
        "eval",
        "--device",
        str(args.device),
        "--ce-checkpoint",
        str(CE_CHECKPOINT),
        "--bilstm-checkpoint",
        str(BILSTM_CHECKPOINT),
        "--lmmse-checkpoint",
        str(LMMSE_CHECKPOINT),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Demo inference complete. Results: {DEFAULT_RESULT_DIR}")


if __name__ == "__main__":
    main()
