"""Demo TX wrapper: create fixed-condition test data with one count argument."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tx_mumimo_e2e_dataset import (
    MuMimoE2EConfig,
    _make_paired_test_datasets,
    _save_npz,
    _snr_name,
    write_config,
)

ROOT = Path(__file__).resolve().parent

DEFAULT_OUT_DIR = ROOT / "datasets" / "demo_clip17_iq05_p2_cpe3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo test data for the fixed clipping-1.7 model."
    )
    parser.add_argument(
        "num_data_pos",
        nargs="?",
        type=int,
        help="Number of test examples to generate.",
    )
    parser.add_argument(
        "--num-data",
        type=int,
        default=None,
        help="Number of test examples to generate.",
    )
    args = parser.parse_args()
    if args.num_data is not None and args.num_data_pos is not None:
        parser.error("Use only one count argument: either 100 or --num-data 100.")
    args.num_data = args.num_data if args.num_data is not None else args.num_data_pos
    if args.num_data is None:
        args.num_data = 1000
    return args


def main() -> None:
    args = parse_args()

    cfg = MuMimoE2EConfig(
        case="clipping",
        clip_ratio=1.7,
        rx_iq_gain_imbalance_db=0.5,
        rx_iq_phase_error_deg=2.0,
        rx_common_phase_error_deg=3.0,
        n_train_frames=0,
        n_val_frames=0,
        n_test_frames_per_snr=int(args.num_data),
        snr_test_db=(0, 5, 10, 15, 20, 25, 30, 35, 40),
    )

    print(f"Generating {args.num_data} demo test frames: {DEFAULT_OUT_DIR}")
    cfg.validate()
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_config(DEFAULT_OUT_DIR, cfg)

    rng = np.random.default_rng(cfg.seed)
    test_sets = _make_paired_test_datasets(
        cfg=cfg,
        n_frames=int(args.num_data),
        snr_db_values=cfg.snr_test_db,
        rng=rng,
    )
    for snr_db in cfg.snr_test_db:
        test = test_sets[float(snr_db)]
        _save_npz(DEFAULT_OUT_DIR / f"test_snr{_snr_name(float(snr_db))}.npz", test)

    print(f"Demo dataset ready: {DEFAULT_OUT_DIR}")


if __name__ == "__main__":
    main()
