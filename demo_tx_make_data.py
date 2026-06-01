"""Demo TX wrapper: create fixed-condition test data with one count argument."""

from __future__ import annotations

import argparse

import numpy as np

from tx_mumimo_e2e_dataset import (
    MuMimoE2EConfig,
    _make_paired_test_datasets,
    _save_npz,
    _snr_name,
    write_config,
)
from environment_config import (
    DEFAULT_ENVIRONMENT_CONFIG,
    dataset_dir,
    load_environment_config,
    tx_defaults,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate demo test data from the shared environment JSON config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_ENVIRONMENT_CONFIG),
        help="Shared environment JSON config.",
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
    if args.num_data is not None and args.num_data <= 0:
        parser.error("Number of test examples must be positive.")
    return args


def main() -> None:
    args = parse_args()
    environment = load_environment_config(args.config)
    if args.num_data is None:
        args.num_data = 1000
    out_dir = dataset_dir(environment["dataset_name"], demo=True)

    config_values = tx_defaults(environment)
    config_values.update(
        n_train_frames=0,
        n_val_frames=0,
        n_test_frames_per_snr=int(args.num_data),
    )
    cfg = MuMimoE2EConfig(
        **config_values,
        snr_train_db=float(config_values["snr_train_db_list"][0]),
    )

    print(f"Generating {args.num_data} demo test frames: {out_dir}")
    cfg.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(out_dir, cfg)

    rng = np.random.default_rng(cfg.seed)
    test_sets = _make_paired_test_datasets(
        cfg=cfg,
        n_frames=int(args.num_data),
        snr_db_values=cfg.snr_test_db,
        rng=rng,
    )
    for snr_db in cfg.snr_test_db:
        test = test_sets[float(snr_db)]
        _save_npz(out_dir / f"test_snr{_snr_name(float(snr_db))}.npz", test)

    print(f"Demo dataset ready: {out_dir}")


if __name__ == "__main__":
    main()
