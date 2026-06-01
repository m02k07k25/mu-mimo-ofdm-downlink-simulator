"""Shared JSON environment configuration for the TX/RX entry points."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_ENVIRONMENT_CONFIG = ROOT / "environment_config.json"

TX_PARAMETER_KEYS = (
    "case",
    "clip_ratio",
    "rx_iq_gain_imbalance_db",
    "rx_iq_phase_error_deg",
    "rx_common_phase_error_deg",
)

TX_DATASET_PARAMETER_KEYS = (
    "modulation",
    "n_users",
    "n_tx",
    "n_rx_per_ue",
    "n_fft",
    "n_cp",
    "n_taps",
    "n_rays_per_path",
    "pdp_decay",
    "carrier_freq_hz",
    "antenna_spacing_lambda",
    "scm_angle_spread_deg",
    "csit_error_var",
    "precoder_norm",
    "pilot_kind",
    "snr_train_db_list",
    "snr_test_db",
    "n_train_frames",
    "n_val_frames",
    "n_test_frames_per_snr",
    "seed",
)


def load_environment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_name = _require_name(config, "dataset_name")
    case = config.get("case")
    if case not in {"linear", "cp_removal", "clipping"}:
        raise ValueError("case must be one of linear, cp_removal, clipping")

    clip_ratio = _require_finite_number(config, "clip_ratio")
    if clip_ratio <= 0.0:
        raise ValueError("clip_ratio must be positive")
    for key in TX_PARAMETER_KEYS[2:]:
        _require_finite_number(config, key)

    tx_dataset = _require_object(config, "tx_dataset")
    if tx_dataset.get("modulation") not in {"16QAM", "64QAM"}:
        raise ValueError("tx_dataset.modulation must be one of 16QAM, 64QAM")
    for key in ("n_users", "n_tx", "n_rx_per_ue", "n_fft", "n_taps", "n_rays_per_path"):
        _require_positive_int(tx_dataset, key)
    if not 1 <= tx_dataset["n_users"] <= tx_dataset["n_tx"]:
        raise ValueError("tx_dataset.n_users must satisfy 1 <= n_users <= n_tx")
    if tx_dataset.get("n_streams") != tx_dataset["n_users"]:
        raise ValueError("tx_dataset.n_streams must equal tx_dataset.n_users")
    if tx_dataset.get("channel_model") != "SCM-style geometric clustered channel":
        raise ValueError("tx_dataset.channel_model must be SCM-style geometric clustered channel")
    _require_non_negative_int(tx_dataset, "n_cp")
    for key in ("pdp_decay", "carrier_freq_hz", "antenna_spacing_lambda"):
        _require_positive_number(tx_dataset, key)
    _require_non_negative_number(tx_dataset, "scm_angle_spread_deg")
    _require_non_negative_number(tx_dataset, "csit_error_var")
    if tx_dataset.get("precoder_norm") not in {"none", "column", "fro"}:
        raise ValueError("tx_dataset.precoder_norm must be one of none, column, fro")
    if tx_dataset.get("pilot_kind") not in {"ones", "qpsk"}:
        raise ValueError("tx_dataset.pilot_kind must be one of ones, qpsk")
    for key in ("snr_train_db_list", "snr_test_db"):
        values = tx_dataset.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"tx_dataset.{key} must be a non-empty list")
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"tx_dataset.{key} values must be finite numbers")
    for key in ("n_train_frames", "n_val_frames", "n_test_frames_per_snr"):
        _require_non_negative_int(tx_dataset, key)
    _require_non_negative_int(tx_dataset, "seed")

    demo = config.get("demo")
    if not isinstance(demo, dict):
        raise ValueError("demo must be an object")
    _require_name(demo, "model_name")

    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    common = _require_object(training, "common")
    ce = _require_object(training, "ce")
    sd = _require_object(training, "sd")

    for key in ("batch_size", "log_every"):
        _require_positive_int(common, key)
    _require_non_negative_int(common, "seed")
    _require_positive_number(common, "lmmse_ridge")

    _require_positive_int(ce, "epochs")
    _require_positive_number(ce, "learning_rate")
    _require_positive_int(ce, "lr_step")
    _require_positive_number(ce, "lr_gamma")
    _require_positive_int(ce, "hidden_dim")
    dropout = _require_non_negative_number(ce, "dropout")
    if dropout >= 1.0:
        raise ValueError("training.ce.dropout must be less than 1")

    _require_positive_int(sd, "epochs")
    _require_positive_number(sd, "learning_rate")
    _require_positive_int(sd, "lr_step")
    _require_positive_number(sd, "lr_gamma")
    _require_positive_int(sd, "group_size")
    hidden_dims = sd.get("bilstm_hidden_dims")
    if not isinstance(hidden_dims, list) or len(hidden_dims) != 3:
        raise ValueError("training.sd.bilstm_hidden_dims must contain three values")
    for hidden_dim in hidden_dims:
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("training.sd.bilstm_hidden_dims values must be positive integers")

    config["dataset_name"] = dataset_name
    return config


def tx_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {key: config[key] for key in TX_PARAMETER_KEYS}
    tx_dataset = config["tx_dataset"]
    defaults.update({key: tx_dataset[key] for key in TX_DATASET_PARAMETER_KEYS})
    defaults["snr_train_db_list"] = tuple(tx_dataset["snr_train_db_list"])
    defaults["snr_test_db"] = tuple(tx_dataset["snr_test_db"])
    return defaults


def rx_training_defaults(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    common = training["common"]
    ce = training["ce"]
    sd = training["sd"]
    return {
        "batch_size": common["batch_size"],
        "lmmse_ridge": common["lmmse_ridge"],
        "seed": common["seed"],
        "log_every": common["log_every"],
        "ce_epochs": ce["epochs"],
        "ce_lr": ce["learning_rate"],
        "ce_lr_step": ce["lr_step"],
        "ce_lr_gamma": ce["lr_gamma"],
        "ce_hidden_dim": ce["hidden_dim"],
        "ce_dropout": ce["dropout"],
        "bilstm_epochs": sd["epochs"],
        "sd_lr": sd["learning_rate"],
        "bilstm_lr_step": sd["lr_step"],
        "sd_lr_gamma": sd["lr_gamma"],
        "group_size": sd["group_size"],
        "bilstm_hidden_dims": tuple(sd["bilstm_hidden_dims"]),
    }


def dataset_dir(dataset_name: str, *, demo: bool = False) -> Path:
    prefix = "demo_" if demo else ""
    return ROOT / "datasets" / f"{prefix}{dataset_name}"


def result_dir(dataset_name: str, *, demo: bool = False) -> Path:
    prefix = "demo_" if demo else ""
    return ROOT / "results" / f"{prefix}{dataset_name}"


def _require_name(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{key} must be a non-empty directory name")
    return value


def _require_finite_number(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _require_object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_non_negative_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _require_positive_number(config: dict[str, Any], key: str) -> float:
    value = _require_finite_number(config, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _require_non_negative_number(config: dict[str, Any], key: str) -> float:
    value = _require_finite_number(config, key)
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return value
