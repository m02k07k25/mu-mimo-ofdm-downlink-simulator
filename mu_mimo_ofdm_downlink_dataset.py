
"""
5 GHz MU-MIMO OFDM Downlink End-to-End 시뮬레이션 코드.

1개 기지국, 8개 송신 안테나, 1~16개 단말, 단말당 4개 수신 안테나를
가정한다. 각 단말은 1개 spatial stream을 수신하며, 한 OFDM symbol에서
동시 ZF precoding 가능한 stream 수는 최대 8개이다.

채널은 다중경로 Rayleigh fading에 5 GHz 도심 소형셀 링크버짓을 결합한다.
거리 기반 path loss, log-normal shadowing, kTB thermal AWGN을 사용하므로
거리가 멀어질수록 수신 전력과 pre-combiner SNR이 낮아진다.
수신부는 perfect CSI 기반으로 OFDM/MIMO/equalization을 전통적으로 처리하고,
AI 학습용 feature에는 equalized symbol x_hat 중심의 detector 입력을 저장한다.

JSON label은 실제 무선 채널로 보내는 것이 아니라 supervised learning
데이터셋 생성을 위해 저장하는 시뮬레이션 정답이다.

실행 예시:
    python mu_mimo_ofdm_downlink_dataset.py

빠른 테스트:
    python mu_mimo_ofdm_downlink_dataset.py --frames 1 --ofdm-symbols 2 --users 1 4 --distance-sweep 10 100 --max-json-records 200
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BOLTZMANN_J_PER_K = 1.380649e-23


# ============================================================
# 1. 시스템 설정
# ============================================================

@dataclass
class SystemConfig:
    """시뮬레이션에 필요한 모든 설정값"""

    # OFDM
    n_fft: int = 64
    n_cp: int = 16
    n_ofdm_symbols: int = 8

    # 64 FFT에서 DC와 guard subcarrier를 제외한 52개 data subcarrier
    # 인덱스 0은 DC로 두고, 1~26, 38~63만 데이터로 사용한다.
    data_idx: Tuple[int, ...] = tuple(list(range(1, 27)) + list(range(38, 64)))

    # 안테나/사용자
    n_tx: int = 8                 # 기지국 송신 안테나 수
    n_rx_per_ue: int = 4          # 각 단말 수신 안테나 수
    max_streams: int = 8          # 한 번에 스케줄링 가능한 최대 사용자 수

    # 채널
    n_taps: int = 6               # 다중경로 tap 수. CP보다 작게 둔다.
    pdp_decay: float = 2.0        # power delay profile 지수 감쇠 계수

    # 5 GHz urban small-cell link budget
    carrier_freq_ghz: float = 5.0
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 30.0     # total BS transmit power across all TX antennas
    rx_noise_figure_db: float = 7.0
    temperature_k: float = 290.0
    distance_min_m: float = 10.0
    distance_max_m: float = 300.0
    path_loss_exponent: float = 3.0
    shadowing_std_db: float = 6.0

    # 변조
    modulation: str = "QPSK"      # "BPSK", "QPSK", "16QAM"

    # 반복
    n_frames: int = 50
    random_seed: int = 42

    # JSONL 저장 제한
    max_json_records: int = 60000

    @property
    def n_data_subcarriers(self) -> int:
        return len(self.data_idx)

    def validate(self) -> None:
        """설정값이 요구 조건과 모순되지 않는지 확인"""
        if self.n_tx != 8:
            raise ValueError("요구 조건상 기지국 송신 안테나는 8개로 두어야 한다.")
        if self.n_rx_per_ue != 4:
            raise ValueError("요구 조건상 각 단말 수신 안테나는 4개로 두어야 한다.")
        if self.max_streams > self.n_tx:
            raise ValueError("동시 stream 수는 송신 안테나 수보다 클 수 없다.")
        if self.n_taps > self.n_cp:
            raise ValueError("다중경로 tap 수는 CP 길이 이하로 두는 것이 안전하다.")
        if self.modulation.upper() not in {"BPSK", "QPSK", "16QAM"}:
            raise ValueError("modulation은 BPSK, QPSK, 16QAM 중 하나여야 한다.")
        if self.carrier_freq_ghz <= 0:
            raise ValueError("carrier_freq_ghz must be positive.")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive.")
        if self.distance_min_m <= 0 or self.distance_max_m < self.distance_min_m:
            raise ValueError("distance range must satisfy 0 < min <= max.")
        if self.path_loss_exponent <= 0:
            raise ValueError("path_loss_exponent must be positive.")
        if self.shadowing_std_db < 0:
            raise ValueError("shadowing_std_db must be non-negative.")


# ============================================================
# 2. 변조 / 복조
# ============================================================

def bits_per_symbol(modulation: str) -> int:
    """변조 방식별 심볼당 비트 수"""
    table = {"BPSK": 1, "QPSK": 2, "16QAM": 4}
    key = modulation.upper()
    if key not in table:
        raise ValueError(f"지원하지 않는 변조 방식: {modulation}")
    return table[key]


def modulate_bits(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    비트를 복소수 심볼로 변조한다.
    평균 심볼 전력은 1로 정규화한다.
    """
    modulation = modulation.upper()
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    bps = bits_per_symbol(modulation)

    if bits.size % bps != 0:
        raise ValueError("비트 길이는 심볼당 비트 수의 배수여야 한다.")

    if modulation == "BPSK":
        # 0 -> +1, 1 -> -1
        return (1.0 - 2.0 * bits.astype(float)).astype(np.complex128)

    if modulation == "QPSK":
        # b0는 실수부 부호, b1은 허수부 부호
        b = bits.reshape(-1, 2)
        symbols = (1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])
        return symbols / math.sqrt(2.0)

    if modulation == "16QAM":
        # Gray에 가까운 4-PAM mapping
        # 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3
        b = bits.reshape(-1, 4)

        def pam_gray(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
            out = np.zeros_like(msb, dtype=float)
            out[(msb == 0) & (lsb == 0)] = 3
            out[(msb == 0) & (lsb == 1)] = 1
            out[(msb == 1) & (lsb == 1)] = -1
            out[(msb == 1) & (lsb == 0)] = -3
            return out

        re = pam_gray(b[:, 0], b[:, 1])
        im = pam_gray(b[:, 2], b[:, 3])
        return (re + 1j * im) / math.sqrt(10.0)

    raise ValueError(f"지원하지 않는 변조 방식: {modulation}")


def demodulate_symbols(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """복소수 심볼을 hard decision으로 비트 복조한다."""
    modulation = modulation.upper()
    symbols = np.asarray(symbols).reshape(-1)

    if modulation == "BPSK":
        return (symbols.real < 0).astype(np.int8)

    if modulation == "QPSK":
        s = symbols * math.sqrt(2.0)
        bits = np.zeros((len(s), 2), dtype=np.int8)
        bits[:, 0] = (s.real < 0).astype(np.int8)
        bits[:, 1] = (s.imag < 0).astype(np.int8)
        return bits.reshape(-1)

    if modulation == "16QAM":
        s = symbols * math.sqrt(10.0)
        bits = np.zeros((len(s), 4), dtype=np.int8)
        bits[:, 0] = (s.real < 0).astype(np.int8)
        bits[:, 1] = (np.abs(s.real) < 2).astype(np.int8)
        bits[:, 2] = (s.imag < 0).astype(np.int8)
        bits[:, 3] = (np.abs(s.imag) < 2).astype(np.int8)
        return bits.reshape(-1)

    raise ValueError(f"지원하지 않는 변조 방식: {modulation}")


def bits_to_class_index(bits: Sequence[int]) -> int:
    """예: QPSK에서 [1, 0] -> 2. JSON label용 class index."""
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return int(idx)


# ============================================================
# 3. OFDM 송수신
# ============================================================

def ofdm_modulate(freq_grid: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    OFDM 변조: IFFT + CP 삽입.

    입력:
        freq_grid: shape = (N_FFT, N_OFDM_SYMBOLS, N_TX)
    출력:
        tx_blocks: shape = (N_OFDM_SYMBOLS, N_FFT + N_CP, N_TX)
    """
    time_no_cp = np.fft.ifft(freq_grid, n=cfg.n_fft, axis=0) * math.sqrt(cfg.n_fft)
    cp = time_no_cp[-cfg.n_cp :, :, :]
    time_with_cp = np.concatenate([cp, time_no_cp], axis=0)
    return np.transpose(time_with_cp, (1, 0, 2))


def ofdm_demodulate(rx_blocks: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    OFDM 복조: CP 제거 + FFT.

    입력:
        rx_blocks: shape = (N_OFDM_SYMBOLS, N_FFT + N_CP, N_RX)
    출력:
        rx_freq: shape = (N_FFT, N_OFDM_SYMBOLS, N_RX)
    """
    no_cp = rx_blocks[:, cfg.n_cp : cfg.n_cp + cfg.n_fft, :]
    rx_freq = np.fft.fft(no_cp, n=cfg.n_fft, axis=1) / math.sqrt(cfg.n_fft)
    return np.transpose(rx_freq, (1, 0, 2))


# ============================================================
# 4. 다중경로 Rayleigh 채널 + AWGN
# ============================================================

def db_to_linear(db_value: float | np.ndarray) -> float | np.ndarray:
    """dB 값을 선형 전력비로 바꾼다."""
    return 10.0 ** (np.asarray(db_value) / 10.0)


def linear_to_db(value: float | np.ndarray, floor: float = 1e-300) -> float | np.ndarray:
    """선형 전력비를 dB로 바꾼다."""
    return 10.0 * np.log10(np.maximum(np.asarray(value), floor))


def dbm_to_watts(dbm: float) -> float:
    """dBm을 watt로 바꾼다."""
    return float(10.0 ** ((dbm - 30.0) / 10.0))


def watts_to_dbm(watts: float | np.ndarray) -> float | np.ndarray:
    """watt를 dBm으로 바꾼다."""
    return linear_to_db(np.maximum(np.asarray(watts), 1e-300)) + 30.0


def path_loss_db(distance_m: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    1 m free-space 기준 log-distance path loss.

    PL(dB) = 32.4 + 20log10(f_GHz) + 10nlog10(d_m / 1m)
    """
    distance_m = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    return (
        32.4
        + 20.0 * np.log10(cfg.carrier_freq_ghz)
        + 10.0 * cfg.path_loss_exponent * np.log10(distance_m)
    )


def thermal_noise_power_watts(cfg: SystemConfig) -> float:
    """수신기 noise figure를 포함한 kTB 열잡음 전력을 watt로 계산한다."""
    noise_figure_linear = float(db_to_linear(cfg.rx_noise_figure_db))
    return BOLTZMANN_J_PER_K * cfg.temperature_k * cfg.bandwidth_hz * noise_figure_linear


def make_user_link_metrics(
    n_users: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
    distance_m: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """사용자별 거리, shadowing, path loss, 채널 전력 이득을 만든다."""
    if distance_m is None:
        distances = rng.uniform(cfg.distance_min_m, cfg.distance_max_m, size=n_users)
    else:
        distances = np.full(n_users, float(distance_m), dtype=float)

    shadowing_db = (
        rng.normal(0.0, cfg.shadowing_std_db, size=n_users)
        if cfg.shadowing_std_db > 0.0
        else np.zeros(n_users, dtype=float)
    )
    path_loss_with_shadowing_db = path_loss_db(distances, cfg) + shadowing_db
    channel_gain_linear = 10.0 ** (-path_loss_with_shadowing_db / 10.0)

    return {
        "distance_m": distances,
        "shadowing_db": shadowing_db,
        "path_loss_db": path_loss_with_shadowing_db,
        "channel_gain_linear": channel_gain_linear,
    }


def normalize_total_tx_power(
    tx_serial: np.ndarray,
    cfg: SystemConfig,
) -> Tuple[np.ndarray, float, float]:
    """
    전체 송신 안테나 평균 합산 전력이 cfg.tx_power_dbm이 되도록 시간 신호를 스케일한다.

    반환값은 정규화된 신호, 적용한 진폭 스케일, 목표 송신 전력[W]이다.
    """
    target_power_w = dbm_to_watts(cfg.tx_power_dbm)
    current_power_w = float(np.mean(np.sum(np.abs(tx_serial) ** 2, axis=1)))
    tx_scale = math.sqrt(target_power_w / max(current_power_w, 1e-300))
    return tx_serial * tx_scale, tx_scale, target_power_w


def generate_multipath_channels(
    n_users: int,
    cfg: SystemConfig,
    rng: np.random.Generator,
    channel_gain_linear: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    다중경로 Rayleigh 채널을 생성한다.

    출력:
        h_time: shape = (K, N_TAPS, N_RX_PER_UE, N_TX)
    """
    tap = np.arange(cfg.n_taps)
    pdp = np.exp(-tap / cfg.pdp_decay)
    pdp = pdp / np.sum(pdp)

    h = (
        rng.standard_normal((n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
        + 1j * rng.standard_normal((n_users, cfg.n_taps, cfg.n_rx_per_ue, cfg.n_tx))
    ) / math.sqrt(2.0)

    h *= np.sqrt(pdp)[None, :, None, None]

    if channel_gain_linear is not None:
        gains = np.asarray(channel_gain_linear, dtype=float).reshape(n_users)
        h *= np.sqrt(gains)[:, None, None, None]

    return h.astype(np.complex128)


def channel_frequency_response(h_time: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    시간영역 impulse response를 부반송파별 주파수 응답으로 변환한다.

    입력:
        h_time: shape = (K, N_TAPS, N_RX_PER_UE, N_TX)
    출력:
        H_f: shape = (K, N_FFT, N_RX_PER_UE, N_TX)
    """
    n_users = h_time.shape[0]
    h_pad = np.zeros((n_users, cfg.n_fft, cfg.n_rx_per_ue, cfg.n_tx), dtype=np.complex128)
    h_pad[:, : cfg.n_taps, :, :] = h_time
    return np.fft.fft(h_pad, n=cfg.n_fft, axis=1)


def apply_channel_one_user(x_time: np.ndarray, h_user: np.ndarray) -> np.ndarray:
    """
    한 단말에 대해 시간영역 다중경로 채널을 적용한다.

    입력:
        x_time: shape = (T, N_TX)
        h_user: shape = (N_TAPS, N_RX_PER_UE, N_TX)
    출력:
        y_time: shape = (T, N_RX_PER_UE)
    """
    n_time = x_time.shape[0]
    n_taps, n_rx, _ = h_user.shape
    y_time = np.zeros((n_time, n_rx), dtype=np.complex128)

    for ell in range(n_taps):
        # y[t] += H[ell] x[t-ell]
        y_time[ell:, :] += x_time[: n_time - ell, :] @ h_user[ell, :, :].T

    return y_time


def add_thermal_awgn(
    y_clean: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    링크버짓의 kTB thermal noise를 복소 AWGN으로 추가한다.

    noise_power_w는 복소수 잡음 전력 E[|n|^2]이며, 수신 신호 전력에 맞춰
    재조정하지 않는다. 따라서 path loss가 커지면 measured SNR이 실제로 낮아진다.
    """
    noise_power_w = thermal_noise_power_watts(cfg)
    noise = (
        rng.standard_normal(y_clean.shape) + 1j * rng.standard_normal(y_clean.shape)
    ) * math.sqrt(noise_power_w / 2.0)

    return y_clean + noise, noise_power_w


# ============================================================
# 5. 수신 결합 및 ZF 프리코딩
# ============================================================

def dominant_rx_combiner(H_user_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    4x8 단말 채널에서 1-stream 수신을 위한 combiner를 만든다.

    입력:
        H_user_k: shape = (N_RX_PER_UE, N_TX)
    출력:
        combiner: shape = (N_RX_PER_UE,)
        h_eff: shape = (N_TX,)
    """
    u, _, _ = np.linalg.svd(H_user_k, full_matrices=False)
    combiner = u[:, 0]
    combiner = combiner / max(np.linalg.norm(combiner), 1e-12)
    h_eff = combiner.conj().T @ H_user_k
    return combiner, h_eff


def zf_precoder(H_eff: np.ndarray) -> np.ndarray:
    """
    Downlink ZF precoder.

    입력:
        H_eff: shape = (N_ACTIVE_USERS, N_TX)
    출력:
        W: shape = (N_TX, N_ACTIVE_USERS)
    """
    n_active, n_tx = H_eff.shape
    if n_active > n_tx:
        raise ValueError("ZF precoding에서 동시 사용자 수는 송신 안테나 수 이하여야 한다.")

    # pseudo-inverse 기반 ZF
    W = np.linalg.pinv(H_eff)

    # 열별 정규화 후 전체 송신 전력 정규화
    for i in range(W.shape[1]):
        norm = np.linalg.norm(W[:, i])
        if norm > 1e-12:
            W[:, i] /= norm
    W /= math.sqrt(n_active)
    return W


def make_user_groups(n_users: int, max_streams: int) -> List[List[int]]:
    """단말 수가 8명을 넘으면 여러 그룹으로 나눈다."""
    return [
        list(range(start, min(start + max_streams, n_users)))
        for start in range(0, n_users, max_streams)
    ]


def build_precoder_context(
    H_f: np.ndarray,
    active_users: Sequence[int],
    subcarrier: int,
) -> Dict:
    """
    특정 사용자 그룹과 특정 부반송파에 대해 combiner와 precoder를 계산한다.
    """
    h_eff_rows = []
    combiners: Dict[int, np.ndarray] = {}
    H_user_map: Dict[int, np.ndarray] = {}

    for user_id in active_users:
        H_user_k = H_f[user_id, subcarrier, :, :]
        combiner, h_eff = dominant_rx_combiner(H_user_k)
        combiners[user_id] = combiner
        H_user_map[user_id] = H_user_k
        h_eff_rows.append(h_eff)

    H_eff = np.vstack(h_eff_rows)
    W = zf_precoder(H_eff)

    return {
        "active_users": list(active_users),
        "H_eff": H_eff,
        "W": W,
        "combiners": combiners,
        "H_user_map": H_user_map,
    }


# ============================================================
# 6. JSONL 데이터셋 보조 함수
# ============================================================

def complex_to_pair(z: complex) -> List[float]:
    """복소수 하나를 [real, imag]로 변환한다."""
    return [float(np.real(z)), float(np.imag(z))]


def complex_array_to_pairs(arr: np.ndarray) -> List:
    """복소수 배열을 [real, imag] 쌍의 nested list로 변환한다."""
    arr = np.asarray(arr)
    return np.stack([arr.real, arr.imag], axis=-1).astype(float).tolist()


def complex_array_to_feature(arr: np.ndarray) -> List[float]:
    """복소수 배열을 [real flatten, imag flatten] 순서의 실수 feature로 변환한다."""
    arr = np.asarray(arr)
    return np.concatenate([arr.real.ravel(), arr.imag.ravel()]).astype(float).tolist()


def make_dataset_record(
    *,
    cfg: SystemConfig,
    frame_id: int,
    distance_m: float,
    path_loss_db_value: float,
    shadowing_db: float,
    rx_power_dbm: float,
    noise_power_true_dbm: float,
    pre_combiner_snr_db: float,
    user_id: int,
    active_users: Sequence[int],
    stream_position: int,
    ofdm_symbol_index: int,
    subcarrier: int,
    rx_vector: np.ndarray,
    H_user_k: np.ndarray,
    W: np.ndarray,
    combiner: np.ndarray,
    desired_gain: complex,
    y_scalar: complex,
    x_hat: complex,
    tx_symbol: complex,
    tx_bits: np.ndarray,
) -> Dict:
    """
    딥러닝 equalizer 학습용 JSON record를 만든다.

    feature_vector는 딥러닝 입력으로 바로 쓰기 쉬운 1차원 실수 벡터이다.
    label은 송신 정답이다.
    """
    active_mask = [0] * cfg.max_streams
    active_users_padded = [-1] * cfg.max_streams
    for i, uid in enumerate(active_users):
        active_mask[i] = 1
        active_users_padded[i] = int(uid)

    W_pad = np.zeros((cfg.n_tx, cfg.max_streams), dtype=np.complex128)
    W_pad[:, : W.shape[1]] = W

    intended_precoder = W[:, stream_position]
    bps = bits_per_symbol(cfg.modulation)

    feature_vector = (
        complex_array_to_feature(np.array([x_hat]))
        + complex_array_to_feature(np.array([desired_gain]))
        + complex_array_to_feature(np.array([y_scalar]))
        + [
            float(abs(desired_gain)),
            float(np.angle(desired_gain)),
            float(noise_power_true_dbm),
            float(pre_combiner_snr_db),
        ]
    )

    return {
        "meta": {
            "frame_id": int(frame_id),
            "distance_m": float(distance_m),
            "path_loss_db": float(path_loss_db_value),
            "shadowing_db": float(shadowing_db),
            "rx_power_dbm": float(rx_power_dbm),
            "noise_power_true_dbm": float(noise_power_true_dbm),
            "pre_combiner_snr_db": float(pre_combiner_snr_db),
            "user_id": int(user_id),
            "active_users": [int(x) for x in active_users],
            "active_users_padded": active_users_padded,
            "active_mask": active_mask,
            "stream_position": int(stream_position),
            "ofdm_symbol_index": int(ofdm_symbol_index),
            "subcarrier_index": int(subcarrier),
            "modulation": cfg.modulation,
            "bits_per_symbol": int(bps),
            "n_tx": int(cfg.n_tx),
            "n_rx_per_ue": int(cfg.n_rx_per_ue),
            "feature_dim": int(len(feature_vector)),
        },
        "input": {
            "feature_vector": feature_vector,

            # 아래 항목들은 디버깅과 해석용이다.
            # 실제 학습에서는 feature_vector만 써도 된다.
            "equalized_symbol_x_hat": complex_to_pair(x_hat),
            "y_scalar": complex_to_pair(y_scalar),
            "rx_vector_4rx": complex_array_to_pairs(rx_vector),
            "channel_H_4x8": complex_array_to_pairs(H_user_k),
            "precoder_W_padded_8x8": complex_array_to_pairs(W_pad),
            "rx_combiner_4": complex_array_to_pairs(combiner),
            "intended_precoder_8": complex_array_to_pairs(intended_precoder),
            "desired_gain": complex_to_pair(desired_gain),
        },
        "label": {
            "tx_bits": np.asarray(tx_bits, dtype=int).reshape(-1).tolist(),
            "tx_symbol": complex_to_pair(tx_symbol),
            "symbol_class": bits_to_class_index(np.asarray(tx_bits, dtype=int).reshape(-1)),
        },
    }


def dataset_schema(cfg: SystemConfig) -> Dict:
    """JSONL 데이터셋의 입출력 구조 설명"""
    feature_dim = (
        2  # equalized_symbol_x_hat
        + 2  # desired_gain
        + 2  # y_scalar
        + 4  # gain_abs, gain_phase, noise_power_true_dbm, pre_combiner_snr_db
    )

    return {
        "format": "JSON Lines",
        "one_line": "one received subcarrier sample for one scheduled user",
        "feature_dim": feature_dim,
        "input_feature_order": [
            "equalized_symbol_x_hat: real, imag",
            "desired_gain: real, imag",
            "y_scalar: real, imag",
            "desired_gain_abs",
            "desired_gain_phase_rad",
            "noise_power_true_dbm",
            "pre_combiner_snr_db",
        ],
        "meta_notes": [
            "This is a perfect-CSI detector dataset: OFDM, MIMO combining, and equalization are done by the conventional receiver first.",
            "channel_H_4x8, rx_combiner_4, and desired_gain are debug fields. The compact feature_vector is centered on x_hat.",
            "pre_combiner_snr_db is based on pre-combiner received power and estimated noise; it is not post-equalizer SINR.",
        ],
        "label": {
            "tx_bits": f"{bits_per_symbol(cfg.modulation)} bits per symbol",
            "tx_symbol": "[real, imag]",
            "symbol_class": "integer class made from tx_bits",
        },
        "system": asdict(cfg),
    }


# ============================================================
# 7. 한 frame End-to-End 시뮬레이션
# ============================================================

def simulate_downlink_frame(
    *,
    cfg: SystemConfig,
    n_users: int,
    distance_m: Optional[float],
    rng: np.random.Generator,
    frame_id: int = 0,
    collect_json_records: bool = False,
    max_records: Optional[int] = None,
) -> Tuple[int, int, List[Dict]]:
    """
    한 frame의 전체 송수신을 수행한다.

    반환:
        error_bits, total_bits, json_records
    """
    cfg.validate()

    if not (1 <= n_users <= 16):
        raise ValueError("단말 수는 1~16 범위여야 한다.")

    bps = bits_per_symbol(cfg.modulation)
    groups = make_user_groups(n_users, cfg.max_streams)

    # --------------------------------------------------------
    # 송수신 관계 1) 모든 단말에 대한 다중경로 채널 생성
    # --------------------------------------------------------
    link_metrics = make_user_link_metrics(n_users, cfg, rng, distance_m=distance_m)
    h_time = generate_multipath_channels(
        n_users,
        cfg,
        rng,
        channel_gain_linear=link_metrics["channel_gain_linear"],
    )
    H_f = channel_frequency_response(h_time, cfg)

    # --------------------------------------------------------
    # 송수신 관계 2) 그룹/부반송파별 combiner, ZF precoder 계산
    # --------------------------------------------------------
    context: Dict[Tuple[int, int], Dict] = {}
    for group_id, active_users in enumerate(groups):
        for subcarrier in cfg.data_idx:
            context[(group_id, int(subcarrier))] = build_precoder_context(
                H_f, active_users, int(subcarrier)
            )

    # --------------------------------------------------------
    # 송신부 1) 비트 생성 -> 변조 -> subcarrier mapping -> precoding
    # --------------------------------------------------------
    tx_grid = np.zeros((cfg.n_fft, cfg.n_ofdm_symbols, cfg.n_tx), dtype=np.complex128)

    # 정답 저장용 배열
    tx_symbols = np.zeros(
        (n_users, cfg.n_data_subcarriers, cfg.n_ofdm_symbols), dtype=np.complex128
    )
    tx_bits = np.zeros(
        (n_users, cfg.n_data_subcarriers, cfg.n_ofdm_symbols, bps), dtype=np.int8
    )
    scheduled = np.zeros(
        (n_users, cfg.n_data_subcarriers, cfg.n_ofdm_symbols), dtype=bool
    )

    for t in range(cfg.n_ofdm_symbols):
        # 단말 수가 8명을 넘으면 OFDM symbol마다 그룹을 번갈아 스케줄링한다.
        # frame_id까지 포함해 그룹을 회전시키므로 OFDM symbol 수가 적어도 여러 frame에서 모든 그룹이 전송된다.
        global_symbol_index = frame_id * cfg.n_ofdm_symbols + t
        group_id = global_symbol_index % len(groups)
        active_users = groups[group_id]

        for data_pos, subcarrier in enumerate(cfg.data_idx):
            ctx = context[(group_id, int(subcarrier))]
            W = ctx["W"]

            s_vec = []
            for user_id in active_users:
                bits_u = rng.integers(0, 2, size=bps, dtype=np.int8)
                sym_u = modulate_bits(bits_u, cfg.modulation)[0]

                tx_bits[user_id, data_pos, t, :] = bits_u
                tx_symbols[user_id, data_pos, t] = sym_u
                scheduled[user_id, data_pos, t] = True
                s_vec.append(sym_u)

            s_vec = np.asarray(s_vec, dtype=np.complex128)
            tx_grid[subcarrier, t, :] = W @ s_vec

    # --------------------------------------------------------
    # 송신부 2) OFDM IFFT + CP 삽입
    # --------------------------------------------------------
    tx_blocks = ofdm_modulate(tx_grid, cfg)
    tx_serial = tx_blocks.reshape(-1, cfg.n_tx)
    tx_serial, tx_scale, _ = normalize_total_tx_power(tx_serial, cfg)

    # --------------------------------------------------------
    # 채널 1) 각 단말로 다중경로 채널 통과 + AWGN
    # --------------------------------------------------------
    rx_freq_all_users: List[np.ndarray] = []
    rx_power_dbm_per_user: List[float] = []
    pre_combiner_snr_db_per_user: List[float] = []
    noise_power_w = thermal_noise_power_watts(cfg)
    noise_power_true_dbm = float(watts_to_dbm(noise_power_w))

    for user_id in range(n_users):
        y_clean = apply_channel_one_user(tx_serial, h_time[user_id])
        rx_power_w = float(np.mean(np.abs(y_clean) ** 2))
        rx_power_dbm = float(watts_to_dbm(rx_power_w))
        y_noisy, _ = add_thermal_awgn(y_clean, cfg, rng)
        pre_combiner_snr_db = float(linear_to_db(rx_power_w / max(noise_power_w, 1e-300)))
        rx_power_dbm_per_user.append(rx_power_dbm)
        pre_combiner_snr_db_per_user.append(pre_combiner_snr_db)

        rx_blocks = y_noisy.reshape(cfg.n_ofdm_symbols, cfg.n_fft + cfg.n_cp, cfg.n_rx_per_ue)
        rx_freq = ofdm_demodulate(rx_blocks, cfg)
        rx_freq_all_users.append(rx_freq)

    # --------------------------------------------------------
    # 수신부 1) combiner 적용 -> gain equalization -> 복조 -> BER 계산
    # --------------------------------------------------------
    error_bits = 0
    total_bits = 0
    json_records: List[Dict] = []

    for t in range(cfg.n_ofdm_symbols):
        global_symbol_index = frame_id * cfg.n_ofdm_symbols + t
        group_id = global_symbol_index % len(groups)
        active_users = groups[group_id]

        for data_pos, subcarrier in enumerate(cfg.data_idx):
            ctx = context[(group_id, int(subcarrier))]
            W = ctx["W"]

            for stream_position, user_id in enumerate(active_users):
                if not scheduled[user_id, data_pos, t]:
                    continue

                rx_freq = rx_freq_all_users[user_id]
                y_vec = rx_freq[subcarrier, t, :]                 # shape = (4,)
                H_user_k = H_f[user_id, subcarrier, :, :]          # shape = (4,8)
                combiner = ctx["combiners"][user_id]

                # 4Rx -> 1개 scalar stream
                y_scalar = combiner.conj().T @ y_vec

                # perfect CSI로 계산한 원하는 stream의 effective channel gain
                desired_gain = tx_scale * (combiner.conj().T @ H_user_k @ W[:, stream_position])

                # gain equalization
                x_hat = y_scalar / (desired_gain + 1e-12)

                bits_hat = demodulate_symbols(np.array([x_hat]), cfg.modulation)
                bits_true = tx_bits[user_id, data_pos, t, :]

                error_bits += int(np.sum(bits_hat != bits_true))
                total_bits += int(bps)

                # 딥러닝 학습용 JSON 라벨 저장
                if collect_json_records:
                    if max_records is None or len(json_records) < max_records:
                        rec = make_dataset_record(
                            cfg=cfg,
                            frame_id=frame_id,
                            distance_m=link_metrics["distance_m"][user_id],
                            path_loss_db_value=link_metrics["path_loss_db"][user_id],
                            shadowing_db=link_metrics["shadowing_db"][user_id],
                            rx_power_dbm=rx_power_dbm_per_user[user_id],
                            noise_power_true_dbm=noise_power_true_dbm,
                            pre_combiner_snr_db=pre_combiner_snr_db_per_user[user_id],
                            user_id=user_id,
                            active_users=active_users,
                            stream_position=stream_position,
                            ofdm_symbol_index=t,
                            subcarrier=int(subcarrier),
                            rx_vector=y_vec,
                            H_user_k=H_user_k,
                            W=W,
                            combiner=combiner,
                            desired_gain=desired_gain,
                            y_scalar=y_scalar,
                            x_hat=x_hat,
                            tx_symbol=tx_symbols[user_id, data_pos, t],
                            tx_bits=bits_true,
                        )
                        json_records.append(rec)

    return error_bits, total_bits, json_records


# ============================================================
# 8. BER 실험 및 JSONL 생성
# ============================================================

def run_ber_experiment(
    *,
    cfg: SystemConfig,
    user_counts: Sequence[int],
    distance_sweep_m: Sequence[float],
) -> Dict[int, List[float]]:
    """단말 수와 거리를 바꾸면서 BER을 계산한다."""
    results: Dict[int, List[float]] = {}

    for n_users in user_counts:
        rng = np.random.default_rng(cfg.random_seed + 1000 * n_users)
        ber_list: List[float] = []

        for distance_m in distance_sweep_m:
            err_sum = 0
            bit_sum = 0

            for frame_id in range(cfg.n_frames):
                err, total, _ = simulate_downlink_frame(
                    cfg=cfg,
                    n_users=n_users,
                    distance_m=float(distance_m),
                    rng=rng,
                    frame_id=frame_id,
                    collect_json_records=False,
                )
                err_sum += err
                bit_sum += total

            ber = err_sum / max(bit_sum, 1)
            ber_list.append(float(ber))
            print(f"[BER] K={n_users:2d}, distance={distance_m:7.1f} m, BER={ber:.4e}")

        results[int(n_users)] = ber_list

    return results


def write_jsonl_dataset(
    *,
    cfg: SystemConfig,
    output_path: Path,
    n_users: int,
    distance_sweep_m: Sequence[float],
    frames_per_distance: int,
    max_records: int,
    records_per_distance: Optional[int] = None,
) -> int:
    """딥러닝 equalizer 학습용 JSONL 파일을 거리별 균등 quota로 생성한다."""
    rng = np.random.default_rng(cfg.random_seed + 9999)
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distances = [float(x) for x in distance_sweep_m]
    if not distances:
        return 0

    if records_per_distance is None:
        quota_per_distance = max(int(max_records), 0) // len(distances)
    else:
        quota_per_distance = max(int(records_per_distance), 0)

    with output_path.open("w", encoding="utf-8") as f:
        for distance_m in distances:
            distance_written = 0
            for frame_id in range(frames_per_distance):
                remaining_for_distance = quota_per_distance - distance_written
                if remaining_for_distance <= 0:
                    break

                _, _, records = simulate_downlink_frame(
                    cfg=cfg,
                    n_users=n_users,
                    distance_m=float(distance_m),
                    rng=rng,
                    frame_id=frame_id,
                    collect_json_records=True,
                    max_records=remaining_for_distance,
                )

                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += len(records)
                distance_written += len(records)

    return written


def save_ber_json(
    path: Path,
    cfg: SystemConfig,
    user_counts: Sequence[int],
    distance_sweep_m: Sequence[float],
    ber_results: Dict[int, List[float]],
) -> None:
    """BER 결과를 JSON으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "user_counts": [int(x) for x in user_counts],
        "distance_sweep_m": [float(x) for x in distance_sweep_m],
        "ber_results": {str(k): [float(v) for v in vals] for k, vals in ber_results.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_ber(
    path: Path,
    distance_sweep_m: Sequence[float],
    ber_results: Dict[int, List[float]],
) -> None:
    """BER vs distance 그래프를 저장한다. matplotlib이 없으면 건너뛴다."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[경고] matplotlib import 실패로 그래프 저장 생략: {exc}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for n_users, ber in ber_results.items():
        plt.semilogy(distance_sweep_m, np.maximum(ber, 1e-7), marker="o", label=f"K={n_users}")

    plt.xlabel("Distance [m]")
    plt.ylabel("BER")
    plt.title("5 GHz Downlink MU-MIMO OFDM: Link Budget + Multipath Rayleigh")
    plt.grid(True, which="both", linestyle=":")
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def print_flow() -> None:
    """송수신 관계에 따른 처리 순서를 출력한다."""
    lines = [
        "1. 거리 sweep 값에 맞춰 사용자별 path loss와 shadowing 계산",
        "2. path loss가 반영된 다중경로 Rayleigh 채널 생성",
        "3. 기지국에서 사용자별 랜덤 비트 생성",
        "4. 비트를 BPSK/QPSK/16QAM 심볼로 변조",
        "5. 단말 수가 8명을 넘으면 사용자 그룹으로 분할",
        "6. perfect CSI 기반으로 BS ZF precoder 계산",
        "7. 변조 심볼을 precoder에 곱해 8Tx 주파수 격자 생성",
        "8. OFDM IFFT와 Cyclic Prefix 삽입",
        "9. 전체 8Tx 평균 전력을 링크버짓 송신 전력으로 정규화",
        "10. 8Tx 신호가 각 단말의 4Rx 채널을 통과",
        "11. kTB와 receiver noise figure 기반 thermal AWGN 추가",
        "12. 수신단에서 CP 제거 후 FFT",
        "13. perfect CSI combiner 적용",
        "14. perfect CSI desired gain으로 equalization",
        "15. hard demodulation 후 BER 계산",
        "16. AI detector 학습용 x_hat feature와 정답 label 저장",
    ]

    print("\n===== 송수신 End-to-End 순서 =====")
    for line in lines:
        print(line)
    print("=================================\n")


# ============================================================
# 9. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downlink MU-MIMO OFDM simulation with JSON labels for DL equalizer"
    )
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="BER 실험용 단말 수 목록")
    parser.add_argument("--distance-sweep", nargs="+", type=float,
                        default=[10, 30, 50, 100, 200, 300],
                        help="BER 실험용 거리[m] 목록")
    parser.add_argument("--frames", type=int, default=50,
                        help="거리 하나당 frame 반복 수")
    parser.add_argument("--ofdm-symbols", type=int, default=8,
                        help="frame 하나당 OFDM symbol 수")
    parser.add_argument("--modulation", type=str, default="QPSK",
                        choices=["BPSK", "QPSK", "16QAM"],
                        help="변조 방식")
    parser.add_argument("--dataset-users", type=int, default=16,
                        help="JSONL 데이터셋 생성 시 사용할 단말 수")
    parser.add_argument("--dataset-frames-per-distance", type=int, default=4,
                        help="JSONL 데이터셋 생성용 거리당 frame 수")
    parser.add_argument("--max-json-records", type=int, default=60000,
                        help="JSONL에 저장할 최대 record 수")
    parser.add_argument("--records-per-distance", type=int, default=None,
                        help="거리별 JSONL record 저장량. 지정하면 max-json-records보다 우선한다.")
    parser.add_argument("--carrier-freq-ghz", type=float, default=5.0,
                        help="carrier frequency [GHz]")
    parser.add_argument("--bandwidth-hz", type=float, default=20e6,
                        help="receiver noise bandwidth [Hz]")
    parser.add_argument("--tx-power-dbm", type=float, default=30.0,
                        help="total BS transmit power across all TX antennas [dBm]")
    parser.add_argument("--rx-noise-figure-db", type=float, default=7.0,
                        help="UE receiver noise figure [dB]")
    parser.add_argument("--temperature-k", type=float, default=290.0,
                        help="receiver noise temperature [K]")
    parser.add_argument("--distance-min-m", type=float, default=10.0,
                        help="minimum random UE distance [m]")
    parser.add_argument("--distance-max-m", type=float, default=300.0,
                        help="maximum random UE distance [m]")
    parser.add_argument("--path-loss-exponent", type=float, default=3.0,
                        help="log-distance path loss exponent")
    parser.add_argument("--shadowing-std-db", type=float, default=6.0,
                        help="log-normal shadowing standard deviation [dB]")
    parser.add_argument("--out-dir", type=str, default="outputs_mu_mimo_ofdm",
                        help="결과 저장 폴더")
    parser.add_argument("--no-plot", action="store_true",
                        help="BER 그래프 생략")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SystemConfig(
        n_ofdm_symbols=args.ofdm_symbols,
        n_frames=args.frames,
        modulation=args.modulation,
        max_json_records=args.max_json_records,
        carrier_freq_ghz=args.carrier_freq_ghz,
        bandwidth_hz=args.bandwidth_hz,
        tx_power_dbm=args.tx_power_dbm,
        rx_noise_figure_db=args.rx_noise_figure_db,
        temperature_k=args.temperature_k,
        distance_min_m=args.distance_min_m,
        distance_max_m=args.distance_max_m,
        path_loss_exponent=args.path_loss_exponent,
        shadowing_std_db=args.shadowing_std_db,
    )
    cfg.validate()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_flow()

    # 설정 저장
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # BER 실험
    print("[시작] BER 실험")
    ber_results = run_ber_experiment(
        cfg=cfg,
        user_counts=args.users,
        distance_sweep_m=args.distance_sweep,
    )

    ber_path = out_dir / "ber_results.json"
    save_ber_json(ber_path, cfg, args.users, args.distance_sweep, ber_results)
    print(f"[저장] BER 결과: {ber_path}")

    if not args.no_plot:
        plot_path = out_dir / "ber_vs_distance.png"
        plot_ber(plot_path, args.distance_sweep, ber_results)
        print(f"[저장] BER 그래프: {plot_path}")

    # 데이터셋 schema 저장
    schema_path = out_dir / "dl_equalizer_dataset_schema.json"
    schema_path.write_text(
        json.dumps(dataset_schema(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[저장] JSONL schema: {schema_path}")

    # 딥러닝 학습용 JSONL 생성
    print("[시작] 딥러닝 equalizer용 JSONL 생성")
    dataset_path = out_dir / "dl_equalizer_dataset.jsonl"
    n_written = write_jsonl_dataset(
        cfg=cfg,
        output_path=dataset_path,
        n_users=args.dataset_users,
        distance_sweep_m=args.distance_sweep,
        frames_per_distance=args.dataset_frames_per_distance,
        max_records=args.max_json_records,
        records_per_distance=args.records_per_distance,
    )
    print(f"[저장] JSONL 데이터셋: {dataset_path} ({n_written} records)")
    print("[완료] label은 실제 송신 신호가 아니라 시뮬레이션용 정답으로 저장된 값이다.")


if __name__ == "__main__":
    main()
