"""
하드웨어 결함 모델.

IQ Imbalance — MATLAB main.m 의 RX 측 결함:
    I_part = real(y)
    Q_part = imag(y)
    y' = (g_i * I) + j*(g_q * Q * cos(phi) - g_i * I * sin(phi))

TX-side clipping (optional) — design3 의 PA saturation:
    amplitude > clip_ratio * σ 인 샘플 클리핑
"""
import numpy as np


def apply_iq_imbalance(y, g_i, g_q, phi):
    """
    MATLAB 수식 1:1.
        y       : complex array
        g_i,g_q : I/Q 이득
        phi     : 위상 오차 (rad)
    """
    I = np.real(y); Q = np.imag(y)
    return (g_i * I) + 1j * (g_q * Q * np.cos(phi) - g_i * I * np.sin(phi))


def apply_clip(x_time, clip_ratio):
    """송신단 PA saturation. amplitude > clip_ratio·σ 면 잘라냄."""
    sigma = np.sqrt(np.mean(np.abs(x_time) ** 2, axis=-1, keepdims=True))
    A_max = clip_ratio * sigma
    amp = np.abs(x_time)
    scale = np.where(amp > A_max, A_max / np.clip(amp, 1e-12, None), 1.0)
    return x_time * scale
