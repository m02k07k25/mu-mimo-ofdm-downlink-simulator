from __future__ import annotations

import math

import numpy as np


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


def _bits_to_ints(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    values = np.zeros(bits.shape[0], dtype=np.int64)
    for bit_pos in range(bits.shape[1]):
        values = (values << 1) | bits[:, bit_pos].astype(np.int64)
    return values


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


def qam_modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    modulation = modulation.upper()
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    bps = bits_per_symbol(modulation)
    if bits.size % bps != 0:
        raise ValueError("Number of bits must be a multiple of bits per symbol")

    axis_bits = bps // 2
    labels = _gray_labels(axis_bits)
    levels = _pam_levels(axis_bits)
    b = bits.reshape(-1, bps)
    re_label = _bits_to_ints(b[:, :axis_bits])
    im_label = _bits_to_ints(b[:, axis_bits:])
    re_index = np.empty_like(re_label)
    im_index = np.empty_like(im_label)
    for index, gray_label in enumerate(labels):
        re_index[re_label == gray_label] = index
        im_index[im_label == gray_label] = index
    symbols = levels[re_index] + 1j * levels[im_index]
    return (symbols / _qam_normalization(modulation)).astype(np.complex64)


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
