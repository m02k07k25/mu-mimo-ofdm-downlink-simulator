"""
base_mod.m / base_demod.m  Python 이식.

mod_scheme : BPSK=1, QPSK=2, 16QAM=4, 64QAM=6, 256QAM=8
입력 bits  : shape (M, L)  — M rows of L bits each
출력 syms  : shape (M, L/mod_scheme)
"""
import numpy as np


def base_mod(data, mod_scheme):
    """MATLAB base_mod 정확 재현."""
    data = np.atleast_2d(data).astype(np.int8)

    if mod_scheme == 1:                                      # BPSK
        return (data * 2 - 1).astype(np.complex128)

    if mod_scheme == 2:                                      # QPSK
        M, L = data.shape
        if L % 2 != 0:
            data = np.concatenate([data, np.zeros((M, 1), dtype=np.int8)], axis=1)
            L += 1
        odd = (data[:, 0:L-1:2] * 2 - 1) * 1j
        even = data[:, 1:L:2] * 2 - 1
        return (odd + even) * 0.7071

    if mod_scheme == 4:                                      # 16-QAM
        M, L = data.shape
        while L % 4 != 0:
            data = np.concatenate([data, np.zeros((M, 1), dtype=np.int8)], axis=1)
            L += 1
        b1 = data[:, 0:L-3:4]; b2 = data[:, 1:L-2:4]
        b3 = data[:, 2:L-1:4]; b4 = data[:, 3:L:4]
        first  = b1 * 4 - 2
        second = (b2 != b1).astype(np.int8) * 2 - 1
        third  = (b3 * 4 - 2) * 1j
        fourth = ((b4 != b3).astype(np.int8) * 2 - 1) * 1j
        return (first + second + third + fourth) * 0.3162

    if mod_scheme == 6:                                      # 64-QAM
        M, L = data.shape
        while L % 6 != 0:
            data = np.concatenate([data, np.zeros((M, 1), dtype=np.int8)], axis=1)
            L += 1
        b1 = data[:, 0:L-5:6]; b2 = data[:, 1:L-4:6]
        b3 = data[:, 2:L-3:6]; b4 = data[:, 3:L-2:6]
        b5 = data[:, 4:L-1:6]; b6 = data[:, 5:L  :6]
        first  = b1 * 8 - 4
        second = (b2 != b1).astype(np.int8) * 4 - 2
        third  = (np.bitwise_xor(np.bitwise_xor(b1, b2), b3)) * 2 - 1
        fourth = (b4 * 8 - 4) * 1j
        fifth  = ((b4 != b5).astype(np.int8) * 4 - 2) * 1j
        sixth  = ((np.bitwise_xor(np.bitwise_xor(b4, b5), b6)) * 2 - 1) * 1j
        return (first + second + third + fourth + fifth + sixth) * 0.1543

    if mod_scheme == 8:                                      # 256-QAM
        M, L = data.shape
        while L % 8 != 0:
            data = np.concatenate([data, np.zeros((M, 1), dtype=np.int8)], axis=1)
            L += 1
        b1 = data[:, 0:L-7:8]; b2 = data[:, 1:L-6:8]
        b3 = data[:, 2:L-5:8]; b4 = data[:, 3:L-4:8]
        b5 = data[:, 4:L-3:8]; b6 = data[:, 5:L-2:8]
        b7 = data[:, 6:L-1:8]; b8 = data[:, 7:L  :8]
        first  = b1 * 16 - 8
        second = (b2 != b1).astype(np.int8) * 8 - 4
        third  = (np.bitwise_xor(np.bitwise_xor(b1, b2), b3)) * 4 - 2
        fourth = (np.bitwise_xor(np.bitwise_xor(np.bitwise_xor(b1, b2), b3), b4)) * 2 - 1
        fifth   = (b5 * 16 - 8) * 1j
        sixth   = ((b6 != b5).astype(np.int8) * 8 - 4) * 1j
        seventh = ((np.bitwise_xor(np.bitwise_xor(b5, b6), b7)) * 4 - 2) * 1j
        eighth  = ((np.bitwise_xor(np.bitwise_xor(np.bitwise_xor(b5, b6), b7), b8)) * 2 - 1) * 1j
        return (first + second + third + fourth +
                fifth + sixth + seventh + eighth) * 0.0767

    raise ValueError(f"Unknown mod_scheme: {mod_scheme}")


def base_demod(mod_data, mod_scheme):
    """MATLAB base_demod 정확 재현 — Gray-coded hard decision."""
    mod_data = np.atleast_2d(mod_data)
    RE = np.real(mod_data); IM = np.imag(mod_data)

    if mod_scheme == 1:
        return (mod_data > 0).astype(np.int8).reshape(-1)

    if mod_scheme == 2:
        M, L = mod_data.shape
        odd = (IM > 0).astype(np.int8)
        even = (RE > 0).astype(np.int8)
        temp = np.stack([odd, even], axis=0)
        return temp.transpose(1, 2, 0).reshape(M, L * 2)

    if mod_scheme == 4:
        M, L = mod_data.shape
        b = [
            (RE > 0).astype(np.int8),
            (np.abs(RE) < 0.6325).astype(np.int8),
            (IM > 0).astype(np.int8),
            (np.abs(IM) < 0.6325).astype(np.int8),
        ]
        temp = np.stack(b, axis=0)
        return temp.transpose(1, 2, 0).reshape(M, L * 4)

    if mod_scheme == 6:
        M, L = mod_data.shape
        b = [
            (RE > 0).astype(np.int8),
            (np.abs(RE) < 0.6172).astype(np.int8),
            ((np.abs(RE) > 0.3086) & (np.abs(RE) < 0.9258)).astype(np.int8),
            (IM > 0).astype(np.int8),
            (np.abs(IM) < 0.6172).astype(np.int8),
            ((np.abs(IM) > 0.3086) & (np.abs(IM) < 0.9258)).astype(np.int8),
        ]
        temp = np.stack(b, axis=0)
        return temp.transpose(1, 2, 0).reshape(M, L * 6)

    if mod_scheme == 8:
        M, L = mod_data.shape
        b = [
            (RE > 0).astype(np.int8),
            (np.abs(RE) < 0.6136).astype(np.int8),
            ((np.abs(RE) > 0.3068) & (np.abs(RE) < 0.9204)).astype(np.int8),
            (((np.abs(RE) > 0.1534) & (np.abs(RE) < 0.4602)) |
             ((np.abs(RE) > 0.7670) & (np.abs(RE) < 1.0738))).astype(np.int8),
            (IM > 0).astype(np.int8),
            (np.abs(IM) < 0.6136).astype(np.int8),
            ((np.abs(IM) > 0.3068) & (np.abs(IM) < 0.9204)).astype(np.int8),
            (((np.abs(IM) > 0.1534) & (np.abs(IM) < 0.4602)) |
             ((np.abs(IM) > 0.7670) & (np.abs(IM) < 1.0738))).astype(np.int8),
        ]
        temp = np.stack(b, axis=0)
        return temp.transpose(1, 2, 0).reshape(M, L * 8)

    raise ValueError(f"Unknown mod_scheme: {mod_scheme}")


if __name__ == "__main__":
    # Sanity: 64-QAM round-trip
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 60), dtype=np.int8)
    sym = base_mod(bits, 6)
    rec = base_demod(sym, 6)
    print(f"64-QAM round-trip 일치: {np.array_equal(bits, rec)}")
    print(f"  sym shape: {sym.shape}")
    print(f"  E|x|² = {np.mean(np.abs(sym)**2):.4f} (1.0 근사)")
