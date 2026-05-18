"""awgn_noise.m  Python 이식."""
import numpy as np


def awgn_noise(hx, SNR, rng=None):
    """
    SNR(dB) 기준 AWGN 추가.
    return: (y, noise_power)
    """
    if rng is None:
        rng = np.random
    p, q = hx.shape
    noise_power = 10 ** (-SNR / 10)
    n = np.sqrt(noise_power / 2) * (rng.standard_normal((p, q)) +
                                    1j * rng.standard_normal((p, q)))
    return hx + n, noise_power
