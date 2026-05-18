"""
등화기 — 7-way 비교용 모두 구현.

[A] Basic Rx             : 그냥 hard demap (등화 없음)
[B] Standard MMSE        : per-subcarrier 1-tap MMSE, 간섭+잡음 분산 사용
[C] WL-MMSE              : Widely-Linear MMSE — IQ image leakage 수학 보상
[D] LMMSE-MMSE           : empirical LMMSE 로 CSI refine 후 MMSE
[E] ZF (ComNet 용)       : x = y / h
"""
import numpy as np


def standard_mmse_per_user(user_Dsym, He_est, Wd, user_idx, noise_var, fft_len):
    """
    MATLAB main.m line 213-222 정확 매칭.
        rx_Dsym_MMSE[k] = user_Dsym[k] * conj(g_kk) / (|g_kk|² + total_noise_var)
    """
    rx = np.zeros(fft_len, dtype=np.complex128)
    for k in range(fft_len):
        H_est_k = He_est[k]                    # (K, K)
        Wd_k = Wd[k]                           # (K, K) (after ZF precoding)
        G_k = H_est_k @ Wd_k                   # (K, K)
        g_kk = G_k[user_idx, user_idx]
        interf_var = np.sum(np.abs(G_k[user_idx, :])**2) - np.abs(g_kk)**2
        total_noise = interf_var + noise_var
        w = np.conj(g_kk) / (np.abs(g_kk)**2 + total_noise)
        rx[k] = user_Dsym[k] * w
    return rx


def wl_mmse_per_user(user_Dsym, He_est, Wd, user_idx,
                    noise_var, fft_len, mu, nu):
    """
    MATLAB main.m line 226-251 정확 매칭.
    augmented receiver:
        y_aug = [y; conj(y)]
        h_aug = [[mu*g_kk,         nu*conj(g_kk)],
                 [conj(nu)*g_kk,   conj(mu)*conj(g_kk)]]
        R_nn  = total_noise * [[|mu|²+|nu|², 2*mu*nu],
                              [2*conj(mu)*conj(nu), |mu|²+|nu|²]]
        W_wl = (h_aug^H · h_aug + R_nn)^-1 · h_aug^H
        x_est_aug = W_wl · y_aug
        rx[k] = x_est_aug[0]
    """
    rx = np.zeros(fft_len, dtype=np.complex128)
    mu2_plus_nu2 = np.abs(mu)**2 + np.abs(nu)**2

    for k in range(fft_len):
        H_est_k = He_est[k]; Wd_k = Wd[k]
        G_k = H_est_k @ Wd_k
        g_kk = G_k[user_idx, user_idx]
        interf_var = np.sum(np.abs(G_k[user_idx, :])**2) - np.abs(g_kk)**2
        total_noise = interf_var + noise_var

        h_aug = np.array([
            [mu * g_kk,              nu * np.conj(g_kk)],
            [np.conj(nu) * g_kk,     np.conj(mu) * np.conj(g_kk)],
        ])
        R_nn = total_noise * np.array([
            [mu2_plus_nu2,                  2 * mu * nu],
            [2 * np.conj(mu) * np.conj(nu), mu2_plus_nu2],
        ])
        y_aug = np.array([user_Dsym[k], np.conj(user_Dsym[k])])
        W_wl = np.linalg.solve(h_aug.conj().T @ h_aug + R_nn,
                                h_aug.conj().T)
        x_aug = W_wl @ y_aug
        rx[k] = x_aug[0]
    return rx


def zf_per_user_via_diag(user_Dsym, h_diag_per_k):
    """
    ZF 등화 — 사용자 d 의 diagonal channel 만 사용해 단순 나누기.
        user_Dsym : (fft_len,)
        h_diag    : (fft_len,)  per-subcarrier diagonal element He[k, d, d]
    return x_zf  : (fft_len,)
    """
    return user_Dsym / h_diag_per_k


def fit_lmmse(h_est_train, h_true_train, ridge=1e-6):
    """
    Empirical LMMSE 회귀 (real-valued).
        h_est_train, h_true_train : (N, K) complex (per-subcarrier diagonal)
    return W : (2K, 2K) real
    """
    N, K = h_est_train.shape
    ri_est  = np.concatenate([h_est_train.real, h_est_train.imag], axis=-1).astype(np.float32)
    ri_true = np.concatenate([h_true_train.real, h_true_train.imag], axis=-1).astype(np.float32)
    XtX = ri_est.T @ ri_est / N
    Xty = ri_est.T @ ri_true / N
    W_T = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0], dtype=np.float32), Xty)
    return W_T.T.astype(np.float32)


def apply_lmmse(h_est, W_lmmse):
    """h_est (N, K) complex → h_refined same shape."""
    N, K = h_est.shape
    ri = np.concatenate([h_est.real, h_est.imag], axis=-1).astype(np.float32)
    out = ri @ W_lmmse.T
    return (out[..., :K] + 1j * out[..., K:]).astype(np.complex128)
