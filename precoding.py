"""
ZF_precoding.m  /  steer_precoding.m  Python 이식.
"""
import numpy as np


def zf_precoding(sym, H):
    """
    MATLAB ZF_precoding 정확 재현.
        sym  : (N_s, fft_len)
        H    : (fft_len, Mr, N_tx)
    return:
        sym_hat : (N_tx, fft_len)
        K       : (fft_len,)          정규화 인자 (사용 안 됨)
        W       : (fft_len, N_tx, N_s) precoding matrix
    """
    fft_len, Mr, N_tx = H.shape
    N_s, fft_len2 = sym.shape
    assert fft_len == fft_len2

    K = np.zeros(fft_len, dtype=np.complex128)
    W = np.zeros((fft_len, N_tx, N_s), dtype=np.complex128)
    sym_hat = np.zeros((N_tx, fft_len), dtype=np.complex128)

    for k in range(fft_len):
        t_H = H[k, :, :]                              # (Mr, N_tx)
        G = np.linalg.pinv(t_H)                       # (N_tx, Mr)  ← MATLAB pinv
        sym_hat[:, k] = G @ sym[:, k]                 # (N_tx,)
        W[k, :, :] = G                                # G shape: (N_tx, Mr)=N_s
        K[k] = np.sqrt(np.trace(G @ G.conj().T))
    return sym_hat, K, W


def steer_precoding(fc, ant, angle, mode=1):
    """
    MATLAB steer_precoding 정확 재현.
        fc    : 중심주파수 (Hz)
        ant   : [N_row, N_col, dy_per_lambda, dz_per_lambda]
        angle : (2, Nrf)  — [theta; phi] (rad)
        mode  : 1 = all-connected, 2 = sub-connected
    return W: precoding 매트릭스
        mode 1: (N, Nrf)
        mode 2: (N*Nrf, Nrf)
    """
    ant = np.asarray(ant, dtype=float)
    angle = np.atleast_2d(angle)
    if ant[0] * ant[1] == 1:
        mode = 2
    _, Nrf = angle.shape

    c = 3e8
    lamda = c / fc
    k_wn = 2 * np.pi / lamda
    dy = ant[2] * lamda
    dz = ant[3] * lamda
    N = int(ant[0] * ant[1])

    # 안테나 위치 행렬
    # MATLAB: temp1 = repmat(0:ant(1)-1, ant(2), 1) → shape (ant(2), ant(1))
    #         temp2 = repmat(0:ant(2)-1, 1, ant(1)) → shape (1, ant(1)*ant(2))
    a1 = int(ant[0]); a2 = int(ant[1])
    temp1 = np.tile(np.arange(a1), (a2, 1))           # (a2, a1)
    temp2 = np.tile(np.arange(a2), (1, a1))           # (1, a1*a2)
    # ant_mat = [0,  temp1(:)*dy,  (temp2.')*dz]
    # temp1(:) in MATLAB is column-major flatten: shape (a2*a1,) with column-major order
    # numpy 등가: temp1.flatten(order='F')
    col_y = temp1.flatten(order='F') * dy             # (N,)
    col_z = temp2.T.flatten() * dz                    # (a1*a2,)
    ant_mat = np.column_stack([np.zeros(N), col_y, col_z])   # (N, 3)

    def trans_f(theta, phi):
        return np.array([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])

    if mode == 1:
        W = np.zeros((N, Nrf), dtype=np.complex128)
        for i in range(Nrf):
            theta = angle[0, i]; phi = angle[1, i]
            W[:, i] = np.exp(-1j * k_wn * (ant_mat @ trans_f(theta, phi)))
        return W
    else:                                              # mode == 2
        W = np.zeros((N * Nrf, Nrf), dtype=np.complex128)
        for i in range(Nrf):
            theta = angle[0, i]; phi = angle[1, i]
            W[i*N:(i+1)*N, i] = np.exp(-1j * k_wn * (ant_mat @ trans_f(theta, phi)))
        return W
