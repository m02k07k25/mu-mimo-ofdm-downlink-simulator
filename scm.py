"""
SCM.m  Python 이식.
3GPP TR 38.900 기반 3D Spatial Channel Model.
main.m 에서 사용하는 메서드만 우선 정확하게 이식:
    ant(N_rx, N_tx)
    FD_channel(sample_len)
    FD_fading(sig, coeff)
그 외 메서드도 가능한 호환되게 유지.
"""
import numpy as np


class SCM:
    """3GPP TR 38.900 SCM (Spatial Channel Model)."""

    def __init__(self):
        # 3D 위치 / 송수신 안테나 방향 초기화
        self.p_src = np.array([0., 0., 0.])
        self.p_dst = np.array([1., 0., 0.])
        self.abr_src = np.array([0., 0., 0.])
        self.abr_dst = np.array([np.pi, 0., 0.])

        # Small-scale 파라미터
        self.fc = 800e6
        self.lamda = None
        self.fs = 20e6
        self.Ts = None
        self.tx_ant = [1, 1, 0.5, 0.5]
        self.rx_ant = [1, 1, 0.5, 0.5]
        self.Ntx = None
        self.Nrx = None
        self.n_path = 7
        self.n_mray = 15
        self.n_ray = None
        self.asd = 3.0
        self.zsd = 3.0
        self.asa = 3.0
        self.zsa = 3.0
        self.xpr_mu = 8.0
        self.xpr_std = 3.0
        self.pdp = None

        # Large-scale
        self.Gt = 0; self.Gr = 0; self.L = 0
        self.distance_rate = 1
        self.exp_beta = 3
        self.sdw_std = 0
        self.los = 0
        self.los_flag = 1
        self.K = 15
        self.No = -174
        self.ZoD_L = np.pi / 2; self.AoD_L = 0
        self.ZoA_L = np.pi / 2; self.AoA_L = 0

        # 방향성 함수 (기본 isotropic 1)
        self.tx_theta = lambda theta: 1.0
        self.tx_phi   = lambda phi:   0.0
        self.rx_theta = lambda theta: 1.0
        self.rx_phi   = lambda phi:   0.0

        # 좌표 변환
        self.cvt_S2R = lambda theta, phi: np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)])

    # ───────── 송수신 안테나 설정 ─────────
    def ant(self, N_rx, N_tx):
        Rx_ant = [N_rx, 1, 0.5, 0.5]
        Tx_ant = [N_tx, 1, 0.5, 0.5]
        self.rx_ant = Rx_ant; self.tx_ant = Tx_ant
        self.Nrx = N_rx; self.Ntx = N_tx
        return Rx_ant, Tx_ant

    # ───────── 안테나 위치 ─────────
    def init_d(self):
        self.Ntx = int(self.tx_ant[0] * self.tx_ant[1])
        tdy = self.tx_ant[2] * self.lamda
        tdz = self.tx_ant[3] * self.lamda
        a1 = int(self.tx_ant[0]); a2 = int(self.tx_ant[1])
        temp1 = np.tile(np.arange(a1), (a2, 1))
        temp2 = np.tile(np.arange(a2), (1, a1))
        col_y = temp1.flatten(order='F') * tdy
        col_z = temp2.T.flatten() * tdz
        self.tx_d = np.column_stack([np.zeros(self.Ntx), col_y, col_z])

        self.Nrx = int(self.rx_ant[0] * self.rx_ant[1])
        rdy = self.rx_ant[2] * self.lamda
        rdz = self.rx_ant[3] * self.lamda
        a1 = int(self.rx_ant[0]); a2 = int(self.rx_ant[1])
        temp3 = np.tile(np.arange(a1), (a2, 1))
        temp4 = np.tile(np.arange(a2), (1, a1))
        col_y = temp3.flatten(order='F') * rdy
        col_z = temp4.T.flatten() * rdz
        self.rx_d = np.column_stack([np.zeros(self.Nrx), col_y, col_z])

    # ───────── path 별 power 할당 ─────────
    def def_pow(self):
        if self.pdp is None:
            pw = np.exp(-(np.arange(1, self.n_path + 1)) / 5.0)
            self.pdp = pw / pw.sum()
        else:
            self.pdp = np.asarray(self.pdp, dtype=float)
            self.n_path = len(self.pdp)
            self.pdp = self.pdp / self.pdp.sum()
        if self.n_ray is None:
            self.n_ray = np.ones(self.n_path, dtype=int) * self.n_mray

    # ───────── cluster·ray 별 angle 생성 ─────────
    def gen_angle(self, rng=None):
        rng = rng or np.random
        # 중심 각도 (4, n_path)
        angle = np.empty((4, self.n_path))
        angle[0, :] = rng.uniform(0, np.pi, size=self.n_path)         # ZoD
        angle[1, :] = -np.pi / 2 + rng.uniform(0, np.pi, size=self.n_path)  # AoD
        angle[2, :] = rng.uniform(0, np.pi, size=self.n_path)         # ZoA
        angle[3, :] = -np.pi / 2 + rng.uniform(0, np.pi, size=self.n_path)  # AoA

        res_angle = []
        for i in range(self.n_path):
            if np.all(self.n_ray == 1):
                tmp_angle = angle.copy()
            else:
                nray = int(self.n_ray[i])
                tmp_angle = rng.standard_normal((4, nray))
                tmp_angle[0] = tmp_angle[0] * (self.zsd * np.pi / 180) + angle[0, i]
                tmp_angle[1] = tmp_angle[1] * (self.asd * np.pi / 180) + angle[1, i]
                tmp_angle[2] = tmp_angle[2] * (self.zsa * np.pi / 180) + angle[2, i]
                tmp_angle[3] = tmp_angle[3] * (self.asa * np.pi / 180) + angle[3, i]
            res_angle.append(tmp_angle)
        return res_angle, angle

    # ───────── per-ray 채널 계수 ─────────
    def _ray_cal(self, sample_len, ZoD, AoD, ZoA, AoA, xpr, vel, rng):
        # 편파 결합 계수
        trx_coef = np.array([[self.rx_theta(ZoA), self.rx_phi(AoA)]])  # (1, 2)
        if xpr == 0:
            mat = np.exp(2j * np.pi * rng.uniform(size=(1,))) * np.array([[1, 0], [0, -1]])
        else:
            r = rng.uniform(size=(2, 2))
            mat = np.exp(2j * np.pi * r) * np.array([[1, 1 / np.sqrt(xpr)],
                                                     [1 / np.sqrt(xpr), 1]])
        trx_coef = trx_coef @ mat
        tx_pol = np.array([self.tx_theta(ZoD), self.tx_phi(AoD)])      # (2,)
        # MATLAB 결과는 스칼라 (1×2)·(2,) = 스칼라가 아니라 ... 다시 트레이스:
        # trx_coef = trx_coef * sub_rx * sub_tx.'
        # 여기서 sub_rx (Nrx,) , sub_tx (Ntx,)
        scalar = trx_coef @ tx_pol                                     # (1,)
        # 안테나 array response
        rx_r = self.cvt_S2R(ZoA, AoA)
        sub_rx = np.exp(2j * np.pi * (self.rx_d @ rx_r) / self.lamda)   # (Nrx,)
        tx_r = self.cvt_S2R(ZoD, AoD)
        sub_tx = np.exp(2j * np.pi * (self.tx_d @ tx_r) / self.lamda)   # (Ntx,)
        trx_tmp = scalar[0] * np.outer(sub_rx, sub_tx)                 # (Nrx, Ntx)

        # Doppler
        if np.all(vel == 0):
            dop_t = np.ones(sample_len, dtype=np.complex128)
        else:
            t_sample = np.arange(sample_len) * self.Ts
            dop_t = np.exp(2j * np.pi * (rx_r @ vel) / self.lamda * t_sample)

        # broadcast across time
        return dop_t[:, None, None] * trx_tmp[None, :, :]              # (sample_len, Nrx, Ntx)

    # ───────── FD_channel ─────────
    def FD_channel(self, sample_len, i_vel=None, rng=None):
        """
        return:
            r_coeff (n_path, sample_len, Nrx, Ntx) complex
            c_ang   (4, n_path)
            res_ang list of (4, n_ray[i])
        """
        rng = rng or np.random
        # 속도 처리 (km/h → m/s)
        if i_vel is None:
            vel = np.array([0., 0., 0.])
        else:
            vel = np.atleast_1d(np.asarray(i_vel, dtype=float)) * 5 / 18
            if len(vel) < 3:
                vel = np.concatenate([vel, np.zeros(3 - len(vel))])

        self.lamda = 3e8 / self.fc
        self.Ts = 1.0 / self.fs
        self.init_d()

        self.def_pow()
        res_ang, c_ang = self.gen_angle(rng=rng)

        # XPR
        xpr_dB = rng.standard_normal((self.n_path, self.n_mray)) * self.xpr_std + self.xpr_mu
        xpr = 10 ** (xpr_dB / 10)

        # cluster 별 채널
        coeff = np.zeros((self.n_path + 1, sample_len, self.Nrx, self.Ntx),
                         dtype=np.complex128)
        f_idx = -1
        for i in range(self.n_path):
            if self.pdp[i] == 0:
                continue
            if f_idx < 0:
                f_idx = i

            tmp_coeff = np.zeros((sample_len, self.Nrx, self.Ntx),
                                 dtype=np.complex128)
            ang_i = res_ang[i]
            for j in range(int(self.n_ray[i])):
                sub = self._ray_cal(sample_len,
                                    ang_i[0, j], ang_i[1, j],
                                    ang_i[2, j], ang_i[3, j],
                                    xpr[i, j], vel, rng)
                # pas (Power Angular Spectrum) — default uniform 1/sqrt(n_ray)
                pas_w = 1.0 / np.sqrt(ang_i.shape[1])
                tmp_coeff = tmp_coeff + sub * pas_w
            coeff[i] = tmp_coeff * np.sqrt(self.pdp[i])

        # LOS (옵션, 보통 사용 안 함)
        if self.los and self.los_flag:
            Kr = 10 ** (self.K / 10)
            coeff = np.sqrt(1.0 / (Kr + 1)) * coeff
            nlos_tmp = coeff[f_idx].copy()
            coeff_los = self._ray_cal(sample_len, self.ZoD_L, self.AoD_L,
                                      self.ZoA_L, self.AoA_L, 0, vel, rng)
            coeff[f_idx] = nlos_tmp + np.sqrt(Kr / (Kr + 1)) * coeff_los

        # path-loss + shadowing
        p_loss = -10 * self.exp_beta * np.log10(self.distance_rate)
        shadowing = rng.standard_normal() * self.sdw_std
        loss = 10 ** ((p_loss + shadowing) / 10)
        coeff = coeff * np.sqrt(loss)

        r_coeff = coeff[:self.n_path]
        return r_coeff, c_ang, res_ang

    # ───────── FD_fading (시간영역 채널 통과) ─────────
    def FD_fading(self, sig, coeff):
        """
        sig   : (Ntx, sym_len)
        coeff : (tap_len, sym_len, Nrx, Ntx)
        return rx_sig : (Nrx, sym_len + tap_len - 1)
        """
        tap_len, sym_len, N_rx, N_tx = coeff.shape
        # temp[tap, sym, rx] = sum_tx coeff[tap, sym, rx, tx] * sig[tx, sym]
        temp = np.einsum('tsri,is->tsr', coeff, sig)                   # (tap, sym, rx)

        # rx_sig[rx, n] = sum_tap temp[tap, n - tap, rx]   (convolution-like)
        rx_sig = np.zeros((N_rx, sym_len + tap_len - 1), dtype=np.complex128)
        for tap in range(tap_len):
            rx_sig[:, tap:tap + sym_len] += temp[tap].T                # (rx, sym)
        return rx_sig


if __name__ == "__main__":
    # 빠른 sanity check
    rng = np.random.default_rng(0)
    m = SCM()
    m.n_path = 7
    m.ant(4, 8)
    h, c_ang, _ = m.FD_channel(80, rng=rng)
    print(f"H shape: {h.shape}  (expect (7, 80, 4, 8))")
    print(f"c_ang shape: {c_ang.shape}  (expect (4, 7))")
    sig = (rng.standard_normal((8, 80)) + 1j * rng.standard_normal((8, 80)))
    rx = m.FD_fading(sig, h)
    print(f"rx shape: {rx.shape}  (expect (4, 80+7-1=86))")
