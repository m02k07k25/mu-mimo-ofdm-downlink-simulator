"""
design5 — ComNet + WL-MMSE / IQ 30° 융합 시뮬레이션 설정.

두 아이디어를 합친 통합 receiver 비교:
  • MATLAB python_port 의 MU-MIMO 2-user SCM + IQ 30° 시나리오
  • design3/4 의 ComNet 하이브리드 (CE subnet + SD subnet)
"""
import numpy as np


class Config:
    # ───────────── 시스템 (MATLAB 매칭) ─────────────
    FFT_LEN     = 64
    CP_LEN      = 16
    MOD_TYPE    = 6          # 64-QAM
    NUM_USERS   = 2
    N_TX        = 8          # BS 안테나
    N_RX        = 4          # UE 안테나 (per user)
    TX_ANT_CFG  = [8, 1, 0.5, 0.5]
    RX_ANT_CFG  = [4, 1, 0.5, 0.5]
    N_PATH      = 7

    # ───────────── 하드웨어 결함 ─────────────
    AMP_ERR_DB  = 1.0        # I/Q 진폭 비대칭 (1dB)
    PHASE_ERR_DEG = 15.0     # I/Q 위상 (★ 15° — Standard 부품 환경)
    CLIP_RATIO  = None       # None = 클리핑 없음, e.g. 1.6 = TX 클리핑 활성화

    # ───────────── 채널 추정 ─────────────
    CSI_ERROR_VAR = 0.001    # 15° 에선 CSI 오차가 상대적으로 ↑ → 줄여 IQ 효과 분리

    # ───────────── 학습 (15° 튜닝) ─────────────
    TRAIN_ITER  = 5000       # 15° 의 미세한 IQ 패턴 학습 위해 증량
    TRAIN_SNR_MIN = 15.0
    TRAIN_SNR_MAX = 35.0
    EPOCHS      = 30         # 수렴까지 더 길게
    BATCH       = 1024
    LR          = 0.001      # PyTorch Adam 안정 학습률

    # ComNet CE subnet
    CE_EPOCHS   = 50
    CE_LR       = 1e-3
    CE_RIDGE    = 1e-6

    # ───────────── 평가 SNR sweep ─────────────
    SNR_MIN     = 10
    SNR_MAX     = 40         # 15° 천장이 30dB 이하에 도달 → 40 까지 확장
    SNR_STEP    = 2
    LOW_SNR_EVAL_ITER  = 200   # < 24 dB
    HIGH_SNR_EVAL_ITER = 600   # ≥ 24 dB

    # ───────────── 저장 경로 ─────────────
    RESULT_DIR  = 'results'
    SEED        = 42


def get_iq_params():
    """IQ Imbalance 파라미터 — config 와 일치."""
    amp = Config.AMP_ERR_DB
    g_i = 10 ** ((amp / 2) / 20)
    g_q = 10 ** (-(amp / 2) / 20)
    phi = Config.PHASE_ERR_DEG * np.pi / 180
    # WL-MMSE augmented receiver 의 mu, nu
    mu = (g_i + g_q * np.exp(-1j * phi)) / 2
    nu = (g_i - g_q * np.exp(+1j * phi)) / 2
    return dict(g_i=g_i, g_q=g_q, phi=phi, mu=mu, nu=nu)


def get_snr_range():
    return list(range(Config.SNR_MIN, Config.SNR_MAX + 1, Config.SNR_STEP))
