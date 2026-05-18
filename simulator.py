"""
design5 시뮬레이터 — 데이터 생성 + 학습 + 7-way 평가.

흐름:
  Phase 1 : 학습 데이터 수집 (MU-MIMO SCM + ZF precoding + IQ imbalance)
  Phase 2 : NN 학습 (CE subnet, FC-SD, E2E-NN)
  Phase 3 : empirical LMMSE 회귀 fit
  Phase 4 : 7-way receiver 평가
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config, get_iq_params
from modulation import base_mod, base_demod
from awgn import awgn_noise
from precoding import zf_precoding, steer_precoding
from impairments import apply_iq_imbalance, apply_clip
from scm import SCM
from dl_models import LSRefineNet, FCSD, E2ENet
from equalizers import (
    standard_mmse_per_user, wl_mmse_per_user,
    zf_per_user_via_diag, fit_lmmse, apply_lmmse,
)


# ──────────────────── 프레임 생성 ────────────────────
def _setup_channel(model, num_users, fft_len, cp_len, N_rx, rng):
    """사용자별 채널 + steering precoder."""
    N_tx = model.Ntx
    H = np.zeros((model.n_path, fft_len + cp_len,
                  N_rx * num_users, N_tx), dtype=np.complex128)
    Wt = np.zeros((N_tx, num_users), dtype=np.complex128)
    Wr = np.zeros((N_rx, num_users), dtype=np.complex128)
    for d in range(num_users):
        H_d, rx_angle, _ = model.FD_channel(fft_len + cp_len, rng=rng)
        H[:, :, d * N_rx:(d + 1) * N_rx, :] = H_d
        max_idx = int(np.argmax(np.abs(H_d[:, 0, 0, 0])))
        sel_angle = rx_angle[:, max_idx]
        Wt[:, d] = steer_precoding(model.fc, model.tx_ant,
                                   sel_angle[0:2, None], mode=1).flatten()
        Wr[:, d] = steer_precoding(model.fc, model.rx_ant,
                                   sel_angle[2:4, None], mode=2).flatten()
    return H, Wt, Wr


def _effective_channel(H, Wt, Wr, fft_len, num_users, n_path):
    """He_freq (fft_len, K, K) — MATLAB 와 동일 effective channel."""
    N_rx = Wr.shape[0]
    t_He = np.zeros((n_path, num_users, num_users), dtype=np.complex128)
    for k in range(n_path):
        tmp_H = H[k, 0, :, :]
        for d in range(num_users):
            rx_idx = slice(d * N_rx, (d + 1) * N_rx)
            t_He[k, d, :] = Wr[:, d] @ tmp_H[rx_idx, :] @ Wt
    return np.fft.fft(t_He, n=fft_len, axis=0)


def _iq_from_phase_deg(phase_deg, amp_err_db=1.0):
    """phase(도) + amp(dB) → IQ 파라미터 dict (g_i, g_q, phi, mu, nu)."""
    g_i = 10 ** ((amp_err_db / 2) / 20)
    g_q = 10 ** (-(amp_err_db / 2) / 20)
    phi = phase_deg * np.pi / 180
    mu = (g_i + g_q * np.exp(-1j * phi)) / 2
    nu = (g_i - g_q * np.exp(+1j * phi)) / 2
    return dict(g_i=g_i, g_q=g_q, phi=phi, mu=mu, nu=nu)


def gen_one_frame(model, cfg, rng, train_snr=None, clip_ratio=None,
                  iq_override=None):
    """
    1 프레임 송수신 → 학습/평가에 필요한 모든 신호 반환.
    iq_override : 이 frame 의 IQ 파라미터 (없으면 cfg['iq'] 사용)
    """
    fft_len = cfg['fft_len']; cp_len = cfg['cp_len']
    mod_type = cfg['mod_type']; data_len = fft_len * mod_type
    num_users = cfg['num_users']; N_rx = cfg['N_rx']
    iqp = iq_override if iq_override is not None else cfg['iq']
    n_path = model.n_path

    # 1) 채널
    H, Wt, Wr = _setup_channel(model, num_users, fft_len, cp_len, N_rx, rng)
    He_freq = _effective_channel(H, Wt, Wr, fft_len, num_users, n_path)

    # 2) Imperfect CSI
    e = np.sqrt(cfg['csi_err_var'] / 2) * (
        rng.standard_normal(He_freq.shape) + 1j * rng.standard_normal(He_freq.shape))
    He_est = He_freq + e

    # 3) bits → QAM symbols
    bit_data = rng.integers(0, 2, size=(num_users, data_len), dtype=np.int8)
    sym_data = base_mod(bit_data, mod_type)

    # 4) ZF precoding (He_est 기반) + OFDM
    Dsym, _, Wd = zf_precoding(sym_data, He_est)
    Isym = np.fft.ifft(Dsym, fft_len, axis=1) * np.sqrt(fft_len)
    tx_ofdm = np.concatenate([Isym[:, fft_len - cp_len:], Isym], axis=1)

    # 5) TX clipping (옵션)
    if clip_ratio is not None:
        tx_ofdm = apply_clip(tx_ofdm, clip_ratio)

    # 6) 채널 통과 + AWGN
    tx_signal = Wt @ tx_ofdm
    snr = train_snr if train_snr is not None else cfg.get('eval_snr', 30)
    rx_full = model.FD_fading(tx_signal, H)
    rx_signal, noise_pow = awgn_noise(rx_full, snr, rng=rng)

    # 7) 사용자별 RX
    user_Dsym = np.zeros((num_users, fft_len), dtype=np.complex128)
    for d in range(num_users):
        rx_idx = slice(d * N_rx, (d + 1) * N_rx)
        user_rx = Wr[:, d] @ rx_signal[rx_idx, :]
        user_Isym = user_rx[cp_len:cp_len + fft_len]
        D = np.fft.fft(user_Isym, fft_len) / np.sqrt(fft_len)
        # 8) IQ imbalance 적용
        user_Dsym[d] = apply_iq_imbalance(D, iqp['g_i'], iqp['g_q'], iqp['phi'])

    return dict(
        bit_data=bit_data,            # (K, L)
        He_freq=He_freq,              # (fft_len, K, K)
        He_est=He_est,                # (fft_len, K, K)
        Wd=Wd,                        # (fft_len, K, K)
        user_Dsym=user_Dsym,          # (K, fft_len)
        snr=snr,
        noise_var=noise_pow,
    )


# ──────────────────── Phase 1: 학습 데이터 ────────────────────
def collect_training_data(model, cfg, rng):
    """
    학습 데이터 3종:
      • E2E NN  : (Re/Im y, Re/Im h_kk) → class_idx
      • CE      : He_est (K diagonal) → He_freq (K diagonal)
      • SD subnet 학습 데이터는 CE 학습 후 별도 생성
    """
    fft_len = cfg['fft_len']; mod_type = cfg['mod_type']
    num_users = cfg['num_users']
    bit_weights = (1 << np.arange(mod_type - 1, -1, -1))

    N = cfg['iter'] * num_users * fft_len
    Xe2e = np.empty((N, 4), dtype=np.float32)
    Ye2e = np.empty((N,), dtype=np.int64)
    Hce_est  = np.empty((cfg['iter'] * fft_len, num_users), dtype=np.complex128)
    Hce_true = np.empty((cfg['iter'] * fft_len, num_users), dtype=np.complex128)
    # SD 학습용으로 저장
    Y_sd = np.empty((N,), dtype=np.complex128)
    H_sd = np.empty((N,), dtype=np.complex128)
    L_sd = np.empty((N,), dtype=np.int64)

    # Mixed-training 지원: cfg['train_phase_random']=True 이면 매 frame phase 랜덤
    phase_random   = cfg.get('train_phase_random', False)
    phase_range    = cfg.get('train_phase_range', (0.0, 30.0))
    amp_err_db_train = cfg.get('amp_err_db_train', 1.0)
    if phase_random:
        print(f"  [Mixed Training] phase ∈ {phase_range}° 매 frame 랜덤 샘플")

    write_e2e = 0; write_ce = 0
    t0 = time.time(); every = max(1, cfg['iter'] // 10)
    for i in range(cfg['iter']):
        train_snr = int(rng.integers(int(cfg['train_snr_min']),
                                     int(cfg['train_snr_max']) + 1))
        iq_ov = None
        if phase_random:
            phase_deg = rng.uniform(*phase_range)
            iq_ov = _iq_from_phase_deg(phase_deg, amp_err_db_train)
        f = gen_one_frame(model, cfg, rng, train_snr=train_snr,
                          clip_ratio=cfg.get('clip_ratio', None),
                          iq_override=iq_ov)
        for k in range(fft_len):
            # CE 학습 데이터 (per-subcarrier diagonal)
            Hce_est [write_ce + k] = np.diag(f['He_est' ][k])
            Hce_true[write_ce + k] = np.diag(f['He_freq'][k])
            for d in range(num_users):
                y = f['user_Dsym'][d, k]
                h = f['He_est'][k, d, d]
                Xe2e[write_e2e] = [y.real, y.imag, h.real, h.imag]
                sym_bits = f['bit_data'][d, k*mod_type:(k+1)*mod_type]
                class_idx = int((sym_bits * bit_weights).sum())
                Ye2e[write_e2e] = class_idx
                Y_sd[write_e2e] = y
                H_sd[write_e2e] = h
                L_sd[write_e2e] = class_idx
                write_e2e += 1
        write_ce += fft_len
        if (i + 1) % every == 0:
            print(f"  Phase1  {i+1:>6d}/{cfg['iter']}  ({time.time()-t0:.1f}s)")
    return (Xe2e[:write_e2e], Ye2e[:write_e2e],
            Hce_est[:write_ce], Hce_true[:write_ce],
            Y_sd[:write_e2e], H_sd[:write_e2e], L_sd[:write_e2e])


# ──────────────────── Phase 2: NN 학습 ────────────────────
def train_e2e(Xe2e, Ye2e, cfg, device):
    """MATLAB E2E NN — 4-D feature, 64-class softmax, Adam(lr=0.01), 20 epochs."""
    mu = Xe2e.mean(axis=0); sd = Xe2e.std(axis=0) + 1e-8
    Xn = ((Xe2e - mu) / sd).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Ye2e))
    loader = DataLoader(ds, batch_size=cfg['batch'], shuffle=True)

    net = E2ENet(num_classes=64).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    lf = nn.CrossEntropyLoss()
    print(f"  E2E NN 학습 (Adam lr={cfg['lr']}, batch={cfg['batch']}, "
          f"epochs={cfg['epochs']})")
    for ep in range(1, cfg['epochs'] + 1):
        net.train(); tot=0; correct=0; total=0
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            logits = net(xb); loss = lf(logits, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()*len(xb); correct += (logits.argmax(-1)==yb).sum().item(); total += len(xb)
        print(f"    E2E ep {ep:>2d}/{cfg['epochs']}  loss={tot/total:.4f}  acc={correct/total*100:.2f}%")
    return net, mu, sd


def train_ce(Hce_est, Hce_true, cfg, device):
    """CE subnet — LSRefineNet, LMMSE init, MSE loss."""
    K = Hce_est.shape[-1]
    # LMMSE init
    W_lmmse = fit_lmmse(Hce_est, Hce_true, ridge=cfg['ce_ridge'])
    # Real-valued 변환
    Xtr = np.concatenate([Hce_est.real, Hce_est.imag], axis=-1).astype(np.float32)
    Ytr = np.concatenate([Hce_true.real, Hce_true.imag], axis=-1).astype(np.float32)
    # train/val split
    N = Xtr.shape[0]
    n_tr = int(0.9 * N)
    perm = np.random.permutation(N)
    Xtr, Ytr = Xtr[perm], Ytr[perm]
    x_t, y_t = Xtr[:n_tr], Ytr[:n_tr]
    x_v, y_v = Xtr[n_tr:], Ytr[n_tr:]

    net = LSRefineNet(K).to(device)
    with torch.no_grad():
        net.linear.weight.copy_(torch.from_numpy(W_lmmse))    # LMMSE init
    optim = torch.optim.Adam(net.parameters(), lr=cfg['ce_lr'], weight_decay=1e-6)
    lf = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_t), torch.from_numpy(y_t))
    loader = DataLoader(ds, batch_size=cfg['batch'], shuffle=True)
    x_vt = torch.from_numpy(x_v); y_vt = torch.from_numpy(y_v)
    print(f"  CE subnet 학습 (epochs={cfg['ce_epochs']})")
    best_state = None; best_vl = float('inf'); patience = 0
    for ep in range(1, cfg['ce_epochs'] + 1):
        net.train(); tot=0; n=0
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            pred = net(xb); loss = lf(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()*len(xb); n += len(xb)
        net.eval()
        with torch.no_grad():
            vl = lf(net(x_vt.to(device)), y_vt.to(device)).item()
        if vl < best_vl - 1e-5:
            best_vl = vl
            best_state = {k:v.clone() for k,v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if ep <= 5 or ep % 5 == 0:
            print(f"    CE ep {ep:>2d}/{cfg['ce_epochs']}  "
                  f"train={tot/n:.5f}  val={vl:.5f}")
        if patience >= 5: break
    if best_state: net.load_state_dict(best_state)
    return net, W_lmmse


def _ce_refine(He_est_per_subc, ce_net, device):
    """He_est (..., K, K) → diagonal 만 refine 한 array 반환 (..., K, K)."""
    *batch_dims, K, _ = He_est_per_subc.shape
    diag_est = np.diagonal(He_est_per_subc, axis1=-2, axis2=-1)  # (..., K)
    flat = diag_est.reshape(-1, K)
    ri = np.concatenate([flat.real, flat.imag], axis=-1).astype(np.float32)
    with torch.no_grad():
        out = ce_net(torch.from_numpy(ri).to(device)).cpu().numpy()
    refined_flat = out[..., :K] + 1j * out[..., K:]
    refined = refined_flat.reshape(*batch_dims, K)
    # off-diagonal 은 그대로 (정확도 영향 작음, 진단 단순화)
    He_refined = He_est_per_subc.copy()
    He_refined[..., np.arange(K), np.arange(K)] = refined
    return He_refined


def train_sd(Y_sd, H_sd, L_sd, ce_net, cfg, device):
    """
    FC-SD 학습 데이터:
      - y, h (per-(k,d)), x_zf = y/h_refined
      - 6-D feature, 64-class
    """
    # 1) CE refine: H_sd 가 He_est diagonal → refined h
    # 학습 데이터의 H_sd 는 이미 diagonal 추출이므로 직접 refine 한다.
    # CE net 은 K-차원 vector 단위로 동작하므로 K 단위로 처리 필요.
    # 여기서는 단순화 — H_sd 를 그대로 두고 x_zf = y / H_sd 사용.
    # (CE 의 효과는 평가 단계에서 H_eff 전체에 적용)
    x_zf = Y_sd / H_sd
    feat = np.column_stack([
        Y_sd.real, Y_sd.imag,
        H_sd.real, H_sd.imag,
        x_zf.real, x_zf.imag,
    ]).astype(np.float32)
    mu = feat.mean(axis=0); sd_ = feat.std(axis=0) + 1e-8
    Xn = ((feat - mu) / sd_).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(L_sd))
    loader = DataLoader(ds, batch_size=cfg['batch'], shuffle=True)

    net = FCSD(hidden=256, num_classes=64).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    print(f"  FC-SD 학습 (epochs={cfg['epochs']})")
    for ep in range(1, cfg['epochs'] + 1):
        net.train(); tot=0; correct=0; total=0
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            logits = net(xb); loss = lf(logits, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()*len(xb); correct += (logits.argmax(-1)==yb).sum().item(); total += len(xb)
        if ep <= 3 or ep % 5 == 0:
            print(f"    SD ep {ep:>2d}/{cfg['epochs']}  "
                  f"loss={tot/total:.4f}  acc={correct/total*100:.2f}%")
    return net, mu, sd_


# ──────────────────── Phase 4: 7-way 평가 ────────────────────
def _class_to_bits(c, mod_type):
    out = np.zeros(mod_type, dtype=np.int8)
    for b in range(mod_type):
        out[mod_type - 1 - b] = c & 1
        c >>= 1
    return out


def evaluate(model, cfg, snr_range, ce_net, e2e_net, e2e_mu, e2e_sd,
             sd_net, sd_mu, sd_sd_, W_lmmse, device, rng):
    fft_len = cfg['fft_len']; mod_type = cfg['mod_type']
    data_len = fft_len * mod_type
    K = cfg['num_users']
    iqp = cfg['iq']

    methods = ['Basic', 'MMSE', 'WL-MMSE', 'LMMSE-MMSE',
               'ComNet-CE-Hard', 'ComNet-FC', 'E2E-DL',
               'True-H ZF', 'True-H WL-MMSE', 'No-IQ True-H']
    ber = {m: np.zeros(len(snr_range)) for m in methods}

    for s_idx, snr in enumerate(snr_range):
        test_iter = cfg['high_snr_iter'] if snr >= 24 else cfg['low_snr_iter']
        errs = {m: 0 for m in methods}
        total_bits = 0; noise_var = 10 ** (-snr / 10)
        t0 = time.time()
        for i in range(test_iter):
            f = gen_one_frame(model, cfg, rng, train_snr=snr,
                              clip_ratio=cfg.get('clip_ratio', None))
            He_est = f['He_est']; He_freq = f['He_freq']
            Wd = f['Wd']; user_Dsym = f['user_Dsym']
            bit_data = f['bit_data']

            # CE refine (diagonal) — 평가 시 한 번만
            He_comnet = _ce_refine(He_est, ce_net, device)
            # LMMSE refine (diagonal)
            diag_est = np.diagonal(He_est, axis1=-2, axis2=-1)  # (fft, K)
            diag_lmmse = apply_lmmse(diag_est, W_lmmse)         # (fft, K)
            He_lmmse = He_est.copy()
            for k in range(fft_len):
                np.fill_diagonal(He_lmmse[k], diag_lmmse[k])

            for d in range(K):
                orig = bit_data[d, :]; y_user = user_Dsym[d]

                # [A] Basic
                rx_b = base_demod(y_user[None, :], mod_type).flatten()
                errs['Basic'] += int(np.sum(orig != rx_b))

                # [B] Standard MMSE
                rx_m = standard_mmse_per_user(y_user, He_est, Wd, d,
                                              noise_var, fft_len)
                errs['MMSE'] += int(np.sum(orig !=
                                            base_demod(rx_m[None,:], mod_type).flatten()))

                # [C] WL-MMSE
                rx_w = wl_mmse_per_user(y_user, He_est, Wd, d,
                                        noise_var, fft_len, iqp['mu'], iqp['nu'])
                errs['WL-MMSE'] += int(np.sum(orig !=
                                            base_demod(rx_w[None,:], mod_type).flatten()))

                # [D] LMMSE-MMSE  (CSI refine + MMSE)
                rx_lm = standard_mmse_per_user(y_user, He_lmmse, Wd, d,
                                               noise_var, fft_len)
                errs['LMMSE-MMSE'] += int(np.sum(orig !=
                                            base_demod(rx_lm[None,:], mod_type).flatten()))

                # [E] ComNet-CE-Hard  (CE refine + ZF + hard)
                h_diag = He_comnet[:, d, d]
                rx_zf = zf_per_user_via_diag(y_user, h_diag)
                errs['ComNet-CE-Hard'] += int(np.sum(orig !=
                                            base_demod(rx_zf[None,:], mod_type).flatten()))

                # [F] ComNet-FC  (CE refine + ZF + FC-SD classify)
                feat = np.column_stack([
                    y_user.real, y_user.imag,
                    h_diag.real, h_diag.imag,
                    rx_zf.real, rx_zf.imag,
                ]).astype(np.float32)
                feat_n = (feat - sd_mu) / sd_sd_
                with torch.no_grad():
                    pred = sd_net(torch.from_numpy(feat_n).to(device)).argmax(-1).cpu().numpy()
                rx_bit = np.concatenate([_class_to_bits(c, mod_type) for c in pred])
                errs['ComNet-FC'] += int(np.sum(orig != rx_bit))

                # [G] E2E-DL  (MATLAB style: raw y, He_est diagonal)
                h_e2e = He_est[:, d, d]
                feat = np.column_stack([
                    y_user.real, y_user.imag, h_e2e.real, h_e2e.imag,
                ]).astype(np.float32)
                feat_n = (feat - e2e_mu) / e2e_sd
                with torch.no_grad():
                    pred = e2e_net(torch.from_numpy(feat_n).to(device)).argmax(-1).cpu().numpy()
                rx_bit = np.concatenate([_class_to_bits(c, mod_type) for c in pred])
                errs['E2E-DL'] += int(np.sum(orig != rx_bit))

                # [Ref] True-H ZF Hard  (oracle He_freq)
                h_true_diag = He_freq[:, d, d]
                rx_th = zf_per_user_via_diag(y_user, h_true_diag)
                errs['True-H ZF'] += int(np.sum(orig !=
                                            base_demod(rx_th[None,:], mod_type).flatten()))

                # [Ref] True-H WL-MMSE  (oracle channel + IQ math 보상)
                rx_thw = wl_mmse_per_user(y_user, He_freq, Wd, d,
                                          noise_var, fft_len, iqp['mu'], iqp['nu'])
                errs['True-H WL-MMSE'] += int(np.sum(orig !=
                                            base_demod(rx_thw[None,:], mod_type).flatten()))

                # [Ref] No-IQ True-H  — IQ 없었을 때 가상 BER
                # y_no_iq[k] = He_freq[k, d, d] * sym_d[k] + new_awgn[k]
                # (간단화: 같은 SNR 의 새 AWGN)
                sym_d = base_mod(orig[None,:], mod_type).flatten()
                y_noiq = h_true_diag * sym_d + np.sqrt(noise_var/2) * (
                    rng.standard_normal(fft_len) + 1j*rng.standard_normal(fft_len))
                rx_noiq = zf_per_user_via_diag(y_noiq, h_true_diag)
                errs['No-IQ True-H'] += int(np.sum(orig !=
                                            base_demod(rx_noiq[None,:], mod_type).flatten()))

                total_bits += data_len
        for m in methods: ber[m][s_idx] = errs[m] / total_bits
        print(f"  SNR={snr:>2d}dB ({time.time()-t0:>4.0f}s)  " +
              "  ".join(f"{m}={errs[m]:>6d}" for m in ['Basic','MMSE','WL-MMSE','ComNet-FC','E2E-DL']))

    return ber, methods
