"""
design5 — ComNet + WL-MMSE / IQ 30° 융합 엔트리포인트.

사용:
    python3 main.py                          # 빠른 테스트 (--iter 1000)
    python3 main.py --iter 5000              # 중간
    python3 main.py --iter 8000              # MATLAB 원본 학습량
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from config import Config, get_iq_params, get_snr_range
from scm import SCM
from simulator import collect_training_data, train_e2e, train_ce, train_sd, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--iter', type=int, default=Config.TRAIN_ITER)
    p.add_argument('--epochs', type=int, default=Config.EPOCHS)
    p.add_argument('--batch', type=int, default=Config.BATCH)
    p.add_argument('--lr', type=float, default=Config.LR)
    p.add_argument('--ce-epochs', type=int, default=Config.CE_EPOCHS)
    p.add_argument('--ce-lr', type=float, default=Config.CE_LR)
    p.add_argument('--low-snr-iter', type=int, default=Config.LOW_SNR_EVAL_ITER)
    p.add_argument('--high-snr-iter', type=int, default=Config.HIGH_SNR_EVAL_ITER)
    p.add_argument('--clip-ratio', type=float, default=None,
                   help='TX 클리핑 (None=비활성)')
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--out-dir', default=Config.RESULT_DIR)
    p.add_argument('--seed', type=int, default=Config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    cfg = dict(
        fft_len=Config.FFT_LEN, cp_len=Config.CP_LEN,
        mod_type=Config.MOD_TYPE, num_users=Config.NUM_USERS,
        N_rx=Config.N_RX, N_tx=Config.N_TX,
        iq=get_iq_params(),
        csi_err_var=Config.CSI_ERROR_VAR,
        iter=args.iter, epochs=args.epochs, batch=args.batch, lr=args.lr,
        ce_epochs=args.ce_epochs, ce_lr=args.ce_lr,
        ce_ridge=Config.CE_RIDGE,
        train_snr_min=Config.TRAIN_SNR_MIN, train_snr_max=Config.TRAIN_SNR_MAX,
        low_snr_iter=args.low_snr_iter, high_snr_iter=args.high_snr_iter,
        clip_ratio=args.clip_ratio,
    )
    snr_range = get_snr_range()

    print("=" * 70)
    print("design5  —  ComNet + WL-MMSE / IQ 30° 융합")
    print(f"  MU-MIMO {cfg['num_users']} users × Nt={cfg['N_tx']}, Nr={cfg['N_rx']}")
    print(f"  64-QAM, IQ Imbalance {Config.AMP_ERR_DB}dB + {Config.PHASE_ERR_DEG}°")
    print(f"  TX clipping: {cfg['clip_ratio']}")
    print(f"  학습 iter = {cfg['iter']}  (MATLAB 원본: 8000)")
    print(f"  평가: SNR {snr_range} dB")
    print("=" * 70)

    # SCM 모델
    model = SCM(); model.n_path = Config.N_PATH
    model.ant(Config.N_RX, Config.N_TX)

    # ─── Phase 1 ───
    print("\n[Phase 1] 학습 데이터 수집")
    Xe2e, Ye2e, Hce_est, Hce_true, Y_sd, H_sd, L_sd = collect_training_data(model, cfg, rng)
    print(f"  E2E 샘플 {Xe2e.shape},  CE 샘플 {Hce_est.shape}")

    # ─── Phase 2: E2E NN ───
    print("\n[Phase 2a] E2E NN 학습 (MATLAB-style)")
    e2e_net, e2e_mu, e2e_sd = train_e2e(Xe2e, Ye2e, cfg, device)

    # ─── Phase 2b: CE subnet ───
    print("\n[Phase 2b] ComNet CE subnet 학습")
    ce_net, W_lmmse = train_ce(Hce_est, Hce_true, cfg, device)

    # ─── Phase 2c: SD subnet ───
    print("\n[Phase 2c] ComNet FC-SD 학습")
    sd_net, sd_mu, sd_sd_ = train_sd(Y_sd, H_sd, L_sd, ce_net, cfg, device)

    # ─── Phase 4 ───
    print("\n[Phase 4] 7-way + 3-ref 평가")
    ber, methods = evaluate(model, cfg, snr_range,
                            ce_net, e2e_net, e2e_mu, e2e_sd,
                            sd_net, sd_mu, sd_sd_, W_lmmse, device, rng)

    # ─── 저장 ───
    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    res = {'config': {k: (v if not isinstance(v, dict) else
                           {kk: (vv.real.tolist() if hasattr(vv, 'real') and hasattr(vv, 'imag') else vv)
                            for kk, vv in v.items()})
                       for k, v in cfg.items() if k != 'iq'},
           'snr_range': snr_range,
           'ber': {m: ber[m].tolist() for m in methods}}
    with open(out / "ber_results.json", 'w') as f:
        json.dump(res, f, indent=2)
    torch.save({'state_dict': e2e_net.state_dict(), 'mu': e2e_mu, 'sd': e2e_sd},
               out / "e2e_net.pt")
    torch.save({'state_dict': ce_net.state_dict()}, out / "ce_subnet.pt")
    torch.save({'state_dict': sd_net.state_dict(), 'mu': sd_mu, 'sd': sd_sd_},
               out / "fc_sd.pt")
    np.savez(out / "lmmse_W.npz", weight=W_lmmse)

    # ─── 표 ───
    print("\n" + "=" * 100)
    head = f"{'Method':<18s}  " + "  ".join(f"{s:>3d}dB" for s in snr_range)
    print(head); print("-" * len(head))
    for m in methods:
        row = "  ".join(f"{ber[m][i]:>5.1e}" for i in range(len(snr_range)))
        print(f"{m:<18s}  {row}")

    try:
        _save_plot(snr_range, ber, methods, out,
                   phase_deg=Config.PHASE_ERR_DEG,
                   clip_ratio=cfg.get('clip_ratio', None),
                   csi_err=cfg['csi_err_var'])
    except Exception as e:
        print(f"⚠ 그래프 저장 실패: {e}")
    print(f"\n저장: {out}/ber_results.json,  {out}/ber_curves.png")


def _save_plot(snr_range, ber, methods, out_dir, phase_deg=None,
               clip_ratio=None, csi_err=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {
        'Basic'           : dict(color='tab:gray',    marker='o', ls='-',  lw=1.5),
        'MMSE'            : dict(color='tab:olive',   marker='s', ls='-',  lw=1.5),
        'WL-MMSE'         : dict(color='tab:purple',  marker='^', ls='-',  lw=2.0),
        'LMMSE-MMSE'      : dict(color='tab:cyan',    marker='D', ls='-',  lw=1.5),
        'ComNet-CE-Hard'  : dict(color='tab:orange',  marker='v', ls='-',  lw=1.5),
        'ComNet-FC'       : dict(color='tab:red',     marker='*', ls='-',  lw=2.2),
        'E2E-DL'          : dict(color='tab:blue',    marker='P', ls='-',  lw=2.0),
        'True-H ZF'       : dict(color='black',       marker='x', ls='--', lw=1.5),
        'True-H WL-MMSE'  : dict(color='dimgray',     marker='+', ls='--', lw=1.5),
        'No-IQ True-H'    : dict(color='black',       marker='X', ls=':',  lw=1.8),
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in methods:
        ys = np.clip(ber[m], 1e-6, 1.0)
        st = style.get(m, dict(marker='o', ls='-'))
        ax.semilogy(snr_range, ys, label=m, markersize=8, **st)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    # 제목에 실제 손상 강도 표시
    title_parts = ["design5 — 7-way receiver", "64-QAM, MU-MIMO 8×2×4"]
    if phase_deg is not None:
        title_parts.append(f"IQ phase {phase_deg:g}°")
    if clip_ratio is not None:
        title_parts.append(f"TX clip CR={clip_ratio}")
    if csi_err is not None:
        title_parts.append(f"CSI err var={csi_err}")
    ax.set_title("\n".join([title_parts[0], "  |  ".join(title_parts[1:])]),
                 fontsize=12)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=9, ncol=2, loc='best')
    plt.tight_layout()
    p = out_dir / "ber_curves.png"
    plt.savefig(p, dpi=120); plt.close()
    print(f"  ✓ {p}")


if __name__ == "__main__":
    main()
