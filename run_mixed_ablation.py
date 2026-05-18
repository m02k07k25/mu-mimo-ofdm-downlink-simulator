"""
Mixed-training Phase Ablation.

전략:
  • NN 학습 (E2E, CE, SD) 한 번만 — 매 frame phase 를 [0, 30]° 사이 균등 샘플
  • 같은 NN 으로 7가지 위상 (0/5/10/15/20/25/30°) 각각 평가
  • 7개 PNG 저장 + 통합 summary PNG

장점:
  • 한 NN 으로 모든 위상 대응 → 실제 5G receiver 의 deployment 시나리오
  • 학습 1회 → 시간 절약 (~84분 → ~25분)
  • 모든 평가가 같은 학습 모델 사용 → fair comparison

사용:
    python3 run_mixed_ablation.py
    python3 run_mixed_ablation.py --iter 3000        # 학습 강화
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import Config, get_snr_range
from scm import SCM
from simulator import (collect_training_data, train_e2e, train_ce,
                       train_sd, evaluate, _iq_from_phase_deg)
from main import _save_plot


EVAL_PHASES = [0, 5, 10, 15, 20, 25, 30]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--iter', type=int, default=1500,
                   help='학습 데이터 frame 수 (random phase)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--ce-epochs', type=int, default=30)
    p.add_argument('--low-snr-iter', type=int, default=150)
    p.add_argument('--high-snr-iter', type=int, default=400)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--csi-err', type=float, default=0.001)
    p.add_argument('--phase-min', type=float, default=0.0)
    p.add_argument('--phase-max', type=float, default=30.0)
    p.add_argument('--eval-phases', type=float, nargs='+', default=EVAL_PHASES)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--out-dir', default='results_mixed')
    return p.parse_args()


def build_cfg(args):
    """학습 시 random phase 사용하는 cfg."""
    return dict(
        fft_len=Config.FFT_LEN, cp_len=Config.CP_LEN,
        mod_type=Config.MOD_TYPE, num_users=Config.NUM_USERS,
        N_rx=Config.N_RX, N_tx=Config.N_TX,
        iq=_iq_from_phase_deg(15.0, Config.AMP_ERR_DB),  # 평가 시 override 됨
        csi_err_var=args.csi_err,
        iter=args.iter, epochs=args.epochs, batch=Config.BATCH, lr=args.lr,
        ce_epochs=args.ce_epochs, ce_lr=Config.CE_LR, ce_ridge=Config.CE_RIDGE,
        train_snr_min=Config.TRAIN_SNR_MIN, train_snr_max=Config.TRAIN_SNR_MAX,
        low_snr_iter=args.low_snr_iter, high_snr_iter=args.high_snr_iter,
        clip_ratio=None,
        # ★ Mixed training 플래그
        train_phase_random=True,
        train_phase_range=(args.phase_min, args.phase_max),
        amp_err_db_train=Config.AMP_ERR_DB,
    )


def evaluate_at_phase(model, cfg, snr_range, phase_deg,
                     ce_net, e2e_net, e2e_mu, e2e_sd,
                     sd_net, sd_mu, sd_sd_, W_lmmse, device, rng):
    """cfg['iq'] 를 해당 위상으로 임시 교체해 평가."""
    saved_iq = cfg['iq']
    cfg['iq'] = _iq_from_phase_deg(phase_deg, Config.AMP_ERR_DB)
    try:
        ber, methods = evaluate(model, cfg, snr_range,
                                ce_net, e2e_net, e2e_mu, e2e_sd,
                                sd_net, sd_mu, sd_sd_, W_lmmse, device, rng)
    finally:
        cfg['iq'] = saved_iq
    return ber, methods


def save_per_phase_plot(snr_range, ber, methods, out_dir, phase_deg, csi_err):
    """위상별 PNG 저장."""
    tag = f"{int(phase_deg):02d}deg"
    tmp = out_dir / f"_tmp_{tag}"; tmp.mkdir(exist_ok=True)
    _save_plot(snr_range, ber, methods, tmp,
               phase_deg=phase_deg, clip_ratio=None, csi_err=csi_err)
    dst = out_dir / f"ber_curves_{tag}.png"
    (tmp / "ber_curves.png").rename(dst)
    tmp.rmdir()
    print(f"    ✓ {dst}")


def save_summary(all_results, out_dir, snr_target=30):
    """통합 비교 PNG 2장."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phases = [r['phase_deg'] for r in all_results]
    snr_range = all_results[0]['snr_range']
    if snr_target not in snr_range:
        snr_target = snr_range[len(snr_range)//2]
    snr_idx = snr_range.index(snr_target)

    # 1) Method 별 4 subplot - 각 패널에 7 위상 선
    key_methods = ['MMSE', 'WL-MMSE', 'ComNet-FC', 'E2E-DL']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), key_methods):
        for r in all_results:
            ys = np.clip(r['ber'][m], 1e-6, 1.0)
            ax.semilogy(snr_range, ys, marker='o', markersize=5,
                        label=f"{int(r['phase_deg'])}°")
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(title="IQ phase", fontsize=8)
    plt.suptitle(f"Mixed-Training Ablation — 각 receiver 의 위상별 BER 추세\n"
                 f"(단일 NN, train phase ∈ [0°, 30°] 랜덤)", fontsize=12)
    plt.tight_layout()
    p = out_dir / "summary_per_method.png"
    plt.savefig(p, dpi=120); plt.close()
    print(f"  ✓ {p}")

    # 2) snr_target 에서 위상별 막대 그래프
    methods_to_show = ['Basic', 'MMSE', 'WL-MMSE',
                       'ComNet-FC', 'E2E-DL', 'True-H WL-MMSE']
    width = 0.13
    x = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(methods_to_show):
        ys = [max(r['ber'][m][snr_idx], 1e-6) for r in all_results]
        ax.bar(x + i*width, ys, width, label=m)
    ax.set_yscale('log')
    ax.set_xticks(x + width*(len(methods_to_show)-1)/2)
    ax.set_xticklabels([f"{int(p)}°" for p in phases])
    ax.set_xlabel("IQ phase distortion (학습 안 한 위상도 포함)")
    ax.set_ylabel(f"BER @ SNR={snr_target} dB (log)")
    ax.set_title(f"위상별 BER @ {snr_target}dB — Mixed-trained NN 평가")
    ax.grid(True, which='both', ls=':', alpha=0.5, axis='y')
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    p = out_dir / f"summary_bar_{snr_target}dB.png"
    plt.savefig(p, dpi=120); plt.close()
    print(f"  ✓ {p}")


def main():
    args = parse_args()
    np.random.seed(Config.SEED); torch.manual_seed(Config.SEED)
    rng = np.random.default_rng(Config.SEED)
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    cfg = build_cfg(args)
    snr_range = get_snr_range()

    print("=" * 72)
    print("Mixed-Training Phase Ablation")
    print(f"  학습 phase: 매 frame [{args.phase_min}, {args.phase_max}]° 균등 랜덤")
    print(f"  평가 phase: {args.eval_phases}")
    print(f"  iter={args.iter}, epochs={args.epochs}, ce_epochs={args.ce_epochs}")
    print(f"  CSI err var={args.csi_err}, lr={args.lr}")
    print(f"  결과: {out}/")
    print("=" * 72)

    model = SCM(); model.n_path = Config.N_PATH
    model.ant(Config.N_RX, Config.N_TX)

    # ─── Phase 1+2: 1회 학습 ───
    print("\n[1/3] Mixed-phase 학습 데이터 수집")
    t0 = time.time()
    Xe2e, Ye2e, Hce_est, Hce_true, Y_sd, H_sd, L_sd = \
        collect_training_data(model, cfg, rng)
    print(f"\n[2/3] NN 학습 (3종)")
    e2e_net, e2e_mu, e2e_sd = train_e2e(Xe2e, Ye2e, cfg, device)
    ce_net,  W_lmmse        = train_ce (Hce_est, Hce_true, cfg, device)
    sd_net,  sd_mu, sd_sd_  = train_sd (Y_sd, H_sd, L_sd, ce_net, cfg, device)
    train_time = time.time() - t0
    print(f"\n  학습 완료 ({train_time:.0f}s)")

    # 학습 모델 저장
    torch.save({'state_dict': e2e_net.state_dict(),
                'mu': e2e_mu, 'sd': e2e_sd}, out / "e2e_net.pt")
    torch.save({'state_dict': ce_net.state_dict()}, out / "ce_subnet.pt")
    torch.save({'state_dict': sd_net.state_dict(),
                'mu': sd_mu, 'sd': sd_sd_}, out / "fc_sd.pt")
    np.savez(out / "lmmse_W.npz", weight=W_lmmse)

    # ─── Phase 3: 7 위상 평가 ───
    print(f"\n[3/3] {len(args.eval_phases)} 위상 평가")
    all_results = []
    for phase_deg in args.eval_phases:
        t_phase = time.time()
        print(f"\n  ── Phase {int(phase_deg)}° 평가 ──")
        ber, methods = evaluate_at_phase(
            model, cfg, snr_range, phase_deg,
            ce_net, e2e_net, e2e_mu, e2e_sd,
            sd_net, sd_mu, sd_sd_, W_lmmse, device, rng)

        # 저장
        res = {'phase_deg': phase_deg, 'snr_range': snr_range,
               'csi_err_var': cfg['csi_err_var'],
               'ber': {m: [float(x) for x in ber[m]] for m in methods}}
        with open(out / f"ber_results_{int(phase_deg):02d}deg.json", 'w') as f:
            json.dump(res, f, indent=2)
        save_per_phase_plot(snr_range, ber, methods, out,
                            phase_deg, cfg['csi_err_var'])
        all_results.append(res)
        print(f"    Phase {int(phase_deg)}° 완료 ({time.time()-t_phase:.0f}s)")

    # 통합 summary
    print(f"\n[요약] Summary PNG 2장 생성")
    save_summary(all_results, out, snr_target=30)

    # 표
    print("\n" + "=" * 90)
    print(f"전체 완료 — 학습 {train_time:.0f}s, 평가 7회 합산")
    print(f"\n위상별 30dB BER (Mixed-trained NN)")
    print("-" * 90)
    snr_idx = snr_range.index(30) if 30 in snr_range else len(snr_range)//2
    print(f"{'phase':<8s}  " + "  ".join(f"{m:>13s}" for m in
          ['Basic','MMSE','WL-MMSE','ComNet-FC','E2E-DL']))
    for r in all_results:
        row = "  ".join(f"{r['ber'][m][snr_idx]:>13.3e}" for m in
                        ['Basic','MMSE','WL-MMSE','ComNet-FC','E2E-DL'])
        print(f"{int(r['phase_deg']):>4d}°    {row}")
    print(f"\n결과 저장: {out}/")


if __name__ == "__main__":
    main()
