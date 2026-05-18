"""
위상 ablation — 0°, 5°, 10°, 15°, 20°, 25°, 30° 각각 학습+평가 → 7 PNG.

각 위상마다:
  • 학습 데이터 수집 + NN 3종 학습 + LMMSE fit
  • 7-way receiver 평가
  • PNG / JSON 저장 (위상별 별도 파일명)
마지막에 통합 summary PNG (DL vs MMSE 의 변화 추이) 도 생성.

사용:
    python3 run_phase_ablation.py                 # iter 1500 디폴트
    python3 run_phase_ablation.py --iter 3000     # 학습 강화
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import Config, get_snr_range
from scm import SCM
from simulator import collect_training_data, train_e2e, train_ce, train_sd, evaluate
from main import _save_plot


PHASE_LIST = [0, 5, 10, 15, 20, 25, 30]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--iter', type=int, default=1500)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--ce-epochs', type=int, default=30)
    p.add_argument('--low-snr-iter', type=int, default=150)
    p.add_argument('--high-snr-iter', type=int, default=400)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--csi-err', type=float, default=0.001)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--out-dir', default='results_phase_ablation')
    p.add_argument('--phases', type=int, nargs='+', default=PHASE_LIST,
                   help='실행할 위상 리스트 (기본 0,5,10,15,20,25,30)')
    return p.parse_args()


def build_cfg(args, phase_deg):
    """phase_deg 가 반영된 IQ 파라미터를 만들어 cfg dict 구성."""
    amp = Config.AMP_ERR_DB
    g_i = 10 ** ((amp / 2) / 20)
    g_q = 10 ** (-(amp / 2) / 20)
    phi = phase_deg * np.pi / 180
    mu  = (g_i + g_q * np.exp(-1j * phi)) / 2
    nu  = (g_i - g_q * np.exp(+1j * phi)) / 2
    iq = dict(g_i=g_i, g_q=g_q, phi=phi, mu=mu, nu=nu)
    return dict(
        fft_len=Config.FFT_LEN, cp_len=Config.CP_LEN,
        mod_type=Config.MOD_TYPE, num_users=Config.NUM_USERS,
        N_rx=Config.N_RX, N_tx=Config.N_TX,
        iq=iq, csi_err_var=args.csi_err,
        iter=args.iter, epochs=args.epochs, batch=Config.BATCH, lr=args.lr,
        ce_epochs=args.ce_epochs, ce_lr=Config.CE_LR, ce_ridge=Config.CE_RIDGE,
        train_snr_min=Config.TRAIN_SNR_MIN, train_snr_max=Config.TRAIN_SNR_MAX,
        low_snr_iter=args.low_snr_iter, high_snr_iter=args.high_snr_iter,
        clip_ratio=None,
        phase_deg=phase_deg,
    )


def run_one_phase(phase_deg, args, out_dir, device, rng):
    """1 위상에 대해 학습 + 평가 + PNG 저장."""
    print("\n" + "=" * 72)
    print(f"  Phase = {phase_deg}°   ({phase_deg}/{30}°)")
    print("=" * 72)

    cfg = build_cfg(args, phase_deg)
    snr_range = get_snr_range()

    # SCM 모델 (각 위상마다 깨끗하게)
    model = SCM(); model.n_path = Config.N_PATH
    model.ant(Config.N_RX, Config.N_TX)

    # Phase 1
    t0 = time.time()
    print(f"  [1/4] 학습 데이터 수집 (iter={cfg['iter']})")
    Xe2e, Ye2e, Hce_est, Hce_true, Y_sd, H_sd, L_sd = \
        collect_training_data(model, cfg, rng)

    # Phase 2
    print(f"  [2/4] NN 학습 (E2E + CE + SD)")
    e2e_net, e2e_mu, e2e_sd = train_e2e(Xe2e, Ye2e, cfg, device)
    ce_net,  W_lmmse        = train_ce (Hce_est, Hce_true, cfg, device)
    sd_net,  sd_mu, sd_sd_  = train_sd (Y_sd, H_sd, L_sd, ce_net, cfg, device)

    # Phase 4
    print(f"  [3/4] 7-way 평가")
    ber, methods = evaluate(model, cfg, snr_range,
                            ce_net, e2e_net, e2e_mu, e2e_sd,
                            sd_net, sd_mu, sd_sd_, W_lmmse, device, rng)

    # 저장
    tag = f"{phase_deg:02d}deg"
    res = {'phase_deg': phase_deg,
           'snr_range': snr_range,
           'csi_err_var': cfg['csi_err_var'],
           'iter': cfg['iter'],
           'ber': {m: [float(x) for x in ber[m]] for m in methods}}
    json_p = out_dir / f"ber_results_{tag}.json"
    with open(json_p, 'w') as f: json.dump(res, f, indent=2)
    print(f"  [4/4] 저장: {json_p}")

    # PNG (제목에 위상 명시)
    png_path = out_dir / f"ber_curves_{tag}.png"
    # _save_plot 가 out_dir 자체에 'ber_curves.png' 저장하므로 임시 폴더 사용
    tmp = out_dir / f"_tmp_{tag}"; tmp.mkdir(exist_ok=True)
    _save_plot(snr_range, ber, methods, tmp,
               phase_deg=phase_deg, clip_ratio=None,
               csi_err=cfg['csi_err_var'])
    (tmp / "ber_curves.png").rename(png_path)
    tmp.rmdir()
    print(f"        {png_path}")

    elapsed = time.time() - t0
    print(f"  Phase {phase_deg}° 완료 ({elapsed:.0f}s)")

    return res


def save_summary(all_results, out_dir):
    """모든 위상의 핵심 receiver BER 추세를 한 그림에."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phases = [r['phase_deg'] for r in all_results]
    snr_range = all_results[0]['snr_range']

    # subplot per method (4개 핵심)
    key_methods = ['MMSE', 'WL-MMSE', 'ComNet-FC', 'E2E-DL']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), key_methods):
        for r in all_results:
            ys = np.clip(r['ber'][m], 1e-6, 1.0)
            ax.semilogy(snr_range, ys, marker='o', markersize=5,
                        label=f"{r['phase_deg']}°")
        ax.set_title(m, fontsize=12)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which='both', ls=':', alpha=0.5)
        ax.legend(title="IQ phase", fontsize=8)
    plt.suptitle("Phase Ablation — Each receiver across IQ phase 0~30°",
                 fontsize=13)
    plt.tight_layout()
    p = out_dir / "summary_per_method.png"
    plt.savefig(p, dpi=120); plt.close()
    print(f"\n  ✓ {p}")

    # 위상별 30dB BER 비교 막대 그래프 (전체 receiver)
    fig, ax = plt.subplots(figsize=(11, 6))
    methods_to_show = ['Basic', 'MMSE', 'WL-MMSE',
                       'ComNet-FC', 'E2E-DL', 'True-H WL-MMSE']
    width = 0.13
    x = np.arange(len(phases))
    snr_target = 30
    if snr_target not in snr_range:
        snr_target = snr_range[len(snr_range)//2]
    snr_idx = snr_range.index(snr_target)

    for i, m in enumerate(methods_to_show):
        ys = [max(r['ber'][m][snr_idx], 1e-6) for r in all_results]
        ax.bar(x + i*width, ys, width, label=m)
    ax.set_yscale('log')
    ax.set_xticks(x + width*(len(methods_to_show)-1)/2)
    ax.set_xticklabels([f"{p}°" for p in phases])
    ax.set_xlabel("IQ phase distortion")
    ax.set_ylabel(f"BER @ SNR={snr_target}dB (log)")
    ax.set_title(f"BER at SNR={snr_target}dB across all phases")
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
    print("=" * 72)
    print(f"Phase Ablation — {len(args.phases)} 위상  ({args.phases})")
    print(f"  iter={args.iter}, epochs={args.epochs}, ce_epochs={args.ce_epochs}")
    print(f"  CSI err var={args.csi_err}, lr={args.lr}")
    print(f"  결과 저장: {out}/")
    print("=" * 72)

    all_results = []
    t_total = time.time()
    for phase in args.phases:
        res = run_one_phase(phase, args, out, device, rng)
        all_results.append(res)

    # 통합 요약
    save_summary(all_results, out)

    # 표 요약
    print("\n" + "=" * 90)
    print(f"전체 완료 ({time.time()-t_total:.0f}s)")
    print(f"\n위상별 30dB BER 요약")
    print("-" * 90)
    snr_range = all_results[0]['snr_range']
    snr_idx = snr_range.index(30) if 30 in snr_range else len(snr_range)//2
    print(f"{'phase':<8s}  " +
          "  ".join(f"{m:>13s}" for m in ['Basic','MMSE','WL-MMSE',
                                          'ComNet-FC','E2E-DL']))
    for r in all_results:
        row = "  ".join(f"{r['ber'][m][snr_idx]:>13.3e}" for m in
                        ['Basic','MMSE','WL-MMSE','ComNet-FC','E2E-DL'])
        print(f"{r['phase_deg']:>4d}°    {row}")


if __name__ == "__main__":
    main()
