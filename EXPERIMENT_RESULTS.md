# 실험 결과

Updated: 2026-05-17

## Summary

SNR-binned LMMSE를 적용한 300 epoch full run을 완료했습니다.

최신 결과:

```text
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse
```

핵심 결론:

- SNR-binned LMMSE는 global LMMSE보다 high-SNR LMMSE channel MSE와 BER을 개선했다.
- 40 dB LMMSE-MMSE는 `3.904e-3 -> 3.033e-3`로 좋아졌다.
- 하지만 40 dB에서 LS-MMSE `2.654e-3`보다 여전히 나쁘다.
- ComNet-BiLSTM은 25 dB 이상에서 LS-MMSE보다 좋다.
- RF-aware True-H WL-MMSE oracle과는 아직 gap이 남아 있다.

## Dataset

```text
outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000
```

설정:

```text
modulation = 64QAM
csit_error_var = 0.001
case = clipping
clip_ratio = 2.0
rx_iq_gain_imbalance_db = 0.2
rx_iq_phase_error_deg = 1.0
rx_common_phase_error_deg = 1.0
n_train_frames = 10000
n_val_frames = 2000
n_test_frames_per_snr = 2000
train_snr_db_list = 15 20 25 30 35 40
test_snr_db = 0 5 10 15 20 25 30 35 40
```

## Run Command

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --result-dir results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse --mode train-all --sd-type bilstm --sd-feature-set rf-reliability --ce-type blend-resmlp --ce-target auto --lmmse-mode snr-binned --bilstm-epochs 300 --device cuda
```

## Compared Runs

### Global LMMSE

```text
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability
```

특징:

```text
CE type = blend-resmlp
SD feature = rf-reliability
CE target = auto -> rf-linear
LMMSE mode = global
```

### SNR-binned LMMSE

```text
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse
```

특징:

```text
CE type = blend-resmlp
SD feature = rf-reliability
CE target = auto -> rf-linear
LMMSE mode = snr-binned
SNR bins = 15 20 25 30 35 40
```

## BER Table

최신 SNR-binned run 기준입니다.

| SNR | LS-MMSE | LMMSE-MMSE | ComNet-CE-ZF-Hard | ComNet-BiLSTM | RF-aware True-H WL-MMSE |
|---:|---:|---:|---:|---:|---:|
| 0  | 3.791e-1 | 3.204e-1 | 3.352e-1 | 3.231e-1 | 2.717e-1 |
| 5  | 2.724e-1 | 2.129e-1 | 2.273e-1 | 2.147e-1 | 1.724e-1 |
| 10 | 1.641e-1 | 1.199e-1 | 1.286e-1 | 1.203e-1 | 8.705e-2 |
| 15 | 8.073e-2 | 5.356e-2 | 5.647e-2 | 5.329e-2 | 2.999e-2 |
| 20 | 2.902e-2 | 1.960e-2 | 1.969e-2 | 1.894e-2 | 8.127e-3 |
| 25 | 9.161e-3 | 8.415e-3 | 7.340e-3 | 7.193e-3 | 3.085e-3 |
| 30 | 4.217e-3 | 5.111e-3 | 3.878e-3 | 3.746e-3 | 1.901e-3 |
| 35 | 3.001e-3 | 3.676e-3 | 2.810e-3 | 2.782e-3 | 1.513e-3 |
| 40 | 2.654e-3 | 3.033e-3 | 2.543e-3 | 2.503e-3 | 1.426e-3 |

## Global vs SNR-Binned

| SNR | Global LMMSE-MMSE | SNR-binned LMMSE-MMSE | Global ComNet-BiLSTM | SNR-binned ComNet-BiLSTM |
|---:|---:|---:|---:|---:|
| 25 | 6.874e-3 | 8.415e-3 | 6.458e-3 | 7.193e-3 |
| 30 | 4.648e-3 | 5.111e-3 | 3.546e-3 | 3.746e-3 |
| 35 | 4.076e-3 | 3.676e-3 | 2.721e-3 | 2.782e-3 |
| 40 | 3.904e-3 | 3.033e-3 | 2.503e-3 | 2.503e-3 |

해석:

- 35/40 dB에서는 SNR-binned LMMSE가 global LMMSE보다 좋아졌다.
- 25/30 dB에서는 global LMMSE BER이 더 좋다. 이 구간에서는 global shrinkage가 오히려 detector BER에 유리하게 작동한 것으로 보인다.
- ComNet-BiLSTM은 40 dB에서 거의 동일하고, 25/30 dB에서는 global 결과가 약간 더 좋았다.
- 따라서 `snr-binned`는 LMMSE estimator sanity 측면에서는 맞지만, 최종 BER 기준으로 모든 SNR에서 무조건 이기는 변경은 아니다.

## Channel MSE

최신 SNR-binned run:

| SNR | LS MSE dB | LMMSE MSE dB | ComNet-CE MSE dB |
|---:|---:|---:|---:|
| 25 | -19.29 | -20.73 | -21.45 |
| 30 | -23.27 | -23.72 | -24.86 |
| 35 | -26.06 | -26.69 | -27.59 |
| 40 | -27.52 | -29.02 | -29.21 |

40 dB에서 global LMMSE MSE는 `-24.92 dB`였고, SNR-binned LMMSE MSE는 `-29.02 dB`입니다. channel estimate 자체는 확실히 개선됐습니다.

중요한 점:

```text
channel MSE 개선 == BER 개선은 아니다.
```

RF impairment, clipping, residual inter-stream interference가 있으면 MSE가 낮은 channel estimate가 plain MMSE detector에서 항상 더 좋은 BER을 만들지는 않습니다.

## 40 dB Interpretation

```text
LS-MMSE                  2.654e-3
LMMSE-MMSE               3.033e-3
ComNet-CE-ZF-Hard        2.543e-3
ComNet-BiLSTM            2.503e-3
RF-aware True-H WL-MMSE  1.426e-3
```

해석:

- ComNet-CE-ZF-Hard와 ComNet-BiLSTM은 LS-MMSE보다 약간 좋다.
- SNR-binned LMMSE는 global LMMSE보다 좋아졌지만 LS-MMSE보다 좋지는 않다.
- RF-aware oracle이 가장 좋으므로 RF-aware feature/correction 방향은 여전히 타당하다.
- 현재 결과는 코드 전체가 망가진 상황이라기보다, RF/clipping mismatch가 남아 있고 direct bit prediction SD가 oracle gap을 다 줄이지 못하는 상황이다.

## Current Assessment

보고서에 쓸 수 있는 정리는 다음과 같습니다.

```text
Clean sanity에서는 기본 구현 문제가 크지 않았다.
Hard setting은 BER floor를 만들 수 있어 main setting을 mild impairment로 낮췄다.
Global LMMSE는 high SNR bias를 만들 수 있어 SNR-binned LMMSE를 추가했다.
SNR-binned LMMSE는 channel MSE와 40 dB LMMSE BER을 개선했다.
ComNet-BiLSTM은 25 dB 이상에서 LS-MMSE를 개선한다.
RF-aware oracle과의 gap은 남아 있어 correction-based neural detector가 다음 과제다.
```

## Next Step

다음 작업은 모델을 더 키우는 것보다 correction-based SD입니다.

```text
baseline bit = LS-MMSE or RF-aware WL-MMSE hard decision
model target = baseline bit correction
final bit = baseline bit XOR predicted correction
```

반드시 같이 기록할 지표:

```text
baseline wrong bits corrected
baseline correct bits damaged
false correction rate
net BER gain
```
