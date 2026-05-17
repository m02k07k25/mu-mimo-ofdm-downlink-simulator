# RX Receiver

`rx_mumimo_receiver.py`는 raw MU-MIMO E2E dataset을 읽고 classical baseline과 ComNet receiver를 학습/평가합니다.

```text
Input: outputs_mumimo_e2e_*/train/val/test .npz
Output: checkpoints, CSV, plots, eval_summary.json
```

## Recommended Command

현재 권장 설정:

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --result-dir results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse --mode train-all --sd-type bilstm --sd-feature-set rf-reliability --ce-type blend-resmlp --ce-target auto --lmmse-mode snr-binned --bilstm-epochs 300 --device cuda
```

## Receiver Flow

```text
rx_p_time -> CP removal -> FFT -> LS channel estimate A_ls
rx_d_time -> CP removal -> FFT -> y_d
A_ls -> LMMSE / ComNet-CE
y_d + channel estimate -> ZF/MMSE/WL-MMSE/ComNet-SD
predicted bits -> BER
```

## LMMSE Mode

```text
--lmmse-mode global
  모든 train SNR을 섞어서 하나의 empirical LMMSE weight를 fit합니다.

--lmmse-mode snr-binned
  train SNR별로 empirical LMMSE weight를 따로 fit합니다.
```

`snr-binned`가 현재 권장값입니다. 기존 global LMMSE는 high SNR에서도 low/mid SNR shrinkage가 남아 LS보다 나쁜 BER을 만들 수 있었습니다. `snr-binned`는 이 bias를 줄입니다.

현재 train SNR bin:

```text
15 20 25 30 35 40 dB
```

test SNR이 train bin에 없으면 가장 가까운 train bin을 사용합니다. 예를 들어 0/5/10 dB test는 15 dB weight를 사용합니다.

## CE Options

### CE target

```text
--ce-target pre-rf
  target = A_eff_true

--ce-target rf-linear
  target = alpha * A_eff_true

--ce-target auto
  RF off -> pre-rf
  RF on  -> rf-linear
```

RF impairment가 켜진 조건에서는 plain complex detector가 pre-RF `A_eff_true`만으로는 mismatch가 생깁니다. main setting에서는 `--ce-target auto`를 사용합니다.

### CE type

```text
--ce-type linear
--ce-type resmlp
--ce-type blend-resmlp
```

현재 권장 CE는 `blend-resmlp`입니다.

```text
A_lmmse = LMMSE(A_ls)
w = sigmoid((snr_db - 27.5) / 4.0)
A_base = w * A_ls + (1 - w) * A_lmmse
A_comnet = A_base + residual_mlp(A_base)
```

low SNR에서는 LMMSE denoising을 더 쓰고, high SNR에서는 LS에 더 가깝게 두는 구조입니다. residual MLP는 이 base channel을 작게 보정합니다.

## SD Feature Sets

```text
--sd-feature-set basic
--sd-feature-set reliability
--sd-feature-set rf-reliability
```

현재 권장 SD feature는 `rf-reliability`입니다.

포함 feature:

```text
plain ZF real/imag
plain MMSE real/imag
RF-aware WL-MMSE real/imag
plain residual matched real/imag
WL-MMSE - plain MMSE delta real/imag
plain residual power
WL delta power
post-MMSE gain magnitude
WL delta magnitude reliability proxy
condition number
noise power
SNR
```

## Baselines

```text
LS-ZF
LS-MMSE
LMMSE-ZF
LMMSE-MMSE
ComNet-CE-ZF-Hard
ComNet-BiLSTM
Pre-RF True-H ZF
Pre-RF True-H MMSE
RF-aware True-H WL-MMSE
Desired-only MRC
```

해석 기준:

- `RF-aware True-H WL-MMSE`는 RF impairment까지 고려한 oracle baseline입니다.
- `Pre-RF True-H`는 RF가 켜진 조건에서는 strict oracle이 아닙니다.
- `Desired-only MRC`가 높은 floor를 보이면 stream interference/RF mismatch가 중요하다는 신호입니다.

## Latest Result Summary

최신 result:

```text
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse
```

40 dB:

```text
LS-MMSE                  2.654e-3
LMMSE-MMSE               3.033e-3
ComNet-CE-ZF-Hard        2.543e-3
ComNet-BiLSTM            2.503e-3
RF-aware True-H WL-MMSE  1.426e-3
```

SNR-binned LMMSE 적용 후 LMMSE-MMSE는 기존 global 결과 `3.904e-3`에서 `3.033e-3`로 좋아졌습니다. 다만 최종 BER에서는 여전히 LS-MMSE보다 약간 나쁩니다. 따라서 이 문제는 단순 channel MSE만이 아니라 RF/clipping mismatch와 detector 목적 함수 차이까지 포함합니다.

## Output Files

```text
eval_summary.json
ber_vs_snr.png
a_mse_vs_snr.png
ber_vs_snr.csv
channel_mse_vs_snr.csv
channel_nmse_vs_snr.csv
diagnostic_vs_snr.csv
train_history_ce.csv
train_history_bilstm_sd.csv
ce_training_curve.png
bilstm_sd_training_curve.png
```

## Next Work

현재 구조는 baseline을 이기기 시작했지만 RF-aware oracle과는 gap이 남아 있습니다.

다음 우선순위:

```text
correction-based SD:
  baseline bit = LS-MMSE or RF-aware WL-MMSE hard decision
  model predicts correction bit
  final bit = baseline bit XOR correction
```

기록할 지표:

```text
baseline wrong bits corrected
baseline correct bits damaged
false correction rate
net BER gain
```
