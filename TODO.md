# TODO

Updated: 2026-05-17

## Done

- raw MU-MIMO E2E TX/RX 경로 구현
- SCM-style clustered multipath channel 구현
- QPSK pilot 적용
- paired test SNR sweep 적용
- RF impairment와 RF-aware True-H WL-MMSE oracle baseline 추가
- BER, bit error count, channel MSE/NMSE, diagnostic CSV 저장 추가
- CE target `auto|pre-rf|rf-linear` 추가
- SD feature `rf-reliability` 추가
- CE type `blend-resmlp` 추가
- main dataset 생성
- `--lmmse-mode global|snr-binned` 추가
- SNR-binned LMMSE full 300 epoch run 완료

## Current Best Result

```text
Dataset:
outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000

Result:
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse
```

설정:

```text
64QAM
csit_error_var = 0.001
case = clipping
clip_ratio = 2.0
rx_iq_gain_imbalance_db = 0.2
rx_iq_phase_error_deg = 1.0
rx_common_phase_error_deg = 1.0
CE = blend-resmlp
SD = BiLSTM
SD feature = rf-reliability
CE target = auto -> rf-linear
LMMSE mode = snr-binned
```

40 dB:

```text
LS-MMSE                  2.654e-3
LMMSE-MMSE               3.033e-3
ComNet-CE-ZF-Hard        2.543e-3
ComNet-BiLSTM            2.503e-3
RF-aware True-H WL-MMSE  1.426e-3
```

현재 결론:

- ComNet-BiLSTM은 25 dB 이상에서 LS-MMSE보다 좋다.
- SNR-binned LMMSE는 global LMMSE보다 high-SNR LMMSE BER과 channel MSE를 개선했다.
- 그래도 high SNR BER에서는 LMMSE-MMSE가 LS-MMSE를 완전히 이기지 못한다.
- RF-aware oracle과는 gap이 남아 있으므로, 다음 단계는 direct bit prediction보다 correction-based SD가 맞다.

## Next Priorities

### 1. Correction-based SD

현재 SD는 bit를 직접 예측한다.

```text
feature -> neural net -> predicted bit
```

다음 단계는 baseline correction 방식이다.

```text
baseline bit = LS-MMSE or RF-aware WL-MMSE hard decision
feature -> neural net -> correction bit
final bit = baseline bit XOR correction
```

추가할 지표:

```text
baseline wrong bits corrected
baseline correct bits damaged
false correction rate
net BER gain
```

목표:

- high SNR에서 baseline이 맞힌 bit를 망치지 않기
- RF-aware oracle과의 gap 줄이기

### 2. Feature Cache

`rf-reliability`는 RF-aware WL-MMSE feature를 RX에서 계산하므로 시간이 든다. TX에서 만들기보다는 RX feature cache가 맞다.

후보 옵션:

```text
--cache-sd-features
```

캐시 위치:

```text
results_*/feature_cache/
```

### 3. Main/Stress 분리

main:

```text
csit_error_var = 0.001
clip_ratio = 2.0
small RF impairment
```

stress:

```text
csit_error_var = 0.005
clip_ratio = 1.6
larger RF impairment
```

main 결과와 stress 결과를 섞어서 해석하지 말 것.

## Commands

Main RX:

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --result-dir results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse --mode train-all --sd-type bilstm --sd-feature-set rf-reliability --ce-type blend-resmlp --ce-target auto --lmmse-mode snr-binned --bilstm-epochs 300 --device cuda
```

Syntax check:

```powershell
python -m py_compile rx_mumimo_receiver.py
```
