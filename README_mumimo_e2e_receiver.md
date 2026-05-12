# Raw MU-MIMO E2E ComNet 수신기

이 문서는 raw end-to-end MU-MIMO 실험 경로를 설명한다.

```text
TX: tx_mumimo_e2e_dataset.py
RX: rx_mumimo_receiver.py
```

이 경로는 `tx_mumimo_comnet_dataset.py -> rx_comnet_receiver.py`로 이어지는
effective-SISO bridge 실험과 다르다. 여기서는 UE 안테나별 raw waveform을 저장하고,
RX가 사용자별 effective channel matrix `A`를 추정한 뒤 ZF/MMSE 및 ComNet 검출을 수행한다.

## 현재 기준

최신 권장 raw MU-MIMO E2E 설정은 다음과 같다.

```text
modulation = 64QAM
n_users = 2
n_streams = 2
n_tx = 8
n_rx_per_ue = 4
n_fft = 64
n_cp = 16
case = clipping
clip_ratio = 3.0
pilot_kind = qpsk
train/val/test = 5000 / 1000 / 1000 per SNR
```

중요한 점은 `pilot_kind=qpsk`가 데이터 변조를 바꾸는 옵션이 아니라는 것이다.

```text
데이터 심볼: 64QAM
BER 복조: 64QAM
채널 추정 파일럿: QPSK phase pilot
```

즉 `x_d_freq`와 `bits`는 64QAM 데이터이고, `x_p_freq`만 채널 추정용 known pilot이다.

## QPSK Pilot을 쓰는 이유

기존 `ones` pilot은 모든 subcarrier가 `1+0j`이다. OFDM IFFT 후 시간영역에서 큰 peak가
생기기 때문에 clipping 조건에서는 파일럿만 심하게 잘린다. 이 경우 SNR을 높여도 LS 채널
추정 MSE가 줄지 않고 BER floor가 생긴다.

확인 결과:

```text
ones pilot + clipping:
  고SNR에서도 LS 채널 MSE가 거의 고정됨
  BER이 10^-3 아래로 내려가지 않음

qpsk pilot + clipping:
  LS 채널 MSE가 SNR에 따라 정상 감소
  25 dB부터 10^-4 수준, 30 dB 이상에서는 0 BER 관측
```

QPSK pilot은 데이터 변조를 QPSK로 낮춘 것이 아니라, clipping에 강한 낮은 PAPR 채널추정
시퀀스를 사용한 것이다.

## TX 모델

BS는 사용자별 dominant-SVD receive direction을 이용해 precoder 설계용 effective channel을
만든다.

```text
c_u[k] = dominant-SVD receive direction from H_tx_est,u[k]
G_tx_est[u,k] = c_u[k]^H H_tx_est,u[k]
W_zf[k] = pinv(G_tx_est[k])
```

각 precoder column은 unit norm으로 정규화한다.

```text
||W_precoder[:, s, k]||^2 = 1
```

실제 RX 데이터에는 UE combiner를 적용하지 않는다. 저장되는 데이터는 UE 안테나별 raw
waveform이다.

clipping case에서는 BS 안테나별 OFDM time-domain symbol을 CP 삽입 전에 clipping한다.

```text
threshold = clip_ratio * RMS(time_symbol)
```

## Dataset Schema

각 `.npz` split에는 다음 항목이 저장된다.

```text
rx_p_time:   complex64 [n_frames, n_streams, n_users, n_rx_per_ue, n_fft+n_cp]
rx_d_time:   complex64 [n_frames, n_users, n_rx_per_ue, n_fft+n_cp]
x_p_freq:    complex64 [n_frames, n_streams, n_streams, n_fft]
x_d_freq:    complex64 [n_frames, n_streams, n_fft]
bits:        int8      [n_frames, n_streams, n_fft, bits_per_symbol]
H_true:      complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_tx]
G_tx_est:    complex64 [n_frames, n_fft, n_streams, n_tx]
W_precoder:  complex64 [n_frames, n_fft, n_tx, n_streams]
A_eff_true:  complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
snr_db:      float32   [n_frames]
noise_power: float32   [n_frames]
```

effective channel은 다음과 같다.

```text
A_eff_true[n,k,u,r,s] =
  sum_tx H_true[n,k,u,r,tx] * W_precoder[n,k,tx,s]
```

## RX 모델

RX는 CP 제거, FFT, LS 채널 추정을 수행한다. 사용자 `u`에 대해 per-subcarrier 모델은 다음과
같다.

```text
Y_u[k] = A_u[k] s[k] + n
```

ZF:

```text
s_hat = (A^H A)^-1 A^H Y
```

MMSE:

```text
s_hat_biased = (A^H A + sigma2 I)^-1 A^H Y
```

MMSE 출력은 QAM hard decision 전에 post-equalization gain으로 보정한다. 이 보정이 없으면
MMSE 출력이 shrinkage된 상태로 복조되어 BER이 부정확해진다.

## ComNet 구조

`rx_mumimo_receiver.py`는 SISO ComNet 구조를 raw MU-MIMO effective channel에 맞게 확장한다.

```text
A_ls
-> MuMimoCERefineNet
-> A_comnet
-> ZF/MMSE baseline 또는 SD subnet
```

모델은 다음 세 부분으로 나뉜다.

```text
MuMimoCERefineNet:
  A_ls -> A_comnet 채널 보정

MuMimoFCSDNet:
  target stream ZF estimate group -> bits

MuMimoBiLSTMSDNet:
  subcarrier sequence feature -> bits
```

비교 곡선:

```text
LS-ZF
LS-MMSE
LMMSE-ZF
LMMSE-MMSE
ComNet-CE-ZF-Hard
ComNet-FC
ComNet-BiLSTM
True-H ZF
True-H MMSE
Desired-only MRC
```

## 최신 실행 명령

10% 크기의 64QAM raw MU-MIMO E2E 데이터 생성:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot --modulation 64QAM --case clipping --clip-ratio 3.0 --pilot-kind qpsk --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
```

CE, FC-SD, BiLSTM-SD 학습 및 평가:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot --result-dir results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_sd150 --mode train-all --sd-type both --sd-epochs 150 --bilstm-epochs 150 --device cuda
```

BiLSTM을 더 길게 학습할 때:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot --result-dir results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_bilstm300_lrstep100 --mode train-sd --sd-type bilstm --bilstm-epochs 300 --sd-lr-step 100 --device cuda
```

최종 결과를 다시 평가할 때:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot --result-dir results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_bilstm300_lrstep100 --mode eval --sd-type both --device cuda
```

## 최신 결과 위치

최신 기준 결과:

```text
results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_bilstm300_lrstep100
```

주요 산출물:

```text
ber_vs_snr.png
a_mse_vs_snr.png
eval_summary.json
train_history_bilstm_sd.csv
```

최신 데이터:

```text
outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot
```

## 최신 결과 요약

QPSK pilot 적용 후 채널 추정 MSE는 SNR에 따라 정상적으로 감소한다.

```text
SNR 25 dB: LS MSE = -18.93 dB
SNR 30 dB: LS MSE = -23.92 dB
SNR 35 dB: LS MSE = -28.93 dB
SNR 40 dB: LS MSE = -33.92 dB
```

BER 결과:

```text
25 dB:
  LS-MMSE = 1.25e-4
  LMMSE-MMSE = 1.12e-4
  ComNet-CE-ZF-Hard = 1.32e-4
  ComNet-FC = 4.43e-4

30 dB:
  LS/LMMSE/ComNet-CE = 0
  ComNet-FC = 9.11e-6

35 dB:
  LS/LMMSE/ComNet-CE = 0
  ComNet-FC = 5.21e-6
  ComNet-BiLSTM = 1.36e-3

40 dB:
  LS/LMMSE/ComNet-CE = 0
  ComNet-FC = 1.30e-6
  ComNet-BiLSTM = 0
```

주의: 현재 SD 학습은 기본적으로 40 dB train split 중심이다. 그래서 BiLSTM은 40 dB에서는 잘
맞지만 10~30 dB 구간 일반화가 약하다. 전체 SNR에서 BiLSTM을 안정적으로 쓰려면 train split을
mixed-SNR로 구성하는 후속 실험이 필요하다.

## 생성 파일

RX 결과 디렉터리에는 다음 파일들이 생성된다.

```text
mumimo_lmmse_estimator.npz
mumimo_ce_refinenet.pt
mumimo_refinenet_fc.pt
mumimo_refinenet_bilstm.pt
train_history_ce.csv
train_history_fc_sd.csv
train_history_bilstm_sd.csv
ce_training_curve.png
fc_sd_training_curve.png
bilstm_sd_training_curve.png
ber_vs_snr.png
a_mse_vs_snr.png
eval_summary.json
```
