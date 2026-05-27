# MU-MIMO OFDM WL-ComNet Simulator

이 저장소는 raw UE antenna-domain MU-MIMO OFDM downlink dataset을 만들고,
CSIT error, clipping, receiver I/Q imbalance 조건에서 WL-aware ComNet
receiver를 학습/평가하는 코드입니다.

```text
TX dataset generator: tx_mumimo_e2e_dataset.py
RX receiver trainer:  rx_mumimo_receiver.py
PHY helper package:   mumimo_phy/
```

## 현재 Receiver 구조

현재 receiver는 receiver I/Q imbalance가 있는 경우를 기준으로
widely-linear (WL) chain에 맞춰져 있습니다.

```text
Pilots -> WL-LS channel estimate
WL-LS -> linear CE layer initialized by train-split WL-LMMSE -> WL-CE
Data + WL-CE -> WL-ZF features -> BiLSTM SD -> predicted bits
```

핵심 기준은 다음과 같습니다.

- CE 입력은 항상 `WL-LS`입니다.
- CE target은 augmented WL channel `(A, B)`입니다.
- CE는 single linear layer이고, training split에서 fit한 empirical
  `WL-LMMSE` weight로 초기화합니다.
- SD는 BiLSTM만 사용합니다. FC-SD는 현재 active flow에 포함하지 않습니다.
- `WL-MMSE`는 proposed detector가 아니라 comparison baseline입니다.

## 최종 실험 세트

아래 명령어들은 공통으로 다음 설정을 사용합니다.

```text
modulation = 64QAM
n_users = 2
n_tx = 8
n_rx_per_ue = 4
n_fft = 64
n_cp = 16
csit_error_var = 0.001
train/val SNR = 40 dB
test SNR sweep = 0, 5, ..., 40 dB
train/val/test frames = 50000 / 10000 / 10000 per SNR
```

### 1. Linear, No I/Q Impairment

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/linear_noiq --case linear --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0
```

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/linear_noiq --result-dir results/linear_noiq --mode train-all --device cuda
```

### 2. Clipping 3.0, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip30_iq05_p2_cpe3 --case clipping --clip-ratio 3.0 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip30_iq05_p2_cpe3 --result-dir results/clip30_iq05_p2_cpe3 --mode train-all --device cuda
```

### 3. Clipping 1.7, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip17_iq05_p2_cpe3 --case clipping --clip-ratio 1.7 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip17_iq05_p2_cpe3 --result-dir results/clip17_iq05_p2_cpe3 --mode train-all --device cuda
```

## 평가 지표

RX script는 result directory 아래에 `eval_summary.json`, CSV, plot을 저장합니다.
주요 BER curve는 다음과 같습니다.

```text
LS-MMSE
LMMSE-MMSE
WL-LS -> WL-ZF
WL-LS -> WL-MMSE
WL-LMMSE -> WL-MMSE
WL-CE -> WL-ZF-BiLSTM
True WL-H -> WL-MMSE
```

Channel estimation 품질은 WL true target `(A, B)` 기준으로 다음 항목을
MSE/NMSE로 기록합니다.

```text
WL-LS
WL-LMMSE
WL-CE
```

`True WL-H -> WL-MMSE`는 true-channel linear WL reference입니다. Clipping까지
최적으로 보상하는 nonlinear upper bound는 아닙니다.

## 주요 출력 파일

```text
config.json
train_snr40.npz
val_snr40.npz
test_snr00.npz ... test_snr40.npz
eval_summary.json
ber_vs_snr.csv / ber_vs_snr.png
channel_mse_vs_snr.csv / channel_nmse_vs_snr.csv
diagnostic_vs_snr.csv
train_history_ce.csv / ce_training_curve.png
train_history_bilstm_sd.csv / bilstm_sd_training_curve.png
```

## 문서

```text
TX_README.md          Dataset generation 설명
RX_README.md          Receiver, CE/SD, metric 설명
mumimo_phy/README.md  공통 PHY helper package 설명
```
