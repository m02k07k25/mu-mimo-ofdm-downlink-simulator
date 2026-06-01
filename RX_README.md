# RX Receiver

`rx_mumimo_receiver.py`는 `tx_mumimo_e2e_dataset.py`로 생성한 raw MU-MIMO
OFDM dataset을 읽고, WL-aware ComNet receiver를 학습/평가합니다.

```text
Input:  datasets/*/*.npz and config.json
Output: checkpoints, train history CSVs, evaluation CSVs, plots, eval_summary.json
```

## 최종 RX 명령어

### Linear, No I/Q Impairment

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/linear_noiq --result-dir results/linear_noiq --mode train-all
```

### Clipping 3.0, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip30_iq05_p2_cpe3 --result-dir results/clip30_iq05_p2_cpe3 --mode train-all
```

### Clipping 1.7, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip17_iq05_p2_cpe3 --result-dir results/clip17_iq05_p2_cpe3 --mode train-all
```

## Receiver 흐름

```text
rx_p_time -> CP removal -> FFT -> WL-LS channel estimate
rx_d_time -> CP removal -> FFT -> received data grid
WL-LS -> empirical WL-LMMSE initialization -> linear CE -> WL-CE
received data + WL-CE -> WL-ZF features -> BiLSTM SD -> predicted bits
predicted bits -> BER
```

## Channel Estimation

현재 CE path는 논문 구조에 맞춰 단순하게 유지합니다.

```text
CE input  = WL-LS
CE target = augmented WL channel (A, B)
CE model  = one linear layer
CE init   = empirical WL-LMMSE weight fitted on the training split only
CE output = WL-CE
```

`--ce-target auto`는 `wl-rf`로 resolve됩니다. 따라서 RF impairment가 0인
경우에도 CE target은 augmented widely-linear representation입니다. No-I/Q
조건에서는 conjugate branch가 0입니다.

### CE 모델 입력값

CE 모델은 user별 `WL-LS` 채널 추정값 전체를 하나의 real-valued vector로
변환해 입력으로 사용합니다.

```text
a_ls shape before vectorization
  = [frames, n_fft, n_users, n_rx_per_ue, 2 * n_streams]

CE input shape
  = [frames * n_users, 2 * n_fft * n_rx_per_ue * (2 * n_streams)]
```

마지막 `2 * n_streams`는 augmented WL channel `(A, B)`의 stream 축입니다.
CE input vector의 바깥쪽 `2 *`는 complex 값을 real/imag로 분리하기 때문에
붙습니다. 기본 설정에서는 다음과 같습니다.

```text
n_fft = 64
n_rx_per_ue = 4
n_streams = 2
CE input_dim = 2 * 64 * 4 * (2 * 2) = 2048
```

CE target도 같은 방식으로 vectorization한 augmented WL true channel
`(A, B)`입니다. 현재 CE 모델은 `input_dim -> input_dim` single linear
layer입니다.

## LMMSE Estimator

기본 LMMSE mode는 다음과 같습니다.

```text
--lmmse-mode snr-binned
```

`snr-binned` 모드에서는 training split에 포함된 SNR별로 WL-LMMSE bin을
fit합니다. Test SNR은 사용 가능한 bin 중 가장 가까운 bin을 사용합니다.

Plain non-WL LMMSE estimator도 training split에서 fit하며, 이는 `LMMSE-MMSE`
comparison baseline에 사용합니다.

## Symbol Detector

제안 neural detector는 BiLSTM만 사용합니다.

```text
WL-CE -> WL-ZF-BiLSTM
```

FC-SD path는 active result metric에서 제거했습니다. 일부 legacy checkpoint
filename은 기존 결과 재평가를 위한 fallback으로만 허용합니다.

### SD 모델 입력값

SD는 CE가 출력한 `WL-CE` 채널과 수신 data grid를 사용해 WL-ZF 검출 결과를
만든 뒤, subcarrier별 9개 특징을 BiLSTM에 입력합니다.

```text
1. WL-ZF equalized target symbol real
2. WL-ZF equalized target symbol imag
3. channel-power normalized matched residual real
4. channel-power normalized matched residual imag
5. log residual power
6. log WL channel gain power
7. log condition number
8. log noise power
9. normalized SNR (= snr_db / 40)
```

즉, SD는 equalized symbol만 사용하는 것이 아닙니다. WL channel gain,
reconstruction residual, channel condition number, noise power, SNR 정보도
함께 사용합니다.

```text
SD input shape
  = [frames * n_streams, n_fft, 9]

default SD input shape
  = [frames * 2, 64, 9]
```

BiLSTM은 subcarrier 축을 sequence 축으로 처리합니다. 출력 target은
`group_size`개 subcarrier의 bit를 묶은 값입니다. 기본 `64QAM`,
`group_size=8`에서는 다음 shape을 사용합니다.

```text
bits_per_symbol = 6
n_groups = n_fft / group_size = 64 / 8 = 8
SD target shape = [frames * n_streams, 8, 8 * 6]
                = [frames * n_streams, 8, 48]
```

## BER Metrics

`eval_summary.json`, CSV, plot에 저장되는 주요 BER metric은 다음과 같습니다.

```text
LS-MMSE
LMMSE-MMSE
WL-LS -> WL-ZF
WL-LS -> WL-MMSE
WL-LMMSE -> WL-MMSE
WL-CE -> WL-ZF-BiLSTM
True WL-H -> WL-MMSE
```

해석 기준은 다음과 같습니다.

- `LS-MMSE`, `LMMSE-MMSE`는 non-WL comparison baseline입니다.
- `WL-LS -> WL-ZF`, `WL-LS -> WL-MMSE`는 pilot LS 기반 WL receiver입니다.
- `WL-LMMSE -> WL-MMSE`는 강한 empirical linear CE baseline입니다.
- `WL-CE -> WL-ZF-BiLSTM`은 proposed neural detector path입니다.
- `True WL-H -> WL-MMSE`는 true-channel linear WL reference이며, clipping까지
  최적으로 보상하는 nonlinear upper bound는 아닙니다.

## Channel Metrics

Channel MSE/NMSE는 WL true target `(A, B)` 기준으로 계산합니다.

```text
WL-LS
WL-LMMSE
WL-CE
```

Channel MSE가 BER과 항상 같은 방향으로 움직이지는 않습니다. 최종 BER에는
detector objective, QAM decision boundary, clipping distortion, WL gain
normalization 영향이 함께 들어갑니다.

## 출력 파일

```text
mumimo_lmmse_estimator.npz
mumimo_plain_lmmse_estimator.npz
mumimo_ce_linear.pt
mumimo_wl_zf_bilstm.pt
train_history_ce.csv
train_history_bilstm_sd.csv
ce_training_curve.png
bilstm_sd_training_curve.png
eval_summary.json
ber_vs_snr.csv
ber_vs_snr.png
a_mse_vs_snr.png
channel_mse_vs_snr.csv
channel_nmse_vs_snr.csv
diagnostic_vs_snr.csv
```
