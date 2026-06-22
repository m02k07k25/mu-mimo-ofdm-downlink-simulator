# RX Receiver

`rx_mumimo_receiver.py`는 `tx_mumimo_e2e_dataset.py`가 생성한 raw MU-MIMO OFDM
데이터셋을 읽어 WL-aware ComNet receiver를 학습하고 평가한다.

```text
Input:  datasets/<dataset_name>/*.npz and config.json
Output: checkpoints, training histories, evaluation summaries, CSVs, plots
```

## 실행 기준

일반 실행은 `environment_config.json`을 먼저 읽어 dataset/result path와 학습
hyperparameter를 정한다. 현재 `environment_config.json`의 `dataset_name`은
`clip17_iq05_p2_cpe3_test`다. 아래 결과 해석 표는 사용자가 확인한 기존 최종
폴더 `results/clip17_iq05_p2_cpe3/eval_summary.json` 기준이다.

## 대표 명령어

`environment_config.json` 기준 전체 학습 및 평가:

```powershell
python rx_mumimo_receiver.py
```

기존 최종 폴더 이름으로 clipping 1.7 실험 재학습 및 평가:

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip17_iq05_p2_cpe3 --result-dir results/clip17_iq05_p2_cpe3 --mode train-all
```

기존 checkpoint로 평가만 다시 수행:

```powershell
python rx_mumimo_receiver.py --dataset-dir datasets/clip17_iq05_p2_cpe3 --result-dir results/clip17_iq05_p2_cpe3 --mode eval
```

데모 inference:

```powershell
python demo_rx_infer_fixed_pt.py
```

## Receiver 흐름

```text
rx_p_time -> CP removal -> FFT -> WL-LS channel estimate
rx_d_time -> CP removal -> FFT -> received data grid
WL-LS -> empirical WL-LMMSE initialization -> linear CE -> WL-CE
received data + WL-CE -> WL-ZF reliability features
WL-ZF features -> BiLSTM SD -> predicted bits
predicted bits -> BER
```

## Channel Estimation

현재 CE path는 RF impairment가 있는 WL 구조에 맞춰 단순하게 고정되어 있다.

```text
CE input  = WL-LS
CE target = augmented WL channel (A, B)
CE model  = one bias-free linear layer
CE init   = empirical WL-LMMSE weight fitted on the training split only
CE output = WL-CE
```

`--ce-target auto`는 `wl-rf`로 resolve된다. RF impairment가 0인 조건에서도
target representation은 augmented WL `(A, B)`이며, 이때 conjugate branch `B`가
0에 가까워진다.

### CE 입력 차원

user별 WL-LS channel estimate 전체를 real-valued vector로 펼쳐 CE 입력으로 쓴다.

```text
a_ls shape before vectorization
  = [frames, n_fft, n_users, n_rx_per_ue, 2 * n_streams]

CE input shape
  = [frames * n_users, 2 * n_fft * n_rx_per_ue * (2 * n_streams)]
```

최종 설정에서는 다음과 같다.

```text
n_fft = 64
n_rx_per_ue = 4
n_streams = 2
CE input_dim = 2 * 64 * 4 * (2 * 2) = 2048
```

`MuMimoCELinearNet`은 `2048 -> 2048` single linear layer다. `hidden_dim`과
`dropout` 설정값은 checkpoint metadata/호환 목적의 속성으로 남아 있지만, 현재
CE forward path에는 hidden layer나 dropout이 들어가지 않는다.

## LMMSE Estimator

기본 LMMSE mode는 `snr-binned`다.

```text
--lmmse-mode snr-binned
```

training split에 포함된 SNR별로 empirical WL-LMMSE weight를 fit하고, test SNR은
가장 가까운 bin을 사용한다. 최종 설정은 train SNR이 40 dB 하나뿐이므로 모든
test SNR에서 40 dB bin의 WL-LMMSE weight가 쓰인다.

plain non-WL LMMSE estimator도 training split에서 별도로 fit하며, 이는
`LMMSE-MMSE` comparison baseline에 사용된다.

## Symbol Detector

제안 neural detector는 BiLSTM SD 하나만 사용한다.

```text
WL-CE -> WL-ZF-BiLSTM
```

FC-SD path는 현재 active metric에서 제거되어 있다. legacy checkpoint 이름은
기존 결과 호환용 fallback으로만 남아 있다.

### SD 입력 feature

SD는 CE가 출력한 `WL-CE` channel과 received data grid로 WL-ZF rough estimate를
만든 뒤, subcarrier별 9개 feature를 BiLSTM에 입력한다.

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

즉 SD는 equalized symbol만 쓰는 detector가 아니라 residual, channel gain,
condition number, noise power, SNR 정보를 함께 쓰는 neural post-detector다.

```text
SD input shape
  = [frames * n_streams, n_fft, 9]

default SD input shape
  = [frames * 2, 64, 9]
```

64QAM, `group_size=8` 기준 target shape는 다음과 같다.

```text
bits_per_symbol = 6
n_groups = n_fft / group_size = 64 / 8 = 8
SD target shape = [frames * n_streams, 8, 8 * 6]
                = [frames * n_streams, 8, 48]
```

BiLSTM hidden dimensions는 `environment_config.json` 기준 `[64, 32, 16]`이다.

### SD loss

기본 SD loss는 BCE가 아니라 sigmoid-MSE다.

```text
--sd-loss mse
loss = MSE(sigmoid(logits), target_bits)
```

`--sd-loss bce`를 명시한 경우에만 `binary_cross_entropy_with_logits`를 사용한다.
현재 기본값과 일반 실행 경로는 `mse`다. 학습 중 best checkpoint 선택 기준은
`val_loss`가 아니라 `val_BER`이며, inference에서는 `sigmoid(logits) > 0.5`로
bit를 결정한다.

## BER Metrics

`eval_summary.json`, CSV, plot에 저장되는 주요 BER metric은 다음과 같다.

```text
LS-MMSE
LMMSE-MMSE
WL-LS -> WL-ZF
WL-LS -> WL-MMSE
WL-LMMSE -> WL-MMSE
WL-CE -> WL-ZF-BiLSTM
True WL-H -> WL-MMSE
```

해석 기준:

- `LS-MMSE`, `LMMSE-MMSE`는 non-WL comparison baseline이다.
- `WL-LS -> WL-ZF`, `WL-LS -> WL-MMSE`는 pilot LS 기반 WL receiver다.
- `WL-LMMSE -> WL-MMSE`는 empirical linear WL-CE baseline이다.
- `WL-CE -> WL-ZF-BiLSTM`은 제안 neural detector path다.
- `True WL-H -> WL-MMSE`는 true-channel linear WL reference다. clipping까지
  완전히 보상하는 nonlinear oracle upper bound는 아니다.

## Clipping 1.7 최종 결과 요약

아래 표는 `results/clip17_iq05_p2_cpe3/eval_summary.json` 기준이다.

| SNR | LS-MMSE | LMMSE-MMSE | WL-LMMSE -> WL-MMSE | WL-CE -> WL-ZF-BiLSTM | True WL-H -> WL-MMSE |
|---:|---:|---:|---:|---:|---:|
| 20 dB | 0.04836 | 0.03717 | 0.03618 | 0.05076 | 0.01870 |
| 25 dB | 0.02743 | 0.01909 | 0.01798 | 0.02511 | 0.01122 |
| 30 dB | 0.02011 | 0.01340 | 0.01235 | 0.01426 | 0.00909 |
| 35 dB | 0.01781 | 0.01177 | 0.01074 | 0.01059 | 0.00847 |
| 40 dB | 0.01710 | 0.01127 | 0.01026 | 0.00957 | 0.00829 |

정리하면 제안 `WL-CE -> WL-ZF-BiLSTM`은 high-SNR 영역에서 error floor를 가장
낮춘다. 35 dB에서 0.01059, 40 dB에서 0.00957을 기록하여 모든 실용 receiver보다
낮고, true-channel linear WL reference인 0.00829에 가까워진다.

40 dB 기준 개선율은 다음과 같다.

| 기준 receiver | BER | 제안 대비 개선율 |
|---|---:|---:|
| WL-LMMSE -> WL-MMSE | 0.01026 | 6.8% |
| LMMSE-MMSE | 0.01127 | 15.1% |
| LS-MMSE | 0.01710 | 44.1% |

다만 20-30 dB에서는 WL-ZF 전단의 noise enhancement 영향으로
`WL-LMMSE -> WL-MMSE`보다 BER이 높다. 따라서 제안 구조의 이득은 clipping
잔차가 noise보다 상대적으로 두드러지는 35 dB 이상 high-SNR 영역에 집중된다.

## Channel Metrics

Channel MSE/NMSE는 WL true target `(A, B)` 기준으로 계산한다.

```text
WL-LS
WL-LMMSE
WL-CE
```

Channel MSE가 BER과 항상 같은 방향으로 움직이지는 않는다. 최종 BER에는 channel
estimation error뿐 아니라 detector objective, QAM decision boundary, clipping
distortion, WL gain normalization, WL-ZF noise enhancement가 함께 반영된다.

## 출력 파일

현재 코드로 학습/평가를 실행하면 다음 파일들이 결과 폴더에 저장된다.

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

기존 `results/clip17_iq05_p2_cpe3` 폴더는 오래된 실행 산출물이라 일부 CSV가 없고
plot/checkpoint/eval_summary 중심으로 남아 있을 수 있다. 수치 검토는
`eval_summary.json`을 기준으로 한다.
