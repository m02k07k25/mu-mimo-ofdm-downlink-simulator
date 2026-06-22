# TX Dataset Generator

`tx_mumimo_e2e_dataset.py`는 raw UE antenna-domain MU-MIMO OFDM 데이터셋을
compressed `.npz` 파일로 생성한다. TX 스크립트는 neural model을 학습하지 않고,
CE/SD 학습과 평가는 `rx_mumimo_receiver.py`에서 수행한다.

## 설정 기준

일반 실행은 프로젝트 루트의 `environment_config.json`을 먼저 읽고, 그 값을
argparse 기본값으로 주입한다. 따라서 `python tx_mumimo_e2e_dataset.py`는 코드
안의 순수 CLI 기본값이 아니라 `environment_config.json` 기준으로 실행된다.
필요하면 CLI 인자로 개별 값을 override할 수 있다.

현재 `environment_config.json`의 `dataset_name`은 `clip17_iq05_p2_cpe3_test`다.
기존 최종 결과 폴더인 `datasets/clip17_iq05_p2_cpe3`와
`results/clip17_iq05_p2_cpe3`는 같은 물리 조건의 clipping 1.7 실험이다.

## 최종 실험 조건

| 항목 | 값 |
|---|---:|
| modulation | 64QAM |
| users / streams | 2 / 2 |
| TX antennas | 8 |
| RX antennas per UE | 4 |
| FFT / CP | 64 / 16 |
| channel | SCM-style geometric clustered channel |
| taps / rays per path | 7 / 15 |
| PDP decay | 5.0 |
| carrier frequency | 800 MHz |
| antenna spacing | 0.5 lambda |
| SCM angle spread | 3 deg |
| CSIT error variance | 0.001 |
| precoder norm | column |
| pilot kind | qpsk |
| train SNR | 40 dB |
| test SNRs | 0, 5, 10, 15, 20, 25, 30, 35, 40 dB |
| train / val / test frames | 50000 / 10000 / 10000 per SNR |
| impairment case | clipping |
| clip ratio | 1.7 |
| RX I/Q gain imbalance | 0.5 dB |
| RX I/Q phase error | 2 deg |
| RX common phase error | 3 deg |
| seed | 7 |

## 대표 명령어

`environment_config.json` 기준으로 생성:

```powershell
python tx_mumimo_e2e_dataset.py
```

기존 최종 폴더 이름으로 clipping 1.7 데이터셋을 명시 생성:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip17_iq05_p2_cpe3 --case clipping --clip-ratio 1.7 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

clipping 3.0 비교 데이터셋:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip30_iq05_p2_cpe3 --case clipping --clip-ratio 3.0 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

linear no-I/Q 비교 데이터셋:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/linear_noiq --case linear --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0
```

데모용 test-only 데이터셋:

```powershell
python demo_tx_make_data.py
```

## 생성 흐름

```text
random bits
-> Gray-coded QAM symbols
-> SCM-style multiuser multipath channel
-> strongest-path angle selection
-> analog TX/RX steering beams
-> H_tx_est = H_true + CSIT error
-> per-subcarrier digital ZF precoder
-> OFDM IFFT
-> optional BS time-domain clipping
-> CP insertion
-> multipath MIMO channel
-> receiver RF impairment
-> AWGN
-> train/val/test .npz files
```

## 주요 모델링

### CSIT Error

TX precoder는 완전한 channel이 아니라 noisy channel estimate로 설계된다.

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

최종 실험에서는 `csit_error_var=0.001`을 사용한다.

### Clipping

`case=clipping`이면 BS antenna별 time-domain OFDM symbol을 CP 삽입 전에
clipping한다.

```text
threshold = clip_ratio * RMS(time_symbol)
```

최종 clipping 1.7 실험에서는 `clip_ratio=1.7`이다.

### Receiver RF Impairment

RX I/Q imbalance와 common phase error는 multipath channel 이후, AWGN 이전에
적용된다.

```text
y_rf = alpha * y + beta * conj(y)
```

이 때문에 RX에서는 일반 complex-linear channel만으로는 충분하지 않고,
widely-linear `(A, B)` representation을 사용한다.

### AWGN and SNR

각 frame의 noise power는 clean received data waveform 기준으로 정해진다.

```text
sigma2 = mean(|clean received data waveform|^2) / 10^(snr_db/10)
noise = sqrt(sigma2/2) * (n_re + j*n_im)
```

test split은 SNR별로 같은 base frame을 공유하고 AWGN scale만 바뀐다.

## 출력 파일

일반 40 dB train/val 설정에서는 다음 파일이 생성된다.

```text
config.json
train_snr40.npz
val_snr40.npz
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

## 주요 배열

```text
rx_p_time              complex64 [frames, streams, users, rx, time_len]
rx_d_time              complex64 [frames, users, rx, time_len]
x_p_freq               complex64 [frames, streams, streams, n_fft]
x_d_freq               complex64 [frames, streams, n_fft]
bits                   int8      [frames, streams, n_fft, bits_per_symbol]
H_true                 complex64 [frames, n_fft, users, rx, n_tx]
G_tx_est               complex64 [frames, n_fft, streams, n_tx]
W_precoder             complex64 [frames, n_fft, n_tx, streams]
W_digital              complex64 [frames, n_fft, streams, streams]
W_tx_analog            complex64 [frames, n_tx, streams]
W_rx_analog            complex64 [frames, users, rx]
A_eff_true             complex64 [frames, n_fft, users, rx, streams]
snr_db                 float32   [frames]
signal_power           float32   [frames]
noise_power            float32   [frames]
desired_power          float32   [frames, users]
inter_stream_power     float32   [frames, users]
effective_sinr_db      float32   [frames, users]
cond_A                 float32   [frames, n_fft, users]
mean_cond_A            float32   [frames]
p95_cond_A             float32   [frames]
```

`linear`와 `clipping`에서는 `time_len = n_fft + n_cp`다. `cp_removal` case에서는
CP를 붙이지 않으므로 `time_len = n_fft`다.
