# TX Dataset Generator

`tx_mumimo_e2e_dataset.py`는 raw UE antenna-domain MU-MIMO OFDM dataset을
compressed `.npz` 파일로 생성합니다. CE/SD neural model은 여기서 만들지
않고, `rx_mumimo_receiver.py`에서 학습합니다.

## 현재 기본 설정

Script 기본값은 최종 제출용 실험군에 맞춰져 있습니다.

```text
modulation = 64QAM
n_users = 2
n_streams = 2
n_tx = 8
n_rx_per_ue = 4
n_fft = 64
n_cp = 16
n_taps = 7
n_rays_per_path = 15
pdp_decay = 5.0
channel_model = SCM-style geometric clustered channel
csit_error_var = 0.001
precoder_norm = column
pilot_kind = qpsk
train_snr_db_list = 40
test_snr_db = 0 5 10 15 20 25 30 35 40
n_train_frames = 50000
n_val_frames = 10000
n_test_frames_per_snr = 10000
```

CLI에서는 channel, RF, clipping 관련 설정만 바꿉니다. SNR과 frame 수는
최종 실험 재현성을 위해 code 기본값으로 고정했습니다.

## 최종 Dataset 생성 명령어

### Linear, No I/Q Impairment

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/linear_noiq --case linear --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0
```

### Clipping 3.0, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip30_iq05_p2_cpe3 --case clipping --clip-ratio 3.0 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

### Clipping 1.7, I/Q 0.5 dB, Phase 2 deg, CPE 3 deg

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip17_iq05_p2_cpe3 --case clipping --clip-ratio 1.7 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
```

## 생성 흐름

```text
random bits
-> Gray QAM symbols
-> SCM-style multiuser multipath channel
-> strongest path angle selection
-> analog TX/RX steering beams
-> H_tx_est = H_true + CSIT error
-> per-subcarrier digital ZF precoder
-> OFDM IFFT / optional clipping / CP insertion
-> multipath MIMO channel
-> receiver RF impairment
-> AWGN
-> train/val/test .npz files
```

## Channel 및 RF model

### CSIT Error

Transmitter는 imperfect channel estimate로 precoder를 설계합니다.

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

최종 실험에서는 `csit_error_var=0.001`을 사용합니다.

### Clipping

`--case clipping`이면 BS antenna별 OFDM time-domain symbol을 cyclic-prefix
삽입 전에 clipping합니다.

```text
threshold = clip_ratio * RMS(time_symbol)
```

### Receiver I/Q Impairment

RF impairment는 multipath channel 이후, AWGN 이전에 적용합니다.

```text
y_rf = alpha * y + beta * conj(y)
```

이 때문에 RX에서는 widely-linear `(A, B)` channel representation을 사용합니다.

## 출력 파일

기본 `40 dB` train/val 설정에서는 다음 파일이 생성됩니다.

```text
config.json
train_snr40.npz
val_snr40.npz
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

## Dataset 배열

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

`time_len = n_fft + n_cp`입니다. 단, `case=cp_removal`일 때는 `time_len = n_fft`입니다.
