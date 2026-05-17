# TX Dataset Generator

`tx_mumimo_e2e_dataset.py`는 raw UE antenna-domain MU-MIMO OFDM waveform을 `.npz` dataset으로 생성합니다.

```text
TX script: tx_mumimo_e2e_dataset.py
RX script: rx_mumimo_receiver.py
Output type: raw_mumimo_e2e
```

## Main Dataset Command

현재 main dataset이 이미 있으면 TX를 다시 실행할 필요는 없습니다. RX feature, CE objective, LMMSE estimator 변경은 RX에서 계산됩니다.

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --modulation 64QAM --case clipping --clip-ratio 2.0 --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0.2 --rx-iq-phase-error-deg 1.0 --rx-common-phase-error-deg 1.0 --n-train-frames 10000 --n-val-frames 2000 --n-test-frames-per-snr 2000
```

## Main Setting

```text
modulation = 64QAM
n_users = 2
n_streams = n_users = 2
n_tx = 8
n_rx_per_ue = 4
n_fft = 64
n_cp = 16
n_taps = 7
n_rays_per_path = 15
pdp_decay = 5.0
channel_model = SCM-style clustered multipath
csit_error_var = 0.001
case = clipping
clip_ratio = 2.0
pilot_kind = qpsk
rx_iq_gain_imbalance_db = 0.2
rx_iq_phase_error_deg = 1.0
rx_common_phase_error_deg = 1.0
train_snr_db_list = 15 20 25 30 35 40
test_snr_db = 0 5 10 15 20 25 30 35 40
```

## Generation Flow

```text
random bits
-> QAM symbols
-> SCM-style multiuser multipath MIMO channel
-> strongest path angle selection
-> analog TX/RX steering beams
-> H_tx_est = H_true + CSIT error
-> digital ZF precoder
-> OFDM IFFT / clipping / CP insertion
-> multipath MIMO channel
-> RX RF impairment
-> AWGN
-> train/val/test .npz
```

## Channel and Error Models

### SCM Multipath

`mumimo_phy/scm.py`가 user별 clustered multipath channel을 만듭니다.

```text
h_time[user, tap, rx_ant, tx_ant]
H_true[subcarrier, user, rx_ant, tx_ant]
```

각 path는 random center angle, angle spread, random phase, exponential PDP를 가집니다.

### CSIT Error

TX는 실제 channel이 아니라 오차가 들어간 `H_tx_est`로 precoder를 만듭니다.

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

main setting에서는 `csit_error_var=0.001`을 사용합니다. 기존 `0.005`는 stress setting으로 분리하는 편이 맞습니다.

### Clipping

`case=clipping`이면 BS antenna별 OFDM time-domain symbol을 CP 삽입 전에 clipping합니다.

```text
threshold = clip_ratio * RMS(time_symbol)
```

main setting은 `clip_ratio=2.0`입니다. `clip_ratio=1.6`은 BER floor가 커서 stress setting으로 보는 것이 적절합니다.

### RF Impairment

RX front-end에서 I/Q gain imbalance, I/Q phase error, common phase rotation을 적용합니다.

widely-linear 관계:

```text
y_rf = alpha * y + beta * conj(y)
```

이 때문에 RX에서 plain complex channel 하나만으로는 RF impairment를 완전히 설명하기 어렵습니다. 그래서 RX에서는 `rf-reliability` feature와 `RF-aware True-H WL-MMSE` oracle baseline을 함께 봅니다.

## Output Files

```text
config.json
train_snr15_40mixed.npz
val_snr15_40mixed.npz
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

## Dataset Arrays

```text
rx_p_time              complex64 [n_frames, n_streams, n_users, n_rx_per_ue, time_len]
rx_d_time              complex64 [n_frames, n_users, n_rx_per_ue, time_len]
x_p_freq               complex64 [n_frames, n_streams, n_streams, n_fft]
x_d_freq               complex64 [n_frames, n_streams, n_fft]
bits                   int8      [n_frames, n_streams, n_fft, bits_per_symbol]
H_true                 complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_tx]
G_tx_est               complex64 [n_frames, n_fft, n_streams, n_tx]
W_precoder             complex64 [n_frames, n_fft, n_tx, n_streams]
A_eff_true             complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
snr_db                 float32   [n_frames]
noise_power            float32   [n_frames]
desired_power          float32   [n_frames, n_users]
inter_stream_power     float32   [n_frames, n_users]
effective_sinr_db      float32   [n_frames, n_users]
cond_A                 float32   [n_frames, n_fft, n_users]
mean_cond_A            float32   [n_frames]
p95_cond_A             float32   [n_frames]
```

`time_len = n_fft + n_cp` unless `case=cp_removal`.

## Notes

- TX는 receiver-specific neural feature를 만들지 않습니다.
- `rf-reliability` feature는 RX에서 CE output, RF params, `rx_d_time`을 이용해 계산합니다.
- `--lmmse-mode snr-binned`도 RX에서 train split을 읽어 fit하므로 TX 재생성이 필요 없습니다.
