# tx_mumimo_comnet_dataset.py

`tx_mumimo_comnet_dataset.py` creates a SISO-compatible ComNet dataset from a
downlink MU-MIMO OFDM simulation. It does not create raw per-antenna UE receive
waveforms. Instead, it applies conventional MU-MIMO precoding and UE combining,
then saves each user's post-combining effective scalar stream in the same `.npz`
shape expected by `rx_comnet_receiver.py`.

In short:

```text
MU-MIMO channel + ZF precoding + UE combining
-> effective scalar stream per user
-> SISO-compatible rx_p_time / rx_d_time
-> existing ComNet receiver
```

## Quick Start

Clean CSIT 64QAM smoke-sized dataset:

```powershell
python tx_mumimo_comnet_dataset.py `
  --out-dir outputs_mumimo_64qam_smoke `
  --modulation 64QAM `
  --n-users 2 `
  --n-train-frames 20 `
  --n-val-frames 8 `
  --n-test-frames-per-snr 8 `
  --snr-test-db 20 40 `
  --csit-error-var 0.0
```

Run the existing SISO ComNet receiver:

```powershell
python rx_comnet_receiver.py `
  --dataset-dir outputs_mumimo_64qam_smoke `
  --result-dir results_mumimo_64qam_smoke `
  --mode train-all `
  --sd-type fc `
  --ce-epochs 1 `
  --sd-epochs 1 `
  --device auto
```

Residual MUI experiment:

```powershell
python tx_mumimo_comnet_dataset.py `
  --out-dir outputs_mumimo_comnet_64qam_csit005 `
  --modulation 64QAM `
  --n-users 2 `
  --csit-error-var 0.005
```

## Model

Default configuration:

```text
N_FFT = 64
N_CP = 16
modulation = 64QAM
BS antennas = 8
UE antennas = 4 per user
users = 2
streams = 1 per user
channel = frequency-selective Rayleigh MIMO
combiner = dominant SVD vector from H_true
precoder = ZF from combiner-projected H_tx_est
power normalization = total Frobenius norm fixed
```

The BS channel estimate is

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

The combiner is computed from `H_true` in v1 so that `csit_error_var` primarily
tests BS precoder mismatch and residual multi-user interference.

For each subcarrier:

```text
g_est,u[k] = c_u[k]^H H_tx_est,u[k]
G_est[k] = stacked g_est,u[k]
W_raw[k] = G_est[k]^H (G_est[k] G_est[k]^H)^-1
W[k] = W_raw[k] / ||W_raw[k]||_F
```

Pilot symbols are sent sequentially per stream. Data symbols are sent for all
users simultaneously, so CSIT error can leave residual MUI in the data symbol.

## Output Files

The output directory contains:

```text
config.json
train_snr40.npz
val_snr40.npz
test_snr00.npz
test_snr05.npz
...
```

Core arrays match `rx_comnet_receiver.py`:

```text
rx_p_time: complex64, [N_eff, n_fft+n_cp]
rx_d_time: complex64, [N_eff, n_fft+n_cp]
x_p_freq:  complex64, [N_eff, n_fft]
x_d_freq:  complex64, [N_eff, n_fft]
h_true:    complex64, [N_eff, n_fft]
bits:      int8,      [N_eff, n_fft, bits_per_symbol]
snr_db:    float32,   [N_eff]
```

`N_eff = n_frames * n_users`.

Extra debug arrays are saved and ignored by the existing receiver:

```text
frame_id
user_id
h_eff_all
signal_power
mui_power
noise_power
effective_sinr_db
```

`rx_p_time` and `rx_d_time` are SISO-compatible synthetic waveforms reconstructed
from post-combining effective frequency-domain symbols. They are not raw
per-antenna UE receive waveforms.

## Interpreting True-H ZF-Hard

For this dataset, `h_true` is the desired effective scalar channel:

```text
h_true[u,k] = c_u[k]^H H_true,u[k] W[:,u,k]
```

Other users' streams are not folded into `h_true`. Therefore `True-H ZF-Hard` in
`rx_comnet_receiver.py` is a desired-channel oracle, not an optimal detector that
removes residual MUI. With `csit_error_var > 0`, high-SNR BER floors can be a
normal consequence of residual MUI rather than a receiver bug.

## Sanity Checks

Useful implementation checks:

```text
Noiseless pilot:
  FFT(rx_p_time) / x_p_freq ~= h_true

Noiseless clean CSIT data:
  csit_error_var = 0
  FFT(rx_d_time) ~= h_true * x_d_freq

Noiseless CSIT-error data:
  csit_error_var > 0
  FFT(rx_d_time) - h_true * x_d_freq has nonzero residual MUI power
```
