# Raw End-to-End MU-MIMO Receiver

This path is the raw end-to-end MU-MIMO experiment path.

It is separate from the effective-SISO bridge:

```text
Bridge path:
  tx_mumimo_comnet_dataset.py -> rx_comnet_receiver.py
  MU-MIMO channel -> post-combining effective SISO stream -> SISO ComNet RX

Raw E2E path:
  tx_mumimo_e2e_dataset.py -> rx_mumimo_receiver.py
  MU-MIMO channel -> raw UE antenna waveform -> MU-MIMO RX
```

In the bridge path, inter-user interference is unmodeled residual interference
from the SISO receiver's point of view. In this raw E2E path, the receiver
estimates the full effective channel matrix `A`, so local ZF/MMSE can suppress
other streams within each UE's antenna dimension.

## TX Model

Default v1 configuration:

```text
n_tx = 8
n_users = 2
n_streams = n_users
n_rx_per_ue = 4
n_fft = 64
n_cp = 16
UE streams = 1 stream per UE
channel = tap-domain Rayleigh multipath -> FFT
pilot = orthogonal stream slots
power = per-stream fixed
```

The raw UE waveform is stored before UE combining. The BS ZF precoder still
needs one channel row per stream, so TX uses a precoder-design receive direction:

```text
c_u[k] = dominant-SVD receive direction from H_tx_est,u[k]
G_tx_est[u,k] = c_u[k]^H H_tx_est,u[k]
W_zf[k] = G_tx_est[k]^H (G_tx_est[k] G_tx_est[k]^H)^-1
```

Each precoder column is normalized:

```text
||W_precoder[:, s, k]||^2 = 1
```

With clean CSIT, `G_tx_est @ W_precoder` should have small off-diagonal terms.
The diagonal terms do not need to be 1 because of column normalization.

## Dataset Schema

Each split `.npz` contains:

```text
rx_p_time:   complex64 [n_frames, n_streams, n_users, n_rx_per_ue, n_fft+n_cp]
rx_d_time:   complex64 [n_frames, n_users, n_rx_per_ue, n_fft+n_cp]
x_p_freq:    complex64 [n_frames, n_streams, n_streams, n_fft]
x_d_freq:    complex64 [n_frames, n_streams, n_fft]
bits:        int8      [n_frames, n_streams, n_fft, bits_per_symbol]
H_true:      complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_tx]
G_tx_est:    complex64 [n_frames, n_fft, n_users, n_tx]
W_precoder:  complex64 [n_frames, n_fft, n_tx, n_streams]
A_eff_true:  complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
snr_db:      float32   [n_frames]
noise_power: float32   [n_frames]
```

`x_p_freq[n, p, s, k]` is the symbol sent by stream `s` during pilot slot `p`.
For the v1 orthogonal pilot, only `p == s` entries are nonzero.

The effective channel is:

```text
A_eff_true[n,k,u,r,s] =
  sum_tx H_true[n,k,u,r,tx] * W_precoder[n,k,tx,s]
```

## SNR Definition

Noise power is the complex per-antenna variance:

```text
n = sqrt(sigma2 / 2) * (randn + j randn)
E[|n|^2] = sigma2
```

For each frame:

```text
signal_power = mean(|A_eff_true @ x_d_freq|^2)
noise_power = signal_power / 10^(snr_db / 10)
```

The dataset also stores:

```text
desired_power
inter_stream_power
effective_sinr_db
cond_A
mean_cond_A
p95_cond_A
```

`inter_stream_power` is the pre-detection power from other streams. It is not
treated as unmodeled interference by the raw MU-MIMO receiver, because local
ZF/MMSE sees the full effective channel matrix.

## RX Model

`rx_mumimo_receiver.py` removes CP, performs FFT, estimates `A_ls` from the
orthogonal pilots, and evaluates local per-UE detection:

```text
Y_u[k] = A_u[k] s[k] + n
```

ZF:

```text
s_hat = (A^H A)^-1 A^H Y
```

MMSE with unit stream symbol power:

```text
s_hat = (A^H A + sigma2 I)^-1 A^H Y
```

v1 assumes `n_streams <= n_rx_per_ue` for local ZF. If this condition is not
met, ZF is disabled and regularized MMSE remains available.

Baselines:

```text
LS-ZF
LS-MMSE
True-H ZF
True-H MMSE
Desired-only MRC
```

`Desired-only MRC` uses only the target stream channel and ignores other streams,
so it is an interference-ignorant baseline.

The neural v1 receiver trains one SDNet per UE:

```text
mumimo_sdnet_user0.pt
mumimo_sdnet_user1.pt
...
```

Each SDNet uses grouped subcarriers. With `group_size=8` and 64QAM, each training
sample predicts `8 * 6 = 48` target bits for one UE.

## Quick Start

16QAM smoke dataset:

```powershell
python tx_mumimo_e2e_dataset.py `
  --out-dir outputs_mumimo_e2e_16qam_smoke `
  --modulation 16QAM `
  --n-users 2 `
  --n-tx 8 `
  --n-rx-per-ue 4 `
  --n-train-frames 20 `
  --n-val-frames 8 `
  --n-test-frames-per-snr 8 `
  --snr-test-db 20 40 `
  --csit-error-var 0.0
```

Train and evaluate:

```powershell
python rx_mumimo_receiver.py `
  --dataset-dir outputs_mumimo_e2e_16qam_smoke `
  --result-dir results_mumimo_e2e_16qam_smoke `
  --mode train-all `
  --device cuda
```

Noiseless sanity:

```powershell
python tx_mumimo_e2e_dataset.py `
  --out-dir outputs_mumimo_e2e_16qam_noiseless_smoke `
  --modulation 16QAM `
  --n-train-frames 1 `
  --n-val-frames 1 `
  --n-test-frames-per-snr 1 `
  --snr-train-db inf `
  --snr-test-db inf

python rx_mumimo_receiver.py `
  --dataset-dir outputs_mumimo_e2e_16qam_noiseless_smoke `
  --result-dir results_mumimo_e2e_16qam_noiseless_smoke `
  --mode eval `
  --device cpu
```
