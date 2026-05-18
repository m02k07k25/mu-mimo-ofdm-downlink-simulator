# Experiment Results

Updated: 2026-05-18

## Scope

This file summarizes the latest 64QAM MU-MIMO E2E receiver runs for three impairment settings and two CSIT-error settings.

Common RX setting:

```text
rx_mumimo_receiver.py
mode = train-all
sd_type = bilstm
sd_feature_set = rf-reliability
ce_type = blend-resmlp
ce_target = auto
lmmse_mode = snr-binned
bilstm_epochs = 300
device = cuda
```

## Run Matrix

| Run family | case | clip_ratio | IQ gain | IQ phase | common phase | train | val | test per SNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_clip_no_iq | linear | unused | 0 dB | 0 deg | 0 deg | 100000 | 20000 | 20000 |
| clip20_rfcase2 | clipping | 2.0 | 0.2 dB | 1 deg | 1 deg | 50000 | 10000 | 10000 |
| clip30_rfcase2 | clipping | 3.0 | 0.2 dB | 1 deg | 1 deg | 50000 | 10000 | 10000 |

Two CSIT settings were run:

```text
csit_error_var = 0.005
csit_error_var = 0.001
```

The `no_clip_no_iq` run has twice the data size of the two clipping runs, so it is a clean sanity reference rather than a fully data-size-matched ablation.

## Result Directories

CSIT error `0.005`:

```text
results_mumimo_e2e_64qam_no_clip_no_iq_train100000_blend_ce_rf_reliability_snr_lmmse
results_mumimo_e2e_64qam_clip20_rfcase2_train50000_blend_ce_rf_reliability_snr_lmmse
results_mumimo_e2e_64qam_clip30_rfcase2_train50000_blend_ce_rf_reliability_snr_lmmse
```

CSIT error `0.001`:

```text
results_mumimo_e2e_64qam_csit001_no_clip_no_iq_train100000_blend_ce_rf_reliability_snr_lmmse
results_mumimo_e2e_64qam_csit001_clip20_rfcase2_train50000_blend_ce_rf_reliability_snr_lmmse
results_mumimo_e2e_64qam_csit001_clip30_rfcase2_train50000_blend_ce_rf_reliability_snr_lmmse
```

## BER Summary

ComNet-BiLSTM BER, `csit_error_var = 0.005`:

| SNR | no_clip_no_iq | clip20_rfcase2 | clip30_rfcase2 |
|---:|---:|---:|---:|
| 0 | 3.249e-1 | 3.219e-1 | 3.223e-1 |
| 5 | 2.060e-1 | 2.094e-1 | 2.095e-1 |
| 10 | 1.099e-1 | 1.139e-1 | 1.121e-1 |
| 15 | 4.397e-2 | 4.856e-2 | 4.542e-2 |
| 20 | 1.445e-2 | 1.800e-2 | 1.510e-2 |
| 25 | 4.788e-3 | 7.350e-3 | 5.005e-3 |
| 30 | 1.550e-3 | 3.900e-3 | 1.673e-3 |
| 35 | 4.913e-4 | 2.881e-3 | 5.991e-4 |
| 40 | 1.544e-4 | 2.587e-3 | 2.578e-4 |

ComNet-BiLSTM BER, `csit_error_var = 0.001`:

| SNR | no_clip_no_iq | clip20_rfcase2 | clip30_rfcase2 |
|---:|---:|---:|---:|
| 0 | 3.230e-1 | 3.179e-1 | 3.233e-1 |
| 5 | 2.055e-1 | 2.073e-1 | 2.093e-1 |
| 10 | 1.086e-1 | 1.122e-1 | 1.109e-1 |
| 15 | 4.194e-2 | 4.659e-2 | 4.353e-2 |
| 20 | 1.297e-2 | 1.643e-2 | 1.359e-2 |
| 25 | 4.259e-3 | 6.722e-3 | 4.481e-3 |
| 30 | 1.477e-3 | 3.773e-3 | 1.604e-3 |
| 35 | 4.840e-4 | 2.846e-3 | 5.750e-4 |
| 40 | 1.575e-4 | 2.574e-3 | 2.533e-4 |

40 dB comparison:

| Run | CSIT | LS-MMSE | LMMSE-MMSE | ComNet-CE-ZF-Hard | ComNet-BiLSTM | RF-aware True-H WL-MMSE |
|---|---:|---:|---:|---:|---:|---:|
| no_clip_no_iq | 0.005 | 1.540e-4 | 2.107e-4 | 1.555e-4 | 1.544e-4 | 5.104e-5 |
| no_clip_no_iq | 0.001 | 1.537e-4 | 2.570e-4 | 1.551e-4 | 1.575e-4 | 5.124e-5 |
| clip20_rfcase2 | 0.005 | 2.785e-3 | 2.629e-3 | 2.689e-3 | 2.587e-3 | 1.611e-3 |
| clip20_rfcase2 | 0.001 | 2.773e-3 | 2.569e-3 | 2.666e-3 | 2.574e-3 | 1.608e-3 |
| clip30_rfcase2 | 0.005 | 3.954e-4 | 3.674e-4 | 3.620e-4 | 2.578e-4 | 5.859e-5 |
| clip30_rfcase2 | 0.001 | 3.911e-4 | 4.302e-4 | 3.630e-4 | 2.533e-4 | 5.651e-5 |

## Channel MSE

Channel MSE at high SNR, `csit_error_var = 0.005`:

| SNR | no_clip CE | no_clip LMMSE | clip20 CE | clip20 LMMSE | clip30 CE | clip30 LMMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | -21.88 | -22.13 | -21.64 | -21.73 | -21.78 | -21.87 |
| 30 | -25.47 | -25.86 | -24.82 | -25.13 | -25.36 | -25.60 |
| 35 | -29.91 | -30.11 | -27.73 | -28.44 | -29.37 | -29.87 |
| 40 | -34.78 | -34.80 | -29.62 | -30.98 | -32.99 | -34.59 |

Channel MSE at high SNR, `csit_error_var = 0.001`:

| SNR | no_clip CE | no_clip LMMSE | clip20 CE | clip20 LMMSE | clip30 CE | clip30 LMMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | -22.25 | -22.59 | -22.00 | -22.16 | -22.14 | -22.30 |
| 30 | -25.71 | -26.28 | -25.06 | -25.53 | -25.60 | -25.96 |
| 35 | -29.98 | -30.34 | -27.83 | -28.71 | -29.46 | -29.98 |
| 40 | -34.79 | -34.87 | -29.65 | -31.24 | -33.03 | -34.56 |

Lowering `csit_error_var` from `0.005` to `0.001` improves channel MSE slightly, especially around 25-30 dB, but it does not materially change the BER floor in the clipping cases.

## Interpretation

1. `clip_ratio = 2.0` is the dominant hard case.
   At 40 dB, ComNet-BiLSTM stays around `2.58e-3` for both CSIT settings. The RF-aware oracle is also high at about `1.61e-3`, so the floor is not explained by channel-estimation error alone.

2. `clip_ratio = 3.0` is a mild impairment.
   It is close to no-clipping through most SNRs. At 40 dB it lands around `2.5e-4`, much better than clip20 but still above the clean/oracle floor.

3. Reducing CSIT error is not the main lever for the current BER floor.
   The 0.001 runs are slightly better in mid/high SNR, but the final BER story is nearly unchanged. The impairment severity, especially clipping at 2.0, dominates.

4. The neural detector is only modestly better than CE-ZF-Hard.
   ComNet-BiLSTM helps most visibly in clip30 high SNR, but it still does not close the gap to `RF-aware True-H WL-MMSE`.

5. The clean run validates the basic chain.
   In no-clipping/no-IQ, LS-MMSE and ComNet-BiLSTM reach about `1.5e-4` at 40 dB, while the oracle is about `5e-5`. The implementation looks sane; the remaining issue is detector/equalizer structure under impairment.

## Next Parameters To Run

### 1. Data-size matched clean reference

The clean run currently uses 100k/20k/20k while clipping runs use 50k/10k/10k. Add a matched clean run before claiming exact clean-vs-clipped deltas.

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_no_clip_no_iq_train50000 --modulation 64QAM --case linear --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0 --n-train-frames 50000 --n-val-frames 10000 --n-test-frames-per-snr 10000
```

### 2. RF-only and clipping-only ablations

These isolate which part of the 4-error bundle matters most.

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_rfcase2_no_clip_train50000 --modulation 64QAM --case linear --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0.2 --rx-iq-phase-error-deg 1.0 --rx-common-phase-error-deg 1.0 --n-train-frames 50000 --n-val-frames 10000 --n-test-frames-per-snr 10000
```

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_clip20_no_iq_train50000 --modulation 64QAM --case clipping --clip-ratio 2.0 --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0 --n-train-frames 50000 --n-val-frames 10000 --n-test-frames-per-snr 10000
```

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_clip30_no_iq_train50000 --modulation 64QAM --case clipping --clip-ratio 3.0 --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0 --rx-iq-phase-error-deg 0 --rx-common-phase-error-deg 0 --n-train-frames 50000 --n-val-frames 10000 --n-test-frames-per-snr 10000
```

### 3. Clip-ratio transition sweep

The transition is between 2.0 and 3.0. Run 2.5 next.

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_clip25_rfcase2_train50000 --modulation 64QAM --case clipping --clip-ratio 2.5 --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0.2 --rx-iq-phase-error-deg 1.0 --rx-common-phase-error-deg 1.0 --n-train-frames 50000 --n-val-frames 10000 --n-test-frames-per-snr 10000
```

RX template for each new dataset:

```powershell
python rx_mumimo_receiver.py --dataset-dir DATASET_DIR --result-dir RESULT_DIR --mode train-all --sd-type bilstm --sd-feature-set rf-reliability --ce-type blend-resmlp --ce-target auto --lmmse-mode snr-binned --bilstm-epochs 300 --device cuda
```

## Improvement Directions

1. Add a correction-based detector instead of only direct bit prediction.

```text
baseline bit = LS-MMSE or RF-aware WL-MMSE hard decision
model target = baseline_bit XOR true_bit
final bit = baseline_bit XOR predicted_correction
```

Track:

```text
baseline wrong bits corrected
baseline correct bits damaged
false correction rate
net BER gain
```

2. Add RF-aware or widely-linear features.
   For RF/IQ runs, the meaningful oracle is `RF-aware True-H WL-MMSE`. The learned receiver should receive conjugate-leakage features or have a widely-linear equalizer path.

3. Prioritize detector/equalizer structure over CE-only improvements.
   CE MSE is already close to LMMSE, but BER remains far from oracle. Reducing channel MSE alone is unlikely to remove the current floor.

4. Keep future ablations controlled.
   Use the same frame counts, `csit_error_var`, seed, SNR list, and receiver epochs when comparing clipping/RF cases.
