# TX 데이터셋 생성 상세 설명

이 문서는 현재 TX 스크립트인 `tx_mumimo_e2e_dataset.py`를 설명합니다. 파일명은 과거 `tx_mumimo_comnet_dataset.py` 경로를 설명하던 이름이지만, 현재 루트에 있는 실제 생성기는 raw end-to-end MU-MIMO 데이터셋 생성기입니다.

```text
입력: argparse 옵션
출력: raw UE antenna-domain MU-MIMO OFDM .npz 데이터셋
사용 RX: rx_mumimo_receiver.py
```

## 한 줄 요약

TX는 multiuser MIMO channel을 만들고, 송신단이 알고 있다고 가정한 `H_tx_est`로 hybrid steering + ZF precoder를 만든 뒤, pilot/data OFDM waveform을 raw UE antenna-domain으로 저장합니다.

```text
bits -> QAM -> precoding -> OFDM -> multipath channel -> RF impairment -> AWGN -> .npz
```

## 기본 명령

smoke run:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_16qam_smoke --modulation 16QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 20 --n-val-frames 8 --n-test-frames-per-snr 8 --snr-test-db 20 40
```

권장 규모 run:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40 --modulation 64QAM --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
```

## 주요 옵션

```text
--out-dir                         출력 directory
--modulation                      16QAM 또는 64QAM
--n-users                         사용자 수, 기본 2
--n-tx                            BS 송신 안테나 수, 기본 8
--n-rx-per-ue                     사용자별 UE 수신 안테나 수, 기본 4
--n-fft                           OFDM FFT 크기, 기본 64
--n-cp                            cyclic prefix 길이, 기본 16
--n-taps                          SCM path/tap 수, 기본 7
--n-rays-per-path                 path당 ray 수, 기본 15
--pdp-decay                       tap power decay, 기본 5.0
--carrier-freq-hz                 carrier frequency, 기본 800e6
--antenna-spacing-lambda          array spacing in wavelength, 기본 0.5
--scm-angle-spread-deg            ray angle spread, 기본 3.0
--snr-train-db                    legacy single train/val SNR
--snr-train-db-list               mixed train/val SNR 목록, 기본 15 20 25 30 35 40
--snr-test-db                     test SNR sweep, 기본 0 5 10 15 20 25 30 35 40
--n-train-frames                  train frame 수, 기본 50000
--n-val-frames                    validation frame 수, 기본 10000
--n-test-frames-per-snr           test SNR별 frame 수, 기본 10000
--csit-error-var                  H_tx_est = H_true + E의 complex error variance, 기본 0.005
--precoder-norm                   none, column, fro 중 하나, 기본 column
--case                            linear, cp_removal, clipping 중 하나, 기본 clipping
--clip-ratio                      clipping threshold = clip_ratio * RMS, 기본 1.6
--pilot-kind                      ones 또는 qpsk, 기본 qpsk
--rx-iq-gain-imbalance-db         RX I/Q gain imbalance, 기본 0.5 dB
--rx-iq-phase-error-deg           RX I/Q quadrature phase error, 기본 3 deg
--rx-common-phase-error-deg       RX common phase rotation, 기본 5 deg
--seed                            random seed, 기본 7
```

## 데이터 생성 흐름

1. `MuMimoE2EConfig`가 실험 설정을 보관하고 검증합니다.
2. `ScmChannelGenerator`가 사용자별 time-domain MIMO multipath channel `h_time`을 만듭니다.
3. `channel_frequency_response()`가 `H_true[k,u,r,t]`를 계산합니다.
4. `add_csit_error()`가 `H_true`에 complex Gaussian error를 더해 `H_tx_est`를 만듭니다.
5. `hybrid_steering_beams()`가 strongest path angle로 analog TX/RX steering beam을 만듭니다.
6. `hybrid_zf_precoder_context()`가 `H_tx_est` 기준으로 subcarrier별 digital ZF precoder를 만듭니다.
7. random bit를 QAM symbol `x_d_freq`로 변환합니다.
8. `precoded_tx_frequency()`가 stream-domain symbol을 BS antenna-domain frequency signal로 바꿉니다.
9. `ofdm_modulate_freq()`가 IFFT, clipping, CP 삽입을 수행합니다.
10. `apply_multipath_mimo()`가 multipath MIMO channel을 시간 영역에서 convolution합니다.
11. `apply_rf_impairments()`가 RX I/Q imbalance와 common phase rotation을 적용합니다.
12. `add_awgn()`이 SNR에 맞춰 complex AWGN을 추가합니다.
13. train/val/test `.npz`와 `config.json`을 저장합니다.

## 채널 모델

채널은 `mumimo_phy/scm.py`의 SCM-style geometric clustered channel입니다.

```text
h_time shape = [n_users, n_taps, n_rx_per_ue, n_tx]
H_true shape = [n_fft, n_users, n_rx_per_ue, n_tx]
```

각 user는 독립적인 clustered multipath channel을 갖습니다. 각 tap/path에는 center angle 4개가 있습니다.

```text
tx_theta, tx_phi, rx_theta, rx_phi
```

각 path 주변에 `n_rays_per_path`개의 ray를 angle spread로 흩뿌리고, random phase와 exponential PDP를 곱해 MIMO tap matrix를 만듭니다.

```text
pdp[path] = exp(-(path+1) / pdp_decay)
pdp = pdp / sum(pdp)
```

strongest path는 `abs(h_user[:,0,0])`가 가장 큰 tap으로 고릅니다. 그 path의 angle이 analog steering beam 생성에 사용됩니다.

## Precoder 모델

TX는 실제 채널 `H_true`가 아니라 CSIT error가 들어간 `H_tx_est`를 보고 precoder를 만듭니다.

```text
H_tx_est = H_true + E
E = CN(0, csit_error_var)
```

analog beam:

```text
W_tx_analog[:, user] = steering vector from selected TX angle
W_rx_analog[user, :] = steering vector from selected RX angle
```

effective estimated channel:

```text
G_tx_est[k,user,:] = W_rx_analog[user,:].T @ H_tx_est[k,user,:,:]
H_eff_est[k] = G_tx_est[k] @ W_tx_analog
```

digital ZF:

```text
W_digital[k] = pinv(H_eff_est[k])
W_precoder[k] = normalize(W_tx_analog @ W_digital[k])
```

기본 normalization은 `column`입니다. 각 stream precoder column의 norm을 1로 맞춥니다. `fro`는 전체 Frobenius norm을 1로 맞추고, `none`은 normalization을 하지 않습니다.

실제 RX가 보는 pre-RF effective channel은 다음으로 저장됩니다.

```text
A_eff_true[k,user,rx_ant,stream] = sum_tx H_true[k,user,rx_ant,tx] * W_precoder[k,tx,stream]
```

## Pilot 설계

pilot은 stream별 orthogonal slot 구조입니다.

```text
x_p_freq[frame, pilot_slot, stream, subcarrier]
```

`pilot_slot == stream`인 위치에만 pilot symbol이 들어가고 나머지 stream은 0입니다. 그래서 RX는 pilot slot별로 각 stream의 channel column을 따로 추정할 수 있습니다.

`--pilot-kind ones`:

```text
모든 subcarrier pilot = 1 + 0j
```

`--pilot-kind qpsk`:

```text
subcarrier별 phase가 {1, j, -1, -j} 중 하나
```

기본값은 `qpsk`입니다. all-ones pilot은 OFDM 시간 영역에서 큰 peak를 만들 수 있고, clipping case에서 pilot 자체가 강하게 왜곡되어 LS channel estimation floor를 만들 수 있습니다. QPSK pilot은 데이터 변조를 바꾸는 것이 아니라 채널 추정용 known pilot의 PAPR 문제를 줄이기 위한 설정입니다.

## Nonlinear/RF/Noise 처리

`--case linear`:

```text
IFFT -> CP 삽입 -> channel
```

`--case clipping`:

```text
IFFT -> per-BS-antenna clipping -> CP 삽입 -> channel
threshold = clip_ratio * RMS(time_symbol)
```

`--case cp_removal`:

```text
IFFT -> CP 없이 저장, RX FFT도 sample 0부터 시작
```

RX RF impairment는 channel 통과 후, AWGN 추가 전에 적용됩니다.

```text
y_rf = alpha * y_channel + beta * conj(y_channel)
```

AWGN은 data clean waveform power 기준 SNR로 정해집니다. pilot에도 같은 noise power가 적용됩니다.

```text
noise_power = mean(|y_d_clean|^2) / 10^(snr_db/10)
```

## Train/Val/Test 정책

train/val은 `snr_train_db_list`에서 frame마다 SNR을 random choice합니다. 기본값은 mixed-SNR입니다.

```text
15 20 25 30 35 40
```

test는 paired base frame 정책을 씁니다. 같은 frame index의 `test_snr00.npz`, `test_snr05.npz`, ..., `test_snr40.npz`는 같은 channel, bits, precoder, clean waveform을 공유하고 AWGN scale만 다릅니다.

이 정책은 SNR별 BER curve를 비교할 때 channel draw 차이를 줄입니다.

## Output files

```text
config.json
train_snr15_40mixed.npz
val_snr15_40mixed.npz
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

## Dataset schema

주요 배열:

```text
rx_p_time      complex64 [n_frames, n_streams, n_users, n_rx_per_ue, time_len]
rx_d_time      complex64 [n_frames, n_users, n_rx_per_ue, time_len]
x_p_freq       complex64 [n_frames, n_streams, n_streams, n_fft]
x_d_freq       complex64 [n_frames, n_streams, n_fft]
bits           int8      [n_frames, n_streams, n_fft, bits_per_symbol]
H_true         complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_tx]
G_tx_est       complex64 [n_frames, n_fft, n_streams, n_tx]
W_precoder     complex64 [n_frames, n_fft, n_tx, n_streams]
W_digital      complex64 [n_frames, n_fft, n_streams, n_streams]
W_tx_analog    complex64 [n_frames, n_tx, n_streams]
W_rx_analog    complex64 [n_frames, n_users, n_rx_per_ue]
A_eff_true     complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
snr_db         float32   [n_frames]
signal_power   float32   [n_frames]
desired_power  float32   [n_frames, n_users]
inter_stream_power float32 [n_frames, n_users]
noise_power    float32   [n_frames]
effective_sinr_db float32 [n_frames, n_users]
cond_A         float32   [n_frames, n_fft, n_users]
mean_cond_A    float32   [n_frames]
p95_cond_A     float32   [n_frames]
```

`time_len`은 `case=cp_removal`이면 `n_fft`, 그 외에는 `n_fft + n_cp`입니다.

`rx_p_time`과 `rx_d_time`은 raw UE antenna-domain waveform입니다. 즉 기존 SISO ComNet용 scalar effective stream이 아니라, 사용자별/안테나별 수신 시간 파형입니다.

## Sanity checks

noiseless와 RF impairment off 조건에서는 pilot LS가 `A_eff_true`에 가까워야 합니다.

```text
FFT(rx_p_time[:, slot]) / x_p_freq[:, slot, slot] ~= A_eff_true[..., slot]
```

clean CSIT와 충분한 RX antenna 조건에서 inter-stream interference가 작아야 합니다.

```text
csit_error_var = 0
inter_stream_power << desired_power
```

CSIT error를 키우면 residual interference가 증가하고, high-SNR에서도 BER floor가 생길 수 있습니다.

```text
csit_error_var = 0.005 또는 더 큼
inter_stream_power 증가
effective_sinr_db 감소
```

## 함수 설명

### Config/CLI

`MuMimoE2EConfig`

실험 설정 dataclass입니다. FFT/CP, modulation, antenna 수, SCM parameter, SNR, CSIT error, clipping, RF impairment, seed를 보관합니다. `n_streams` property는 현재 `n_users`와 같게 둡니다.

`MuMimoE2EConfig.validate()`

설정값의 범위를 검사합니다. 예를 들어 `n_taps <= n_cp`, modulation choice, non-negative error variance, pilot 종류 등을 확인합니다.

`parse_args()`

command-line 옵션을 정의하고 argparse namespace를 반환합니다.

`build_config(args)`

argparse 결과를 `MuMimoE2EConfig`로 변환합니다. legacy `--snr-train-db`와 mixed `--snr-train-db-list` 우선순위를 처리합니다.

`main()`

CLI entry point입니다. argument를 읽고 config를 만든 뒤 `generate_all()`을 호출합니다.

### Channel/precoder

`add_csit_error(H_true, cfg, rng)`

`H_true`에 complex Gaussian CSIT error를 더해 `H_tx_est`를 만듭니다. `csit_error_var=0`이면 복사본을 그대로 반환합니다.

`build_scm_generator(cfg)`

TX/RX array 설정과 SCM parameter를 이용해 `ScmChannelGenerator`를 생성합니다.

`modulate_ofdm(freq_symbol, cfg)`

frequency-domain signal에 OFDM modulation을 적용합니다. 내부적으로 IFFT, clipping, CP 처리를 수행합니다.

`apply_rx_rf_impairments(waveform, cfg)`

RX I/Q gain imbalance, I/Q phase error, common phase rotation을 수신 waveform에 적용합니다.

`make_orthogonal_pilots(cfg, n_frames)`

stream별 orthogonal pilot slot을 만듭니다. `ones` 또는 `qpsk` pilot을 선택합니다.

### Dataset allocation/metrics

`_empty_split(cfg, n_frames)`

split 하나에 필요한 모든 output 배열을 shape에 맞춰 0으로 할당합니다. `x_p_freq` pilot도 여기서 생성됩니다.

`_condition_numbers(A_eff_true)`

subcarrier/user별 effective channel matrix의 condition number를 계산합니다.

`_effective_sinr_db(desired_power, inter_stream_power, noise_power)`

사용자별 effective SINR을 dB로 계산합니다.

### Frame generation

`_make_clean_frame(cfg, x_p_freq, bps, scm_generator, rng)`

frame 하나의 clean waveform과 metadata를 만듭니다. channel 생성, CSIT error 추가, analog/digital precoder 생성, data/pilot OFDM modulation, multipath channel, RF impairment, desired/interference power 계산이 모두 여기서 수행됩니다.

`_write_noisy_frame(data, frame_index, clean, y_d_time, y_p_time, snr_value, noise_power)`

clean frame metadata와 noisy waveform을 output 배열의 지정 frame index에 씁니다.

`_apply_scaled_awgn(clean, unit_noise, noise_power)`

paired test에서 같은 unit noise realization을 SNR별 scale만 바꿔 적용합니다.

`_make_split_dataset(cfg, n_frames, snr_db, rng)`

train 또는 validation split을 만듭니다. SNR 값이 여러 개이면 frame마다 random SNR을 선택합니다.

`_make_paired_test_datasets(cfg, n_frames, snr_db_values, rng)`

test SNR sweep 전체를 만듭니다. 같은 clean frame과 unit noise를 공유하고 SNR별 noise scale만 바꿉니다.

### Save/naming

`_save_npz(path, dataset)`

dataset dict를 compressed `.npz`로 저장하고 shape summary를 출력합니다.

`_snr_name(snr_db)`

SNR 값을 파일명용 문자열로 바꿉니다. 예: `0 -> 00`, `40 -> 40`.

`_snr_set_name(snr_db)`

train/val SNR set 이름을 만듭니다. mixed SNR이면 `15_40mixed` 같은 이름을 반환합니다.

`write_config(out_dir, cfg)`

`config.json`을 저장합니다. waveform type, channel model, pilot design, RF impairment coefficient, test SNR policy 같은 receiver용 metadata도 함께 기록합니다.

`generate_all(cfg, out_dir)`

전체 dataset 생성 orchestration 함수입니다. config 저장, train 생성, val 생성, paired test 생성, 파일 저장을 순서대로 수행합니다.
