# RX 학습/평가 상세 설명

이 문서는 `rx_mumimo_receiver.py`를 설명합니다. 이 receiver는 `tx_mumimo_e2e_dataset.py`가 만든 raw UE antenna-domain MU-MIMO OFDM 데이터셋을 읽고, classical baseline과 ComNet 계열 receiver를 학습/평가합니다.

```text
입력: outputs_mumimo_e2e_*/config.json, train/val/test .npz
출력: checkpoint, train history, BER/MSE summary, plot
```

## 한 줄 요약

RX는 pilot waveform에서 effective channel `A`를 추정하고, data waveform에서 각 사용자 stream의 QAM symbol 또는 bit를 복원합니다.

```text
rx_p_time -> FFT -> LS channel estimate -> LMMSE/ComNet CE refinement
rx_d_time -> FFT -> ZF/MMSE 또는 ComNet SD -> bits -> BER
```

## 기본 명령

smoke run:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_16qam_smoke --result-dir results_mumimo_e2e_16qam_smoke --mode train-all --sd-type both --ce-epochs 1 --sd-epochs 1 --bilstm-epochs 1 --device cuda
```

권장 run:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40_bilstm --mode train-all --sd-type bilstm --sd-feature-set reliability --ce-type resmlp --bilstm-epochs 300 --device cuda
```

이미 학습된 checkpoint로 평가만 다시 실행:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40_bilstm --mode eval --sd-type bilstm --device cuda
```

## 주요 옵션

```text
--dataset-dir             TX가 만든 dataset directory
--result-dir              checkpoint/result 저장 directory
--mode                    train-all, train-ce, train-sd, eval
--sd-type                 fc, bilstm, both
--sd-loss                 mse 또는 bce
--sd-feature-set          basic 또는 reliability
--ce-type                 linear 또는 resmlp
--ce-init                 identity 또는 lmmse
--ce-checkpoint           CE checkpoint path override
--fc-checkpoint           FC-SD checkpoint path override
--bilstm-checkpoint       BiLSTM-SD checkpoint path override
--lmmse-checkpoint        LMMSE estimator cache path override
--ce-epochs              CE epoch 수, 기본 50
--sd-epochs              FC-SD epoch 수, 기본 50
--bilstm-epochs          BiLSTM-SD epoch 수, 기본 300
--batch-size             batch size, 기본 512
--ce-lr                  CE learning rate, 기본 1e-3
--sd-lr                  SD learning rate, 기본 1e-3
--bilstm-lr              BiLSTM 전용 learning rate, 기본은 --sd-lr 사용
--ce-lr-step             CE StepLR step, 기본 25
--sd-lr-step             SD StepLR step, 기본 25
--bilstm-lr-step         BiLSTM StepLR step, 기본 100
--ce-lr-gamma            CE StepLR gamma, 기본 0.5
--sd-lr-gamma            SD/BiLSTM StepLR gamma, 기본 0.5
--group-size             subcarrier group size, 기본 8
--hidden-dim             FC-SD hidden dim, 기본 256
--ce-hidden-dim          ResMLP CE hidden dim, 기본 512
--ce-dropout             ResMLP CE dropout, 기본 0.05
--bilstm-hidden-dims     BiLSTM hidden dims 3개, 기본 64 32 16
--lmmse-ridge            empirical LMMSE ridge, 기본 1e-6
--eps                    numerical epsilon, 기본 1e-8
--device                 auto, cpu, cuda 등
--seed                   random seed, 기본 7
--log-every              log interval, 기본 10
```

## 입력 데이터 전처리

TX 데이터셋의 raw time-domain waveform은 RX에서 FFT로 frequency-domain으로 바뀝니다.

```text
rx_p_time -> y_p
rx_d_time -> y_d
```

shape는 전처리 후 다음처럼 정리됩니다.

```text
y_p      complex64 [n_frames, n_streams, n_users, n_rx_per_ue, n_fft]
y_d      complex64 [n_frames, n_fft, n_users, n_rx_per_ue]
a_ls     complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
a_true   complex64 [n_frames, n_fft, n_users, n_rx_per_ue, n_streams]
bits     int8      [n_frames, n_streams, n_fft, bits_per_symbol]
x_d_freq complex64 [n_frames, n_streams, n_fft]
```

pilot은 stream별 orthogonal slot 구조이므로 LS channel estimate는 slot별로 계산됩니다.

```text
a_ls[..., stream_id] = FFT(rx_p_time[:, stream_id]) / x_p_freq[:, stream_id, stream_id]
```

## Receiver 모델

per-subcarrier, per-user 관점에서 data model은 다음과 같습니다.

```text
y_u[k] = A_u[k] s[k] + n
```

여기서 `y_u[k]`는 사용자 `u`의 수신 안테나 벡터이고, `A_u[k]`는 `n_rx_per_ue x n_streams` effective channel matrix입니다. target bit는 user `u`의 stream `u`에 대응합니다.

## Channel estimation

**LS**

pilot에서 바로 얻는 기본 추정입니다.

```text
A_ls = y_p / x_p
```

**Empirical LMMSE**

train split의 `A_ls -> A_true` 관계를 real/imag vector regression으로 맞춥니다.

```text
weight = argmin ||X weight^T - Y||^2 + ridge
A_lmmse = weight(A_ls)
```

**ComNet CE**

`A_ls`를 입력으로 받아 `A_true`에 가까운 `A_comnet`을 출력합니다.

```text
A_ls -> MuMimoCERefineNet 또는 MuMimoCEResMLPNet -> A_comnet
```

`--ce-type linear`는 선형 layer 하나입니다. `--ce-type resmlp`는 LMMSE/identity base linear output에 residual MLP를 더합니다.

## Symbol/bit detection

**ZF**

```text
s_hat = (A^H A + eps I)^-1 A^H y
```

**MMSE**

```text
s_hat = (A^H A + sigma2 I)^-1 A^H y
```

MMSE는 QAM hard decision 전에 post-equalization gain으로 shrinkage를 보정합니다.

**RF-aware WL-MMSE**

RX RF impairment가 켜진 경우 I/Q imbalance 때문에 mirror-subcarrier leakage가 생깁니다. `RF-aware True-H WL-MMSE`는 `A_true`와 RF coefficient `alpha`, `beta`를 사용해 subcarrier와 mirror subcarrier를 함께 푸는 oracle baseline입니다.

```text
y_rf[k] = alpha * y[k] + beta * conj(y[-k])
```

**ComNet-FC**

subcarrier를 `group_size`개씩 묶고 feature를 FC network에 넣어 bit logits를 출력합니다.

```text
[group_size, feature_dim] -> FC -> [group_size * bits_per_symbol]
```

**ComNet-BiLSTM**

전체 subcarrier sequence를 BiLSTM 3층에 넣고, group 단위 bit logits를 출력합니다.

```text
[n_fft, feature_dim] -> BiLSTM -> [n_fft/group_size, group_size * bits_per_symbol]
```

## SD feature set

`--sd-feature-set basic`

FC는 ZF estimate의 real/imag만 사용합니다. BiLSTM basic은 ZF, MMSE, noise, SNR 정보를 사용합니다.

`--sd-feature-set reliability`

현재 기본값입니다. feature dimension은 11입니다.

```text
ZF real
ZF imag
MMSE real
MMSE imag
matched residual real
matched residual imag
log residual power
MMSE gain magnitude
log condition number
log noise power
normalized SNR
```

이 feature는 단순 symbol 위치뿐 아니라 detector reliability 정보를 같이 줍니다.

## 평가 baseline

`eval_summary.json`과 `ber_vs_snr.png`에 기록되는 BER 항목은 다음과 같습니다.

```text
LS-ZF                       A_ls + ZF
LS-MMSE                     A_ls + MMSE
LMMSE-ZF                    A_lmmse + ZF
LMMSE-MMSE                  A_lmmse + MMSE
ComNet-CE-ZF-Hard           A_comnet + ZF + hard QAM demodulation
ComNet-FC                   A_comnet + FC bit detector
ComNet-BiLSTM               A_comnet + BiLSTM bit detector
Pre-RF True-H ZF            pre-RF A_eff_true + ZF
Pre-RF True-H MMSE          pre-RF A_eff_true + MMSE
RF-aware True-H WL-MMSE     RF impairment를 반영한 widely-linear True-H MMSE oracle
Desired-only MRC            target stream channel만 쓰는 sanity baseline
```

`a_mse_vs_snr.png`에는 channel estimate 품질이 들어갑니다.

```text
LS          mean(|A_ls - A_true|^2)
LMMSE       mean(|A_lmmse - A_true|^2)
ComNet-CE   mean(|A_comnet - A_true|^2)
```

## 출력 파일

```text
mumimo_lmmse_estimator.npz       empirical LMMSE channel estimator
mumimo_ce_refinenet.pt           CE network checkpoint
mumimo_refinenet_fc.pt           FC-SD checkpoint
mumimo_refinenet_bilstm.pt       BiLSTM-SD checkpoint
train_history_ce.csv             CE training log
train_history_fc_sd.csv          FC-SD training log
train_history_bilstm_sd.csv      BiLSTM-SD training log
ce_training_curve.png            CE train/val loss curve
fc_sd_training_curve.png         FC-SD train/val/BER curve
bilstm_sd_training_curve.png     BiLSTM-SD train/val/BER curve
eval_summary.json                full evaluation summary
ber_vs_snr.png                   BER plot
a_mse_vs_snr.png                 channel MSE plot
```

matplotlib import가 실패하면 plot 대신 CSV가 저장됩니다.

## Mode별 동작

`--mode train-all`

LMMSE estimator를 만들거나 로드하고, CE를 학습한 뒤, 선택한 SD network를 학습하고, 마지막에 test SNR sweep을 평가합니다.

`--mode train-ce`

LMMSE estimator를 만들거나 로드하고, CE만 학습한 뒤 평가합니다. SD checkpoint는 새로 학습하지 않습니다.

`--mode train-sd`

기존 CE checkpoint를 로드하고, SD만 학습한 뒤 평가합니다.

`--mode eval`

기존 CE/SD checkpoint를 로드해 test set만 평가합니다. SD checkpoint가 없으면 해당 ComNet-FC/BiLSTM 항목은 skip됩니다.

## 함수 설명

### CLI/QAM helper

`parse_args()`

RX command-line 옵션을 정의합니다.

`bits_per_symbol(modulation)`

QPSK, 16QAM, 64QAM의 symbol당 bit 수를 반환합니다.

`_pam_levels(axis_bits)`, `_gray_labels(axis_bits)`, `_ints_to_bits(values, width)`, `_qam_normalization(modulation)`

QAM demodulation에 필요한 PAM level, Gray code label, bit 변환, normalization factor를 계산하는 내부 helper입니다.

`qam_demodulate(symbols, modulation)`

complex QAM symbol을 nearest constellation point로 hard decision하고 bit array로 변환합니다.

### Network class

`MuMimoCERefineNet`

CE용 linear network입니다. real/imag vectorized `A_ls`를 입력받아 같은 차원의 refined channel vector를 출력합니다.

`MuMimoCEResMLPNet`

base linear layer와 residual MLP를 더한 CE network입니다. 마지막 residual layer를 0으로 초기화해 처음에는 base estimator에 가깝게 시작합니다.

`build_ce_model(ce_type, input_dim, hidden_dim, dropout)`

`linear` 또는 `resmlp` CE model을 생성합니다.

`MuMimoFCSDNet`

subcarrier group 단위 FC symbol detector입니다. 입력은 `group_size * feature_dim`, 출력은 `group_size * bits_per_symbol` logits입니다.

`MuMimoBiLSTMSDNet`

subcarrier sequence용 3-layer bidirectional LSTM detector입니다. sequence feature를 받아 group 단위 bit logits를 출력합니다.

### Load/preprocess

`resolve_device(device_arg)`

`auto`이면 CUDA 가능 여부에 따라 `cuda` 또는 `cpu`를 고릅니다.

`load_config(dataset_dir)`

`config.json`을 읽고 modulation 문자열을 정규화합니다.

`find_one(dataset_dir, pattern)`

glob pattern에 맞는 파일이 정확히 하나인지 확인하고 path를 반환합니다.

`load_npz(path)`

`.npz` 파일을 dict 형태로 읽습니다.

`ofdm_demodulate(rx_time, cfg)`

CP 제거 후 FFT를 수행합니다. `case=cp_removal`이면 sample 0부터 FFT를 수행합니다.

`preprocess_split(raw, cfg, eps)`

raw `.npz`를 receiver가 쓰는 `y_p`, `y_d`, `a_ls`, `a_true`, `bits`, `noise_power`, `cond_A` dict로 변환합니다.

### Channel vector conversion/metrics

`ce_feature_dim(cfg)`

CE input/output vector dimension을 계산합니다.

`ce_complex_to_ri(a_eff)`

complex effective channel을 user별 real/imag concatenated vector로 변환합니다.

`ce_ri_to_complex(values, n_frames, n_fft, n_users, n_rx, n_streams)`

real/imag vector를 complex effective channel tensor로 복원합니다.

`ce_ri_to_complex_like(values, like)`

기준 tensor `like`의 shape를 사용해 `ce_ri_to_complex()`를 호출합니다.

`bit_error_rate(pred_bits, true_bits)`

bit mismatch 평균을 계산합니다.

`hard_demod_stream_grid(symbols, modulation)`

stream grid symbol을 QAM hard demodulation해 bit grid로 바꿉니다.

`channel_mse(a_hat, a_true)`, `channel_nmse(a_hat, a_true)`, `to_db(value)`

channel MSE, normalized MSE, dB 변환을 수행합니다.

### Linear/RF detector

`linear_detect(y_d, a_eff, noise_power, method, eps)`

ZF 또는 MMSE로 full stream estimate를 계산합니다. MMSE는 diagonal gain으로 output shrinkage를 보정합니다.

`linear_detect_with_gain(y_d, a_eff, noise_power, method, eps)`

`linear_detect()`와 유사하지만 detector response gain도 함께 반환합니다. SD reliability feature 생성에 사용됩니다.

`_complex_channel_real_matrix(a_eff)`, `_conjugated_output_real_matrix(base)`, `_apply_complex_scalar_to_real_rows(base, scalar)`, `_complex_grid_to_real(values)`, `_block_gain_compensate_real(estimates, response, n_symbols, eps)`, `_solve_real_mmse(b_matrix, y_real, noise_power, method, eps, n_symbols)`

widely-linear RF-aware detector를 real-valued block linear system으로 풀기 위한 내부 helper입니다.

`rf_iq_wl_detect(y_d, a_eff, noise_power, cfg, method, eps)`

I/Q imbalance와 common phase rotation을 반영한 widely-linear ZF/MMSE detector입니다. mirror subcarrier pair를 함께 풉니다.

`target_user_streams(full_stream_estimates)`

full stream estimate에서 user `u`의 target stream `u`만 꺼내 `[n_frames, n_users, n_fft]`로 정리합니다.

`desired_only_mrc(y_d, a_eff, eps)`

target stream channel만 사용해 matched combining을 수행합니다. 다른 stream 간섭 제거는 하지 않는 sanity baseline입니다.

`ber_for_user_grid(symbols, bits, modulation)`

user grid symbol을 hard demodulation하고 BER를 계산합니다.

`detector_ber(...)`

linear detector를 실행하고 target stream BER를 계산합니다.

`rf_iq_wl_detector_ber(...)`

RF-aware widely-linear detector를 실행하고 target stream BER를 계산합니다.

### Logging/save helper

`should_log(epoch, epochs, log_every)`

현재 epoch에서 log를 출력할지 결정합니다.

`write_history(path, rows, columns)`

training history CSV를 저장합니다.

`save_training_plot(path, rows, title, include_ber)`

training loss와 validation BER plot을 저장합니다. matplotlib이 없으면 skip합니다.

### LMMSE/CE training

`fit_lmmse_weight(train_data, ridge)`

train split에서 `A_ls -> A_true` empirical LMMSE weight를 fitting합니다. sample 수가 feature 수보다 적으면 안정성을 위해 identity를 반환합니다.

`save_lmmse_weight(path, weight_ri, cfg, ridge)`, `load_lmmse_weight(path)`

LMMSE estimator를 저장/로드합니다.

`get_lmmse_weight(dataset_dir, cfg, args, checkpoint_path)`

LMMSE checkpoint가 있으면 로드하고, 없으면 train split으로 fitting한 뒤 저장합니다.

`apply_lmmse_weight(a_ls, weight_ri)`

저장된 LMMSE weight를 `A_ls`에 적용해 `A_lmmse`를 만듭니다.

`train_ce(cfg, train_data, val_data, args, device, checkpoint_path, lmmse_weight)`

CE model을 학습합니다. `--ce-init lmmse`이면 LMMSE weight를 base layer 초기값으로 사용합니다.

`load_ce_model(path, cfg, device)`

CE checkpoint를 로드합니다. checkpoint metadata가 부족한 경우 state_dict shape로 model type을 추론합니다.

`predict_ce(model, a_ls, device, batch_size)`

CE model inference를 batch 단위로 수행해 `A_comnet`을 반환합니다.

### SD feature/training

`sd_loss_value(logits, target, sd_loss)`

SD loss를 계산합니다. `mse`는 sigmoid probability와 target bit의 MSE, `bce`는 BCE-with-logits입니다.

`validate_sd_feature_set(feature_set)`, `sd_feature_dim(feature_set, sd_kind)`, `infer_sd_feature_set(feature_dim, sd_kind, fallback)`

SD feature set 이름과 dimension을 검증/추론합니다.

`normalized_log_feature(values, floor, lo, hi, scale)`, `frame_feature(values, n_users, n_fft)`, `condition_feature(cond_a, n_frames, n_users, n_fft)`

noise power, residual power, condition number 같은 scalar/grid feature를 network 입력 범위로 정규화합니다.

`make_sd_features(y_d, a_hat, noise_power, snr_db, cond_a, feature_set, sd_kind, eps)`

ZF/MMSE estimate, residual, gain, condition, SNR/noise를 조합해 SD input feature tensor를 만듭니다.

`make_fc_sd_arrays(...)`

FC-SD 학습용으로 subcarrier group을 flatten합니다.

`make_bilstm_sd_arrays(...)`

BiLSTM-SD 학습용으로 전체 subcarrier sequence와 group label을 만듭니다.

`train_fc_sd(cfg, train_data, val_data, ce_model, args, device, checkpoint_path)`

CE output channel을 사용해 FC-SD를 학습하고 checkpoint/history/plot을 저장합니다.

`train_bilstm_sd(cfg, train_data, val_data, ce_model, args, device, checkpoint_path)`

CE output channel을 사용해 BiLSTM-SD를 학습하고 checkpoint/history/plot을 저장합니다.

### SD load/inference

`infer_fc_feature_dim(state_dict, group_size)`

FC-SD checkpoint의 첫 layer shape에서 feature dimension을 추론합니다.

`load_fc_sd_model(path, cfg, group_size, device, args_feature_set)`

FC-SD checkpoint를 로드합니다.

`infer_bilstm_hidden_dims(state_dict)`, `infer_bilstm_feature_dim(state_dict)`

BiLSTM checkpoint state_dict에서 hidden dimension과 input feature dimension을 추론합니다.

`load_bilstm_sd_model(path, cfg, group_size, device, args_feature_set)`

BiLSTM-SD checkpoint를 로드합니다.

`predict_fc_sd_bits(...)`

FC-SD model로 bit prediction을 수행하고 원래 bit grid shape로 복원합니다.

`predict_bilstm_sd_bits(...)`

BiLSTM-SD model로 bit prediction을 수행하고 원래 bit grid shape로 복원합니다.

### Evaluation/save

`format_ber(value)`

BER 값을 log 출력용 문자열로 포맷합니다.

`evaluate_one(path, cfg, ce_model, fc_model, bilstm_model, lmmse_weight, args, device)`

test file 하나를 평가합니다. LS/LMMSE/ComNet CE, ZF/MMSE, RF-aware WL-MMSE, optional FC/BiLSTM BER와 channel MSE/NMSE, condition summary를 계산합니다.

`save_eval_summary(result_dir, cfg, results)`

SNR별 평가 결과를 `eval_summary.json`으로 저장합니다.

`save_metric_csv(path, summary, section)`

plot 생성이 어려울 때 metric을 CSV로 저장합니다.

`ordered_metric_names(names, preferred)`

plot legend 순서를 고정하기 위한 helper입니다.

`save_eval_plots(result_dir, summary)`

`ber_vs_snr.png`와 `a_mse_vs_snr.png`를 저장합니다.

`evaluate_all(dataset_dir, cfg, ce_model, fc_model, bilstm_model, lmmse_weight, args, device)`

모든 `test_snr*.npz`를 순회 평가하고 summary/plot을 저장합니다.

`main()`

전체 RX entry point입니다. config와 checkpoint path를 정하고, mode에 따라 LMMSE/CE/SD 학습 또는 로드, 최종 평가를 실행합니다.
