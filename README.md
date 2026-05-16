# MU-MIMO OFDM Downlink Simulator

이 프로젝트는 downlink MU-MIMO OFDM 송수신 과정을 시뮬레이션하고, classical receiver와 ComNet 계열 receiver를 비교하는 실험 코드입니다. 현재 주 경로는 raw UE antenna-domain waveform을 저장하는 end-to-end MU-MIMO 경로입니다.

```text
TX: tx_mumimo_e2e_dataset.py
RX: rx_mumimo_receiver.py
PHY: mumimo_phy/
```

처음 보는 기준으로 말하면, TX는 여러 사용자에게 동시에 OFDM 신호를 보내는 기지국 역할을 합니다. TX가 채널, precoder, pilot, data, noise, RF impairment가 포함된 `.npz` 데이터셋을 만들고, RX는 그 데이터셋에서 pilot을 이용해 채널을 추정한 뒤 각 사용자의 bit를 복원합니다. 성능은 BER, effective channel MSE/NMSE, condition number로 확인합니다.

## 현재 기준 설정

기본 설정은 다음과 같습니다.

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
channel_model = SCM-style clustered multipath
csit_error_var = 0.005
case = clipping
clip_ratio = 1.6
pilot_kind = qpsk
train_snr_db_list = 15 20 25 30 35 40
test_snr_db = 0 5 10 15 20 25 30 35 40
rx_iq_gain_imbalance_db = 0.5
rx_iq_phase_error_deg = 3.0
rx_common_phase_error_deg = 5.0
```

## 전체 흐름

```text
random bits
-> 16QAM/64QAM symbol mapping
-> SCM-style multiuser multipath MIMO channel 생성
-> strongest path angle 기반 analog TX/RX steering beam 생성
-> H_true에 CSIT error를 더해 H_tx_est 생성
-> H_tx_est 기준 digital ZF precoder 생성
-> stream-domain pilot/data를 BS antenna-domain frequency grid로 precoding
-> OFDM IFFT, clipping, cyclic prefix 삽입
-> multipath MIMO channel 통과
-> receiver RF impairment 적용
-> AWGN 추가
-> train/val/test .npz 저장
-> RX에서 CP 제거, FFT, LS/LMMSE/ComNet CE 채널 추정
-> ZF/MMSE/ComNet-FC/ComNet-BiLSTM 검출
-> BER와 channel MSE 저장
```

## 채널과 에러 모델

**SCM-style clustered multipath channel**

`mumimo_phy/scm.py`의 `ScmChannelGenerator`가 사용자별로 여러 path와 path별 여러 ray를 생성합니다. 각 path는 송신/수신 angle, angle spread, random phase, exponential PDP decay를 갖습니다. 시간 영역 채널은 다음 형태입니다.

```text
h_time[user, tap, rx_ant, tx_ant]
```

주파수 영역 채널은 FFT를 통해 다음 형태로 저장됩니다.

```text
H_true[subcarrier, user, rx_ant, tx_ant]
```

**Multipath**

multipath는 신호가 여러 지연 tap을 통해 수신되는 현상입니다.

```text
y[t] = h0*x[t] + h1*x[t-1] + h2*x[t-2] + ...
```

`n_taps <= n_cp`이면 normal OFDM 조건에서 CP가 multipath 지연을 흡수할 수 있습니다. `cp_removal` case는 CP를 생략해 CP 제거/ISI 조건을 실험하기 위한 옵션입니다.

**CSIT error**

CSIT는 transmitter가 precoder를 만들 때 알고 있다고 가정하는 채널 정보입니다. 실제로는 송신기가 완벽한 채널을 알 수 없으므로 TX에서는 다음과 같이 오차를 넣습니다.

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

기본값 `csit_error_var=0.005`는 precoder mismatch를 만들고, ZF가 사용자 간 간섭을 완전히 제거하지 못하게 합니다. 이 residual inter-stream interference는 high-SNR BER floor의 원인이 될 수 있습니다.

**Channel estimation error**

RX는 pilot으로 effective channel `A`를 추정합니다. LS 추정은 개념적으로 다음과 같습니다.

```text
A_ls ~= FFT(rx_p_time) / x_p_freq
```

오차 원인은 AWGN, clipping으로 인한 pilot 왜곡, RF impairment, finite pilot 구조, CSIT mismatch가 만든 residual interference입니다. RX 결과에서는 `A_eff_true`와 `A_ls`, `A_lmmse`, `A_comnet`의 MSE/NMSE를 비교합니다.

**RF impairment**

RX front-end에는 I/Q imbalance와 common phase rotation이 기본으로 들어갑니다.

```text
rx_iq_gain_imbalance_db = 0.5
rx_iq_phase_error_deg = 3.0
rx_common_phase_error_deg = 5.0
```

I/Q imbalance는 widely-linear 모델로 볼 수 있습니다.

```text
y_rf = alpha * y_channel + beta * conj(y_channel)
```

그래서 `A_eff_true`는 RF 적용 전의 linear effective channel이고, RF impairment가 켜져 있으면 plain True-H ZF/MMSE도 완전 oracle은 아닙니다. 이 조건을 반영하기 위해 RX에는 `RF-aware True-H WL-MMSE` baseline이 있습니다.

**AWGN**

noise variance는 clean data waveform power와 SNR로 계산됩니다.

```text
noise_power = mean(|y_clean|^2) / 10^(snr_db/10)
noise = sqrt(noise_power/2) * (n_real + j*n_im)
```

test split은 paired base frame 정책을 사용합니다. 즉 `test_snr*.npz`는 같은 bits/channel/precoder/clean waveform을 공유하고, SNR별 AWGN scale만 달라집니다. SNR sweep 비교가 더 안정적입니다.

## 파일 구조

```text
tx_mumimo_e2e_dataset.py              raw MU-MIMO E2E 데이터셋 생성기
rx_mumimo_receiver.py                 raw MU-MIMO receiver 학습/평가
mumimo_phy/                           OFDM, QAM, noise, SCM, beamforming, RF impairment 공통 모듈
copy/                                 MATLAB reference 코드
old_comnet/                           이전 SISO/effective-SISO ComNet 코드
README_tx_mumimo_comnet_dataset.md    TX 데이터셋 생성 상세 설명
README_mumimo_e2e_receiver.md         RX 학습/평가 상세 설명
TODO.md                               진행 상황과 실험 TODO
```

## 권장 환경

프로젝트 터미널의 기본 Python은 다음 환경으로 기록되어 있습니다.

```text
C:\Python313\python.exe
Python 3.13.3
numpy 2.3.3
torch 2.8.0+cu128
torch.cuda.is_available() == True
```

확인 명령:

```powershell
python -c "import sys, numpy, torch; print(sys.version); print(numpy.__version__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 빠른 실행

smoke dataset 생성:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_16qam_smoke --modulation 16QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 20 --n-val-frames 8 --n-test-frames-per-snr 8 --snr-test-db 20 40
```

smoke receiver 학습/평가:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_16qam_smoke --result-dir results_mumimo_e2e_16qam_smoke --mode train-all --sd-type both --ce-epochs 1 --sd-epochs 1 --bilstm-epochs 1 --device cuda
```

권장 규모 dataset 생성:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40 --modulation 64QAM --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
```

권장 receiver 학습/평가:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip16_mixed15_40_bilstm --mode train-all --sd-type bilstm --sd-feature-set reliability --ce-type resmlp --bilstm-epochs 300 --device cuda
```

## 주요 출력

TX output directory:

```text
config.json
train_snr15_40mixed.npz
val_snr15_40mixed.npz
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

RX result directory:

```text
mumimo_lmmse_estimator.npz
mumimo_ce_refinenet.pt
mumimo_refinenet_fc.pt
mumimo_refinenet_bilstm.pt
train_history_ce.csv
train_history_fc_sd.csv
train_history_bilstm_sd.csv
ce_training_curve.png
fc_sd_training_curve.png
bilstm_sd_training_curve.png
eval_summary.json
ber_vs_snr.png
a_mse_vs_snr.png
```

## 결과 해석

`ber_vs_snr.png`는 SNR별 bit error rate입니다. 아래로 갈수록 좋습니다.

```text
LS-ZF                       LS channel estimate + ZF detector
LS-MMSE                     LS channel estimate + MMSE detector
LMMSE-ZF                    empirical LMMSE channel estimate + ZF detector
LMMSE-MMSE                  empirical LMMSE channel estimate + MMSE detector
ComNet-CE-ZF-Hard           ComNet CE channel estimate + ZF + hard QAM decision
ComNet-FC                   ComNet CE + FC symbol detector
ComNet-BiLSTM               ComNet CE + BiLSTM symbol detector
Pre-RF True-H ZF            pre-RF A_eff_true를 쓰는 ZF baseline
Pre-RF True-H MMSE          pre-RF A_eff_true를 쓰는 MMSE baseline
RF-aware True-H WL-MMSE     RF I/Q mirror leakage까지 반영한 widely-linear oracle baseline
Desired-only MRC            target stream만 matched combining하는 sanity baseline
```

`a_mse_vs_snr.png`는 effective channel estimate가 `A_eff_true`와 얼마나 다른지 dB로 보여줍니다. 낮을수록 좋습니다.

```text
LS          pilot에서 바로 계산한 A_ls
LMMSE       train split에서 맞춘 empirical LMMSE 보정
ComNet-CE   CE network가 보정한 A_comnet
```

`eval_summary.json`에는 BER뿐 아니라 `bit_errors`와 `total_bits`도 저장됩니다. BER가 0으로 보이면 실제로 error count가 0인지, 아니면 test bit 수가 부족해서 관측되지 않은 것인지 같이 확인해야 합니다.

## 세부 문서

TX 데이터셋 생성의 배열 shape, 채널/precoder/noise 저장 방식, 함수 설명은 `README_tx_mumimo_comnet_dataset.md`를 보세요.

RX 학습/평가 흐름, CE/SD network 구조, baseline 의미, 함수 설명은 `README_mumimo_e2e_receiver.md`를 보세요.
