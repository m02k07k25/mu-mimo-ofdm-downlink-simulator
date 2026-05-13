# MU-MIMO OFDM Downlink Simulator

이 프로젝트는 MU-MIMO OFDM downlink 통신을 시뮬레이션하고, 수신기에서 ZF/MMSE baseline과 ComNet 계열 딥러닝 수신기를 학습/평가하는 코드입니다.

처음 보는 사람 기준으로 말하면, 송신기 `TX`는 여러 사용자에게 동시에 OFDM 신호를 보내는 데이터셋을 만들고, 수신기 `RX`는 그 데이터셋을 읽어서 각 사용자의 원래 bit를 얼마나 잘 복원하는지 BER로 평가합니다.

## 현재 상태

- `tx_mumimo_e2e_dataset.py`는 raw UE antenna-domain MU-MIMO OFDM 데이터셋을 생성합니다.
- 채널은 `mumimo_phy/scm.py`의 SCM-style clustered multipath channel을 사용합니다.
- MATLAB `copy/SCM.m`, `copy/steer_precoding.m`, `copy/ZF_precoding.m`에서 쓰던 기본 아이디어를 Python 모듈로 분리했습니다.
- 기본 `csit_error_var`는 `0.005`입니다.
- 기본 pilot은 `qpsk`입니다. 데이터 변조는 그대로 `16QAM` 또는 `64QAM`입니다.
- `rx_mumimo_receiver.py`는 pilot으로 effective channel을 추정하고, ZF/MMSE/ComNet receiver를 학습 및 평가합니다.

## 폴더 구조

```text
tx_mumimo_e2e_dataset.py       raw MU-MIMO E2E 데이터셋 생성기
rx_mumimo_receiver.py          raw MU-MIMO receiver 학습/평가
mumimo_phy/                    채널, 빔포밍, OFDM, QAM, noise 공통 모듈
copy/                          원본 MATLAB 참고 코드
old_comnet/                    이전 SISO/effective-SISO ComNet 코드
TODO.md                        진행상황과 다음 작업
```

## 권장 환경

사용자는 Python 3.9 기반 `torch_mk` conda 환경을 만드는 것을 요청했습니다. RTX 4090에서는 PyTorch CUDA wheel을 쓰면 됩니다.

환경 생성:

```powershell
conda create -n torch_mk python=3.9 pip -y --solver=classic
conda activate torch_mk
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy matplotlib
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

검증에서 `torch.cuda.is_available()`가 `True`이고 GPU 이름이 `NVIDIA GeForce RTX 4090`이면 됩니다.

## 빠른 실행

TX 데이터셋 생성:

```powershell
conda activate torch_mk
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --modulation 64QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
```

RX 학습/평가:

```powershell
conda activate torch_mk
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip20_reliability --mode train-all --sd-type both --sd-feature-set reliability --ce-type resmlp --ce-hidden-dim 512 --ce-dropout 0.05 --sd-epochs 150 --bilstm-epochs 300 --device cuda
```

작게 동작 확인만 할 때:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_16qam_smoke --modulation 16QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 20 --n-val-frames 8 --n-test-frames-per-snr 8 --snr-test-db 20 40
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_16qam_smoke --result-dir results_mumimo_e2e_16qam_smoke --mode train-all --sd-type both --ce-epochs 1 --sd-epochs 1 --bilstm-epochs 1 --device cuda
```

## 전체 모델 흐름

### 1. TX 데이터 생성

```text
random bits
-> QAM symbols
-> SCM multipath MIMO channel 생성
-> strongest path angle 선택
-> analog TX/RX steering beam 설계
-> CSIT error가 들어간 H_tx_est 생성
-> digital ZF precoder 생성
-> OFDM IFFT, clipping, CP 삽입
-> multipath MIMO channel 통과
-> AWGN noise 추가
-> .npz dataset 저장
```

저장되는 주요 배열:

```text
rx_p_time      pilot 수신 파형
rx_d_time      data 수신 파형
x_p_freq       알려진 pilot symbol
x_d_freq       송신 data symbol
bits           정답 bit label
H_true         실제 채널
G_tx_est       송신기가 알고 있다고 가정한 effective channel
W_precoder     최종 precoder
A_eff_true     RX 평가용 실제 effective channel
noise_power    noise 분산
```

### 2. RX 학습/평가

```text
rx_p_time, x_p_freq
-> OFDM FFT
-> LS channel estimation
-> optional LMMSE channel estimation
-> ComNet CE channel refinement
-> ZF/MMSE linear detector
-> FC/BiLSTM symbol detector with optional residual/reliability features
-> hard demodulation
-> BER 계산
```

결과는 보통 `results_*` 폴더에 저장됩니다.

```text
ber_vs_snr.png
a_mse_vs_snr.png
eval_summary.json
train_history_*.csv
```

### 결과 그래프 선 의미

`ber_vs_snr.png`는 SNR별 BER, Bit Error Rate를 그립니다. 아래로 갈수록 좋은 결과입니다.

```text
LS-ZF                  pilot으로 구한 LS channel estimate + ZF linear detector
LS-MMSE                pilot으로 구한 LS channel estimate + MMSE linear detector
LMMSE-ZF               LS estimate를 empirical LMMSE로 보정한 channel + ZF detector
LMMSE-MMSE             LMMSE channel estimate + MMSE detector, 주요 classical baseline
ComNet-CE-ZF-Hard      ComNet CE가 보정한 channel + ZF detector + hard QAM decision
ComNet-FC              ComNet CE channel + FC symbol detector
ComNet-BiLSTM          ComNet CE channel + BiLSTM symbol detector
True-H ZF              실제 effective channel A_eff_true를 알고 있다고 가정한 ZF oracle baseline
True-H MMSE            실제 effective channel A_eff_true를 알고 있다고 가정한 MMSE oracle baseline
Desired-only MRC       target stream channel만 matched combining하고 stream 간섭 제거는 하지 않는 sanity baseline
```

`a_mse_vs_snr.png`는 channel estimate가 실제 effective channel `A_eff_true`와 얼마나 다른지 MSE를 dB로 그립니다. 낮을수록 channel 추정이 더 정확합니다.

```text
LS                     pilot에서 바로 계산한 channel estimate A_ls
LMMSE                  LS estimate를 empirical LMMSE로 선형 보정한 channel estimate
ComNet-CE              CE network가 보정한 channel estimate A_comnet
```

`eval_summary.json`에는 같은 값이 숫자로 저장됩니다. BER가 비슷할 때는 `bit_errors`와 `total_bits`를 같이 봐야 실제 error count 차이를 확인할 수 있습니다.

## 핵심 개념

### Pilot

Pilot은 수신기가 채널을 추정할 수 있도록 송신기가 일부러 보내는 알려진 신호입니다.

수신기는 pilot의 원래 값 `x_p_freq`를 알고 있습니다. 수신된 pilot `y_p`와 비교하면 채널을 대략 계산할 수 있습니다.

```text
y_p = channel * x_p + noise
channel ~= y_p / x_p
```

이 프로젝트에서 `--pilot-kind qpsk`는 pilot을 QPSK phase symbol로 보낸다는 뜻입니다. 데이터 변조를 QPSK로 바꾼다는 뜻이 아닙니다. 데이터는 `--modulation 64QAM`이면 계속 64QAM입니다.

`ones` pilot은 모든 subcarrier가 `1+0j`라서 OFDM 시간영역에서 peak가 커질 수 있습니다. clipping이 켜져 있으면 pilot 자체가 심하게 찌그러져 채널 추정이 나빠질 수 있습니다. 그래서 기본값은 `qpsk`입니다.

### CSIT Error

CSIT는 Channel State Information at Transmitter의 약자입니다. 송신기가 precoder를 만들기 위해 알고 있다고 가정하는 채널 정보입니다.

현실에서는 송신기가 실제 채널 `H_true`를 완벽하게 알 수 없습니다. 그래서 코드에서는 아래처럼 오차를 추가합니다.

```text
H_tx_est = H_true + E
E[|E|^2] = csit_error_var
```

`csit_error_var`가 클수록 송신기가 잘못된 채널을 보고 precoder를 만들게 됩니다. 그러면 ZF precoding이 완벽하게 간섭을 지우지 못하고, 사용자 간 간섭이 남습니다.

현재 기본값:

```text
csit_error_var = 0.005
```

의미:

```text
0.0    송신기가 채널을 완벽히 안다고 가정
0.005  MATLAB reference와 비슷한 작은 송신 채널 오차
더 큼   precoder mismatch 증가, BER floor 가능성 증가
```

### Channel Estimation Error

RX 쪽 channel estimation error는 수신기가 pilot으로 계산한 채널 `A_ls`가 실제 채널 `A_eff_true`와 다른 정도입니다.

주요 원인:

- noise 때문에 pilot이 흔들림
- clipping 때문에 pilot waveform이 찌그러짐
- CSIT error 때문에 data 구간에 residual interference가 생김
- multipath와 MIMO channel이 복잡해서 추정 문제가 어려워짐

평가는 보통 channel MSE 또는 NMSE로 봅니다.

```text
MSE = mean(|A_est - A_true|^2)
```

### Multipath

Multipath는 송신 신호가 한 경로로만 오는 것이 아니라, 반사/산란 때문에 여러 지연 경로로 도착하는 현상입니다.

간단히 말하면 같은 신호가 여러 복사본으로 늦게 도착합니다.

```text
received[t] = h0*x[t] + h1*x[t-1] + h2*x[t-2] + ...
```

이 프로젝트에서는 SCM-style channel이 여러 path와 path당 여러 ray를 만듭니다.

관련 옵션:

```text
--n-taps            path/tap 개수
--n-rays-per-path   path마다 ray 개수
--pdp-decay         뒤쪽 path 전력이 얼마나 빨리 줄어드는지
--scm-angle-spread-deg path 주변 ray angle spread
```

### Multipath Error

엄밀히 말하면 multipath 자체는 에러가 아니라 채널 특성입니다. 하지만 수신기 입장에서는 다음 문제를 만듭니다.

- 심볼이 시간축에서 퍼져서 서로 겹침
- subcarrier마다 채널 크기와 위상이 달라짐
- MIMO에서는 사용자와 안테나 사이의 channel matrix가 복잡해짐

OFDM은 이 문제를 줄이기 위해 CP, cyclic prefix를 붙입니다. `n_taps <= n_cp`이면 대부분의 multipath 지연을 CP가 흡수해서 FFT 이후에는 subcarrier별 곱셈 문제로 바꿀 수 있습니다.

### Clipping

OFDM 신호는 순간 peak가 클 수 있습니다. Clipping은 너무 큰 peak를 잘라내는 처리입니다.

```text
threshold = clip_ratio * RMS(time_symbol)
```

`clip_ratio`가 작을수록 더 많이 자릅니다.

```text
3.0  약한 clipping
2.0  중간 clipping
1.6  강한 clipping
```

Clipping은 PAPR을 줄이는 대신 신호를 왜곡합니다. 특히 pilot이 왜곡되면 channel estimation이 나빠질 수 있습니다.

### ZF와 MMSE

ZF, Zero Forcing은 다른 stream의 간섭을 0으로 만들려고 하는 검출/precoding 방식입니다. 채널이 나쁘거나 noise가 크면 noise까지 키울 수 있습니다.

MMSE는 간섭 제거와 noise 증폭 사이에서 균형을 잡습니다. 낮은 SNR에서는 보통 ZF보다 안정적입니다.

## 주요 옵션

```text
--modulation              16QAM 또는 64QAM
--n-users                 사용자 수, 기본 2
--n-tx                    BS 송신 안테나 수, 기본 8
--n-rx-per-ue             사용자별 수신 안테나 수, 기본 4
--n-taps                  SCM path/tap 수, 기본 7
--n-rays-per-path         path당 ray 수, 기본 15
--csit-error-var          송신 채널 추정 오차 분산, 기본 0.005
--case                    linear, cp_removal, clipping
--clip-ratio              clipping threshold 계수
--pilot-kind              ones 또는 qpsk
--snr-train-db            train/val SNR
--snr-test-db             test SNR sweep
--ce-type                 CE 모델 선택, 기본 resmlp
--ce-hidden-dim           Residual MLP CE hidden size, 기본 512
--ce-dropout              Residual MLP CE dropout, 기본 0.05
--sd-feature-set          basic 또는 reliability, 기본 reliability
--bilstm-hidden-dims      BiLSTM hidden size 3개, 기본 64 32 16
--bilstm-lr-step          BiLSTM 전용 LR decay step, 기본 100
```

## 함수 설명

간략한 역할만 정리하면 다음과 같습니다.

```text
tx_mumimo_e2e_dataset.py
  build_scm_generator       SCM channel generator 설정
  make_orthogonal_pilots    stream별 orthogonal pilot 생성
  _make_split_dataset       train/val/test split 하나 생성
  generate_all              전체 dataset 생성

mumimo_phy/scm.py
  ScmChannelGenerator       clustered multipath MIMO channel 생성
  channel_frequency_response time-domain tap을 frequency-domain H로 변환
  apply_multipath_mimo      time-domain waveform에 multipath MIMO channel 적용

mumimo_phy/beamforming.py
  steering_precoder         MATLAB식 steering vector 생성
  zf_precoder               pseudo-inverse 기반 ZF precoder 생성
  hybrid_zf_precoder_context analog steering + digital ZF 결합

mumimo_phy/ofdm.py
  ofdm_modulate_freq        IFFT, clipping, CP 처리

rx_mumimo_receiver.py
  preprocess_split          pilot/data를 FFT하고 LS channel 추정
  linear_detect             ZF/MMSE 검출
  MuMimoCEResMLPNet         LS/LMMSE channel estimate에 residual MLP 보정
  MuMimoFCSDNet             FC 기반 symbol detector
  MuMimoBiLSTMSDNet         BiLSTM 기반 symbol detector
  make_sd_features          ZF/MMSE/residual/reliability SD 입력 feature 생성
```

## 현재 권장 실험

64QAM, SCM channel, CSIT error 0.005, clipping ratio 2.0:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --modulation 64QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip20_reliability --mode train-all --sd-type both --sd-feature-set reliability --ce-type resmlp --ce-hidden-dim 512 --ce-dropout 0.05 --sd-epochs 150 --bilstm-epochs 300 --device cuda
```

## 주의

- `outputs_*`, `results_*`는 용량이 커질 수 있습니다.
- `csit_error_var`는 TX precoder 설계 오차입니다. RX pilot 추정 오차와 같은 개념이 아닙니다.
- `pilot_kind=qpsk`는 pilot만 QPSK phase를 쓴다는 뜻입니다.
- BER이 0으로 찍히는 경우는 실제로 완벽하다는 뜻이 아니라, 해당 test bit 수 안에서 error가 관측되지 않았다는 뜻입니다.
