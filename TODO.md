# TODO: Raw MU-MIMO E2E ComNet

작성 기준: 2026-05-13

## 완료된 작업

- `tx_mumimo_e2e_dataset.py`로 raw UE antenna-domain MU-MIMO OFDM 데이터셋을 생성한다.
- `rx_mumimo_receiver.py`로 raw MU-MIMO ComNet 수신기를 학습/평가한다.
- SCM-style clustered multipath channel을 `mumimo_phy/scm.py`로 분리했다.
- MATLAB reference의 기본 물리계층 함수를 Python 모듈로 정리했다.
  - `copy/SCM.m` -> `ScmChannelGenerator`
  - `copy/steer_precoding.m` -> `steering_precoder`
  - `copy/ZF_precoding.m` -> `zf_precoder`
  - `copy/awgn_noise.m` -> `add_awgn`
- analog TX/RX steering beam과 per-subcarrier digital ZF precoder를 적용했다.
- BiLSTM-SD 기본 hidden size를 `20 10 6`에서 `64 32 16`으로 키우고, BiLSTM 전용 LR step 기본값을 `100`, epoch 기본값을 `300`으로 조정했다.
- CE 기본 모델을 기존 linear 보정에서 `base + residual MLP` 구조의 `resmlp`로 바꿨다. 기본값은 `--ce-hidden-dim 512 --ce-dropout 0.05`이다.
- SDNet 입력에 `--sd-feature-set reliability`를 추가해 ZF/MMSE 추정값, full-stream residual, gain, cond(A), noise/SNR feature를 함께 사용한다.
- `csit_error_var` 기본값을 `0.005`로 변경했다.
- clipping 조건에서 all-ones pilot이 channel estimation floor를 만들 수 있음을 확인하고, `--pilot-kind qpsk`를 기본 pilot 방향으로 정리했다.
- `mumimo_phy/` 패키지를 추가해 OFDM, QAM, noise, SCM, beamforming을 모듈화했다.
- 명령어는 PowerShell backtick 줄연결 없이 한 줄로 제공한다는 규칙을 `AGENTS.md`에 추가했다.
- 루트 `README.md`를 추가해 환경, 모델 흐름, 주요 개념, TX/RX 명령어를 정리했다.

## 현재 기준 설정

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
channel_model = SCM-style clustered multipath
csit_error_var = 0.005
case = clipping
clip_ratio = 2.0 또는 3.0
pilot_kind = qpsk
```

현재 권장 명령어:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --modulation 64QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 5000 --n-val-frames 1000 --n-test-frames-per-snr 1000
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_scm_csit005_clip20 --result-dir results_mumimo_e2e_64qam_scm_csit005_clip20_reliability --mode train-all --sd-type both --sd-feature-set reliability --ce-type resmlp --ce-hidden-dim 512 --ce-dropout 0.05 --sd-epochs 150 --bilstm-epochs 300 --device cuda
```

## 다음 작업

### 1. 환경 정리

- 사용자가 `torch_mk` conda 환경을 직접 생성한다.
- Python 3.9 + PyTorch CUDA 환경에서 다음을 확인한다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

- `torch.cuda.is_available() == True` 확인 후 smoke run을 수행한다.

### 2. SCM 구현 검증

- MATLAB `copy/SCM.m` 결과와 Python `ScmChannelGenerator`의 통계적 특성을 비교한다.
- 확인할 항목:
  - path power decay
  - strongest path angle 선택
  - steering vector 크기와 위상
  - frequency response shape
  - raw received waveform shape

### 3. Clip Ratio Ablation

다음 조합을 비교한다.

```text
case=linear
case=clipping, clip_ratio=3.0, pilot_kind=qpsk
case=clipping, clip_ratio=2.0, pilot_kind=qpsk
case=clipping, clip_ratio=1.6, pilot_kind=qpsk
case=clipping, clip_ratio=2.0, pilot_kind=ones
```

비교 지표:

```text
BER vs SNR
channel MSE vs SNR
effective SINR
error count / total bit count
```

### 4. CSIT Error Ablation

`csit_error_var`별 성능을 비교한다.

```text
0.0
0.001
0.005
0.01
0.02
```

목적:

- 송신 precoder mismatch가 BER floor에 미치는 영향 확인
- residual multi-user interference 증가 확인
- `True-H` baseline과 LS/LMMSE/ComNet 차이 확인

### 5. Mixed-SNR 학습

현재 train split은 기본적으로 `snr_train_db=40` 단일 SNR 중심이다.

추가할 기능:

- train SNR list 옵션 추가
- frame마다 다른 SNR로 train split 생성
- 예시 옵션:

```text
--snr-train-db-list 0 5 10 15 20 25 30 35 40
```

기대 효과:

- BiLSTM이 40 dB에 과하게 맞춰지는 문제 완화
- 10~30 dB 구간 일반화 개선

### 6. RX 결과 기록 개선

- `eval_summary.json`에 error count와 total bit count를 저장한다.
- BER이 0일 때도 실제 관측 error 수를 구분한다.
- 결과 plot에 설정값을 함께 기록한다.

### 7. 문서 보강

- `README_mumimo_e2e_receiver.md`의 깨진 한글을 UTF-8로 정리한다.
- effective-SISO bridge와 raw MU-MIMO E2E 경로 차이를 그림 또는 표로 정리한다.
- `mumimo_phy/README.md`에 MATLAB reference와 Python 구현 차이를 더 명확히 적는다.

## 최소 검증 명령어

문법 검사:

```powershell
python -m py_compile tx_mumimo_e2e_dataset.py rx_mumimo_receiver.py mumimo_phy\__init__.py mumimo_phy\beamforming.py mumimo_phy\modulation.py mumimo_phy\noise.py mumimo_phy\ofdm.py mumimo_phy\scm.py
```

TX smoke:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_16qam_smoke --modulation 16QAM --case clipping --clip-ratio 2.0 --pilot-kind qpsk --n-train-frames 20 --n-val-frames 8 --n-test-frames-per-snr 8 --snr-test-db 20 40
```

RX smoke:

```powershell
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_16qam_smoke --result-dir results_mumimo_e2e_16qam_smoke --mode train-all --sd-type both --ce-epochs 1 --sd-epochs 1 --bilstm-epochs 1 --device cuda
```
