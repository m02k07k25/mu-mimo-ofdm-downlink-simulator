# tx_comnet_ofdm_dataset.py

`tx_comnet_ofdm_dataset.py`는 ComNet 방식 SISO OFDM receiver 실험을 위한 데이터셋 생성기다. 이 파일은 송신기와 채널만 담당한다. LS, ZF, LMMSE, 딥러닝 모델 학습은 하지 않는다.

ComNet 논문은 OFDM receiver를 channel estimation subnet과 signal detection subnet으로 나누는 model-driven DL 구조를 제안한다. 이 파일은 그 구조에서 receiver가 볼 수 있는 입력인 pilot/data 수신 파형과 학습 label을 만든다.

논문 참고: https://arxiv.org/abs/1810.09082

## 실행

논문형 비교용 64QAM clipping 데이터셋:

```powershell
python tx_comnet_ofdm_dataset.py
```

16QAM 디버깅 데이터셋:

```powershell
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet_16qam --modulation 16QAM
```

case를 명시해서 따로 만들 수도 있다.

```powershell
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet_64qam_linear --case linear
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet_64qam_cp --case cp_removal
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet_64qam_clipping --case clipping --clip-ratio 1.6
```

빠른 테스트용:

```powershell
python tx_comnet_ofdm_dataset.py `
  --out-dir outputs_comnet_smoke `
  --n-train-frames 200 `
  --n-val-frames 60 `
  --n-test-frames-per-snr 60 `
  --snr-test-db 0 20 40
```

## 기본 설정

```text
N_FFT = 64
N_CP = 16
frame = 1 pilot OFDM symbol + 1 data OFDM symbol
subcarriers = 64개 전체 사용
modulation = 64QAM
channel = multipath Rayleigh + AWGN
train SNR = 40 dB
test SNR = 0, 5, 10, 15, 20, 25, 30, 35, 40 dB
case = clipping by default, or linear | cp_removal | clipping
```

`linear`는 LMMSE가 거의 최적으로 동작하는 sanity check 조건이다. 논문처럼 ComNet detector의 의미를 보려면 기본값인 `clipping` 또는 `cp_removal` 같은 nonlinear case를 우선 본다.

`QPSK`와 `16QAM`은 debug option으로 남겨둔다. 논문형 비교 그래프는 `64QAM`을 기본으로 생성한다.

## 신호 흐름

```text
random bits
-> QAM modulation
-> pilot frequency grid xP 생성
-> data frequency grid xD 생성
-> IFFT
-> case별 nonlinear 처리
-> cyclic prefix 삽입 또는 생략
-> frame별 random multipath Rayleigh channel 통과
-> AWGN 추가
-> rx_p_time, rx_d_time 저장
-> h_true, bits 저장
```

중요한 점은 이 파일이 receiver 처리를 하지 않는다는 것이다.

```text
하지 않는 것:
  LS channel estimation
  LMMSE channel estimation
  ZF/MMSE equalization
  hard demod baseline
  ComNet 학습
```

이 분리를 유지해야 receiver 쪽에서 `xP`, `yP`, `yD`만 사용한다는 조건을 검증할 수 있다.

## 출력 파일

예를 들어 기본값인 `--out-dir outputs_comnet_64qam_clipping`이면 다음 파일이 만들어진다.

```text
outputs_comnet_64qam_clipping/
  config.json
  train_snr40.npz
  val_snr40.npz
  test_snr00.npz
  test_snr05.npz
  ...
  test_snr40.npz
```

각 `.npz` 파일 필드:

```text
rx_p_time: complex64, [N, 80] for linear/clipping, [N, 64] for cp_removal
rx_d_time: complex64, [N, 80] for linear/clipping, [N, 64] for cp_removal
x_p_freq:  complex64, [N, 64]
x_d_freq:  complex64, [N, 64]
h_true:    complex64, [N, 64]
bits:      int8,      [N, 64, bits_per_symbol]
snr_db:    float32,   [N]
```

`rx_p_time`, `rx_d_time`은 time-domain 수신 파형이다. `linear`와 `clipping`은 CP가 붙어 있고, `cp_removal`은 CP 없이 저장된다. receiver 파일이 `config.json`의 `case`를 보고 CP 제거 또는 첫 64샘플 FFT를 수행한다.

`h_true`, `x_d_freq`, `bits`는 학습 label 또는 평가용이다. 실제 ComNet inference path에서 입력으로 쓰면 안 된다.

## 데이터셋 의미

현재 데이터셋은 frame 단위로 만들어진다.

```text
1 frame =
  1 pilot OFDM symbol
  1 data OFDM symbol
  1 random channel realization
  64 subcarrier symbols
```

train/val/test가 `.npz` 파일 단위로 분리되므로 기존 JSONL처럼 subcarrier 단위 random split leakage가 생기지 않는다.

## 논문처럼 더 맞추려면 추가할 것

현재 구현은 ComNet 구조 검증용이다. 논문 조건에 더 가깝게 가려면 아래 항목을 추가해야 한다.

```text
1. 채널 모델
   현재: simple multipath Rayleigh
   추가: WINNER II C1 NLOS 2.6 GHz 또는 그에 준하는 channel model

2. modulation
   현재: QPSK/16QAM/64QAM 지원
   논문형 실험: 64QAM 중심

3. 실험 반복
   여러 seed, 충분한 test frame, 평균 BER curve 저장

4. FC-DNN baseline용 dataset
   논문은 model-driven ComNet과 data-driven FC-DNN을 비교한다.
   FC-DNN baseline을 추가하려면 receiver 입력/출력 정의를 별도로 둬야 한다.

5. MU-MIMO 확장
   현재: SISO OFDM frame만 생성
   추가: 다중 송신/수신 안테나 차원, 사용자별 pilot/resource mapping,
   per-subcarrier MIMO channel tensor, stream/user별 bit label 저장
   receiver 쪽 detector와 맞물리도록 dataset schema를 새로 정의해야 한다.
```

## 권장 실행 순서

빠르게 pipeline을 확인할 때만 16QAM을 쓴다.

```powershell
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet_16qam --modulation 16QAM --case linear
python rx_comnet_receiver.py --dataset-dir outputs_comnet_16qam --result-dir results_comnet_16qam --mode train-all --device cuda
```

논문형 비교는 64QAM clipping case로 한다.

```powershell
python tx_comnet_ofdm_dataset.py
python rx_comnet_receiver.py --mode train-all --device cuda
```
