# rx_comnet_receiver.py

`rx_comnet_receiver.py`는 `tx_comnet_ofdm_dataset.py`가 만든 `.npz` 데이터셋을 읽어서 기존 OFDM receiver baseline과 ComNet 기반 딥러닝 receiver를 학습/평가한다.

ComNet 논문은 OFDM receiver를 channel estimation subnet과 signal detection subnet으로 나누고, 기존 통신 알고리즘의 중간 결과를 딥러닝 subnet이 보정하는 구조를 사용한다. 현재 구현도 이 흐름을 따른다.

논문 참고: https://arxiv.org/abs/1810.09082

## 실행

전체 학습과 평가:

```powershell
python rx_comnet_receiver.py `
  --mode train-all `
  --sd-type both `
  --device cuda
```

FC-SD만 학습:

```powershell
python rx_comnet_receiver.py `
  --result-dir results_comnet_64qam_clipping_fc `
  --mode train-all `
  --sd-type fc `
  --device cuda
```

BiLSTM-SD만 학습:

```powershell
python rx_comnet_receiver.py `
  --result-dir results_comnet_64qam_clipping_bilstm `
  --mode train-all `
  --sd-type bilstm `
  --device cuda
```

평가만 다시 실행:

```powershell
python rx_comnet_receiver.py `
  --mode eval `
  --sd-type both `
  --device cuda
```

## 전체 수신 흐름

```text
dataset load
-> rx_p_time, rx_d_time
-> CP 제거
-> FFT
-> yP, yD
-> LS channel estimation: h_ls = yP / xP
-> LMMSE channel estimation baseline
-> CE subnet: h_comnet = LSRefineNet(h_ls)
-> ZF/MMSE equalization
-> hard demod or SD subnet
-> BER / channel MSE 평가
```

실제 ComNet inference path에서 쓰는 값:

```text
xP
yP
yD
```

label 또는 평가용으로만 쓰는 값:

```text
h_true
x_d_freq
bits
```

## 비교 항목

BER 그래프와 `eval_summary.json`에는 다음 항목이 들어간다.

```text
LS-ZF
LMMSE-ZF
LMMSE-MMSE
ComNet-CE-Hard
ComNet-FC
ComNet-BiLSTM
True-H ZF-Hard
No-Clip True-H Ref
```

각 항목 의미:

```text
LS-ZF:
  LS 채널 추정 후 ZF equalization + hard QAM decision.

LMMSE-ZF:
  LMMSE로 보정한 채널을 사용하고, equalization은 ZF 사용.

LMMSE-MMSE:
  LMMSE 채널 추정 + one-tap MMSE equalization.

ComNet-CE-Hard:
  LSRefineNet으로 채널 추정값을 보정한 뒤 ZF + hard QAM decision.

ComNet-FC:
  LSRefineNet + ZF + FC-SD bit detector.

ComNet-BiLSTM:
  LSRefineNet + ZF + BiLSTM-SD sequence bit detector.

True-H ZF-Hard:
  정답 채널 h_true로 ZF equalization 후 hard QAM decision을 한 기준선.
  실제 receiver에서는 h_true를 사용할 수 없으므로 channel-estimation oracle reference다.
  clipping/cp_removal 같은 nonlinear case에서는 최적 detector 상한선이 아니다.
  따라서 nonlinear case에서 ComNet-FC/BiLSTM이 이 값을 넘을 수 있다.

No-Clip True-H Ref:
  clipping case에서만 추가되는 counterfactual reference다.
  같은 채널과 같은 noise realization을 쓰되, 송신 time-domain clipping이 없었다면
  True-H ZF-Hard 성능이 얼마였는지 계산한다.
  실제 receiver 알고리즘이 아니라 "CR clipping 손실이 없었을 때의 목표 성능"이다.
  송신 data symbol 자체를 직접 맞히는 label leakage 기준은 아니다.

```

CE MSE 그래프에는 다음 항목이 들어간다.

```text
LS
LMMSE
ComNet-CE
```

## 모델 구성

### CE subnet: LSRefineNet

```text
input:
  h_ls complex [N, 64]

real-valued 변환:
  concat(real(h_ls), imag(h_ls)) -> [N, 128]

network:
  Linear(128, 128, bias=False)

output:
  h_comnet complex [N, 64]

loss:
  MSE(h_comnet, h_true)
```

초기화:

```text
--ce-init identity
  LS passthrough에서 시작

--ce-init lmmse
  train split에서 fit한 empirical LMMSE matrix로 시작
```

현재 LMMSE는 analytical channel covariance 식이 아니라, train split의 `(h_ls, h_true)` 통계로 ridge regression 형태의 real-valued matrix를 fit한다.

### FC-SD

```text
input:
  x_zf = yD / h_comnet
  8개 연속 subcarrier 단위 group
  complex [N, 8] -> real [N, 16]

network:
  Linear(16, 120)
  ReLU
  Linear(120, 8 * bits_per_symbol)

output:
  group별 bit probability
```

loss:

```text
--sd-loss mse
  sigmoid(logits)와 bit label 사이 MSE

--sd-loss bce
  BCEWithLogitsLoss
```

논문 재현을 우선하면 `mse`가 기본값이다.

### BiLSTM-SD

```text
input sequence:
  [N, 64, 6]

per subcarrier feature:
  real(yD)
  imag(yD)
  real(h_comnet)
  imag(h_comnet)
  real(x_zf)
  imag(x_zf)

network:
  BiLSTM(6 -> 20, bidirectional)
  BiLSTM(40 -> 10, bidirectional)
  BiLSTM(20 -> 6, bidirectional)
  8-subcarrier group reshape
  Linear(12 * 8, 8 * bits_per_symbol)

output:
  [N, 8 groups, 8 * bits_per_symbol]
```

BiLSTM은 linear case에서도 실행되지만, 논문적으로는 CP removal이나 clipping 같은 nonlinear case에서 의미가 더 커진다.

## 출력 파일

```text
results_comnet_64qam_clipping_practical/
  ce_refinenet.pt
  lmmse_estimator.npz
  zf_refinenet_fc.pt
  zf_refinenet_bilstm.pt
  train_history_ce.csv
  train_history_sd.csv
  train_history_bilstm_sd.csv
  ce_mse_vs_snr.png
  ber_vs_snr.png
  eval_summary.json
```

먼저 확인할 파일:

```text
ber_vs_snr.png
ce_mse_vs_snr.png
eval_summary.json
```

## 학습 데이터 구성

train split:

```text
train_snr40.npz
```

용도:

```text
LMMSE estimator fit
CE subnet 학습
FC-SD 학습
BiLSTM-SD 학습
```

validation split:

```text
val_snr40.npz
```

용도:

```text
CE validation loss
SD validation loss
SD validation BER
```

test split:

```text
test_snr00.npz
test_snr05.npz
...
test_snr40.npz
```

용도:

```text
SNR별 BER
SNR별 CE MSE / NMSE
```

현재 기본 실험은 64QAM clipping case, train SNR 40 dB, test SNR sweep 0-40 dB이다. `linear` case는 LMMSE가 거의 이상적인 기준선에 가까워지는 sanity check 조건이고, ComNet detector의 차이는 nonlinear case에서 더 의미 있게 본다.

## 논문과 아직 다른 점

현재 구현은 ComNet receiver 구조, 64QAM 기본값, LMMSE CE init, FC/BiLSTM SD group output, CP removal/clipping case를 지원한다. 논문 결과에 더 가깝게 만들려면 아래가 남아 있다.

```text
1. 채널 모델 고도화
   현재: simple multipath Rayleigh
   추가: WINNER II C1 NLOS 2.6 GHz 또는 논문 조건에 가까운 channel model

2. analytical LMMSE
   현재: train data 기반 empirical LMMSE
   추가: channel covariance 기반 analytical LMMSE estimator

3. FC-DNN baseline
   논문은 ComNet과 data-driven FC-DNN도 비교한다.
   현재 구현에는 FC-DNN black-box baseline이 없다.

4. complexity 비교
   parameter count
   inference memory
   runtime

5. MU-MIMO 확장
   현재: SISO OFDM receiver만 구현
   추가: per-subcarrier MIMO channel matrix, user/stream별 pilot 처리,
   MIMO ZF/MMSE baseline, MU interference를 고려한 ComNet CE/SD 구조,
   stream별 BER/NMSE 평가 항목이 필요하다.
```

## 권장 최종 실험 명령

64QAM clipping 데이터셋 생성:

```powershell
python tx_comnet_ofdm_dataset.py `
  --out-dir outputs_comnet_64qam_clipping `
  --case clipping `
  --n-train-frames 50000 `
  --n-val-frames 10000 `
  --n-test-frames-per-snr 10000
```

현재 권장 개발/검증 학습:

```powershell
python rx_comnet_receiver.py `
  --dataset-dir outputs_comnet_64qam_clipping `
  --result-dir results_comnet_64qam_clipping_practical `
  --mode train-all `
  --sd-type both `
  --batch-size 1000 `
  --device cuda
```

`--dataset-dir outputs_comnet_64qam_clipping`, `--result-dir results_comnet_64qam_clipping_practical`, `64QAM`, `--ce-init lmmse`, `--ce-epochs 200`, `--sd-epochs 500`은 현재 기본값이다.
`--bilstm-epochs`를 생략하면 `--sd-epochs`와 같은 500 epoch로 학습한다.

논문형 긴 학습을 다시 돌릴 때는 epoch와 scheduler step을 명시한다.

```powershell
python rx_comnet_receiver.py `
  --dataset-dir outputs_comnet_64qam_clipping `
  --result-dir results_comnet_64qam_clipping_long `
  --mode train-all `
  --sd-type both `
  --batch-size 1000 `
  --ce-epochs 2000 `
  --sd-epochs 5000 `
  --bilstm-epochs 5000 `
  --ce-lr-step 1000 `
  --sd-lr-step 2000 `
  --device cuda
```

## 결과 해석 기준

정상적인 결과라면 대체로 다음 경향을 기대한다.

```text
CE MSE:
  LMMSE가 LS보다 좋아야 한다.
  ComNet-CE는 충분히 학습하면 LS 또는 LMMSE에 가까워지거나 더 좋아질 수 있다.

BER:
  LMMSE-ZF / LMMSE-MMSE는 LS-ZF보다 대체로 좋아야 한다.
  ComNet-CE-Hard는 LS-ZF보다 좋아지는지 확인한다.
  ComNet-FC와 ComNet-BiLSTM은 충분히 학습해야 의미가 있다.
  linear case에서 True-H ZF-Hard보다 일반 알고리즘이 지속적으로 좋으면 구현 오류를 의심한다.
  clipping/cp_removal case에서는 True-H ZF-Hard가 최적 detector 상한선이 아니므로 DL detector가 더 좋을 수 있다.
  No-Clip True-H Ref는 clipping이 없었을 때의 counterfactual 목표 성능이다.
```
