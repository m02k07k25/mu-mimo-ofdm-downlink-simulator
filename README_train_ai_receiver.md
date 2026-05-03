# train_ai_receiver.py

## 인자

```bash
python train_ai_receiver.py \
  --dataset outputs_mu_mimo_ofdm/dl_equalizer_dataset.jsonl \
  --epochs 30 \
  --batch-size 512 \
  --lr 1e-3 \
  --hidden-dims 256 128 \
  --val-ratio 0.2 \
  --model-out outputs_mu_mimo_ofdm/ai_receiver.pt \
  --result-dir result_model \
  --device auto \
  --seed 7
```

- `--dataset`: `mu_mimo_ofdm_downlink_dataset.py`가 생성한 JSONL 파일
- `--epochs`: 학습 epoch 수
- `--batch-size`: mini-batch 크기
- `--lr`: AdamW learning rate
- `--hidden-dims`: MLP hidden layer 크기 목록
- `--val-ratio`: validation split 비율
- `--model-out`: 저장할 PyTorch checkpoint 경로
- `--result-dir`: loss 그래프와 보조 결과를 저장할 폴더
- `--device`: `auto`, `cpu`, `cuda` 등
- `--seed`: train/validation split seed

## 목적

`dl_equalizer_dataset.jsonl`의 `input.feature_vector`를 읽어 AI detector를 학습한다.

현재 데이터셋은 perfect CSI 수신기가 OFDM/MIMO/equalization을 먼저 처리한 뒤 만든 `x_hat` 중심 feature를 사용한다. 따라서 이 모델은 raw OFDM/MIMO 복원기가 아니라, equalized symbol을 보고 `symbol_class`를 고르는 detector다.

## 입력과 출력

입력:

```text
feature_vector
```

기본 feature는 다음 순서다.

```text
x_hat.real
x_hat.imag
desired_gain.real
desired_gain.imag
y_scalar.real
y_scalar.imag
abs(desired_gain)
angle(desired_gain)
noise_power_true_dbm
pre_combiner_snr_db
```

출력:

```text
label.symbol_class
```

QPSK는 4-class, 16QAM은 16-class로 자동 처리한다.

## 학습 처리

1. JSONL을 읽어 `feature_vector`, `symbol_class`, `tx_bits`, `distance_m`을 로드한다.
2. seed 고정 random split으로 train/validation을 나눈다.
3. train feature 평균/표준편차로 standardization한다.
4. PyTorch MLP classifier를 학습한다.
5. 매 epoch마다 train loss, validation loss, train accuracy, validation accuracy, SER, BER을 출력한다.
6. 마지막에 거리별 validation BER을 출력한다.
7. 모델 weight, feature mean/std, class 수, bits-per-symbol, loss history를 checkpoint로 저장한다.
8. `result_model/loss_curve.png`에 x축 epoch, y축 loss 그래프를 자동 저장한다.

## 모델 구조

기본 모델은 MLP다.

```text
input_dim
→ Linear(input_dim, 256)
→ ReLU
→ Linear(256, 128)
→ ReLU
→ Linear(128, num_classes)
```

기본 perfect CSI detector dataset에서는 `input_dim=10`이다. QPSK면 `num_classes=4`, 16QAM이면 `num_classes=16`이다.

## 실행 예시

빠른 확인:

```bash
python train_ai_receiver.py --dataset outputs_mu_mimo_ofdm/dl_equalizer_dataset.jsonl --epochs 2 --batch-size 256
```

일반 학습:

```bash
python train_ai_receiver.py --dataset outputs_mu_mimo_ofdm/dl_equalizer_dataset.jsonl --epochs 30 --batch-size 512
```

GPU 강제:

```bash
python train_ai_receiver.py --dataset outputs_mu_mimo_ofdm/dl_equalizer_dataset.jsonl --device cuda
```

## PyTorch

PyTorch가 없으면 스크립트는 자동 설치하지 않고 설치 안내만 출력한다.

RTX 3070 + `C:\Python313\python.exe` 환경에서는 CUDA wheel 설치 예시는 다음과 같다.

```powershell
C:\Python313\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

확인:

```powershell
C:\Python313\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 해석

QPSK에서 validation accuracy가 25% 근처면 랜덤 수준이다. perfect CSI `x_hat` feature에서는 충분한 데이터가 있으면 QPSK는 쉽게 학습되어야 한다. 16QAM은 class가 16개이고 constellation 간격이 좁으므로 QPSK보다 더 많은 데이터와 epoch가 필요하다.
