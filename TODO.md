# TODO: Raw MU-MIMO E2E ComNet

작성 기준: 2026-05-13

## 현재 완료된 것

- `tx_mumimo_e2e_dataset.py`로 raw UE antenna-domain MU-MIMO OFDM 데이터 생성 가능.
- `rx_mumimo_receiver.py`로 raw MU-MIMO ComNet 수신기 학습/평가 가능.
- SISO ComNet 구조를 raw MU-MIMO에 맞게 확장.
  - `MuMimoCERefineNet`
  - `MuMimoFCSDNet`
  - `MuMimoBiLSTMSDNet`
- empirical LMMSE channel estimator 추가.
- BER 비교 곡선 정리.
  - `LS-ZF`
  - `LS-MMSE`
  - `LMMSE-ZF`
  - `LMMSE-MMSE`
  - `ComNet-CE-ZF-Hard`
  - `ComNet-FC`
  - `ComNet-BiLSTM`
  - `True-H ZF`
  - `True-H MMSE`
  - `Desired-only MRC`
- MMSE detector post-equalization gain 보정 추가.
- clipping 조건에서 all-ones pilot이 BER floor를 만드는 문제 확인.
- `--pilot-kind qpsk` 추가.
  - 데이터 변조는 64QAM 유지.
  - 채널 추정용 `x_p_freq`만 QPSK phase pilot 사용.
- 최신 10% 데이터 기준으로 10^-3 이하 BER 확인.

## 최신 기준 산출물

최신 데이터:

```text
outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot
```

최신 결과:

```text
results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_bilstm300_lrstep100
```

주요 파일:

```text
ber_vs_snr.png
a_mse_vs_snr.png
eval_summary.json
train_history_bilstm_sd.csv
```

## 남은 작업

### 1. Mixed-SNR 학습 데이터 지원

현재 train split은 기본적으로 `snr_train_db=40` 단일 SNR 중심이다.

이 때문에 BiLSTM은 40 dB에서는 잘 맞지만, 10~30 dB 구간 일반화가 약하다.
전체 SNR에서 BiLSTM을 안정적으로 쓰려면 TX에서 train split을 여러 SNR로 섞어야 한다.

할 일:

- `tx_mumimo_e2e_dataset.py`에 train SNR list 옵션 추가.
  - 예: `--snr-train-db 0 5 10 15 20 25 30 35 40`
  - 또는 별도 옵션 `--snr-train-db-list`
- train split 내부에 `snr_db`가 frame별로 섞이도록 생성.
- RX SD 학습이 mixed-SNR train split을 그대로 사용하도록 확인.
- BiLSTM이 10~40 dB 전 구간에서 FC/LMMSE와 비교 가능한지 재평가.

### 2. BiLSTM 학습 조건 정리

현재 BiLSTM은 `--bilstm-epochs 300 --sd-lr-step 100`에서 고SNR 성능이 좋아졌다.
하지만 기본값 50 epoch로는 부족하다.

할 일:

- BiLSTM 전용 기본값을 문서화.
  - 권장: `--bilstm-epochs 300 --sd-lr-step 100`
- FC와 BiLSTM의 LR step을 분리할지 검토.
  - 예: `--fc-lr-step`, `--bilstm-lr-step`
- `sd_loss=mse`와 `sd_loss=bce` 비교 결과를 표로 정리.
- feature normalization 개선 검토.
  - ZF/MMSE estimate scaling
  - noise/SNR feature scaling

### 3. Pilot / clipping ablation 정식 실험

현재 결론은 다음과 같다.

```text
ones pilot + clipping:
  pilot clipping 때문에 채널추정 MSE floor 발생

qpsk pilot + clipping:
  채널추정 MSE가 SNR에 따라 정상 감소
```

할 일:

- 아래 조합을 동일 frame 수로 비교.
  - `pilot_kind=ones`, `clip_ratio=1.6`
  - `pilot_kind=ones`, `clip_ratio=3.0`
  - `pilot_kind=qpsk`, `clip_ratio=1.6`
  - `pilot_kind=qpsk`, `clip_ratio=3.0`
  - `case=linear`
- `a_mse_vs_snr.png`와 `ber_vs_snr.png`를 한 문서에 비교 정리.
- QPSK pilot이 데이터 변조를 바꾸지 않는다는 설명을 README와 발표 자료에 유지.

### 4. CE subnet 개선

최신 결과에서 high-SNR 영역은 LS/LMMSE가 이미 매우 좋다.
ComNet-CE가 40 dB에서 LS보다 약간 나빠지는 구간이 있다.

할 일:

- CE early stopping 추가 검토.
- CE loss에 residual learning 구조 적용 검토.
  - 현재: `A_ls -> A_comnet`
  - 후보: `A_comnet = A_ls + DeltaA`
- CE를 SNR별로 학습하거나 mixed-SNR로 학습했을 때 MSE 비교.
- LMMSE 초기화와 identity 초기화 비교 표 작성.

### 5. 평가 신뢰도 개선

현재 test는 SNR별로 서로 다른 랜덤 frame을 생성한다.
낮은 BER 구간에서는 35 dB와 40 dB 사이에 작은 통계 흔들림이 생길 수 있다.

할 일:

- 같은 channel/bit/noise direction에서 SNR만 바꾸는 diagnostic sweep 추가.
- 낮은 BER 측정을 위해 test frame 수 증가 실험.
  - 현재 10% 실험: 1000 frame per SNR
  - 정식 논문/보고용: 10000 frame per SNR 이상 권장
- `eval_summary.json`에 bit error count와 total bit count도 저장.
- BER가 0일 때 plotting용 floor와 실제 error count를 구분해서 기록.

### 6. 결과 디렉터리 정리

현재 실험 과정에서 여러 output/result 디렉터리가 생성되어 있다.

할 일:

- 유지할 최종 결과:
  - `outputs_mumimo_e2e_64qam_10pct_clip30_qpskpilot`
  - `results_mumimo_e2e_64qam_10pct_clip30_qpskpilot_bilstm300_lrstep100`
- 중간 실험 결과는 필요하면 보관 폴더로 이동하거나 삭제 대상 목록 작성.
- `.gitignore`에서 대용량 `outputs_*`, `results_*`가 제외되는지 확인.
- 커밋에는 코드와 문서만 포함하고, 대용량 데이터/체크포인트는 포함하지 않는 것을 권장.

### 7. README / 보고용 문서 보강

할 일:

- `README_mumimo_e2e_receiver.md`에 최신 결과 표 유지.
- 실험 명령어를 "빠른 재현"과 "정식 재현"으로 분리.
- 최종 그래프 스크린샷 또는 결과 표를 보고용 문서에 정리.
- effective-SISO bridge와 raw MU-MIMO E2E 경로 차이를 그림으로 정리.

### 8. 최소 검증 명령

코드 수정 후 최소 확인:

```powershell
python -m py_compile tx_mumimo_e2e_dataset.py rx_mumimo_receiver.py
```

권장 smoke:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_16qam_smoke --modulation 16QAM --case clipping --clip-ratio 3.0 --pilot-kind qpsk --n-train-frames 20 --n-val-frames 8 --n-test-frames-per-snr 8 --snr-test-db 20 40
python rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_16qam_smoke --result-dir results_mumimo_e2e_16qam_smoke --mode train-all --sd-type both --device cuda --ce-epochs 1 --sd-epochs 1 --bilstm-epochs 1
```

## 다음 우선순위

1. Mixed-SNR train split 추가.
2. BiLSTM을 mixed-SNR로 재학습.
3. `eval_summary.json`에 error count 저장.
4. pilot/clipping ablation 결과 표 정리.
5. 최종 64QAM 10% 결과를 README와 발표 자료에 반영.

## 실험 결과 기록

결과 폴더 삭제 전 최소 기록. 공통 설정은 `64QAM`, `n_users=2`, `n_tx=8`, `n_rx_per_ue=4`, `n_fft=64`, `n_cp=16`, `train/val/test=5000/1000/1000 per SNR`, `csit_error_var=0.0`.

| 실험 | 주요 설정 | 40 dB 핵심 결과 |
| --- | --- | --- |
| linear sanity | `case=linear`, 기존 pilot | LS/LMMSE/ComNet-CE/True-H BER = 0, FC = 1.16e-3, BiLSTM = 4.33e-3 |
| clipping 강함 | `case=clipping`, `clip_ratio=1.6`, ones pilot | LS-MMSE = 2.04e-1, LMMSE-MMSE = 7.55e-2, ComNet-CE = 7.57e-2, FC = 8.62e-2, BiLSTM = 8.87e-2, True-H = 4.14e-4 |
| clipping 완화 | `case=clipping`, `clip_ratio=2.5`, ones pilot | LS-MMSE = 1.10e-1, LMMSE-MMSE = 3.11e-2, ComNet-CE = 3.14e-2, FC = 3.21e-2, BiLSTM = 3.50e-2, True-H = 0 |
| clipping 완화 | `case=clipping`, `clip_ratio=3.0`, ones pilot | LS-MMSE = 6.00e-2, LMMSE-MMSE = 1.48e-2, ComNet-CE = 1.51e-2, FC = 1.61e-2, BiLSTM = 1.27e-2, True-H = 0 |
| 최종 기준 | `case=clipping`, `clip_ratio=3.0`, `pilot_kind=qpsk`, BiLSTM 300 epoch, `sd_lr_step=100` | LS/LMMSE/ComNet-CE/True-H/BiLSTM BER = 0, FC = 1.30e-6 |

최종 기준 상세:

| SNR | LS-MMSE | LMMSE-MMSE | ComNet-CE-ZF-Hard | ComNet-FC | ComNet-BiLSTM | LS MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 dB | 7.22e-3 | 7.18e-3 | 7.53e-3 | 1.01e-2 | 1.73e-1 | -13.95 dB |
| 25 dB | 1.25e-4 | 1.12e-4 | 1.32e-4 | 4.43e-4 | 1.35e-1 | -18.93 dB |
| 30 dB | 0 | 0 | 0 | 9.11e-6 | 6.21e-2 | -23.92 dB |
| 35 dB | 0 | 0 | 0 | 5.21e-6 | 1.36e-3 | -28.93 dB |
| 40 dB | 0 | 0 | 0 | 1.30e-6 | 0 | -33.92 dB |

비고: `qpsk`는 데이터 변조가 아니라 채널 추정용 pilot이다. 데이터와 BER 복조는 모두 `64QAM`이다. BiLSTM은 현재 40 dB train split 중심이라 10~30 dB 일반화가 약하고, mixed-SNR train이 다음 우선순위다.
