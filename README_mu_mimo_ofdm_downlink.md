# mu_mimo_ofdm_downlink_dataset.py

## 인자

```bash
python mu_mimo_ofdm_downlink_dataset.py \
  --users 1 2 4 8 16 \
  --distance-sweep 10 30 50 100 200 300 \
  --frames 50 \
  --ofdm-symbols 8 \
  --modulation QPSK \
  --dataset-users 16 \
  --dataset-frames-per-distance 4 \
  --max-json-records 60000 \
  --records-per-distance 10000 \
  --carrier-freq-ghz 5.0 \
  --bandwidth-hz 20000000 \
  --tx-power-dbm 30 \
  --rx-noise-figure-db 7 \
  --temperature-k 290 \
  --distance-min-m 10 \
  --distance-max-m 300 \
  --path-loss-exponent 3.0 \
  --shadowing-std-db 6.0 \
  --out-dir outputs_mu_mimo_ofdm \
  --no-plot
```

- `--users`: BER 실험에 사용할 K, 즉 UE 수 목록
- `--distance-sweep`: BER/JSONL 생성에 사용할 거리 목록
- `--frames`: BER 통계용 거리당 frame 수. 최종 그래프는 200 이상 권장
- `--ofdm-symbols`: frame당 data OFDM symbol 수
- `--modulation`: `BPSK`, `QPSK`, `16QAM`
- `--dataset-users`: JSONL 생성에 사용할 UE 수
- `--dataset-frames-per-distance`: JSONL 생성용 거리당 frame 수
- `--max-json-records`: 전체 JSONL 최대 record 수
- `--records-per-distance`: 거리별 record 수. 지정하면 `max-json-records // 거리수` 대신 사용
- `--carrier-freq-ghz`, `--bandwidth-hz`, `--tx-power-dbm`, `--rx-noise-figure-db`, `--temperature-k`: 링크버짓 파라미터
- `--distance-min-m`, `--distance-max-m`: random distance를 쓸 때의 범위
- `--path-loss-exponent`, `--shadowing-std-db`: path loss/shadowing 파라미터
- `--out-dir`: 출력 폴더
- `--no-plot`: BER 그래프 생략

## 목적

5GHz 도심 소형셀 downlink MU-MIMO OFDM 링크를 시뮬레이션하고, **perfect CSI 기반 AI detector 검증용 JSONL 데이터셋**을 만든다.

현재 구조에서는 OFDM, MIMO combining, equalization은 전통 수식으로 처리한다. AI는 이미 equalization된 `x_hat`을 보고 symbol class를 고르는 detector 역할만 한다.

## 시스템 구성

- 기지국: 1개
- 기지국 송신 안테나: 8Tx
- 단말 수: `K = n_users`, 1~16개
- 단말 수신 안테나: UE당 4Rx
- stream: UE당 1 spatial stream
- 최대 동시 stream: 8개
- OFDM: 64 FFT, CP 16, data subcarrier 52개
- 기본 변조: QPSK
- precoding: perfect CSI 기반 downlink ZF
- 수신: perfect CSI 기반 combiner와 desired gain equalization

## 처리 흐름

1. 거리별 path loss와 shadowing을 만든다.
2. path loss가 반영된 multipath Rayleigh channel을 만든다.
3. 부반송파별 true channel `H_f`를 계산한다.
4. true channel로 SVD combiner와 ZF precoder를 계산한다.
5. bit 생성, modulation, OFDM IFFT, CP 삽입을 수행한다.
6. 전체 8Tx 송신 전력을 `tx_power_dbm`에 맞춰 정규화한다.
7. 채널과 thermal AWGN을 통과시킨다.
8. 수신단에서 CP 제거, FFT, combiner, gain equalization으로 `x_hat`을 만든다.
9. 전통 hard demodulation으로 baseline BER을 계산한다.
10. AI detector 학습용 JSONL에 `x_hat` 중심 feature와 정답 label을 저장한다.

## JSONL feature

`feature_vector`는 작고 명확하게 유지한다.

```text
[
  x_hat.real,
  x_hat.imag,
  desired_gain.real,
  desired_gain.imag,
  y_scalar.real,
  y_scalar.imag,
  abs(desired_gain),
  angle(desired_gain),
  noise_power_true_dbm,
  pre_combiner_snr_db
]
```

디버깅용으로 `input`에는 `rx_vector_4rx`, `channel_H_4x8`, `rx_combiner_4`, `desired_gain`도 저장하지만, 학습 스크립트는 기본적으로 `feature_vector`만 사용한다.

## 출력 파일

기본 출력 폴더는 `outputs_mu_mimo_ofdm`이다.

- `config.json`: 시뮬레이션 설정
- `ber_results.json`: 거리별 baseline BER
- `ber_vs_distance.png`: BER 그래프
- `dl_equalizer_dataset_schema.json`: JSONL schema
- `dl_equalizer_dataset.jsonl`: AI detector 학습용 데이터셋

## 실행 예시

빠른 검증:

```bash
python mu_mimo_ofdm_downlink_dataset.py --frames 2 --ofdm-symbols 2 --users 1 4 --distance-sweep 10 100 --max-json-records 400 --no-plot
```

AI 학습용 데이터셋 생성:

```bash
python mu_mimo_ofdm_downlink_dataset.py --frames 50 --ofdm-symbols 8 --distance-sweep 10 30 50 100 200 300 --max-json-records 60000
```

16QAM 데이터셋 생성:

```bash
python mu_mimo_ofdm_downlink_dataset.py --modulation 16QAM --frames 100 --max-json-records 120000
```

## 한계

이 파일은 perfect CSI 검증 단계다. 실제 수신기는 채널을 완벽히 알 수 없으므로, pilot 기반 noisy CSI 단계는 별도 실험으로 확장해야 한다.
