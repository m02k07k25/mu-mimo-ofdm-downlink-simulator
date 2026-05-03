# 5GHz MU-MIMO OFDM Downlink 시뮬레이터

파일: `mu_mimo_ofdm_downlink_dataset.py`

이 코드는 5GHz 도심 소형셀 downlink MU-MIMO OFDM 링크를 end-to-end로 시뮬레이션한다. BER을 거리별로 계산하고, 딥러닝 equalizer 학습에 쓸 수 있는 JSONL record도 생성한다.

## 시스템 구성

- 기지국: 1개
- 기지국 송신 안테나: 8Tx
- 단말 수: `K = n_users`, 1~16개
- 단말 수신 안테나: UE당 4Rx
- stream: UE당 1 spatial stream
- 최대 동시 stream: 8개
- OFDM: 64 FFT, CP 16, data subcarrier 52개
- 기본 변조: QPSK, 선택 가능: BPSK/QPSK/16QAM
- precoding: downlink ZF precoding
- 수신 결합: UE별 4Rx SVD 기반 dominant combiner

`K`는 사용자 수다. 코드 내부에서는 대부분 `n_users`로 쓰이며, 출력 로그에서는 `K=1`, `K=4`처럼 표시된다.

## 처리 흐름

1. 거리 sweep 값에 맞춰 사용자별 거리, shadowing, path loss를 계산한다.
2. Rayleigh multipath tap을 만들고 path loss 전력 이득을 채널에 곱한다.
3. 시간영역 채널을 FFT해서 부반송파별 `H_f`를 만든다.
4. 각 UE의 4x8 채널에서 SVD combiner를 만들고 1-stream effective channel로 줄인다.
5. active user들의 effective channel을 모아 ZF precoder `W`를 계산한다.
6. 랜덤 bit를 만들고 BPSK/QPSK/16QAM 심볼로 변조한다.
7. 변조 심볼에 precoder를 곱해 8Tx OFDM frequency grid를 만든다.
8. IFFT와 CP 삽입으로 시간영역 OFDM 신호를 만든다.
9. 전체 8Tx 평균 합산 송신 전력이 `tx_power_dbm`이 되도록 정규화한다.
10. 각 UE의 4Rx multipath 채널을 통과시킨다.
11. `k*T*B*NF` 기반 thermal AWGN을 추가한다.
12. 수신단에서 CP 제거, FFT, combiner, desired gain equalization을 수행한다.
13. hard demodulation으로 bit를 복원하고 원래 bit와 비교해 BER을 계산한다.
14. JSONL 생성 모드에서는 수신 벡터, 채널, precoder, 링크버짓 meta, 정답 label을 저장한다.

## 거리 감쇠와 노이즈 모델

기본값은 5GHz 도심 소형셀 기준이다.

- carrier frequency: 5GHz
- bandwidth: 20MHz
- BS 총 송신 전력: 30dBm
- UE receiver noise figure: 7dB
- temperature: 290K
- 거리 sweep: 10, 30, 50, 100, 200, 300m
- path loss exponent: 3.0
- shadowing standard deviation: 6dB

Path loss는 다음 식을 쓴다.

```text
PL(dB) = 32.4 + 20log10(f_GHz) + 10*n*log10(d_m) + shadowing_dB
channel_gain_linear = 10^(-PL/10)
```

AWGN은 수신 신호 전력에 맞춰 재조정하지 않는다. 노이즈 전력은 다음과 같이 고정된다.

```text
noise_power_W = k * T * B * NF_linear
```

따라서 거리가 멀어지면 path loss가 커지고, 수신 전력과 measured SNR이 실제로 낮아진다. 예를 들어 shadowing을 0dB로 두면 10m에서 100m로 갈 때 path loss가 30dB 증가하고 수신 전력도 약 30dB 감소한다.

## 실행 예시

기본 실행:

```bash
python mu_mimo_ofdm_downlink_dataset.py
```

빠른 테스트:

```bash
python mu_mimo_ofdm_downlink_dataset.py --frames 1 --ofdm-symbols 2 --users 1 4 --distance-sweep 10 100 --max-json-records 200 --no-plot
```

거리 sweep 변경:

```bash
python mu_mimo_ofdm_downlink_dataset.py --distance-sweep 20 50 100 200 300
```

링크버짓 파라미터 변경:

```bash
python mu_mimo_ofdm_downlink_dataset.py --tx-power-dbm 33 --bandwidth-hz 10000000 --rx-noise-figure-db 5 --path-loss-exponent 3.2 --shadowing-std-db 4
```

Anaconda에만 `numpy`가 설치된 환경에서는 다음처럼 실행할 수 있다.

```powershell
& 'C:\Users\m02k0\anaconda3\python.exe' .\mu_mimo_ofdm_downlink_dataset.py --frames 1 --ofdm-symbols 2 --users 1 4 --distance-sweep 10 100 --no-plot
```

## 출력 파일

기본 출력 폴더는 `outputs_mu_mimo_ofdm`이다.

- `config.json`: 시뮬레이션 설정
- `ber_results.json`: 거리 sweep과 BER 결과
- `ber_vs_distance.png`: BER vs distance 그래프
- `dl_equalizer_dataset_schema.json`: JSONL schema 설명
- `dl_equalizer_dataset.jsonl`: equalizer 학습용 record

JSONL meta에는 다음 링크버짓 값이 포함된다.

- `distance_m`
- `path_loss_db`
- `shadowing_db`
- `rx_power_dbm`
- `noise_power_dbm`
- `measured_snr_db`

`label`은 실제 무선 시스템에서 수신기가 알 수 있는 값이 아니라, 시뮬레이션에서 supervised learning 데이터셋을 만들기 위해 저장한 송신 정답이다.

## 한계

현재 코드는 동기화 오차, 채널 추정 오차, CFO, phase noise, mobility/Doppler, channel coding, HARQ, scheduling policy, 실제 3GPP CDL/TDL 채널 모델은 포함하지 않는다. BER은 uncoded hard decision 기준이다.
