# TODO: End-to-End MU-MIMO ComNet Extension

작성일: 2026-05-12

## 현재 상황

- `tx_mumimo_comnet_dataset.py`가 추가되어 downlink MU-MIMO OFDM 채널, UE combiner, BS ZF precoder, CSIT error를 포함한 dataset을 생성할 수 있다.
- 현재 MU-MIMO TX 출력은 raw multi-antenna UE waveform이 아니다.
- 현재 출력은 각 UE stream을 post-combining effective SISO stream으로 바꾼 뒤, 기존 `rx_comnet_receiver.py`가 읽을 수 있는 shape로 저장한다.
- 따라서 현재 구조는 다음과 같다.

```text
MU-MIMO channel + precoding + UE combining
-> per-user effective scalar stream
-> SISO-compatible ComNet dataset
-> existing SISO rx_comnet_receiver.py
```

- `rx_comnet_receiver.py`는 아직 full MU-MIMO receiver가 아니다.
- `results_mumimo_smoke/`에는 작은 smoke 학습/평가 결과가 있지만, 이것은 full end-to-end MU-MIMO RX 검증이 아니라 SISO-compatible bridge 검증이다.
- `README_tx_mumimo_comnet_dataset.md`에는 현재 v1 dataset의 의미와 한계가 문서화되어 있다.

## 중요한 결론

목표가 end-to-end MU-MIMO라면 현재 상태는 완료가 아니다.

현재 구현은 "MU-MIMO 환경에서 생성한 effective SISO stream을 기존 SISO ComNet RX로 처리"하는 단계이다.  
end-to-end MU-MIMO라고 하려면 RX도 raw multi-antenna, multi-user 신호를 직접 처리하도록 바꿔야 한다.

## 해야 할 일

### 1. End-to-end MU-MIMO dataset schema 정의

- `tx_mumimo_comnet_dataset.py` 또는 새 TX script가 raw UE antenna-domain 신호를 저장하도록 확장한다.
- 후보 array shape:

```text
rx_p_time:   complex64, [n_frames, n_users, n_rx_per_ue, n_fft+n_cp]
rx_d_time:   complex64, [n_frames, n_users, n_rx_per_ue, n_fft+n_cp]
x_p_freq:    complex64, [n_frames, n_users, n_fft]
x_d_freq:    complex64, [n_frames, n_users, n_fft]
bits:        int8,      [n_frames, n_users, n_fft, bits_per_symbol]
H_true:      complex64, [n_frames, n_fft, n_users, n_rx_per_ue, n_tx]
W_precoder:  complex64, [n_frames, n_fft, n_tx, n_streams]
snr_db:      float32,   [n_frames]
```

- `n_streams`는 우선 `n_users`로 둔다. 즉 UE당 1 stream.
- 기존 effective SISO용 `h_true: [N_eff, n_fft]`와 raw MU-MIMO용 `H_true`를 혼동하지 않도록 config에 `waveform_type`을 명확히 기록한다.

### 2. TX 시뮬레이터를 raw MU-MIMO 송수신으로 확장

- 현재는 `h_eff_all`을 만든 뒤 scalar frequency-domain stream을 OFDM time waveform으로 재구성한다.
- end-to-end용 TX에서는 다음을 직접 시뮬레이션해야 한다.

```text
s[k]                     : [n_streams]
x_tx[k] = W[k] s[k]      : [n_tx]
y_u[k] = H_u[k] x_tx[k] + noise_u[k] : [n_rx_per_ue]
```

- pilot도 UE별 scalar effective pilot이 아니라, RX가 channel/stream 정보를 추정할 수 있는 multi-antenna pilot 구조로 저장해야 한다.
- clean CSIT, imperfect CSIT, residual MUI 조건을 모두 재현 가능하게 유지한다.

### 3. MU-MIMO RX script 추가

- 기존 `rx_comnet_receiver.py`를 직접 크게 바꾸기보다, 우선 `rx_mumimo_receiver.py`를 새로 만드는 쪽이 안전하다.
- 새 RX는 raw `rx_*_time`을 FFT하여 `[frame, user, rx_ant, subcarrier]` 신호를 처리해야 한다.
- baseline receiver를 먼저 구현한다.

```text
LS channel estimation
per-user combining
ZF / MMSE linear detection
True-H oracle baseline
BER vs SNR
```

- baseline이 맞은 뒤 neural receiver를 붙인다.

### 4. Neural MU-MIMO receiver 설계

- 1차 목표는 UE별 detector:

```text
input:
  y_u[k] across rx antennas
  estimated H_u[k] or effective channel features
  optional W[k]

output:
  bits for user u, all subcarriers
```

- 이후 확장 후보:

```text
joint multi-user detector
attention or transformer over subcarriers/users/antennas
BiLSTM over subcarriers with antenna/channel features
```

- 기존 ComNet CE/SD 구조를 그대로 쓸 수 있는지, 또는 MU-MIMO용 feature tensor로 바꿔야 하는지 분리해서 검증한다.

### 5. 검증 항목

- Noiseless clean CSIT에서 True-H linear detector BER가 0에 가까운지 확인한다.
- `csit_error_var = 0`에서 residual MUI가 거의 제거되는지 확인한다.
- `csit_error_var > 0`에서 high-SNR BER floor가 재현되는지 확인한다.
- raw waveform FFT 결과가 `H_u[k] W[k] s[k] + noise`와 일치하는지 확인한다.
- 기존 SISO-compatible v1 결과와 end-to-end raw MU-MIMO 결과를 구분해서 저장한다.

### 6. 실험 산출물 정리

- smoke 출력 디렉터리는 commit 대상에서 제외할지 결정한다.
- full experiment용 output/result directory naming을 정한다.

```text
outputs_mumimo_effective_siso_*
results_mumimo_effective_siso_*
outputs_mumimo_e2e_raw_*
results_mumimo_e2e_raw_*
```

- README도 다음 두 문서로 분리하는 것을 고려한다.

```text
README_tx_mumimo_comnet_dataset.md      # current effective-SISO bridge
README_mumimo_e2e_receiver.md           # future raw MU-MIMO RX path
```

## 완료 기준

- raw MU-MIMO dataset을 생성할 수 있다.
- `rx_mumimo_receiver.py`가 raw multi-antenna UE waveform을 직접 읽는다.
- baseline ZF/MMSE/True-H 결과가 sanity check를 통과한다.
- neural MU-MIMO receiver 학습 및 평가가 가능하다.
- BER vs SNR plot과 `eval_summary.json`이 end-to-end raw MU-MIMO 기준으로 생성된다.
- 문서에서 effective SISO bridge와 end-to-end MU-MIMO RX를 명확히 구분한다.
