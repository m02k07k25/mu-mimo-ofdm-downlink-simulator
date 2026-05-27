# mumimo_phy

`mumimo_phy`는 MU-MIMO OFDM dataset generator와 receiver script가 공유하는
physical-layer primitive를 모아둔 package입니다.

## Module 구성

- `scm.py`: geometric clustered MU-MIMO channel 생성, frequency response,
  time-domain multipath 적용.
- `beamforming.py`: array steering vector, hybrid analog beam,
  per-subcarrier digital ZF precoding.
- `ofdm.py`: OFDM modulation/demodulation, cyclic prefix 처리, clipping,
  precoded frequency-domain transmit mapping.
- `modulation.py`: Gray-coded QPSK/16QAM/64QAM modulation helper와 hard
  demodulation.
- `noise.py`: SNR 변환과 complex AWGN helper.
- `helper/impairments.py`: receiver I/Q imbalance, common phase rotation,
  widely-linear RF impairment coefficient.

## 역할 경계

이 package는 재사용 가능한 PHY operation만 담당합니다. Dataset policy,
ComNet training, CE/SD model, evaluation metric, plotting은
`tx_mumimo_e2e_dataset.py` 또는 `rx_mumimo_receiver.py`에 둡니다.
