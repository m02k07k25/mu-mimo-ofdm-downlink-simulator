# mumimo_phy

`tx_mumimo_e2e_dataset.py`에서 쓰는 MU-MIMO 물리계층 공통 모듈입니다.

목표는 TX 스크립트 안에 채널, 빔포밍, OFDM, QAM, noise 구현이 길게 섞이지 않도록 분리하는 것입니다.

## 모듈

- `scm.py`: SCM-style clustered multipath MIMO channel 생성 및 time-domain channel 적용.
- `beamforming.py`: MATLAB식 steering vector, analog beam, digital ZF precoder.
- `ofdm.py`: IFFT, clipping, CP insertion.
- `modulation.py`: 16QAM/64QAM Gray QAM modulation.
- `noise.py`: SNR 변환과 AWGN 추가.

## MATLAB Mapping

- `copy/SCM.m` -> `ScmChannelGenerator`
- `copy/steer_precoding.m` -> `steering_precoder`
- `copy/ZF_precoding.m` -> `zf_precoder`
- `copy/awgn_noise.m` -> `add_awgn`

## 구현 흐름

```text
ScmChannelGenerator
-> selected strongest path angle
-> steering_precoder for TX/RX analog beams
-> hybrid_zf_precoder_context
-> W_precoder
-> OFDM waveform
-> apply_multipath_mimo
```

## 함수 요약

```text
ScmChannelGenerator.generate_multiuser
  사용자별 SCM channel tap과 path angle 생성

channel_frequency_response
  time-domain channel tap을 subcarrier별 frequency-domain channel로 변환

apply_multipath_mimo
  송신 안테나 waveform에 multipath MIMO channel을 적용

steering_precoder
  angle과 antenna array 설정으로 steering beam 생성

hybrid_zf_precoder_context
  analog steering beam 위에서 per-subcarrier digital ZF precoder 생성

ofdm_modulate_freq
  frequency-domain symbol을 time-domain OFDM symbol로 변환

add_awgn
  complex Gaussian noise 추가
```
