# MU-MIMO OFDM ComNet

raw UE antenna-domain MU-MIMO OFDM downlink dataset을 생성하고, classical receiver와 ComNet 계열 receiver를 비교하는 프로젝트입니다.

```text
TX: tx_mumimo_e2e_dataset.py
RX: rx_mumimo_receiver.py
PHY modules: mumimo_phy/
```

## Project Flow

```text
TX dataset generation
-> raw pilot/data OFDM waveform 저장
-> RX에서 LS/LMMSE/ComNet-CE channel estimate 생성
-> ZF/MMSE/RF-aware WL-MMSE/ComNet-SD로 bit 검출
-> BER, channel MSE/NMSE, diagnostic CSV/plot 저장
```

현재 clean sanity에서는 QAM bit ordering, stream/user indexing, FFT/IFFT normalization, CP 제거 위치, pilot indexing, MMSE gain compensation 쪽의 큰 오류 가능성은 낮습니다.

## Main Dataset

현재 main dataset:

```text
outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000
```

설정:

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
csit_error_var = 0.001
case = clipping
clip_ratio = 2.0
pilot_kind = qpsk
rx_iq_gain_imbalance_db = 0.2
rx_iq_phase_error_deg = 1.0
rx_common_phase_error_deg = 1.0
n_train_frames = 10000
n_val_frames = 2000
n_test_frames_per_snr = 2000
train_snr_db_list = 15 20 25 30 35 40
test_snr_db = 0 5 10 15 20 25 30 35 40
```

## Generate Dataset

TX 데이터는 이미 만들어져 있으면 다시 생성할 필요가 없습니다.

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe tx_mumimo_e2e_dataset.py --out-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --modulation 64QAM --case clipping --clip-ratio 2.0 --csit-error-var 0.001 --rx-iq-gain-imbalance-db 0.2 --rx-iq-phase-error-deg 1.0 --rx-common-phase-error-deg 1.0 --n-train-frames 10000 --n-val-frames 2000 --n-test-frames-per-snr 2000
```

## Train/Evaluate RX

현재 권장 RX 설정:

```powershell
C:\Users\m02k0\anaconda3\envs\incheon_traffic_gpu\python.exe rx_mumimo_receiver.py --dataset-dir outputs_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000 --result-dir results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse --mode train-all --sd-type bilstm --sd-feature-set rf-reliability --ce-type blend-resmlp --ce-target auto --lmmse-mode snr-binned --bilstm-epochs 300 --device cuda
```

핵심 RX 옵션:

```text
CE type = blend-resmlp
CE target = auto -> RF on이면 rf-linear
SD feature = rf-reliability
LMMSE mode = snr-binned
```

`snr-binned` LMMSE는 train SNR별로 empirical LMMSE weight를 따로 fit합니다. test SNR이 train bin에 없으면 가장 가까운 train SNR bin을 사용합니다.

## Latest Result

최신 full run:

```text
results_mumimo_e2e_64qam_csit001_clip20_rfsmall_train10000_blend_ce_rf_reliability_snr_lmmse
```

40 dB:

```text
LS-MMSE                  2.654e-3
LMMSE-MMSE               3.033e-3
ComNet-CE-ZF-Hard        2.543e-3
ComNet-BiLSTM            2.503e-3
RF-aware True-H WL-MMSE  1.426e-3
```

해석:

- ComNet-BiLSTM은 25 dB 이상에서 LS-MMSE보다 좋습니다.
- SNR-binned LMMSE는 global LMMSE보다 high-SNR channel MSE와 LMMSE BER을 개선했습니다.
- 그래도 high SNR BER 기준 LMMSE-MMSE가 LS-MMSE를 완전히 이기지는 못합니다. channel MSE와 최종 BER 최적점이 다르고 RF/clipping mismatch가 남아 있기 때문입니다.
- RF-aware True-H WL-MMSE oracle과는 아직 gap이 있으므로 다음 단계는 correction-based SD입니다.

자세한 수치는 [실험결과.md](./실험결과.md)를 보세요.

## Output Metrics

RX 결과에는 다음 값이 저장됩니다.

```text
BER per method
bit_errors / total_bits
channel MSE/NMSE
desired_power_mean
inter_stream_power_mean
interference_to_desired_ratio_db
effective_sinr_db_mean
effective_sinr_db_p10
cond_A_mean
cond_A_p95
noise_power_mean
```

생성 파일:

```text
eval_summary.json
ber_vs_snr.png
a_mse_vs_snr.png
ber_vs_snr.csv
channel_mse_vs_snr.csv
channel_nmse_vs_snr.csv
diagnostic_vs_snr.csv
train_history_ce.csv
train_history_bilstm_sd.csv
```

## Documents

```text
README_tx_mumimo_comnet_dataset.md   TX dataset 상세 설명
README_mumimo_e2e_receiver.md        RX receiver 상세 설명
실험결과.md                          최신 실험 결과와 해석
TODO.md                              현재 문제와 다음 작업
```
