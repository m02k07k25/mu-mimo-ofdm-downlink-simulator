# MU-MIMO OFDM WL-ComNet Simulator

이 저장소는 raw UE antenna-domain MU-MIMO OFDM downlink 데이터셋을 만들고,
CSIT error, clipping, receiver I/Q imbalance 조건에서 WL-aware ComNet
receiver를 학습하고 평가합니다.

```text
TX dataset generator: tx_mumimo_e2e_dataset.py
RX receiver trainer:  rx_mumimo_receiver.py
Demo TX wrapper:      demo_tx_make_data.py
Demo RX wrapper:      demo_rx_infer_fixed_pt.py
PHY helper package:   mumimo_phy/
```

## 환경설정

일반 TX/RX와 데모 TX/RX는 프로젝트 루트의 `environment_config.json`을
공통으로 사용합니다. 시연 또는 실험 전에 이 파일만 수정하면 됩니다.

### 채널 및 RF 조건

```json
{
  "dataset_name": "clip17_iq05_p2_cpe3",
  "case": "clipping",
  "clip_ratio": 1.7,
  "rx_iq_gain_imbalance_db": 0.5,
  "rx_iq_phase_error_deg": 2.0,
  "rx_common_phase_error_deg": 3.0
}
```

| 항목 | 설명 |
| --- | --- |
| `dataset_name` | 일반 데이터셋과 결과 폴더 이름입니다. 데모 폴더에는 자동으로 `demo_` 접두사가 붙습니다. |
| `case` | `linear`, `cp_removal`, `clipping` 중 하나를 선택합니다. |
| `clip_ratio` | clipping threshold에 사용하는 RMS 배수입니다. |
| `rx_iq_gain_imbalance_db` | 수신단 I/Q gain imbalance입니다. |
| `rx_iq_phase_error_deg` | 수신단 I/Q phase error입니다. |
| `rx_common_phase_error_deg` | 수신단 common phase error입니다. |

### TX 데이터셋 설정

```json
{
  "tx_dataset": {
    "modulation": "64QAM",
    "n_users": 2,
    "n_streams": 2,
    "n_tx": 8,
    "n_rx_per_ue": 4,
    "n_fft": 64,
    "n_cp": 16,
    "n_taps": 7,
    "n_rays_per_path": 15,
    "pdp_decay": 5.0,
    "carrier_freq_hz": 800000000.0,
    "antenna_spacing_lambda": 0.5,
    "scm_angle_spread_deg": 3.0,
    "channel_model": "SCM-style geometric clustered channel",
    "csit_error_var": 0.001,
    "precoder_norm": "column",
    "pilot_kind": "qpsk",
    "snr_train_db_list": [40],
    "snr_test_db": [0, 5, 10, 15, 20, 25, 30, 35, 40],
    "n_train_frames": 50000,
    "n_val_frames": 10000,
    "n_test_frames_per_snr": 10000,
    "seed": 7
  }
}
```

`tx_dataset`은 일반 TX 데이터셋 생성 조건입니다. `n_streams`는 현재
구조에서 `n_users`와 동일해야 합니다. `channel_model`은 현재 지원하는
SCM-style geometric clustered channel을 명시합니다.

### 데모 모델 선택

```json
{
  "demo": {
    "model_name": "clip17_iq05_p2_cpe3"
  }
}
```

`demo.model_name`은 데모 RX가 사용할 학습 완료 모델을 선택합니다.
`results/<model_name>/` 아래의 CE, BiLSTM, LMMSE checkpoint를 읽습니다.
데모용 데이터셋 이름과 독립적으로 설정할 수 있습니다.

### RX 학습 설정

```json
{
  "training": {
    "common": {
      "batch_size": 512,
      "lmmse_ridge": 0.000001,
      "seed": 7,
      "log_every": 10
    },
    "ce": {
      "epochs": 50,
      "learning_rate": 0.001,
      "lr_step": 25,
      "lr_gamma": 0.5,
      "hidden_dim": 512,
      "dropout": 0.05
    },
    "sd": {
      "epochs": 100,
      "learning_rate": 0.001,
      "lr_step": 100,
      "lr_gamma": 0.5,
      "group_size": 8,
      "bilstm_hidden_dims": [64, 32, 16]
    }
  }
}
```

`training.common`은 공통 RX 설정입니다. `training.ce`는 channel estimator,
`training.sd`는 BiLSTM symbol detector 학습 설정입니다. `lr_step`과
`lr_gamma`는 learning-rate scheduler에 사용됩니다.

## 실행 명령

환경설정 파일의 기본값을 사용하면 별도 인자 없이 실행할 수 있습니다.
RX는 device를 자동으로 선택하므로 `--device cuda`를 지정할 필요가 없습니다.

### 일반 TX/RX

```powershell
python tx_mumimo_e2e_dataset.py
python rx_mumimo_receiver.py
```

### 데모 TX/RX

```powershell
python demo_tx_make_data.py
python demo_rx_infer_fixed_pt.py
```

데모 데이터 개수만 줄이려면 숫자를 인자로 전달합니다.

```powershell
python demo_tx_make_data.py 100
```

기존 CLI 인자를 사용한 개별 override도 지원합니다.

```powershell
python tx_mumimo_e2e_dataset.py --clip-ratio 3.0
python rx_mumimo_receiver.py --mode eval
```

## Receiver 구조

```text
Pilots -> WL-LS channel estimate
WL-LS -> linear CE layer initialized by train-split WL-LMMSE -> WL-CE
Data + WL-CE -> WL-ZF features -> BiLSTM SD -> predicted bits
```

- CE 입력은 `WL-LS`입니다.
- CE target은 augmented WL channel `(A, B)`입니다.
- CE는 single linear layer이며 empirical `WL-LMMSE` weight로 초기화합니다.
- SD는 BiLSTM을 사용합니다.
- `WL-MMSE`는 proposed detector가 아니라 comparison baseline입니다.

## 주요 출력 파일

```text
datasets/<dataset_name>/config.json
datasets/<dataset_name>/train_snr40.npz
datasets/<dataset_name>/val_snr40.npz
datasets/<dataset_name>/test_snr00.npz ... test_snr40.npz
results/<dataset_name>/eval_summary.json
results/<dataset_name>/ber_vs_snr.csv
results/<dataset_name>/ber_vs_snr.png
results/<dataset_name>/channel_mse_vs_snr.csv
results/<dataset_name>/channel_nmse_vs_snr.csv
results/<dataset_name>/diagnostic_vs_snr.csv
results/<dataset_name>/train_history_ce.csv
results/<dataset_name>/ce_training_curve.png
results/<dataset_name>/train_history_bilstm_sd.csv
results/<dataset_name>/bilstm_sd_training_curve.png
```

## 세부 문서

```text
TX_README.md          데이터셋 생성 설명
RX_README.md          Receiver, CE/SD, metric 설명
mumimo_phy/README.md  공통 PHY helper package 설명
```
