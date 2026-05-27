# Project Environment Notes

- 권장 Python: `C:\Python313\python.exe`
- Project terminal의 기본 `python`도 이 환경을 가리키도록 설정되어 있습니다.
- 확인된 runtime: Python `3.13.3`
- 확인된 package:
  - `numpy 2.3.3`
  - `torch 2.8.0+cu128`
  - `torch.cuda.is_available() == True`
- 사용자가 따로 요청하지 않으면 ComNet script 실행에는 이 기본 Python 환경을
  사용합니다.

대표 명령어:

```powershell
python tx_mumimo_e2e_dataset.py --out-dir datasets/clip17_iq05_p2_cpe3 --case clipping --clip-ratio 1.7 --rx-iq-gain-imbalance-db 0.5 --rx-iq-phase-error-deg 2 --rx-common-phase-error-deg 3
python rx_mumimo_receiver.py --dataset-dir datasets/clip17_iq05_p2_cpe3 --result-dir results/clip17_iq05_p2_cpe3 --mode train-all --device cuda
```

명령어 형식:

- 사용자에게 명령어를 줄 때는 PowerShell backtick line continuation 없이
  single-line command로 제공합니다.
