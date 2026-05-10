# Project Environment Notes

- Preferred Python: `C:\Python313\python.exe`
- This is the default `python` on the project terminal.
- Verified runtime: Python `3.13.3`
- Verified packages:
  - `numpy 2.3.3`
  - `torch 2.8.0+cu128`
  - `torch.cuda.is_available() == True`
- Prefer this default Python environment for running the ComNet scripts unless the user asks otherwise.

Common commands:

```powershell
python tx_comnet_ofdm_dataset.py --out-dir outputs_comnet --modulation 16QAM
python rx_comnet_receiver.py --dataset-dir outputs_comnet --result-dir results_comnet --mode train-all --device cuda
```
