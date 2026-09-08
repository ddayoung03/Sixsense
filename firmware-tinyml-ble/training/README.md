# Retraining — horn/noise/siren classifier

Goal: fix weak real-world **horn** detection. The v3.2 model was trained on
mostly isolated, cleanly recorded horn clips with only gentle added noise
(SNR 18–35 dB), so it learned "horn = clear tone over near-silence" and misses
short horns buried in street noise. This pipeline rebuilds every horn/siren
training window as *random-offset event + real background bed at realistic
(often low) SNR*, with the event frequently occupying only a fraction of the
2 s window (`dataset.py:DEFAULT_CONFIG`).

The output model keeps the **exact same** I/O and op set as v3.2
(`log_mel_input` 64×126×1 INT8 → 3-class INT8 softmax; ops SUB, CONV_2D,
MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX), so `main/model_data.h` is a
drop-in swap. Only the INT8 quant constants in `main.cpp` change —
`apply_to_firmware.py` patches them.

## Environment (WSL2 Ubuntu, one time)

Verified on the existing `Ubuntu` (26.04) WSL distro with an RTX 5060.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
mkdir -p ~/sixsense-train && cd ~/sixsense-train && uv venv --python 3.12
source .venv/bin/activate
uv pip install "tensorflow[and-cuda]" librosa soundfile scikit-learn
```

Then, from **this** directory, always `source env.sh` first — it activates the
venv and sets `LD_LIBRARY_PATH` (TF 2.21 doesn't auto-wire the pip CUDA libs in
WSL).

```bash
cd /mnt/c/Users/PTY/orca/Sixsense/firmware-tinyml-ble/training
source env.sh && python check_gpu.py       # -> GPUs: [PhysicalDevice('/physical_device:GPU:0')]
```

Notes:
- **Do not set `CUDA_VISIBLE_DEVICES=""`** — TF 2.21 double-frees in
  `_initialize_physical_devices` when CUDA is disabled in this WSL. Train on the
  GPU (it's fast: ~25 s for 40 epochs on the full set anyway).
- RTX 50xx (Blackwell / sm_120) isn't in TF 2.21's prebuilt kernels, so the
  first GPU op JIT-compiles from PTX (a few seconds, then cached in `~/.nv`).

## Data layout

```
data/raw/horn/*.wav          car-horn events (short bursts fine)
data/raw/siren/*.wav         siren events
data/raw/noise/*.wav         non-alert street / car / crowd sound
data/raw/backgrounds/*.wav   OPTIONAL beds mixed under horn/siren (also folded
                             into the noise class; falls back to noise/ clips)
```

- `python download_public_data.py --out data/raw` grabs the ESC-50 supplement
  (only ~40 horn + ~40 siren clips — **not enough on its own**).
- Add the bulk **AI-Hub 자동차 소음** wavs into `horn/ siren/ noise/`. That set
  is what the original v3.2 used (`model_original/dataset_manifest.csv` lists the
  filenames).

## Run

```bash
source env.sh
python features.py ../main/mel_filterbank.h                 # sanity: pipeline == firmware
python build_features.py --data-root data/raw --copies 10   # -> artifacts/features.npz
python train.py --epochs 60                                  # -> artifacts/*
```

`train.py` writes `model_int8.tflite`, `model_data.h`, `model_info.json`,
`quant_constants.txt`, `report.txt` into `artifacts/` and fails loudly if the
model gains an op the firmware can't run or the tensor arena gets too big.

## Ship

```bash
python apply_to_firmware.py --dry-run     # preview
python apply_to_firmware.py               # patches main/model_data.h, main.cpp, model_info.json (.bak kept)
cd .. && idf.py build flash monitor
```

Check `artifacts/report.txt` (HORN recall + confusion matrix) before flashing.
