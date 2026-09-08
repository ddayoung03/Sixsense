"""Fast end-to-end smoke test on synthetic audio: no real data needed.
Generates tone-ish horn/siren and noise clips, runs build_features + train,
and asserts the exported TFLite has the firmware's 6-op set and INT8 I/O.
Run:  source env.sh && python smoke_test.py   (accuracy is meaningless here;
this only proves the pipeline runs and the model stays firmware-compatible.)
"""
import subprocess, sys, os
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path("data/_smoke")
SR = 16000


def wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), SR)


rng = np.random.default_rng(0)
for i in range(12):
    t = np.linspace(0, 0.6, int(SR * 0.6), endpoint=False)
    honk = 0.5 * (np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t))
    wav(ROOT / "horn" / f"h{i}.wav", honk)

    ts = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    sweep = 0.5 * np.sin(2 * np.pi * (700 + 150 * np.sin(2 * np.pi * 0.5 * ts)) * ts)
    wav(ROOT / "siren" / f"s{i}.wav", sweep)

    wav(ROOT / "noise" / f"n{i}.wav", 0.2 * rng.standard_normal(int(SR * 2.5)))

env = os.environ.copy()
# NOTE: do NOT set CUDA_VISIBLE_DEVICES="" -- TF 2.21 in this WSL double-frees
# in _initialize_physical_devices when CUDA is disabled. Let it use the GPU.
run = lambda a: subprocess.run([sys.executable, *a], check=True, env=env)
run(["build_features.py", "--data-root", str(ROOT), "--out", "artifacts/_smoke.npz",
     "--copies", "6", "--noise-copies", "4"])
run(["train.py", "--features", "artifacts/_smoke.npz", "--outdir", "artifacts/_smoke",
     "--epochs", "4", "--batch", "32"])

import tensorflow as tf
raw = Path("artifacts/_smoke/model_data.h").read_text()
assert "sound_classifier_model_len" in raw
it = tf.lite.Interpreter(model_path="artifacts/_smoke/model_int8.tflite")
it.allocate_tensors()
ops = {d["op_name"] for d in it._get_ops_details()} - {"DELEGATE"}  # XNNPACK = runtime only
print("ops:", sorted(ops))
allowed = {"SUB", "CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}
extra = ops - allowed
assert not extra, f"model uses ops the firmware resolver lacks: {extra}"
idet = it.get_input_details()[0]
odet = it.get_output_details()[0]
assert idet["dtype"] == np.int8 and odet["dtype"] == np.int8
assert tuple(idet["shape"]) == (1, 64, 126, 1), idet["shape"]
print("SMOKE OK  input", idet["shape"], idet["dtype"], " ops match firmware")
