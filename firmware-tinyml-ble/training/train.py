"""Stage 2: train, INT8-quantise, and emit the firmware artifacts.

Outputs (into --outdir):
  model_int8.tflite        the quantised model
  model_data.h             drop-in replacement for main/model_data.h
  model_info.json          updated feature / quant metadata
  quant_constants.txt      EXPECTED_INPUT_SCALE / ZERO_POINT for main.cpp
  report.txt               held-out accuracy + confusion matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model, convert_int8
from constants import (SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, TIME_FRAMES,
                       FMIN, FMAX, MIN_DB, MAX_DB, RMS_TARGET_PCM16)

CLASSES = ["horn", "noise", "siren"]


def c_array(raw: bytes) -> str:
    lines, row = [], []
    for i, b in enumerate(raw):
        row.append(f"0x{b:02x}")
        if len(row) == 12:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    body = "\n".join(lines)
    return (
        "#pragma once\n#include <stdint.h>\n\n"
        f"alignas(16) const unsigned char sound_classifier_model[] = {{\n{body}\n}};\n"
        f"const unsigned int sound_classifier_model_len = {len(raw)};\n"
    )


FIRMWARE_OPS = {"SUB", "CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}


def tflite_io_quant(model_bytes: bytes):
    it = tf.lite.Interpreter(model_content=model_bytes)
    it.allocate_tensors()

    ops = {d["op_name"] for d in it._get_ops_details()} - {"DELEGATE"}
    extra = ops - FIRMWARE_OPS
    if extra:
        raise SystemExit(f"model uses ops the firmware resolver lacks: {extra}\n"
                         f"(main.cpp registers exactly {sorted(FIRMWARE_OPS)})")

    arena_est = sum(int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
                    for d in it.get_tensor_details() if d["shape"].size)
    if arena_est > 280_000:
        print(f"WARNING: rough arena estimate {arena_est} B is close to the "
              f"firmware's 300000 B TENSOR_ARENA_SIZE; shrink filters/dense_units")
    else:
        print(f"rough arena estimate: {arena_est} B (firmware cap 300000)")

    idet, odet = it.get_input_details()[0], it.get_output_details()[0]
    return (float(idet["quantization"][0]), int(idet["quantization"][1]),
            float(odet["quantization"][0]), int(odet["quantization"][1]), it)


def evaluate(interpreter, x_float, y):
    idet = interpreter.get_input_details()[0]
    odet = interpreter.get_output_details()[0]
    s, z = idet["quantization"]
    preds = []
    for i in range(len(x_float)):
        q = np.clip(np.round(x_float[i] / s + z), -128, 127).astype(np.int8)
        interpreter.set_tensor(idet["index"], q[None, ...])
        interpreter.invoke()
        preds.append(int(np.argmax(interpreter.get_tensor(odet["index"])[0])))
    preds = np.asarray(preds)
    rep = classification_report(y, preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(y, preds)
    return preds, rep, cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.npz")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filters", default="8,16,24",
                    help="conv filter counts (shrink if arena warning fires)")
    ap.add_argument("--dense-units", type=int, default=24)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    d = np.load(args.features, allow_pickle=True)
    X, y, feature_mean = d["X"], d["y"], float(d["feature_mean"])
    print(f"X={X.shape}  class counts={np.bincount(y)}  feature_mean={feature_mean:.3f}")

    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed)

    cw = len(y_tr) / (len(CLASSES) * np.bincount(y_tr))
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("class_weight:", class_weight)

    model = build_model(feature_mean=feature_mean, n_classes=len(CLASSES),
                        filters=tuple(int(f) for f in args.filters.split(",")),
                        dense_units=args.dense_units)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=5e-5),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                         monitor="val_accuracy"),
    ]
    model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=args.epochs,
              batch_size=args.batch, class_weight=class_weight, callbacks=cbs)

    raw = convert_int8(model, x_tr)
    in_s, in_z, out_s, out_z, it = tflite_io_quant(raw)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "model_int8.tflite").write_bytes(raw)
    (outdir / "model_data.h").write_text(c_array(raw))

    preds, rep, cm = evaluate(it, x_te, y_te)
    horn_recall = cm[0, 0] / cm[0].sum()
    report = (
        f"model bytes: {len(raw)}\n"
        f"input  quant: scale={in_s:.10f} zero={in_z}\n"
        f"output quant: scale={out_s:.10f} zero={out_z}\n\n{rep}\n\n"
        f"confusion matrix (rows=true {CLASSES}):\n{cm}\n\n"
        f"HORN recall: {horn_recall:.4f}\n"
    )
    (outdir / "report.txt").write_text(report)
    print(report)

    (outdir / "quant_constants.txt").write_text(
        f"// paste into main.cpp\n"
        f"constexpr float EXPECTED_INPUT_SCALE = {in_s:.16f}f;\n"
        f"constexpr int   EXPECTED_INPUT_ZERO_POINT = {in_z};\n"
        f"constexpr float EXPECTED_OUTPUT_SCALE = {out_s:.16f}f;\n"
        f"constexpr int   EXPECTED_OUTPUT_ZERO_POINT = {out_z};\n")

    info = {
        "version": "honk_robust_v4_lowsnr",
        "classes": CLASSES,
        "sample_rate": SAMPLE_RATE, "window_seconds": 2.0,
        "sample_count": SAMPLE_RATE * 2,
        "feature_type": "log_mel_spectrogram",
        "n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS,
        "fmin": FMIN, "fmax": FMAX, "min_db": MIN_DB, "max_db": MAX_DB,
        "time_frames": TIME_FRAMES,
        "feature_mean_sub": feature_mean,
        "minimum_confidence": 0.75,
        "runtime_rms_normalization": {
            "enabled": True, "gate_pcm16": 150.0,
            "target_pcm16": RMS_TARGET_PCM16,
            "min_gain": 0.25, "max_gain": 8.0, "peak_limit_float": 0.98,
        },
        "tflite_input_quantization": {"scale": in_s, "zero_point": in_z},
        "tflite_output_quantization": {"scale": out_s, "zero_point": out_z},
    }
    (outdir / "model_info.json").write_text(json.dumps(info, indent=2))
    print("wrote", outdir / "model_data.h", "and", outdir / "model_info.json")


if __name__ == "__main__":
    main()
