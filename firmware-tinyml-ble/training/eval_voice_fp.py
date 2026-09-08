"""Targeted check: does the trained model still call human speech "horn"?

Overall accuracy/recall don't answer this directly because the speech clips
are folded into the much larger "noise" class in features.npz. This runs the
exported INT8 tflite model on the speech clips alone (as plain 2 s windows,
same as they're trained) and reports the predicted-class breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import CLASSES, load_mono_16k, plain_window, iter_wavs
from features import logmel_from_pcm16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-dir", default="data/raw/backgrounds/speech")
    ap.add_argument("--model", default="artifacts/model_int8.tflite")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    files = iter_wavs(args.speech_dir)
    if not files:
        raise SystemExit(f"no speech clips found under {args.speech_dir}")

    it = tf.lite.Interpreter(model_path=args.model)
    it.allocate_tensors()
    idet = it.get_input_details()[0]
    odet = it.get_output_details()[0]
    s, z = idet["quantization"]

    counts = {c: 0 for c in CLASSES}
    horn_examples = []
    for fp in files:
        clip = load_mono_16k(fp)
        pcm = plain_window(clip, rng)
        feat = logmel_from_pcm16(pcm)
        q = np.clip(np.round(feat / s + z), -128, 127).astype(np.int8)
        it.set_tensor(idet["index"], q[None, ..., None])
        it.invoke()
        pred = int(np.argmax(it.get_tensor(odet["index"])[0]))
        counts[CLASSES[pred]] += 1
        if CLASSES[pred] == "horn":
            horn_examples.append(fp.name)

    n = len(files)
    print(f"speech clips evaluated: {n}")
    for c in CLASSES:
        print(f"  predicted {c:6s}: {counts[c]:4d}  ({100 * counts[c] / n:.1f}%)")
    if horn_examples:
        print(f"\nstill misclassified as horn ({len(horn_examples)}):")
        for name in horn_examples[:20]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
