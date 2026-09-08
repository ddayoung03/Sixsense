"""Stage 1: turn raw event / background clips into a cached feature tensor.

Input directory layout (pass --data-root):

    <root>/horn/*.wav          horn events (short bursts ok)
    <root>/siren/*.wav         siren events
    <root>/noise/*.wav         non-alert street / car / crowd sound
    <root>/backgrounds/*.wav   OPTIONAL beds mixed under horn/siren; also
                               folded into the noise class. Falls back to the
                               noise clips when absent.

Each raw event -> N augmented 2 s windows (--copies), features extracted with
the firmware-exact pipeline, all written to one .npz.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from constants import N_MELS, TIME_FRAMES
from features import logmel_from_pcm16
from dataset import (CLASSES, DEFAULT_CONFIG, load_mono_16k, event_window,
                     plain_window, iter_wavs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="artifacts/features.npz")
    ap.add_argument("--copies", type=int, default=8,
                    help="augmented windows per raw horn/siren clip")
    ap.add_argument("--noise-copies", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    root = Path(args.data_root)
    rng = random.Random(args.seed)

    bg_dir = root / "backgrounds"
    has_bg = bg_dir.is_dir() and any(iter_wavs(bg_dir))
    bg_files = iter_wavs(bg_dir) if has_bg else iter_wavs(root / "noise")
    print(f"loading {len(bg_files)} background beds ...")
    backgrounds = [load_mono_16k(p) for p in bg_files]

    # noise-class source clips = noise/ plus backgrounds/ (if present)
    noise_files = iter_wavs(root / "noise") + (iter_wavs(bg_dir) if has_bg else [])

    X, y = [], []
    for label, cls in enumerate(CLASSES):
        t0 = time.time()
        if cls == "noise":
            print(f"[noise] {len(noise_files)} clips x {args.noise_copies}")
            for fp in noise_files:
                try:
                    clip = load_mono_16k(fp)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {fp.name}: {e}"); continue
                for _ in range(args.noise_copies):
                    X.append(logmel_from_pcm16(plain_window(clip, rng)))
                    y.append(label)
        else:
            files = iter_wavs(root / cls)
            cfg = DEFAULT_CONFIG[cls]
            print(f"[{cls}] {len(files)} clips x {args.copies}")
            for fp in files:
                try:
                    ev = load_mono_16k(fp)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {fp.name}: {e}"); continue
                for _ in range(args.copies):
                    pcm = event_window(ev, cfg, rng, backgrounds, to_pcm16=True)
                    X.append(logmel_from_pcm16(pcm))
                    y.append(label)
        print(f"  done in {time.time() - t0:.1f}s")

    X = np.asarray(X, dtype=np.float32).reshape(-1, N_MELS, TIME_FRAMES, 1)
    y = np.asarray(y, dtype=np.int64)
    feature_mean = float(X.mean())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, feature_mean=feature_mean,
                        classes=np.array(CLASSES))
    print(f"saved {args.out}: X={X.shape} y={np.bincount(y)} "
          f"feature_mean={feature_mean:.3f}")


if __name__ == "__main__":
    main()
