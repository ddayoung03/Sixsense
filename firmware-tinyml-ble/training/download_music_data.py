"""Fetch music clips and drop them into <out>/backgrounds/music/.

Why: false positives were reported for horn while music was playing (phone
speaker / YouTube). The noise class had no music examples at all - the
sustained, harmonically-rich tones common in music (especially brass/synth)
can look similar to a car horn's steady tone in log-mel space.

Placed under backgrounds/ (not noise/) for the same reason as
download_speech_data.py: folded into the noise class as plain 2 s windows,
AND used as SNR-mixed beds under horn/siren events (a horn heard over a car
radio is realistic too).

Source: GTZAN genre collection, mirrored on Hugging Face (marsyas/gtzan) since
the original opihi.cs.uvic.ca host is down. 1000 x 30 s clips across 10
genres, .au format - converted to .wav here since dataset.py's iter_wavs()
only looks for .wav/.flac/.ogg/.mp3. Only a random subsample is kept.
"""

from __future__ import annotations

import argparse
import random
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"


def fetch_gtzan(out: Path, limit: int, seed: int, keep_dir: Path) -> None:
    tar_path = keep_dir / "genres.tar.gz"
    if not tar_path.exists():
        print("downloading GTZAN genres (~1.2 GB) ...")
        keep_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GTZAN_URL, tar_path)
    else:
        print(f"reusing already-downloaded {tar_path}")

    extract_dir = keep_dir / "genres"
    if not extract_dir.exists():
        print("extracting ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(keep_dir)

    aus = sorted(list(extract_dir.rglob("*.au")) + list(extract_dir.rglob("*.wav")))
    print(f"found {len(aus)} music clips")
    rng = random.Random(seed)
    rng.shuffle(aus)
    chosen = aus[:limit]

    dst_dir = out / "backgrounds" / "music"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in chosen:
        try:
            data, sr = sf.read(str(fp), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"  skip {fp.name}: {e}")
            continue
        sf.write(str(dst_dir / f"gtzan_{fp.parent.name}_{fp.stem}.wav"), data, sr)
        n += 1
    print(f"  converted {n} music clips into {dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=500,
                    help="random subsample of clips to keep")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work-dir", default="data/_music_download",
                    help="scratch dir for the tarball/extraction (safe to delete after)")
    args = ap.parse_args()
    fetch_gtzan(Path(args.out), args.limit, args.seed, Path(args.work_dir))


if __name__ == "__main__":
    main()
