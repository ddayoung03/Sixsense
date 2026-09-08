"""Fetch human speech clips and drop them into <out>/backgrounds/speech/.

Why: false positives were landing on "horn" almost every time, and the noise
class had zero human-voice examples (AI Hub's noise categories are all
vehicle/transit sounds; ESC-50 has no speech category either). A shouted or
sustained vowel has strong, fairly stable harmonics like a car horn, so the
model had never been taught to tell them apart.

Placing the clips under backgrounds/ (not noise/) does double duty, per
build_features.py:
  - folded into the noise class as plain 2 s windows, AND
  - used as SNR-mixed beds under horn/siren events, so the model also sees
    "horn burst over background chatter" during training.

Source: LibriSpeech dev-clean (openslr.org, CC BY 4.0), continuous read
speech, ~40 speakers. Only a random subsample is kept so speech doesn't
dwarf the existing traffic/street backgrounds.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

LIBRISPEECH_DEV_CLEAN = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def fetch_librispeech(out: Path, limit: int, seed: int, keep_tar_dir: Path) -> None:
    tar_path = keep_tar_dir / "dev-clean.tar.gz"
    if not tar_path.exists():
        print("downloading LibriSpeech dev-clean (~337 MB) ...")
        keep_tar_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LIBRISPEECH_DEV_CLEAN, tar_path)
    else:
        print(f"reusing already-downloaded {tar_path}")

    extract_dir = keep_tar_dir / "LibriSpeech" / "dev-clean"  # tar's own top-level dir
    if not extract_dir.exists():
        print("extracting ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(keep_tar_dir)

    flacs = sorted(extract_dir.rglob("*.flac"))
    print(f"found {len(flacs)} speech clips")
    rng = random.Random(seed)
    rng.shuffle(flacs)
    chosen = flacs[:limit]

    dst_dir = out / "backgrounds" / "speech"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fp in chosen:
        shutil.copy(fp, dst_dir / f"librispeech_{fp.stem}.flac")
    print(f"  copied {len(chosen)} speech clips into {dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=700,
                    help="random subsample of clips to keep (avoid drowning out other backgrounds)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work-dir", default="data/_speech_download",
                    help="scratch dir for the tarball/extraction (safe to delete after)")
    args = ap.parse_args()
    fetch_librispeech(Path(args.out), args.limit, args.seed, Path(args.work_dir))


if __name__ == "__main__":
    main()
