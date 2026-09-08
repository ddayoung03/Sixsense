"""Fetch freely-downloadable horn / siren / street-noise clips and lay them out
under <out>/{horn,siren,noise,backgrounds}/ for build_features.py.

Sources (research / non-commercial use):
  ESC-50            CC BY-NC 3.0   github.com/karoldvl/ESC-50
                    -> car_horn, siren, + engine/street/wind noise
  (UrbanSound8K is better but is form-gated; add it manually under the same
   folders if you have it - car_horn / siren_ / street_music / jackhammer /
   engine_idling / drilling / air_conditioner)

This only covers the *public* supplement. The bulk horn/siren data is your
AI-Hub 자동차 소음 set - drop those wavs into the same folders.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ESC50_ZIP = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# ESC-50 category -> our folder
ESC50_MAP = {
    "car_horn": "horn",
    "siren": "siren",
    "engine": "backgrounds",
    "airplane": "backgrounds",
    "wind": "backgrounds",
    "rain": "backgrounds",
    "crackling_fire": "backgrounds",
    "footsteps": "noise",
    "clapping": "noise",
    "coughing": "noise",
    "washing_machine": "noise",
    "vacuum_cleaner": "backgrounds",
}


def fetch_esc50(out: Path) -> None:
    print("downloading ESC-50 (~600 MB) ...")
    data = urllib.request.urlopen(ESC50_ZIP, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    meta_txt = zf.read("ESC-50-master/meta/esc50.csv").decode()
    rows = list(csv.DictReader(io.StringIO(meta_txt)))
    audio_prefix = "ESC-50-master/audio/"
    n = 0
    for r in rows:
        folder = ESC50_MAP.get(r["category"])
        if not folder:
            continue
        dst = out / folder / f"esc50_{r['category']}_{r['filename']}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(audio_prefix + r["filename"]) as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
        n += 1
    print(f"  extracted {n} ESC-50 clips into {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("horn", "siren", "noise", "backgrounds"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fetch_esc50(out)
    print("\nNow add your AI-Hub 자동차 소음 wavs:")
    print(f"  {out}/horn/   <- 경적 clips")
    print(f"  {out}/siren/  <- 사이렌 clips")
    print(f"  {out}/noise/  <- 일반 도로/자동차 소음 (non-alert)")


if __name__ == "__main__":
    main()
