"""Pull horn / siren / noise wavs straight out of the AI-Hub '도시 소리 데이터'
(dataSetSn=585) 교통소음 zips into data/raw/, without a full extraction.

zip internal filenames are EUC-KR; `unzip` isn't installed, so this uses
Python's zipfile and decodes names itself. Only the wavs we keep are written
to disk, so the 3 GB VS_1 zip can be deleted afterwards.

Category -> class map (edit CATEGORY_MAP to taste):
  1.자동차/1.차량경적      -> horn
  2.이륜자동차/4.이륜차경적 -> horn
  1.자동차/2.차량사이렌     -> siren
  everything else 주행음/비행기/헬리콥터/기차/지하철 -> noise
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

CATEGORY_MAP = {
    "1.차량경적": "horn",
    "4.이륜차경적": "horn",
    "2.차량사이렌": "siren",
    "3.차량주행음": "noise",
    "5.이륜차주행음": "noise",
    "6.비행기": "noise",
    "7.헬리콥터": "noise",
    "8.기차": "noise",
    "9.지하철": "noise",
}


def decode(name: str) -> str:
    try:
        return name.encode("cp437").decode("euc-kr")
    except Exception:
        return name


def classify(path: str) -> str | None:
    for token, cls in CATEGORY_MAP.items():
        if token in path:
            return cls
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, nargs="+", help="one or more *.교통소음.zip")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="0 = no cap; otherwise keep at most N wavs per class")
    args = ap.parse_args()

    out = Path(args.out)
    counts = {"horn": 0, "siren": 0, "noise": 0}

    for zpath in args.zip:
        zf = zipfile.ZipFile(zpath)
        for info in zf.infolist():
            if info.is_dir():
                continue
            dn = decode(info.filename)
            if not dn.lower().endswith(".wav"):
                continue
            cls = classify(dn)
            if cls is None:
                continue
            if args.limit_per_class and counts[cls] >= args.limit_per_class:
                continue
            dst = out / cls / f"aihub_{Path(dn).name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            counts[cls] += 1

    print("extracted:", counts, "->", out)
    total_mb = sum(p.stat().st_size for p in out.rglob("aihub_*.wav")) / 1e6
    print(f"kept {sum(counts.values())} wavs, {total_mb:.0f} MB")
    print("safe to delete the source zip now")


if __name__ == "__main__":
    main()
