"""Sort a downloaded AI-Hub audio dump into data/raw/{horn,siren,noise}/.

Two modes:

  --from-manifest   Reproduce exactly the v3.2 training subset: copy only the
                    files whose names appear in model_original/dataset_manifest.csv,
                    into the class that manifest assigns them. Safest starting point.

  --by-folder h=<substr> s=<substr> n=<substr>
                    You classify by path: any file whose path (lowercased)
                    contains the given substring goes to that class. Use this to
                    pull in MORE than the v3.2 subset once you know the layout,
                    e.g.  --by-folder h=경적 h=horn s=사이렌 s=siren n=주행 n=noise

Copies (not moves) so the original download stays intact. Skips duplicates.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

FW = Path(__file__).resolve().parents[1]
MANIFEST = FW / "model_original" / "dataset_manifest.csv"
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}


def load_manifest_map() -> dict[str, str]:
    m: dict[str, str] = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # original_name may collide across source zips but the class is the
            # same for all horn_*/siren_*/noise_* groups, so last write is fine
            m[row["original_name"].strip()] = row["class"].strip()
    return m


def copy_into(src: Path, cls: str, out: Path, seen: set[str]) -> bool:
    if src.name in seen:
        return False
    dst = out / cls / f"aihub_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    seen.add(src.name)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the extracted AI-Hub download")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--from-manifest", action="store_true")
    ap.add_argument("--by-folder", nargs="*", default=[],
                    help="tokens like h=경적 s=사이렌 n=주행 (repeatable per class)")
    args = ap.parse_args()

    src_root = Path(args.src)
    out = Path(args.out)
    wavs = [p for p in src_root.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    print(f"found {len(wavs)} audio files under {src_root}")

    counts = {"horn": 0, "siren": 0, "noise": 0}
    seen: set[str] = set()

    if args.from_manifest:
        mapping = load_manifest_map()
        for p in wavs:
            cls = mapping.get(p.name)
            if cls and copy_into(p, cls, out, seen):
                counts[cls] += 1
    elif args.by_folder:
        rules: dict[str, list[str]] = {"horn": [], "siren": [], "noise": []}
        key = {"h": "horn", "s": "siren", "n": "noise"}
        for tok in args.by_folder:
            k, _, v = tok.partition("=")
            rules[key[k]].append(v.lower())
        for p in wavs:
            path_l = str(p).lower()
            for cls, subs in rules.items():
                if any(s in path_l for s in subs):
                    if copy_into(p, cls, out, seen):
                        counts[cls] += 1
                    break
    else:
        raise SystemExit("pass --from-manifest or --by-folder")

    print("copied:", counts, "-> ", out)
    print("next:  python build_features.py --data-root", out, "--copies 10")


if __name__ == "__main__":
    main()
