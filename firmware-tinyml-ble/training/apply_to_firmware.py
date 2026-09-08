"""Stage 3: drop the freshly trained model into the firmware tree.

  - main/model_data.h                <- artifacts/model_data.h
  - main.cpp EXPECTED_INPUT_SCALE / EXPECTED_INPUT_ZERO_POINT
                                     <- artifacts/model_int8.tflite quant
  - model_original/model_info.json   <- artifacts/model_info.json (backup kept)

Run from the training/ directory after train.py. Use --dry-run to preview.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

FW = Path(__file__).resolve().parents[1]      # firmware-tinyml-ble/


def patch_main_cpp(main_cpp: Path, in_scale: float, in_zero: int,
                   out_scale: float, out_zero: int, dry: bool) -> None:
    text = main_cpp.read_text()

    # softmax INT8 output quant is always 1/256 / -128; verify, don't rewrite.
    m = re.search(r"EXPECTED_OUTPUT_SCALE = ([0-9.eE+-]+)f;", text)
    m2 = re.search(r"EXPECTED_OUTPUT_ZERO_POINT = (-?\d+);", text)
    if abs(float(m.group(1)) - out_scale) > 1e-6 or int(m2.group(1)) != out_zero:
        raise SystemExit(f"output quant changed ({out_scale}/{out_zero}); "
                         f"main.cpp has {m.group(1)}/{m2.group(1)} - patch manually")

    repls = {
        r"(constexpr float EXPECTED_INPUT_SCALE = )[0-9.eE+-]+f;":
            rf"\g<1>{in_scale:.16f}f;",
        r"(constexpr int EXPECTED_INPUT_ZERO_POINT = )-?\d+;":
            rf"\g<1>{in_zero};",
    }
    for pat, rep in repls.items():
        new, n = re.subn(pat, rep, text)
        if n != 1:
            raise SystemExit(f"main.cpp: pattern matched {n} times, expected 1:\n  {pat}")
        text = new
    if dry:
        print(f"[dry-run] would set EXPECTED_INPUT_SCALE={in_scale:.16f}f "
              f"EXPECTED_INPUT_ZERO_POINT={in_zero} in {main_cpp}")
        return
    main_cpp.write_text(text)
    print(f"patched {main_cpp}: input quant -> {in_scale:.10f} / {in_zero}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    art = Path(args.artifacts)

    info = json.loads((art / "model_info.json").read_text())
    q_in = info["tflite_input_quantization"]
    q_out = info["tflite_output_quantization"]
    print(f"new input quant : scale={q_in['scale']:.10f} zero={q_in['zero_point']}")
    print(f"new output quant: scale={q_out['scale']:.10f} zero={q_out['zero_point']}")

    dst_h = FW / "main" / "model_data.h"
    dst_info = FW / "model_original" / "model_info.json"
    if not args.dry_run:
        shutil.copy(dst_h, dst_h.with_suffix(".h.bak"))
        shutil.copy(dst_info, dst_info.with_suffix(".json.bak"))
        shutil.copy(art / "model_data.h", dst_h)
        shutil.copy(art / "model_info.json", dst_info)
        print(f"copied model_data.h ({(art / 'model_data.h').stat().st_size} B) and model_info.json")
    else:
        print(f"[dry-run] would copy model_data.h -> {dst_h}")

    patch_main_cpp(FW / "main" / "main.cpp",
                   q_in["scale"], q_in["zero_point"],
                   q_out["scale"], q_out["zero_point"], args.dry_run)

    print("\nnext: cd .. && idf.py build flash monitor")


if __name__ == "__main__":
    main()
