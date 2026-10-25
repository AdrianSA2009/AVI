"""Verifikasi jalur realtime: gerbang blokir tanpa-tangan + akurasi huruf S.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\test_realtime_letters.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realtime_test as rt  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GATE_SIZE = 320


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/processed/detect"))
    ap.add_argument("--with-hand", type=Path, default=Path("data/processed/detect_hands"))
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args(argv)

    class Args:
        pass

    a = Args()
    a.model = rt.find_best_pt(1)
    a.conf = args.conf
    a.iou = 0.45
    a.onnx = False
    predict = rt.make_predictor(a)
    focus = rt.make_hand_gate(Path("weights/hand_landmarker.task"), 0.3)

    source = {p.stem for p in (args.src / "images").glob("*") if p.suffix.lower() in IMG_EXT}
    hand = {p.stem for p in (args.with_hand / "images").glob("*") if p.suffix.lower() in IMG_EXT}
    no_hand = sorted(source - hand)
    rng = random.Random(4)
    picked = rng.sample(no_hand, min(args.samples, len(no_hand)))

    leaked = 0
    for stem in picked:
        image = cv2.imread(str(args.src / "images" / f"{stem}.jpg"))
        if image is None:
            continue
        focused = focus(image)
        if focused is None:
            continue
        crop, _ = focused
        if predict(crop):
            leaked += 1
    print(f"tanpa tangan: {leaked}/{len(picked)} masih lolos gerbang+crop "
          f"({100 * leaked / len(picked):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
