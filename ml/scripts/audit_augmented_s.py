"""Cek apakah varian with_augmentation berisi gambar S yang sah (bukan duplikat).

Sebelum menambah data, ukur: (1) apakah gambarnya unik vs no_augmentation,
(2) apakah labelnya benar-benar S secara visual (uji model),
(3) apakah datanya membantu mengenali frame webcam/video.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\audit_augmented_s.py
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path("data/raw/Labeled_Official_Indonesian_Sign_Language_(SIBI)_Alphabet/"
            "Labeled Official Indonesian Sign Language (SIBI) Alphabet")


def dhash(image, size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | int(tiny[y, x] > tiny[y, x + 1])
    return bits


def collect(variant_dir: Path, letter: str) -> list[Path]:
    multi = "multi" in variant_dir.name
    out = []
    images_dir = variant_dir / "images"
    if not images_dir.is_dir():
        return out
    for p in images_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        if multi:
            m = re.search(r"__([A-Za-z])_", p.name)
        else:
            m = re.match(r"([A-Za-z])_", p.name)
        if m and m.group(1).upper() == letter:
            out.append(p)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letter", default="S")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_augmented_s"))
    args = ap.parse_args(argv)

    no_aug = collect(ROOT / "sibi_alphabet_multi_bg_no_augmentation", args.letter)
    with_aug = collect(ROOT / "sibi_alphabet_multi_bg_with_augmentation", args.letter)
    print(f"{args.letter}: no_augmentation={len(no_aug)} gambar, "
          f"with_augmentation={len(with_aug)} gambar")

    if not with_aug:
        print("tidak ada data augmentasi untuk diaudit")
        return 1

    rng = random.Random(42)
    aug_pick = rng.sample(with_aug, min(args.samples, len(with_aug)))
    base_pick = rng.sample(no_aug, min(args.samples, len(no_aug))) if no_aug else []

    base_hashes = [dhash(cv2.imread(str(p))) for p in base_pick if cv2.imread(str(p)) is not None]
    dup = 0
    tiles = []
    for p in aug_pick:
        image = cv2.imread(str(p))
        if image is None:
            continue
        h = dhash(image)
        if base_hashes and min((h ^ b).bit_count() for b in base_hashes) <= 4:
            dup += 1
        if len(tiles) < 16:
            tiles.append(cv2.resize(image, (240, 240)))
    print(f"  dari {len(aug_pick)} sampel augmentasi: {dup} mirip-duplikat dengan no_augmentation")

    args.out.mkdir(parents=True, exist_ok=True)
    if tiles:
        width = 4
        rows = [cv2.hconcat(tiles[i:i + width]) for i in range(0, len(tiles), width)]
        if rows[-1].shape[1] != rows[0].shape[1]:
            pad = rows[0].shape[1] - rows[-1].shape[1]
            rows[-1] = cv2.copyMakeBorder(rows[-1], 0, 0, 0, pad,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(str(args.out / f"{args.letter}_augmented_samples.jpg"), cv2.vconcat(rows))
        print(f"  montage: {args.out / f'{args.letter}_augmented_samples.jpg'}")

    # uji apakah model mengenali ini sebagai huruf yang sama
    from ultralytics import YOLO
    weights = Path("runs/hands_fold_1/weights/best.pt")
    if weights.is_file():
        model = YOLO(str(weights))
        correct = total = 0
        for p in aug_pick[:40]:
            image = cv2.imread(str(p))
            if image is None:
                continue
            boxes = model.predict(image, imgsz=640, conf=0.35, device="0", verbose=False)[0].boxes
            top = model.names[int(boxes.cls[boxes.conf.argmax()])] if len(boxes) else "(none)"
            total += 1
            correct += int(top == args.letter)
        print(f"  model hands_fold_1 mengenali {correct}/{total} gambar augmentasi sebagai {args.letter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
