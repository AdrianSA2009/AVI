"""Tambahkan varian with_augmentation (yang belum terpakai) ke dataset training.

ingest_detect.py sengaja hanya memakai *_no_augmentation untuk mencegah leakage.
Akibatnya ~972 gambar S (dan ribuan gambar huruf lain) tidak terlatih, padahal
itu variasi yang justru dibutuhkan untuk kondisi webcam/video.

Yang dilakukan:
- Ambil SEMUA varian Labeled_Official (no_augmentation + with_augmentation).
- Buang duplikat persis antar varian (dHash Hamming 0).
- Bangun ulang box dengan landmark tangan MediaPipe (konsisten dengan
  relabel_hands.py), supaya label tidak mengurung wajah.
- Grup = klaster near-duplicate, agar tidak bocor antar fold.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\ingest_augmented.py --limit 400
    python scripts\\ingest_augmented.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path("data/raw/Labeled_Official_Indonesian_Sign_Language_(SIBI)_Alphabet/"
            "Labeled Official Indonesian Sign Language (SIBI) Alphabet")
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # tanpa J, Z (dinamis)
CLASS_ID = {letter: i for i, letter in enumerate(LETTERS)}
GATE_SIZE = 320


def dhash(image, size=8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | int(tiny[y, x] > tiny[y, x + 1])
    return bits


def collect_pairs(variant: Path) -> list[tuple[Path, Path]]:
    """(gambar, label) untuk semua kelas alfabet di satu varian.

    Dua tata letak ditemukan di dataset ini:
      flat   : images/train/bg_01__S_100.jpg  + labels/train/bg_01__S_100.txt
      nested : images/train/S/bg_01__S_100.jpg + labels/train/S/bg_01__S_100.txt
    Keduanya harus didukung: varian with_augmentation memakai nested.
    """
    pairs: list[tuple[Path, Path]] = []
    images_root = variant / "images"
    if not images_root.is_dir():
        return pairs
    for split_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        labels_root = variant / "labels" / split_dir.name
        for image in sorted(split_dir.rglob("*")):
            if image.suffix.lower() not in IMG_EXT:
                continue
            rel = image.relative_to(split_dir)
            label = (labels_root / rel).with_suffix(".txt")
            if not label.is_file() and rel.parent != Path("."):
                # subfolder huruf: nama hurufnya = folder induk
                label = labels_root / rel.parent / f"{image.stem}.txt"
            pairs.append((image, label if label.is_file() else Path()))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=Path("data/processed/augmented_frames"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_augmented.json"))
    ap.add_argument("--model", type=Path, default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--padding", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="0 = semua")
    ap.add_argument("--report", type=Path, default=Path("outputs/ingest_augmented.csv"))
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision

    if not args.root.is_dir():
        raise SystemExit(f"Folder tidak ada: {args.root}")
    if not args.model.is_file():
        raise SystemExit(f"Model tangan tidak ada: {args.model}")

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            num_hands=2,
            min_hand_detection_confidence=args.conf,
            min_hand_presence_confidence=args.conf))

    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    per_letter: Counter[str] = Counter()
    per_variant: Counter[str] = Counter()
    seen_hashes: list[int] = []
    dropped_dup = dropped_nohand = dropped_nolabel = 0
    processed = 0

    for variant in sorted(p for p in args.root.iterdir() if p.is_dir()):
        pairs = collect_pairs(variant)
        print(f"{variant.name}: {len(pairs)} gambar", flush=True)
        for image_path, label_path in pairs:
            processed += 1
            if args.limit and processed > args.limit:
                break
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            if not label_path.is_file():
                dropped_nolabel += 1
                continue
            height, width = image.shape[:2]
            small = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            result = detector.detect(Image(image_format=ImageFormat.SRGB,
                                           data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if not result.hand_landmarks:
                dropped_nohand += 1
                continue
            boxes = []
            for landmarks in result.hand_landmarks:
                xs = [p.x for p in landmarks]
                ys = [p.y for p in landmarks]
                x1, x2 = min(xs) * width, max(xs) * width
                y1, y2 = min(ys) * height, max(ys) * height
                px, py = (x2 - x1) * args.padding, (y2 - y1) * args.padding
                boxes.append((max(0.0, x1 - px), max(0.0, y1 - py),
                              min(float(width), x2 + px), min(float(height), y2 + py)))
            if not boxes:
                dropped_nohand += 1
                continue

            hash_value = dhash(image)
            if any((hash_value ^ old).bit_count() == 0 for old in seen_hashes):
                dropped_dup += 1
                continue
            seen_hashes.append(hash_value)

            letter = (image_path.name.split("__")[-1][:1].upper()
                      if "__" in image_path.name else image_path.name[:1].upper())
            class_id = CLASS_ID[letter]
            stem = f"{variant.name[:26]}_{image_path.stem[:60]}"
            dest_image = out_images / f"{stem}.jpg"
            dest_label = out_labels / f"{stem}.txt"
            cv2.imwrite(str(dest_image), image)
            dest_label.write_text("\n".join(
                f"{class_id} {(b[0] + b[2]) / 2 / width:.6f} {(b[1] + b[3]) / 2 / height:.6f} "
                f"{(b[2] - b[0]) / width:.6f} {(b[3] - b[1]) / height:.6f}" for b in boxes)
                + "\n", encoding="utf-8")

            per_letter[letter] += 1
            per_variant[variant.name] += 1
            manifest.append({"path": dest_image.as_posix(), "kelas": letter,
                             "grup": f"dup:{hash_value:016x}", "sumber": "Augmented_Official"})
            if processed % 1000 == 0:
                print(f"  {processed} diproses, {len(manifest)} disimpan", flush=True)
        if args.limit and processed > args.limit:
            break

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    import csv
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["letter", "count"])
        writer.writeheader()
        for letter, count in sorted(per_letter.items()):
            writer.writerow({"letter": letter, "count": count})

    summary = {
        "processed": processed,
        "saved": len(manifest),
        "dropped_duplicate": dropped_dup,
        "dropped_no_hand": dropped_nohand,
        "dropped_no_label": dropped_nolabel,
        "per_letter": dict(sorted(per_letter.items())),
        "per_variant": dict(per_variant),
        "manifest": str(args.manifest),
        "note": "Grup = klaster dHash; jalankan merge_manifests.py lalu make_folds.py",
    }
    (args.out / "ingest_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("processed", "saved", "dropped_duplicate",
                       "dropped_no_hand", "dropped_no_label")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
