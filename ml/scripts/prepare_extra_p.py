"""Siapkan tambahan foto P dari raw, dengan box tangan MediaPipe.

Baca hanya split train dari sumber folder-class (tanpa anotasi bbox). Label P
berasal dari nama folder. Box berasal dari landmark tangan MediaPipe; gambar tanpa
landmark dibuang, tidak diberi label negatif. Dedupe/near-dedupe dikelompokkan
agar fold train/val tidak bocor.

Sumber:
  data/raw/SIBI_DATASET/train/P
  data/raw/Klasifikasi_Isyarat_Satu_Tangan_SIBI/train/P

Tulis:
  data/processed/p_extra/{images,labels}
  data/processed/manifest_p_extra.json
  outputs/p_extra_audit.jpg

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\prepare_extra_p.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 14  # P in current 24 static classes A-I,K-Y
GATE_SIZE = 320


def dhash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(tiny[y, x] > tiny[y, x + 1])
    return value


def hand_boxes(detector, image, padding):
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

    h, w = image.shape[:2]
    resized = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
    result = detector.detect(Image(image_format=ImageFormat.SRGB,
                                   data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    out = []
    for landmarks in result.hand_landmarks:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x1, x2 = min(xs) * w, max(xs) * w
        y1, y2 = min(ys) * h, max(ys) * h
        px, py = (x2 - x1) * padding, (y2 - y1) * padding
        x1, y1 = max(0.0, x1 - px), max(0.0, y1 - py)
        x2, y2 = min(float(w), x2 + px), min(float(h), y2 + py)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            out.append((x1, y1, x2, y2))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=[
        Path("data/raw/SIBI_DATASET/train/P"),
        Path("data/raw/Klasifikasi_Isyarat_Satu_Tangan_SIBI/train/P"),
    ])
    parser.add_argument("--out", type=Path, default=Path("data/processed/p_extra"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_p_extra.json"))
    parser.add_argument("--model", type=Path, default=Path("weights/hand_landmarker.task"))
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--dhash-distance", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-samples", type=int, default=24)
    parser.add_argument("--audit-out", type=Path, default=Path("outputs/p_extra_audit.jpg"))
    args = parser.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision

    if not args.model.is_file():
        raise SystemExit(f"Model tangan tidak ada: {args.model}")
    candidates = []
    for root in args.roots:
        if not root.is_dir():
            print(f"lewati, folder tidak ada: {root}")
            continue
        source_name = root.parents[1].name
        candidates.extend((p, source_name) for p in sorted(root.rglob("*"))
                          if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not candidates:
        raise SystemExit("Tidak ada foto P pada sumber train")

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            num_hands=2,
            min_hand_detection_confidence=args.conf,
            min_hand_presence_confidence=args.conf,
        ))
    images_dir, labels_dir = args.out / "images", args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest, kept_hashes = [], [], []
    skipped_read = skipped_hand = skipped_dup = 0
    for index, (source_path, source_name) in enumerate(candidates):
        image = cv2.imread(str(source_path))
        if image is None:
            skipped_read += 1
            continue
        boxes = hand_boxes(detector, image, args.padding)
        if not boxes:
            skipped_hand += 1
            continue
        hashed = dhash(image)
        group_index = next((j for j, prev in enumerate(kept_hashes)
                            if (hashed ^ prev).bit_count() <= args.dhash_distance), None)
        if group_index is not None:
            skipped_dup += 1
            continue
        kept_hashes.append(hashed)

        h, w = image.shape[:2]
        stem = f"{source_name}_{source_path.stem}"
        dest_image = images_dir / f"{stem}.jpg"
        dest_label = labels_dir / f"{stem}.txt"
        cv2.imwrite(str(dest_image), image)
        dest_label.write_text("\n".join(
            f"{CLASS_ID} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            for x1, y1, x2, y2 in boxes) + "\n", encoding="utf-8")
        group = f"p-near:{group_index if group_index is not None else len(kept_hashes)-1}"
        manifest.append({"path": dest_image.as_posix(), "kelas": "P",
                         "grup": group, "sumber": f"P_extra_{source_name}"})
        rows.append((dest_image, boxes))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.audit_samples, len(rows)))
    tiles = []
    for path, boxes in sample:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(image, "GT P", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(image, (240, 240)))
    if tiles:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        width = 4
        rows_img = [cv2.hconcat(tiles[i:i + width]) for i in range(0, len(tiles), width)]
        if rows_img[-1].shape[1] < rows_img[0].shape[1]:
            rows_img[-1] = cv2.copyMakeBorder(rows_img[-1], 0, 0, 0,
                rows_img[0].shape[1] - rows_img[-1].shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(str(args.audit_out), cv2.vconcat(rows_img))

    print(json.dumps({
        "candidate_images": len(candidates), "saved": len(manifest),
        "skipped_no_hand": skipped_hand, "skipped_near_duplicate": skipped_dup,
        "skipped_unreadable": skipped_read, "per_source": {
            s: sum(m["sumber"] == f"P_extra_{s}" for m in manifest)
            for s in sorted({m["sumber"].replace("P_extra_", "") for m in manifest})},
        "manifest": str(args.manifest), "audit": str(args.audit_out),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
