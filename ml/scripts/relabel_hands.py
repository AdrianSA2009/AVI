"""Ganti box wajah/badan dengan box landmark tangan (MediaPipe HandLandmarker).

Masalah: box pada dataset gambar lama jauh lebih besar dari tangan (median luas
0.39-0.48 dari frame) sehingga model belajar mengurung wajah/badan.

Strategi: untuk setiap gambar, deteksi landmark tangan, lalu tulis SATU box
per tangan hasil union landmark + padding. Gambar tanpa tangan terdeteksi
DIBUANG (bukan memakai box lama). Kelas tetap dari label sumber.

Sumber tidak diubah: hasil ditulis ke folder baru.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\relabel_hands.py --limit 200
    python scripts\\relabel_hands.py
    python scripts\\relabel_hands.py --padding 0.15 --conf 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DYNAMIC = {"J", "Z"}


def hand_class_letters(labels_yaml: Path) -> dict[int, str]:
    """Peta id kontigu -> huruf, sama dengan ingest_detect.py (J,Z dibuang)."""
    import yaml

    data = yaml.safe_load(labels_yaml.read_text(encoding="utf-8"))
    letters = sorted(str(item["nama"]).upper() for item in data["kelas"]
                     if item["kategori"] == "huruf")
    return {i: letter for i, letter in
            enumerate(letter for letter in letters if letter not in DYNAMIC)}


def make_detector(model: Path, conf: float, num_hands: int):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        num_hands=num_hands,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_boxes(detector, image, padding: float) -> list[tuple[float, float, float, float]]:
    """Box per tangan sebagai xyxy piksel, hasil union landmark + padding."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
    boxes = []
    for landmarks in result.hand_landmarks:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x1, x2 = min(xs) * width, max(xs) * width
        y1, y2 = min(ys) * height, max(ys) * height
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
        x2, y2 = min(float(width), x2 + pad_x), min(float(height), y2 + pad_y)
        if x2 - x1 > 4 and y2 - y1 > 4:
            boxes.append((x1, y1, x2, y2))
    return boxes


def to_yolo_line(class_id: int, box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def class_id_of(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            return int(parts[0])
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/processed/detect"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/detect_hands"))
    ap.add_argument("--manifest-in", type=Path,
                    default=Path("data/processed/manifest_detect.json"))
    ap.add_argument("--manifest-out", type=Path,
                    default=Path("data/processed/manifest_hands.json"))
    ap.add_argument("--model", type=Path, default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--padding", type=float, default=0.15)
    ap.add_argument("--background-ratio", type=float, default=0.0,
                    help="JANGAN dinaikkan: gambar tanpa hasil MediaPipe ternyata "
                         "punya box asli (100%%), jadi label kosong = meracuni training. "
                         "Gerbang tangan MediaPipe dipakai saat inferensi, bukan di sini.")
    ap.add_argument("--report", type=Path, default=Path("outputs/relabel_hands.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = semua gambar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels-yaml", type=Path, default=Path("labels.yaml"),
                    help="sumber daftar kelas untuk menerjemahkan id -> huruf")
    args = ap.parse_args(argv)

    if not args.model.is_file():
        raise SystemExit(f"Model tangan tidak ada: {args.model}")
    if not args.src.is_dir():
        raise SystemExit(f"Sumber tidak ada: {args.src}")
    if not args.labels_yaml.is_file():
        raise SystemExit(f"Daftar kelas tidak ada: {args.labels_yaml}")

    id_to_letter = hand_class_letters(args.labels_yaml)

    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    source_group: dict[str, str] = {}
    if args.manifest_in.is_file():
        for item in json.loads(args.manifest_in.read_text(encoding="utf-8")):
            source_group[Path(item["path"]).stem] = item.get("grup", "")

    images = sorted(p for p in (args.src / "images").rglob("*")
                    if p.suffix.lower() in IMG_EXT)
    if args.limit:
        images = images[:args.limit]
    detector = make_detector(args.model, args.conf, args.num_hands)

    per_source_kept: Counter[str] = Counter()
    per_source_dropped: Counter[str] = Counter()
    per_source_background: Counter[str] = Counter()
    per_class_kept: Counter[str] = Counter()
    rows: list[dict] = []
    manifest: list[dict] = []
    box_areas: list[float] = []
    no_hand: dict[str, list[Path]] = defaultdict(list)

    for index, image_path in enumerate(images, start=1):
        source = image_path.stem.split("__")[0]
        label_path = args.src / "labels" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            per_source_dropped[source] += 1
            continue
        height, width = image.shape[:2]
        boxes = hand_boxes(detector, image, args.padding)
        if not boxes:
            # simpan kandidat background; bagian yang dipakai ditentukan setelah loop
            no_hand[source].append(image_path)
            continue
        class_id = class_id_of(label_path)
        if class_id is None:
            per_source_dropped[source] += 1
            continue

        lines = [to_yolo_line(class_id, box, width, height) for box in boxes]
        dest_image = out_images / f"{image_path.stem}.jpg"
        dest_label = out_labels / f"{image_path.stem}.txt"
        cv2.imwrite(str(dest_image), image)
        dest_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for box in boxes:
            box_areas.append((box[2] - box[0]) * (box[3] - box[1]) / (width * height))

        per_source_kept[source] += 1
        per_class_kept[id_to_letter.get(class_id, str(class_id))] += 1
        rows.append({"image": str(dest_image), "source": source,
                     "class_id": class_id, "hands": len(boxes)})
        manifest.append({"path": dest_image.as_posix(),
                         "kelas": id_to_letter.get(class_id, ""),
                         "grup": source_group.get(image_path.stem,
                                                  f"one:{dest_image.as_posix()}"),
                         "sumber": source})
        if index % 500 == 0:
            print(f"{index}/{len(images)} diproses | disimpan {len(rows)}", flush=True)

    # Contoh negatif (background, label kosong) supaya model bisa memutuskan
    # "tidak ada tangan". Tanpa ini model memberi box pada wajah/badan.
    rng = random.Random(args.seed)
    for source, paths in sorted(no_hand.items()):
        rng.shuffle(paths)
        take = round(len(paths) * args.background_ratio)
        for image_path in paths[:take]:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            dest_image = out_images / f"{image_path.stem}.jpg"
            cv2.imwrite(str(dest_image), image)
            (out_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            per_source_background[source] += 1
            rows.append({"image": str(dest_image), "source": source,
                         "class_id": -1, "hands": 0})
            manifest.append({"path": dest_image.as_posix(), "kelas": "",
                             "grup": source_group.get(image_path.stem,
                                                      f"one:{dest_image.as_posix()}"),
                             "sumber": source})
        per_source_dropped[source] = len(paths) - take

    box_areas.sort()

    def quant(fraction: float) -> float:
        return box_areas[int(len(box_areas) * fraction)] if box_areas else 0.0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source", "class_id", "hands"])
        writer.writeheader()
        writer.writerows(rows)
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False),
                                 encoding="utf-8")

    summary = {
        "images_in": len(images),
        "images_kept": len(rows),
        "images_with_hand": sum(per_source_kept.values()),
        "images_background": sum(per_source_background.values()),
        "images_dropped": len(images) - len(rows),
        "kept_per_source": dict(per_source_kept),
        "background_per_source": dict(per_source_background),
        "dropped_per_source": dict(per_source_dropped),
        "background_ratio": args.background_ratio,
        "padding": args.padding,
        "conf": args.conf,
        "box_area_fraction": {"p10": round(quant(0.1), 4), "median": round(quant(0.5), 4),
                              "p90": round(quant(0.9), 4)},
        "report_csv": str(args.report),
        "manifest": str(args.manifest_out),
    }
    (args.out / "relabel_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(rows) < 100:
        print("PERINGATAN: gambar tersimpan sedikit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
