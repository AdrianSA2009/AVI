"""Ekstrak frame video SIBI jadi data training in-domain (box tangan MediaPipe).

Masalah: dataset gambar adalah close-up terang, sedangkan video SIBI wide shot
gelap dengan tangan hanya ~1.2% frame (dataset ~4.6%). Model jadi gagal pada
huruf tertentu (S, D, H, P, T).

Solusi: ambil frame dari video SIBI sendiri, box dari landmark tangan
MediaPipe (bukan dari prediksi YOLO yang keliru), label dari nama folder video.
Grup = nama video, supaya tidak bocor antar split.

Baca : data/raw/.../sibi-dataset-dib-example-face-blurred/alphabet/<huruf>/*.mp4
Tulis: data/processed/sibi_video_frames/{images,labels}
       data/processed/manifest_sibi_video.json

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\extract_sibi_video.py --per-video 12
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}
GATE_SIZE = 320


def hand_boxes(detector, image, padding: float):
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

    height, width = image.shape[:2]
    small = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
    result = detector.detect(Image(image_format=ImageFormat.SRGB,
                                   data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    boxes = []
    for landmarks in result.hand_landmarks:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x1, x2 = min(xs) * width, max(xs) * width
        y1, y2 = min(ys) * height, max(ys) * height
        pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
        x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
        x2, y2 = min(float(width), x2 + pad_x), min(float(height), y2 + pad_y)
        if x2 - x1 > 8 and y2 - y1 > 8:
            boxes.append((x1, y1, x2, y2))
    return boxes


def to_line(class_id: int, box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/sibi_video_frames"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_sibi_video.json"))
    ap.add_argument("--model", type=Path, default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--labels-yaml", type=Path, default=Path("labels.yaml"))
    ap.add_argument("--per-video", type=int, default=12)
    ap.add_argument("--padding", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-blur", type=float, default=20.0)
    ap.add_argument("--report", type=Path, default=Path("outputs/sibi_video_frames.csv"))
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision

    if not args.model.is_file():
        raise SystemExit(f"Model tangan tidak ada: {args.model}")
    if not args.source.is_dir():
        raise SystemExit(f"Sumber video tidak ada: {args.source}")

    data = yaml.safe_load(args.labels_yaml.read_text(encoding="utf-8"))
    letters = sorted(str(item["nama"]).upper() for item in data["kelas"]
                     if item["kategori"] == "huruf")
    id_of = {letter: i for i, letter in
             enumerate(x for x in letters if x not in DYNAMIC)}

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            num_hands=2,
            min_hand_detection_confidence=args.conf,
            min_hand_presence_confidence=args.conf,
        ))

    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    rows: list[dict] = []
    manifest: list[dict] = []
    per_letter: Counter[str] = Counter()
    skipped_blur = skipped_nohand = 0

    for video in videos:
        letter = video.parent.name.upper()
        if letter in DYNAMIC or letter not in id_of:
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 3:
            cap.release()
            continue
        step = max(1, total // (args.per_video + 1))
        index = taken = 0
        while taken < args.per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur < args.min_blur:
                    skipped_blur += 1
                else:
                    boxes = hand_boxes(detector, frame, args.padding)
                    if not boxes:
                        skipped_nohand += 1
                    else:
                        height, width = frame.shape[:2]
                        stem = f"{video.stem}_f{index:04d}"
                        image_path = images_dir / f"{stem}.jpg"
                        label_path = labels_dir / f"{stem}.txt"
                        cv2.imwrite(str(image_path), frame)
                        label_path.write_text(
                            "\n".join(to_line(id_of[letter], b, width, height)
                                      for b in boxes) + "\n", encoding="utf-8")
                        per_letter[letter] += 1
                        rows.append({"image": str(image_path), "video": video.name,
                                     "letter": letter, "frame": index,
                                     "blur": round(blur, 2), "boxes": len(boxes)})
                        manifest.append({
                            "path": image_path.as_posix(),
                            "kelas": letter,
                            "grup": f"vid:{video.stem}",
                            "sumber": "SIBI_video",
                        })
                        taken += 1
            index += 1
        cap.release()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "video", "letter",
                                                    "frame", "blur", "boxes"])
        writer.writeheader()
        writer.writerows(rows)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = {
        "frames_saved": len(rows),
        "videos_used": len({r["video"] for r in rows}),
        "skipped_blur": skipped_blur,
        "skipped_no_hand": skipped_nohand,
        "per_letter": dict(sorted(per_letter.items())),
        "letters_missing": sorted(set(id_of) - set(per_letter)),
        "manifest": str(args.manifest),
        "report_csv": str(args.report),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
