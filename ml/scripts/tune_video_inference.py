"""Cari konfigurasi inferensi terbaik untuk video SIBI (tangan kecil di frame).

Masalah: luas tangan di video SIBI hanya ~1.2% frame, sedangkan di dataset
training ~4.6%. Model gagal pada skala kecil.

Menguji kombinasi: model x imgsz x crop-tangan, lalu melaporkan akurasi top-1
per huruf pada frame video.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\tune_video_inference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}
GATE_SIZE = 320


def hand_crop(detector, image, pad: float):
    height, width = image.shape[:2]
    small = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    result = detector.detect(Image(image_format=ImageFormat.SRGB,
                                   data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    if not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1, x2 = min(xs) * width, max(xs) * width
    y1, y2 = min(ys) * height, max(ys) * height
    pad_x, pad_y = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return image[y1:y2, x1:x2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--models", nargs="+", default=[
        "runs/hands_fold_1/weights/best.pt",
        "runs/handsc_fold_1/weights/best.pt",
    ])
    ap.add_argument("--model-sizes", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--gate-model", type=Path, default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--out", type=Path, default=Path("outputs/video_inference_tuning.json"))
    ap.add_argument("--frames", type=int, default=6,
                    help="jumlah frame tersebar merata per video")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision
    from ultralytics import YOLO

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.gate_model)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        ))

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT
                    and p.parent.name.upper() not in DYNAMIC)

    # muat frame tersebar merata sekali, dipakai semua konfigurasi
    samples: list[tuple[str, object]] = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 1:
            cap.release()
            continue
        step = max(1, total // (args.frames + 1))
        index = 0
        taken = 0
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                samples.append((video.parent.name.upper(), frame))
                taken += 1
            index += 1
        cap.release()

    results = []
    for weights in args.models:
        if not Path(weights).is_file():
            continue
        model = YOLO(weights)
        for size in args.model_sizes:
            for use_crop in (False, True):
                correct = total_n = 0
                per_letter: dict[str, list[int]] = {}
                for truth, frame in samples:
                    target = hand_crop(detector, frame, 1.2) if use_crop else frame
                    if target is None:
                        total_n += 1
                        per_letter.setdefault(truth, [0, 0])[1] += 1
                        continue
                    boxes = model.predict(target, imgsz=size, conf=args.conf,
                                          device=args.device, verbose=False)[0].boxes
                    top = model.names[int(boxes.cls[boxes.conf.argmax()])] if len(boxes) else ""
                    total_n += 1
                    correct += int(top == truth)
                    entry = per_letter.setdefault(truth, [0, 0])
                    entry[1] += 1
                    entry[0] += int(top == truth)
                results.append({
                    "weights": weights,
                    "imgsz": size,
                    "hand_crop": use_crop,
                    "accuracy": round(correct / total_n, 4) if total_n else 0.0,
                    "correct": correct,
                    "frames": total_n,
                    "per_letter": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_letter.items())},
                })
                print(f"{Path(weights).parent.parent.name:16} imgsz={size:5} crop={use_crop!s:5} "
                      f"acc={results[-1]['accuracy']:.4f} ({correct}/{total_n})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    best = max(results, key=lambda r: (r["accuracy"], r["imgsz"])) if results else None
    if best:
        print(f"\nTERBAIK: {best['weights']} imgsz={best['imgsz']} crop={best['hand_crop']} "
              f"acc={best['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
