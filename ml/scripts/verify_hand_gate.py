"""Bukti gerbang tangan menekan false positive wajah.

Untuk gambar yang TANPA tangan, ukur: berapa yang masih lolos bila YOLO
digabung gerbang MediaPipe HandLandmarker.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\verify_hand_gate.py --weights runs/hands_fold_1/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GATE_SIZE = 320


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/hands_fold_1/weights/best.pt"))
    ap.add_argument("--gate-model", type=Path,
                    default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--src", type=Path, default=Path("data/processed/detect"))
    ap.add_argument("--with-hand", type=Path, default=Path("data/processed/detect_hands"))
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--gate-conf", type=float, default=0.3)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    gate = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.gate_model)),
            num_hands=1,
            min_hand_detection_confidence=args.gate_conf,
            min_hand_presence_confidence=args.gate_conf,
        ))

    source = {p.stem for p in (args.src / "images").glob("*")
              if p.suffix.lower() in IMG_EXT}
    hand = {p.stem for p in (args.with_hand / "images").glob("*")
            if p.suffix.lower() in IMG_EXT}
    no_hand = sorted(source - hand)
    rng = random.Random(args.seed)
    picked = rng.sample(no_hand, min(args.samples, len(no_hand)))

    yolo_fp = gate_fp = 0
    for stem in picked:
        image = cv2.imread(str(args.src / "images" / f"{stem}.jpg"))
        if image is None:
            continue
        boxes = model.predict(image, imgsz=640, conf=args.conf,
                              device=args.device, verbose=False)[0].boxes
        if len(boxes) == 0:
            continue
        yolo_fp += 1
        small = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
        result = gate.detect(Image(image_format=ImageFormat.SRGB,
                                   data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if result.hand_landmarks:
            gate_fp += 1

    summary = {
        "weights": str(args.weights),
        "images_without_hand": len(picked),
        "fp_yolo_only": yolo_fp,
        "fp_yolo_plus_gate": gate_fp,
        "reduction": round(1 - gate_fp / yolo_fp, 4) if yolo_fp else 0.0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
