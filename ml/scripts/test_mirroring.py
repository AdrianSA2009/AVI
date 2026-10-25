"""Uji hipotesis mirroring: apakah frame video SIBI terbalik vs dataset gambar?

Dua bukti:
1. handedness MediaPipe (kiri/kanan) per sumber, dan
2. prediksi setelah gambar dicermin (flip horizontal).

Kalau flip mengubah prediksi secara sistematis, berarti orientasi penting dan
kedua sumber tidak searah.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\test_mirroring.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realtime_test as rt  # noqa: E402

GATE_SIZE = 320


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("runs/hands_fold_1/weights/best.pt"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/processed/detect_hands/images"))
    ap.add_argument("--video-frames", type=Path, default=Path("data/processed/sibi_video_frames/images"))
    ap.add_argument("--letters", nargs="+", default=["S", "M", "N"])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="weights/hand_landmarker.task"),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        ))

    a = type("A", (), {})()
    a.model, a.conf, a.iou, a.onnx = args.weights, args.conf, 0.45, False
    predict = rt.make_predictor(a)

    def handedness(image):
        small = cv2.resize(image, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)
        result = detector.detect(Image(image_format=ImageFormat.SRGB,
                                       data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if not result.handedness:
            return "-"
        return result.handedness[0][0].category_name

    import yaml
    names = {int(k): str(v).upper() for k, v in yaml.safe_load(
        Path("datasets/sibi/hands_run1.yaml").read_text(encoding="utf-8"))["names"].items()}

    for letter in args.letters:
        # dataset gambar
        ds_paths = []
        for path in sorted(args.image_dir.glob("*.jpg")):
            label = path.parent.parent / "labels" / f"{path.stem}.txt"
            if not label.is_file():
                continue
            lines = [l for l in label.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines and names.get(int(lines[0].split()[0])) == letter:
                ds_paths.append(path)
            if len(ds_paths) >= args.samples:
                break

        vid_paths = sorted(args.video_frames.glob(f"*alphabet-{letter.lower()}_*.jpg"))[:args.samples]

        print(f"\n=== huruf {letter} ===")
        for tag, paths in (("dataset", ds_paths), ("video", vid_paths)):
            hand = Counter()
            normal = Counter()
            flipped = Counter()
            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                hand[handedness(image)] += 1
                r1 = predict(image)
                normal[max(r1, key=lambda d: d["conf"])["cls"] if r1 else "(none)"] += 1
                r2 = predict(cv2.flip(image, 1))
                flipped[max(r2, key=lambda d: d["conf"])["cls"] if r2 else "(none)"] += 1
            print(f"  {tag:8} tangan={dict(hand)}")
            print(f"  {tag:8} normal  : {dict(normal.most_common(4))}")
            print(f"  {tag:8} di-flip : {dict(flipped.most_common(4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
