"""Cek konsistensi label S/M antar sumber pada dataset box-tangan.

Kalau model benar untuk S dari satu sumber tapi salah untuk sumber lain,
berarti label S/M tidak konsisten antar sumber (masalah data, bukan tuning).

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\check_class_consistency.py --letters S M N A E
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/hands_fold_1/weights/best.pt"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/detect_hands"))
    ap.add_argument("--data-yaml", type=Path, default=Path("datasets/sibi/hands_run1.yaml"))
    ap.add_argument("--letters", nargs="+", default=["S", "M", "N", "A", "E"])
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("outputs/class_consistency.csv"))
    args = ap.parse_args(argv)

    names = {int(k): str(v).upper() for k, v in
             yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))["names"].items()}

    by_letter_source: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for image_path in sorted((args.dataset / "images").glob("*")):
        if image_path.suffix.lower() not in IMG_EXT:
            continue
        label = args.dataset / "labels" / f"{image_path.stem}.txt"
        if not label.is_file():
            continue
        lines = [l for l in label.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue
        letter = names.get(int(lines[0].split()[0]))
        if letter not in args.letters:
            continue
        source = image_path.stem.split("__")[0]
        by_letter_source[(letter, source)].append(image_path)

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    rng = random.Random(args.seed)

    rows = []
    for (letter, source), paths in sorted(by_letter_source.items()):
        pick = rng.sample(paths, min(args.samples, len(paths)))
        predictions: Counter[str] = Counter()
        for path in pick:
            image = cv2.imread(str(path))
            if image is None:
                continue
            boxes = model.predict(image, imgsz=640, conf=args.conf,
                                 device=args.device, verbose=False)[0].boxes
            top = model.names[int(boxes.cls[boxes.conf.argmax()])] if len(boxes) else "(none)"
            predictions[top] += 1
        total = sum(predictions.values())
        correct = predictions[letter]
        rows.append({"letter": letter, "source": source, "n": total,
                     "correct": correct,
                     "accuracy": round(correct / total, 3) if total else 0.0,
                     "top_wrong": ", ".join(f"{k}:{v}" for k, v in
                                            predictions.most_common(3) if k != letter)})
        print(f"  {letter:2} {source:18} benar {correct:3}/{total:3} "
              f"({rows[-1]['accuracy']:.2f})  salah jadi: {rows[-1]['top_wrong'] or '-'}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["letter", "source", "n", "correct",
                                                    "accuracy", "top_wrong"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
