"""Uji false positive: gambar TANPA tangan tidak boleh dapat box.

Ini cek yang menangkap masalah "wajah terdeteksi sebagai huruf". Gambar tanpa
tangan = gambar sumber yang tidak punya label di dataset hasil relabel.

Baca : --src (dataset sumber, punya images/ + labels/)
       --with-hand (dataset relabel, isinya gambar bertangan)
Tulis: ringkasan JSON + montage kasus salah

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\check_false_positives.py --weights runs/hands_fold_1/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/hands_fold_1/weights/best.pt"))
    ap.add_argument("--src", type=Path, default=Path("data/processed/detect"),
                    help="dataset sumber (berisi gambar tanpa tangan)")
    ap.add_argument("--with-hand", type=Path, default=Path("data/processed/detect_hands"),
                    help="dataset relabel (isinya gambar bertangan)")
    ap.add_argument("--out", type=Path, default=Path("outputs/false_positive_check"))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"Bobot tidak ada: {args.weights}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    source_stems = {p.stem for p in (args.src / "images").glob("*")
                    if p.suffix.lower() in IMG_EXT}
    hand_stems = {p.stem for p in (args.with_hand / "images").glob("*")
                  if p.suffix.lower() in IMG_EXT}
    no_hand = sorted(source_stems - hand_stems)
    if not no_hand:
        raise SystemExit("Tidak ada gambar tanpa tangan untuk diuji")

    rng = random.Random(args.seed)
    picked = rng.sample(no_hand, min(args.samples, len(no_hand)))

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confs: list[float] = []
    tiles = []
    for stem in picked:
        path = args.src / "images" / f"{stem}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            continue
        source = stem.split("__")[0]
        result = model.predict(image, imgsz=640, conf=args.conf,
                               device=args.device, verbose=False)[0]
        stats[source][1] += 1
        if len(result.boxes):
            stats[source][0] += 1
            confs.append(float(max(result.boxes.conf)))
            if len(tiles) < 24:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls,
                                          result.boxes.conf):
                    x1, y1, x2, y2 = (int(v) for v in box.tolist())
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(image, f"{model.names[int(cls)]} {float(conf):.2f}",
                                (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2)
                tiles.append(cv2.resize(image, (280, 280)))

    total = sum(v[1] for v in stats.values())
    flagged = sum(v[0] for v in stats.values())
    summary = {
        "weights": str(args.weights),
        "no_hand_images_evaluated": total,
        "false_positive_images": flagged,
        "false_positive_rate": round(flagged / total, 4) if total else 0.0,
        "mean_false_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "per_source": {k: {"flagged": v[0], "n": v[1]} for k, v in sorted(stats.items())},
        "target": "false_positive_rate serendah mungkin; sebelumnya 0.76",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if tiles:
        width = 6
        rows = [cv2.hconcat(tiles[i:i + width]) for i in range(0, len(tiles), width)]
        if rows[-1].shape[1] != rows[0].shape[1]:
            pad = rows[0].shape[1] - rows[-1].shape[1]
            rows[-1] = cv2.copyMakeBorder(rows[-1], 0, 0, 0, pad,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(str(args.out / "false_positives.jpg"), cv2.vconcat(rows))
    print(json.dumps({k: summary[k] for k in
                      ("no_hand_images_evaluated", "false_positive_images",
                       "false_positive_rate", "mean_false_confidence")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
