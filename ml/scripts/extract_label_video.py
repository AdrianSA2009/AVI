"""Extract frames and pseudo-label video frames for static SIBI YOLO fine-tuning.

No training occurs. Original videos and weights remain untouched. Frames whose
prior YOLO model returns no sufficiently confident box are copied to review/.
For accepted boxes, preserve predicted coordinates but force GT class from the
video folder (alphabet letters; J/Z excluded because dynamic and unsupported).

PowerShell, from C:\\AVI\\ml:
    python scripts\\extract_label_video.py
    python scripts\\extract_label_video.py --fps-sample 4 --conf 0.15 --blur-threshold 30
    python scripts\\extract_label_video.py --visualize 60

Output under datasets/video_yolo/: images/, labels/, review/, metadata.csv,
review_samples.jpg. Training/splits deliberately not started until manual review.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}


def d_hash(gray: np.ndarray) -> int:
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(tiny[y, x] > tiny[y, x + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("data/video_test/alphabet"))
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/detect/runs/detect/fold_1/weights/best.pt"))
    ap.add_argument("--data-yaml", type=Path, default=Path("datasets/sibi/run1.yaml"))
    ap.add_argument("--out", type=Path, default=Path("datasets/video_yolo"))
    ap.add_argument("--fps-sample", type=float, default=3.0)
    ap.add_argument("--blur-threshold", type=float, default=30.0)
    ap.add_argument("--dhash-distance", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--visualize", type=int, default=60,
                    help="random annotated samples in montage")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak aktif; cek torch +cu128")
    if not args.weights.is_file():
        raise SystemExit(f"Weights tidak ditemukan: {args.weights}")
    if args.fps_sample <= 0:
        raise SystemExit("--fps-sample harus > 0")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    data = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    names = {int(k): str(v).upper() for k, v in data["names"].items()}
    name_to_id = {v: k for k, v in names.items()}
    model = YOLO(str(args.weights))

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"Video tidak ditemukan: {args.source}")
    img_dir, label_dir, review_dir = (args.out / "images", args.out / "labels",
                                      args.out / "review")
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    review_rows: list[dict] = []
    dup_hashes: dict[str, list[int]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for vi, video in enumerate(videos):
        rel = video.relative_to(args.source)
        letter = rel.parts[0].upper() if len(rel.parts) >= 2 else ""
        if letter in DYNAMIC or letter not in name_to_id:
            counts[f"excluded_{letter or 'unknown'}"] += 1
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            counts["video_unreadable"] += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        interval = max(1, round(fps / args.fps_sample))
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = frame_no
            frame_no += 1
            if current % interval:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < args.blur_threshold:
                counts["blur"] += 1
                continue
            ph = d_hash(gray)
            group = str(video)
            if any(hamming(ph, old) <= args.dhash_distance for old in dup_hashes[group]):
                counts["near_duplicate"] += 1
                continue
            result = model.predict(frame, device=0, conf=args.conf,
                                   verbose=False, imgsz=640)[0]
            stem = f"v{vi:04d}_f{current:06d}"
            if result.boxes is None or len(result.boxes) == 0:
                dest = review_dir / f"{stem}.jpg"
                cv2.imwrite(str(dest), frame)
                review_rows.append({"image": str(dest), "video": str(video),
                                    "frame": current, "truth": letter,
                                    "blur": blur, "reason": "no_box_over_conf"})
                counts["review_no_box"] += 1
                continue
            bi = int(result.boxes.conf.argmax())
            conf = float(result.boxes.conf[bi])
            # Keep only best box; use video-folder ground truth, never predicted class.
            xywhn = result.boxes.xywhn[bi].cpu().tolist()
            label_text = " ".join([str(name_to_id[letter]),
                                   *(f"{v:.6f}" for v in xywhn)])
            out_image = img_dir / f"{stem}.jpg"
            out_label = label_dir / f"{stem}.txt"
            cv2.imwrite(str(out_image), frame)
            out_label.write_text(label_text + "\n", encoding="utf-8")
            if conf < args.conf:
                review_rows.append({"image": str(out_image), "video": str(video),
                                    "frame": current, "truth": letter,
                                    "blur": blur, "reason": "low_confidence"})
                counts["review_low_conf"] += 1
            else:
                rows.append({"image": str(out_image), "label": str(out_label),
                             "video": str(video), "frame": current,
                             "label_class": letter, "box_conf": conf,
                             "blur_laplacian": blur, "fps": fps})
                counts["accepted"] += 1
                dup_hashes[group].append(ph)
        cap.release()
        if (vi + 1) % 25 == 0:
            print(f"Video {vi + 1}/{len(videos)}", flush=True)

    fields = ["image", "label", "video", "frame", "label_class", "box_conf",
              "blur_laplacian", "fps"]
    with (args.out / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (args.out / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "video", "frame", "truth", "blur", "reason"])
        writer.writeheader(); writer.writerows(review_rows)

    # Seeded montage: accepted box overlays, randomly sampled for manual audit.
    rng = random.Random(args.seed)
    picks = rng.sample(rows, min(args.visualize, len(rows))) if rows else []
    if picks:
        cols, tile = 6, 180
        cell_h = tile + 28
        canvas = Image.new("RGB", (cols * tile, ((len(picks) + cols - 1) // cols) * cell_h), "black")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(picks):
            img = cv2.imread(row["image"])
            lab = Path(row["label"]).read_text(encoding="utf-8").split()
            h, w = img.shape[:2]
            _, xc, yc, bw, bh = map(float, lab)
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb = cv2.cvtColor(cv2.resize(img, (tile, tile)), cv2.COLOR_BGR2RGB)
            x, y = (i % cols) * tile, (i // cols) * cell_h
            canvas.paste(Image.fromarray(rgb), (x, y))
            draw.text((x + 2, y + tile + 2),
                      f"{row['label_class']} conf={row['box_conf']:.2f}", fill="white")
        canvas.save(args.out / "review_samples.jpg", quality=88)

    report = {"source": str(args.source), "weights": str(args.weights),
              "videos_seen": len(videos), "accepted_pseudo_labels": len(rows),
              "review_items": len(review_rows), "counts": dict(counts),
              "class_distribution": dict(Counter(r["label_class"] for r in rows)),
              "training_started": False,
              "caveat": "Boxes auto-predicted; assigned class forced to folder label. "
                        "Manually audit/correct labels.csv and review_samples.jpg before use."}
    (args.out / "extraction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("No training started; review pseudo-labels first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
