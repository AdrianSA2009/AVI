"""Uji cepat model box-tangan pada frame video SIBI: render overlay + ringkas kelas.

Baca : folder video berlabel kelas (nama subfolder = huruf)
Tulis: <out>/overlay.jpg (montage) dan <out>/report.json

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\check_video_hands.py --weights runs/hands_fold_1/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}
IMGSZ = 640


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--weights", type=Path,
                    default=Path("runs/hands_fold_1/weights/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/video_hands_check"))
    ap.add_argument("--frames", type=int, default=3, help="frame per video")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"Bobot tidak ada: {args.weights}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    tiles = []
    for video in videos:
        truth = video.parent.name.upper()
        if truth in DYNAMIC:
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        picked = 0
        index = 0
        while picked < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if total > args.frames:
                step = max(1, total // (args.frames + 1))
                if index % step or index == 0:
                    index += 1
                    continue
            result = model.predict(frame, imgsz=IMGSZ, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            boxes = result.boxes
            classes = [model.names[int(c)] for c in boxes.cls] if boxes is not None else []
            confs = [float(c) for c in boxes.conf] if boxes is not None else []
            top = classes[confs.index(max(confs))] if confs else ""
            rows.append({"video": video.name, "truth": truth, "frame": index,
                         "predicted": top, "classes": classes,
                         "max_conf": round(max(confs), 4) if confs else 0.0,
                         "n_boxes": len(classes)})
            annotated = frame.copy()
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = (int(v) for v in box.tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"{model.names[int(cls)]} {float(conf):.2f}",
                            (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)
            cv2.putText(annotated, f"GT {truth}", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            tiles.append(cv2.resize(annotated, (320, 320)))
            picked += 1
            index += 1
        cap.release()

    if tiles:
        width = 4
        rows_img = [cv2.hconcat(tiles[i:i + width])
                    for i in range(0, len(tiles), width)]
        if rows_img[-1].shape[1] != rows_img[0].shape[1]:
            pad = rows_img[0].shape[1] - rows_img[-1].shape[1]
            rows_img[-1] = cv2.copyMakeBorder(
                rows_img[-1], 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(str(args.out / "overlay.jpg"), cv2.vconcat(rows_img))

    correct = sum(1 for r in rows if r["predicted"] == r["truth"])
    summary = {
        "videos": len({r["video"] for r in rows}),
        "frames": len(rows),
        "frames_with_box": sum(1 for r in rows if r["n_boxes"]),
        "top1_correct": correct,
        "top1_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "wrong": [{"truth": r["truth"], "predicted": r["predicted"],
                   "video": r["video"]} for r in rows if r["predicted"] != r["truth"]],
        "truth_counts": dict(Counter(r["truth"] for r in rows)),
        "overlay": str(args.out / "overlay.jpg"),
    }
    (args.out / "report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("videos", "frames", "frames_with_box",
                       "top1_accuracy")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
