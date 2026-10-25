"""Evaluasi checkpoint deployment pada P holdout dan frame video SIBI.

Foto P diukur pada split valid/test raw yang tidak masuk manifest training.
Video SIBI hanya satu signer/klip per kelas; hasilnya sanity-check in-domain,
bukan bukti generalisasi antarsigner.

Pemakaian:
    python scripts\\evaluate_deploy.py --weights runs\\augp_deploy\\deploy.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("runs/augp_deploy/deploy.pt"))
    ap.add_argument("--p-root", type=Path, default=Path("data/raw/SIBI_DATASET"))
    ap.add_argument("--video-root", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--out", type=Path, default=Path("outputs/deploy_evaluation.json"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint tidak ada: {args.weights}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    p_results = {}
    for split in ("valid", "test"):
        files = sorted((args.p_root / split / "P").glob("*.jpg"))
        pred = Counter()
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                pred["(unreadable)"] += 1
                continue
            boxes = model.predict(image, imgsz=args.imgsz, conf=args.conf,
                                  device=args.device, verbose=False)[0].boxes
            if len(boxes):
                idx = int(boxes.conf.argmax())
                pred[model.names[int(boxes.cls[idx])]] += 1
            else:
                pred["(none)"] += 1
        p_results[split] = {
            "n": len(files),
            "predictions": dict(pred),
            "P_correct": pred["P"],
            "P_recall": round(pred["P"] / len(files), 4) if files else 0.0,
        }

    videos = sorted(p for p in args.video_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT
                    and p.parent.name.upper() not in DYNAMIC)
    rows = []
    correct = frame_count = with_box = 0
    for video in videos:
        truth = video.parent.name.upper()
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            continue
        indices = sorted(set([1, total // 3, 2 * total // 3, max(1, total - 2)]))
        index_set = set(indices)
        frame_idx = 0
        preds = Counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in index_set:
                boxes = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                      device=args.device, verbose=False)[0].boxes
                if len(boxes):
                    i = int(boxes.conf.argmax())
                    pred = model.names[int(boxes.cls[i])]
                    with_box += 1
                else:
                    pred = "(none)"
                preds[pred] += 1
                frame_count += 1
                correct += int(pred == truth)
            frame_idx += 1
        cap.release()
        winner = preds.most_common(1)[0][0] if preds else "(none)"
        rows.append({"truth": truth, "video": video.name,
                     "video_vote": winner, "correct": winner == truth,
                     "frame_predictions": dict(preds)})

    result = {
        "weights": str(args.weights),
        "p_raw_holdout": p_results,
        "video_in_domain_sanity_check": {
            "videos": len(rows),
            "sampled_frames": frame_count,
            "frames_with_box": with_box,
            "frame_accuracy": round(correct / frame_count, 4) if frame_count else 0.0,
            "warning": "satu signer/satu video per kelas; bukan generalisasi antarsigner",
            "per_video": rows,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("P holdout:")
    for split, data in p_results.items():
        print(f"  {split}: P={data['P_correct']}/{data['n']} ({data['P_recall']:.3f}) {data['predictions']}")
    v = result["video_in_domain_sanity_check"]
    print(f"video: {v['frame_accuracy']:.3f} ({v['sampled_frames']} frame), "
          f"box={v['frames_with_box']}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
