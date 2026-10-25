"""Bandingkan model pada frame video SIBI lewat jalur realtime (gerbang + crop).

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\compare_realtime_models.py --models runs/comb_fold_1/weights/best.pt runs/hands_fold_1/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realtime_test as rt  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}
PROBLEM = ["S", "D", "H", "P", "T", "W"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=[
                        "runs/hands_fold_1/weights/best.pt:640:crop",
                        "runs/comb6_fold_1/weights/best.pt:640:crop",
                        "runs/handsc_fold_1/weights/best.pt:1280:full",
                        "runs/comb_fold_1/weights/best.pt:640:crop",
                    ], help="format: path:imgsz:full|crop")
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=Path("outputs/realtime_model_compare.json"))
    args = ap.parse_args(argv)

    focus = rt.make_hand_gate(Path("weights/hand_landmarker.task"), 0.3)

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT
                    and p.parent.name.upper() not in DYNAMIC)

    frames: list[tuple[str, object]] = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // (args.frames + 1))
        index = taken = 0
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                frames.append((video.parent.name.upper(), frame))
                taken += 1
            index += 1
        cap.release()

    results = {}
    for spec in args.models:
        weights, size, form = spec.rsplit(":", 2)
        if not Path(weights).is_file():
            results[spec] = {"error": "tidak ada"}
            continue
        a = type("A", (), {})()
        a.model, a.conf, a.iou, a.onnx = Path(weights), args.conf, 0.45, False
        a.imgsz = int(size)
        predict = rt.make_predictor(a)
        per_letter: dict[str, list[int]] = {}
        correct = total_n = 0
        for truth, frame in frames:
            target = frame
            if form == "crop":
                focused = focus(frame)
                if focused is None:
                    total_n += 1
                    per_letter.setdefault(truth, [0, 0])[1] += 1
                    continue
                target, _ = focused
            dets = predict(target)
            top = max(dets, key=lambda d: d["conf"])["cls"] if dets else ""
            total_n += 1
            correct += int(top == truth)
            e = per_letter.setdefault(truth, [0, 0])
            e[1] += 1
            e[0] += int(top == truth)
        results[spec] = {
            "accuracy": round(correct / total_n, 4) if total_n else 0.0,
            "correct": correct,
            "frames": total_n,
            "failed_letters": [k for k, v in sorted(per_letter.items()) if v[0] < v[1]],
            "per_letter": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_letter.items())},
        }
        print(f"{spec:52} acc={results[spec]['accuracy']:.4f} ({correct}/{total_n}) "
              f"gagal={' '.join(results[spec]['failed_letters']) or '-'}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    best = max(results, key=lambda k: results[k].get("accuracy", 0))
    print(f"\nTERBAIK: {best} acc={results[best]['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
