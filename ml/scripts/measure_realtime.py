"""Ukur akurasi video per huruf lewat jalur kode realtime (opsi dapat diatur).

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\measure_realtime.py --model runs/handsc_fold_1/weights/best.pt --imgsz 1280 --no-hand-crop
    python scripts\\measure_realtime.py --model runs/comb6_fold_1/weights/best.pt --imgsz 640
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None,
                    help="kosongkan untuk memakai prioritas default find_best_pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-hand-crop", action="store_true")
    ap.add_argument("--no-hand-gate", action="store_true")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("outputs/measure_realtime.json"))
    args = ap.parse_args(argv)

    a = type("A", (), {})()
    a.model = args.model or rt.find_best_pt(1)
    a.conf = args.conf
    a.iou = 0.45
    a.onnx = False
    a.imgsz = args.imgsz
    predict = rt.make_predictor(a)
    focus = None if args.no_hand_gate else rt.make_hand_gate(
        Path("weights/hand_landmarker.task"), 0.3)

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT
                    and p.parent.name.upper() not in DYNAMIC)

    per_letter: dict[str, list[int]] = {}
    correct = total = blocked = 0
    for video in videos:
        truth = video.parent.name.upper()
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, count // (args.frames + 1))
        index = taken = 0
        votes: dict[str, int] = {}
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                target, offset = frame, (0, 0)
                if focus is not None:
                    focused = focus(frame)
                    if focused is None:
                        blocked += 1
                        taken += 1
                        index += 1
                        continue
                    if not args.no_hand_crop:
                        target, offset = focused
                dets = predict(target)
                top = max(dets, key=lambda d: d["conf"])["cls"] if dets else "(none)"
                votes[top] = votes.get(top, 0) + 1
                taken += 1
            index += 1
        cap.release()
        winner = max(votes, key=votes.get) if votes else "(none)"
        entry = per_letter.setdefault(truth, [0, 0])
        entry[1] += 1
        entry[0] += int(winner == truth)
        correct += int(winner == truth)
        total += 1
        print(f"  {truth:2} -> {winner:8} ({dict(votes)})", flush=True)

    summary = {
        "model": str(a.model), "imgsz": args.imgsz,
        "hand_crop": not args.no_hand_crop, "hand_gate": focus is not None,
        "video_accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct, "videos": total,
        "gate_blocked_frames": blocked,
        "per_letter": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_letter.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nakurasi video (voting) = {summary['video_accuracy']} ({correct}/{total})")
    failed = [k for k, v in per_letter.items() if v[0] < v[1]]
    print(f"huruf gagal: {' '.join(failed) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
