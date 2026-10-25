"""Uji akhir: konfigurasi realtime baru pada frame webcam + semua huruf video.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\verify_final_config.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realtime_test as rt  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-hand-crop", action="store_true")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--webcam-dir", type=Path, default=Path("data/webcam_test"))
    ap.add_argument("--out", type=Path, default=Path("outputs/verify_final.json"))
    args = ap.parse_args(argv)

    a = type("A", (), {})()
    a.model = args.model or rt.find_best_pt(1)
    a.conf = args.conf
    a.iou = 0.45
    a.onnx = False
    a.imgsz = args.imgsz
    predict = rt.make_predictor(a)
    focus = rt.make_hand_gate(Path("weights/hand_landmarker.task"), 0.3)

    print(f"model: {a.model}\nimgsz: {args.imgsz}  crop: {not args.no_hand_crop}\n")

    # 1. frame webcam nyata
    print("=== frame webcam pengguna ===")
    webcam = []
    for path in sorted(args.webcam_dir.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        focused = focus(image)
        if focused is None:
            print(f"  {path.name}: gerbang blokir")
            continue
        target, offset = focused
        if args.no_hand_crop:
            target = image
            offset = (0, 0)
        dets = predict(target)
        if offset != (0, 0):
            ox, oy = offset
            for d in dets:
                d["x1"] += ox; d["x2"] += ox
                d["y1"] += oy; d["y2"] += oy
        top = max(dets, key=lambda d: d["conf"]) if dets else None
        webcam.append({"image": path.name, "result": top["cls"] if top else "(none)",
                       "conf": round(top["conf"], 3) if top else 0.0})
        print(f"  {path.name}: {top['cls'] if top else '(none)'} "
              f"{top['conf'] if top else 0:.2f}")

    # 2. video
    print("\n=== huruf video SIBI (voting 6 frame) ===")
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
        votes: Counter[str] = Counter()
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                target = frame
                if not args.no_hand_crop:
                    focused = focus(frame)
                    if focused is None:
                        blocked += 1
                        taken += 1
                        index += 1
                        continue
                    target, _ = focused
                dets = predict(target)
                votes[max(dets, key=lambda d: d["conf"])["cls"] if dets else "(none)"] += 1
                taken += 1
            index += 1
        cap.release()
        winner = votes.most_common(1)[0][0] if votes else "(none)"
        entry = per_letter.setdefault(truth, [0, 0])
        entry[1] += 1
        entry[0] += int(winner == truth)
        correct += int(winner == truth)
        total += 1
        mark = "" if winner == truth else "  <--"
        print(f"  {truth:2} -> {winner:8} {dict(votes)}{mark}")

    summary = {
        "model": str(a.model), "imgsz": args.imgsz,
        "hand_crop": not args.no_hand_crop,
        "webcam_results": webcam,
        "video_accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct, "videos": total,
        "gate_blocked": blocked,
        "per_letter": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_letter.items())},
        "failed": [k for k, v in sorted(per_letter.items()) if v[0] < v[1]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nakurasi video (voting) = {summary['video_accuracy']} ({correct}/{total})")
    print(f"huruf gagal: {' '.join(summary['failed']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
