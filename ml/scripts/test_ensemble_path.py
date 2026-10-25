"""Uji jalur ensemble di realtime_test.py pada frame video (bukan tulis ulang).

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\test_ensemble_path.py
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
MEMBERS = [
    "runs/hands_fold_1/weights/best.pt:640:crop",
    "runs/handsc_fold_1/weights/best.pt:1280:full",
    "runs/comb6_fold_1/weights/best.pt:640:crop",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=Path("outputs/test_ensemble_path.json"))
    args = ap.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions, vision

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="weights/hand_landmarker.task"),
            num_hands=1, min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3))
    predict = rt.make_ensemble_predictor(MEMBERS, args.conf, detector)

    per_letter = {}
    correct = total = 0
    for video in sorted(p for p in args.source.rglob("*")
                        if p.is_file() and p.suffix.lower() in VIDEO_EXT
                        and p.parent.name.upper() not in DYNAMIC):
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
                dets = predict(frame)
                votes[dets[0]["cls"] if dets else "(none)"] += 1
                taken += 1
            index += 1
        cap.release()
        winner = votes.most_common(1)[0][0] if votes else "(none)"
        per_letter[truth] = winner
        correct += int(winner == truth)
        total += 1
        print(f"  {truth:2} -> {winner:8} {'OK' if winner == truth else '<-- salah'}")

    summary = {"accuracy": round(correct / total, 4) if total else 0.0,
               "correct": correct, "videos": total,
               "per_letter": per_letter,
               "correct_letters": sorted(k for k, v in per_letter.items() if k == v),
               "wrong_letters": sorted(k for k, v in per_letter.items() if k != v),
               "members": MEMBERS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nakurasi = {summary['accuracy']} ({correct}/{total})")
    print(f"benar: {' '.join(summary['correct_letters'])}")
    print(f"salah: {' '.join(summary['wrong_letters'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
