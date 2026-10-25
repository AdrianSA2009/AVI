"""Status per huruf pada jalur realtime (gerbang + crop) untuk frame video SIBI.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\report_letters_realtime.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realtime_test as rt  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DYNAMIC = {"J", "Z"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("data/raw/Indonesian Sign Language System (SIBI) Dataset/"
                                 "sibi-dataset-dib-example-face-blurred/alphabet"))
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args(argv)

    class Args:
        pass

    a = Args()
    a.model = rt.find_best_pt(1)
    a.conf = args.conf
    a.iou = 0.45
    a.onnx = False
    predict = rt.make_predictor(a)
    focus = rt.make_hand_gate(Path("weights/hand_landmarker.task"), 0.3)

    videos = sorted(p for p in args.source.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT
                    and p.parent.name.upper() not in DYNAMIC)

    report = {}
    for video in videos:
        truth = video.parent.name.upper()
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, frame_count // (args.frames + 1))
        index = taken = 0
        correct = 0
        wrong = []
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                focused = focus(frame)
                if focused is not None:
                    crop, _ = focused
                    dets = predict(crop)
                    top = max(dets, key=lambda d: d["conf"])["cls"] if dets else "(none)"
                    if top == truth:
                        correct += 1
                    else:
                        wrong.append(top)
                    taken += 1
            index += 1
        cap.release()
        report[truth] = (correct, taken, wrong)

    print("Huruf  benar/total  salah jadi")
    for letter in sorted(report):
        correct, taken, wrong = report[letter]
        flag = "" if correct == taken else "   <-- MASALAH"
        print(f"  {letter:2}   {correct}/{taken}        {', '.join(wrong) or '-'}{flag}")
    failed = [l for l, (c, t, _) in report.items() if c < t]
    print(f"\nHuruf belum benar penuh: {' '.join(failed) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
