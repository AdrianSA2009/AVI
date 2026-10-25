"""Uji headless logika realtime_test.py: gerbang + crop + prediksi.

Memastikan crop/gerbang bekerja tanpa membuka kamera, dan memverifikasi
akurasi video SIBI memakai jalur kode yang sama dengan aplikasi realtime.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\test_realtime_pipeline.py
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
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

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
    correct = total = 0
    blocked = 0
    for video in videos:
        truth = video.parent.name.upper()
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, frame_count // (args.frames + 1))
        index = taken = 0
        while taken < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index > 0 and index % step == 0:
                focused = focus(frame)
                total += 1
                if focused is None:
                    blocked += 1
                else:
                    crop, (ox, oy) = focused
                    dets = predict(crop)
                    for d in dets:
                        assert 0 <= d["x1"] + ox <= frame.shape[1] + ox, "offset aneh"
                    top = max(dets, key=lambda d: d["conf"])["cls"] if dets else ""
                    correct += int(top == truth)
                taken += 1
            index += 1
        cap.release()

    print(f"frame dievaluasi: {total}")
    print(f"diblokir gerbang (tak ada tangan): {blocked}")
    print(f"akurasi top-1 jalur realtime (crop): {correct / total if total else 0:.4f}")
    print(f"model: {a.model}")
    assert total > 0, "tidak ada frame dievaluasi"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
