"""Uji detector YOLO gambar pada video alfabet: sampling frame per klip.

Input layout: data/video_test/alphabet/<huruf>/*.{mp4,mov,...}
Model: best.pt fold_1 (fallback fold_2); J/Z dilaporkan out-of-scope karena
model gambar dilatih tanpa kelas dinamis tersebut.

Mengukur video-level majority vote dari frame sampel (bukan realtime temporal
recognition). Pisahkan interpretasi: video-level accuracy, frame detection rate,
coverage, confusion matrix. Hanya gunakan sebagai evaluasi lintas-sumber jika
video tidak menjadi sumber data training.

Jalankan dari C:\\AVI\\ml:
    python scripts\\test_yolo_videos.py
    python scripts\\test_yolo_videos.py --frames 5 --conf 0.25
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def model_path() -> Path:
    for path in (
        Path("runs/detect/runs/detect/fold_1/weights/best.pt"),
        Path("runs/detect/runs/detect/fold_2/weights/best.pt"),
    ):
        if path.is_file():
            return path
    raise SystemExit("YOLO best.pt tidak ditemukan")


def sample_frames(path: Path, count: int) -> list:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    positions = sorted(set(int(x) for x in
                           __import__("numpy").linspace(0, n - 1, min(count, n))))
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/video_test/alphabet"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path,
                        default=Path("runs/audit/yolo_video_test.json"))
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames harus >= 1")
    if not args.data.is_dir():
        raise SystemExit(f"Folder test tidak ada: {args.data}")

    checkpoint = args.model or model_path()
    model = YOLO(str(checkpoint))
    names = {int(k): str(v).upper() for k, v in model.names.items()}
    valid = set(names.values())
    per_class: dict[str, dict] = {}
    matrix: Counter[tuple[str, str]] = Counter()
    all_correct = all_test = covered_correct = covered_total = 0
    excluded: Counter[str] = Counter()
    unreadable = 0

    videos = sorted(p for p in args.data.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    for i, video in enumerate(videos, start=1):
        # Struktur kategori/huruf/arsip-subfolder/klip — label huruf adalah parent
        # kedua relatif dari root; arsip Roboflow menyisipkan satu subfolder uppercase.
        rel = video.relative_to(args.data)
        expected = rel.parts[0].upper() if len(rel.parts) >= 2 else rel.parent.name.upper()
        if expected not in valid:
            excluded[expected] += 1
            continue
        frames = sample_frames(video, args.frames)
        if not frames:
            unreadable += 1
            continue
        results = model.predict(frames, imgsz=640, conf=args.conf,
                                device="0", verbose=False)
        votes: Counter[str] = Counter()
        detected = 0
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            best_idx = int(result.boxes.conf.argmax())
            cls = int(result.boxes.cls[best_idx])
            label = names.get(cls, "?")
            votes[label] += 1
            detected += 1
        pred = votes.most_common(1)[0][0] if votes else "NO_DETECTION"
        all_test += 1
        covered_total += 1
        if detected:
            covered_correct += int(pred == expected)
        all_correct += int(pred == expected)
        matrix[(expected, pred)] += 1
        entry = per_class.setdefault(expected, {"videos": 0, "correct": 0,
                                                "detections": 0, "frames": 0})
        entry["videos"] += 1
        entry["correct"] += int(pred == expected)
        entry["detections"] += detected
        entry["frames"] += len(frames)
        if i % 50 == 0:
            print(f"video {i}/{len(videos)}", flush=True)

    report = {
        "checkpoint": str(checkpoint), "source": str(args.data),
        "sampled_frames_per_video": args.frames, "confidence": args.conf,
        "test_videos_in_scope": all_test,
        "video_level_accuracy_including_no_detection":
            all_correct / all_test if all_test else None,
        "video_level_accuracy_when_any_frame_detected":
            covered_correct / covered_total if covered_total else None,
        "per_class": per_class,
        "confusion_matrix_counts": {
            truth: dict(Counter({pred: n for (t, pred), n in matrix.items() if t == truth}))
            for truth in sorted({t for t, _ in matrix})
        },
        "out_of_scope_classes": dict(excluded),
        "unreadable_videos": unreadable,
        "caveat": "Frame-sampled YOLO detector, not video-sequence model. "
                  "Score is independent only if test clips were never used in training.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Video diuji: {all_test}; akurasi video termasuk no-detection: "
          f"{report['video_level_accuracy_including_no_detection']}")
    print(f"Kelas di luar model: {dict(excluded)}; tidak terbaca: {unreadable}")
    print(f"Laporan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
