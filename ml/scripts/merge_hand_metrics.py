"""Gabungkan metrik 2 fold model box-tangan menjadi satu ringkasan.

Baca : runs/metrics_cv_hands_fold1.json, runs/metrics_cv_hands_fold2.json
Tulis: runs/metrics_cv_hands.json (rata-rata +/- std)

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\merge_hand_metrics.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEYS = ["mAP50", "mAP50_95", "precision", "recall", "F1", "ms_per_img"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold1", type=Path, default=Path("runs/metrics_cv_hands_fold1.json"))
    ap.add_argument("--fold2", type=Path, default=Path("runs/metrics_cv_hands_fold2.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/metrics_cv_hands.json"))
    args = ap.parse_args(argv)

    folds = []
    for path in (args.fold1, args.fold2):
        if not path.is_file():
            raise SystemExit(f"Metrik fold tidak ada: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        folds.append(data["per_fold"][0])

    summary = {
        "per_fold": folds,
        "rata2": {k: statistics.mean(f[k] for f in folds) for k in KEYS},
        "std": {k: statistics.pstdev(f[k] for f in folds) for k in KEYS},
        "config": {
            "dataset": "data/processed/detect_hands (box tangan MediaPipe)",
            "folds": "datasets/sibi/hands_folds.json",
            "epochs": 50,
            "patience": 12,
            "batch": 16,
            "imgsz": 640,
            "seed": 42,
            "model": "yolo11n.pt",
            "device": 0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    for key in KEYS:
        print(f"{key:12} {summary['rata2'][key]:.4f} +/- {summary['std'][key]:.4f}")
    print(f"Ringkasan: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
