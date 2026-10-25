"""Evaluasi checkpoint video pada folder klip berlabel, tanpa mengubah checkpoint.

Folder test harus punya struktur <kategori>/<kelas>/*.mp4, misalnya:
    data/video_test/alphabet/H/klip_01.mp4
    data/video_test/number/10/klip_01.mp4

Prediksi hanya dihitung untuk kelas yang ada di classes.json. Video pada
folder test harus klip baru dan tidak dipakai saat training; bila tidak ada,
program melaporkan hasilnya sebagai evaluasi eksploratif, bukan generalisasi.

Jalankan dari C:\\AVI\\ml:
    python scripts\\test_video.py --data data/video_test
    python scripts\\test_video.py --data "C:\\path\\to\\video_test"

Output: runs/video_sibi/test_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_video import ClipClassifier, VideoDataset

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def read_records(root: Path, classes: list[str], categories: list[str] | None = None
                 ) -> tuple[list[tuple[Path, int]], list[str]]:
    class_to_id = {label: i for i, label in enumerate(classes)}
    records: list[tuple[Path, int]] = []
    unknown: list[str] = []
    for path in sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in VIDEO_EXT):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            unknown.append(str(rel))
            continue
        if categories and rel.parts[0].lower() not in categories:
            continue
        label = f"{rel.parts[0].lower()}:{rel.parts[1].lower()}"
        if label not in class_to_id:
            unknown.append(f"{rel} (label {label!r} tidak ada di checkpoint)")
            continue
        records.append((path, class_to_id[label]))
    return records, unknown


def metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    matrix = [[0 for _ in classes] for _ in classes]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[name] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(len(classes)))
    present = [v for v in per_class.values() if v["support"]]
    return {
        "videos": total,
        "accuracy": round(correct / total, 4) if total else None,
        "macro_precision": round(sum(v["precision"] for v in present) / len(present), 4)
            if present else None,
        "macro_recall": round(sum(v["recall"] for v in present) / len(present), 4)
            if present else None,
        "macro_f1": round(sum(v["f1"] for v in present) / len(present), 4)
            if present else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="folder test: kategori/kelas/clip.mp4")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/video_sibi/best.pt"))
    parser.add_argument("--classes", type=Path,
                        default=Path("runs/video_sibi/classes.json"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--categories", nargs="+", choices=("alphabet", "number", "affix"),
                        default=None, help="pilih kategori yang diuji")
    parser.add_argument("--out", type=Path,
                        default=Path("runs/video_sibi/test_report.json"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Folder test tidak ada: {args.data}")
    if not args.checkpoint.is_file() or not args.classes.is_file():
        raise SystemExit("Checkpoint/classes.json tidak ditemukan; jalankan "
                         "scripts\\train_video.py terlebih dahulu")

    class_map = json.loads(args.classes.read_text(encoding="utf-8"))
    classes = [class_map[str(i)] for i in range(len(class_map))]
    records, unknown = read_records(args.data, classes, args.categories)
    if not records:
        raise SystemExit("Tidak ada video berlabel yang cocok. "
                         "Gunakan struktur kategori/kelas/video.mp4")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ClipClassifier(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    loader = DataLoader(VideoDataset(records), batch_size=args.batch,
                        shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for clips, targets in loader:
            logits = model(clips.to(args.device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.tolist())

    report = metrics(y_true, y_pred, classes)
    report["test_folder"] = str(args.data)
    report["checkpoint"] = str(args.checkpoint)
    report["unknown_or_skipped"] = unknown
    report["caveat"] = (
        "Nilai hanya mengukur generalisasi bila klip test dan signer tidak "
        "muncul pada training. Video test yang sama dengan training tidak sah."
    )
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Video diuji: {report['videos']}")
    print(f"Akurasi: {report['accuracy']} | macro F1: {report['macro_f1']}")
    for name, item in report["per_class"].items():
        if item["support"]:
            print(f"  {name}: {item['support']} klip, "
                  f"P={item['precision']} R={item['recall']} F1={item['f1']}")
    if unknown:
        print(f"Dilewati (struktur/label tak dikenal): {len(unknown)}")
    print(f"Laporan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
