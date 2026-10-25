"""Training 2-Fold Cross-Validation YOLO11n (Jalur A, deteksi gestur statis).

Baca : datasets/sibi/folds.json (hasil make_folds.py) + data/processed/detect/
Tulis: runs/detect/fold_1, fold_2   (model, log, hasil val per fold)
       runs/metrics_cv.json          mAP/P/R/F1 + FPS per fold, rata-rata +/- std

Catatan hardware: RTX 3070 Laptop 8 GB — device=0, AMP aktif.
Kalau OOM: turunkan --batch dulu, lalu --imgsz.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 30 --batch 16 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml


def check_gpu() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak aktif — cek instalasi torch +cu128")
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | torch {torch.__version__}")
    return "0"


def write_dataset_yaml(folds: dict, manifest: list[dict], run: int,
                       out_dir: Path, prefix: str = "",
                       class_map_path: Path | None = None) -> Path:
    """Buat yaml YOLO untuk 1 run: train = fold_X, val = fold_Y (sebaliknya)."""
    train_key, val_key = (f"fold_{run}", f"fold_{3 - run}")
    for key in (train_key, val_key):
        # absolut supaya tak bergantung ke 'path' di yaml
        abs_paths = [Path(p).resolve().as_posix() for p in folds[key]]
        (out_dir / f"{prefix}run{run}_{key}.txt").write_text(
            "\n".join(abs_paths) + "\n", encoding="utf-8")
    letters = sorted({it["kelas"] for it in manifest if it["kelas"]})
    if class_map_path:
        class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        names = {int(k): str(v) for k, v in class_map.items()}
        if set(names.values()) != set(letters) or len(names) != len(letters):
            raise SystemExit("--class-map must map every manifest class exactly once")
        if set(names) != set(range(len(names))):
            raise SystemExit("--class-map IDs must be contiguous from 0")
    else:
        names = {i: letter for i, letter in enumerate(letters)}
    data = {
        "path": out_dir.resolve().as_posix(),
        "train": f"{prefix}run{run}_{train_key}.txt",
        "val": f"{prefix}run{run}_{val_key}.txt",
        "names": names,
    }
    yaml_path = out_dir / f"{prefix}run{run}.yaml"
    yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def eval_fold(model_path: Path) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    r = m.val(verbose=False, device="0")
    p, rec = float(r.box.mp), float(r.box.mr)
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "precision": p,
        "recall": rec,
        "F1": f1,
        "ms_per_img": float(r.speed["inference"]),
        "names": {int(k): v for k, v in r.names.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=Path, default=Path("datasets/sibi/folds.json"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_detect.json"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15,
                    help="epoch tanpa perbaikan sebelum berhenti")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/sibi"),
                    help="folder yaml + daftar gambar per split")
    ap.add_argument("--prefix", default="", help="awalan nama yaml/run")
    ap.add_argument("--metrics-out", type=Path, default=Path("runs/metrics_cv.json"),
                    help="ringkasan metrik; default menimpa baseline gambar lama")
    ap.add_argument("--resume", action="store_true",
                    help="lanjutkan run parsial dari weights/last.pt bila ada")
    ap.add_argument("--run-folds", default="1,2",
                    help="fold yang dijalankan, mis. '1' atau '1,2'")
    ap.add_argument("--aug", default="",
                    help="augmentasi tambahan, mis. 'scale=0.9,hsv_v=0.6,degrees=15'")
    ap.add_argument("--class-map", type=Path, default=None,
                    help="JSON id->class map untuk mempertahankan ID pretrained")
    args = ap.parse_args(argv)

    device = check_gpu()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_dir = args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    results = []
    run_folds = [int(x) for x in str(args.run_folds).split(",") if x.strip()]
    for run in run_folds:
        name = f"{args.prefix}fold_{run}"
        last = Path("runs") / name / "weights" / "last.pt"
        print(f"\n===== FOLD RUN {run} "
              f"(train=fold_{run}, val=fold_{3 - run}) =====")
        if args.resume and last.is_file():
            print(f"RESUME dari {last}")
            model = YOLO(str(last))
            model.train(resume=True, device=device, workers=args.workers)
        else:
            data_yaml = write_dataset_yaml(folds, manifest, run, out_dir, args.prefix,
                                           args.class_map)
            model = YOLO(args.model)
            extra = {}
            for item in str(args.aug).split(","):
                if "=" in item:
                    key, _, value = item.partition("=")
                    extra[key.strip()] = float(value)
            model.train(
                data=str(data_yaml), epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, amp=True, device=device,
                workers=args.workers, seed=42, patience=args.patience,
                project=str(Path("runs").resolve()),  # absolut: hindari path nested
                name=name, exist_ok=True, **extra,
            )
        metrics = eval_fold(Path(model.trainer.best))
        results.append(metrics)
        print(f"  mAP50={metrics['mAP50']:.4f} "
              f"mAP50-95={metrics['mAP50_95']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['F1']:.4f} "
              f"({metrics['ms_per_img']:.1f} ms/img)")

    keys = ["mAP50", "mAP50_95", "precision", "recall", "F1", "ms_per_img"]
    summary = {
        "per_fold": results,
        "rata2": {k: statistics.mean(r[k] for r in results) for k in keys},
        "std": {k: statistics.pstdev(r[k] for r in results) for k in keys},
        "config": {"model": args.model, "epochs": args.epochs,
                   "batch": args.batch, "imgsz": args.imgsz, "seed": 42,
                   "patience": args.patience},
    }
    Path("runs").mkdir(exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== RATA-RATA 2 FOLD =====")
    for k in keys:
        print(f"  {k}: {summary['rata2'][k]:.4f} +/- {summary['std'][k]:.4f}")
    print(f"Metrik: {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
