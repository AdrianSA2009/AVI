"""Latih satu checkpoint deployment dari seluruh manifest terkurasi.

Ini bukan validasi silang: seluruh manifest dipakai untuk training, val=False.
Default memakai seluruh data static26 A-Z dan checkpoint output baru; jangan timpa
checkpoint deployment 24 kelas. Jangan laporkan metrik training sebagai akurasi generalisasi. Verifikasi model
terpisah lewat webcam/video sesudah training.

Output tidak menimpa fold CV:
  <output>/weights/last.pt (checkpoint penuh)
  <output>/args.yaml, results.csv, dll.
  datasets/sibi/<nama-output>_all.yaml + <nama-output>_all.txt

Khusus model A-Z, seluruh 26 kelas memakai manifest static26. Checkpoint awal
harus 26 kelas dengan mapping cocok; output disimpan di runs/static26_deploy/.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\train_deploy_full.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml


def resolve_class_ids(manifest_names: set[str], class_map: dict[str, str] | None) -> dict[str, int]:
    if class_map is None:
        ids = {name: index for index, name in enumerate(sorted(manifest_names))}
    else:
        ids = {str(name).upper(): int(class_id) for class_id, name in class_map.items()}
    if set(ids) != manifest_names or set(ids.values()) != set(range(len(ids))):
        raise ValueError("class map must map every manifest class to contiguous IDs")
    names = set(ids)
    if len(names) not in {24, 26} or (len(names) == 26 and names != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
        raise ValueError("expected 24 deployment classes or complete A-Z class set")
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_static26_full.json"))
    ap.add_argument("--run-list", type=Path, default=None,
                    help="optional complete image list; uses manifest images if omitted")
    ap.add_argument("--data-dir", type=Path, default=Path("datasets/sibi"))
    ap.add_argument("--model", type=Path,
                    default=Path("runs/static26b_fold_2/weights/best.pt"),
                    help="26-class pretrained checkpoint; output is separate")
    ap.add_argument("--out", type=Path, default=Path("runs/static26_deploy"))
    ap.add_argument("--class-map", type=Path,
                    default=Path("datasets/sibi/static26_class_map.json"),
                    help="JSON ID-to-class mapping for the training manifest")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="scale=0.9,hsv_v=0.7,degrees=10")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak aktif; cek PyTorch +cu128")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest tidak ada: {args.manifest}")
    if not args.model.is_file():
        raise SystemExit(f"Checkpoint awal tidak ada: {args.model}")
    if args.out.resolve() == Path("runs/augp_deploy").resolve():
        raise SystemExit("Output path is reserved for the existing 24-class deployment")
    if args.out.is_dir() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory must be empty to protect existing checkpoints: {args.out}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [i for i in items if i.get("kelas")]
    if not items:
        raise SystemExit("Manifest tidak memiliki item berkelas")

    # Validate the complete dataset before starting training.
    missing = []
    for item in items:
        image = Path(item["path"])
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if not image.is_file() or not label.is_file():
            missing.append(str(image if not image.is_file() else label))
            if len(missing) >= 5:
                break
    if missing:
        raise SystemExit("File dataset hilang:\n" + "\n".join(missing))

    manifest_names = {str(i["kelas"]).upper() for i in items}
    class_map = json.loads(args.class_map.read_text(encoding="utf-8")) if args.class_map else None
    try:
        ids = resolve_class_ids(manifest_names, class_map)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    names = sorted(ids, key=ids.get)

    image_paths = [str(Path(i["path"]).resolve()) for i in items]
    if args.run_list:
        if not args.run_list.is_file():
            raise SystemExit(f"Daftar gambar tidak ada: {args.run_list}")
        image_paths = [line.strip() for line in args.run_list.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        if len(image_paths) != len(items) or len(set(image_paths)) != len(image_paths):
            raise SystemExit("Daftar gambar harus mencakup manifest tepat satu kali")
        listed = set(image_paths)
        if listed != {str(Path(i["path"]).resolve()) for i in items}:
            raise SystemExit("Daftar gambar berbeda dari manifest; training dibatalkan")

    if args.out.resolve() == Path("runs/augp_deploy").resolve():
        raise SystemExit("Output path is reserved for the existing 24-class deployment")
    if args.out.is_dir() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory must be empty to protect existing checkpoints: {args.out}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    list_file = args.data_dir / f"{args.out.name}_all.txt"
    list_file.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    data_file = args.data_dir / f"{args.out.name}_all.yaml"
    data_file.write_text(yaml.safe_dump({
        "path": args.data_dir.resolve().as_posix(),
        "train": list_file.resolve().as_posix(),
        # Ultralytics requires val key even when val=False; never evaluated.
        "val": list_file.resolve().as_posix(),
        "names": {ids[name]: name for name in names},
    }, sort_keys=False), encoding="utf-8")

    extra = {}
    for spec in args.aug.split(","):
        if "=" in spec:
            key, _, value = spec.partition("=")
            extra[key.strip()] = float(value)

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    print(f"Training full-data: {len(items)} gambar | {len(names)} kelas | {names}")
    print("val=False: model final memakai seluruh manifest; hasil ini bukan metrik CV.")
    model.train(
        data=str(data_file.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        seed=args.seed,
        device=0,
        amp=True,
        val=False,
        plots=False,
        project=str(args.out.parent.resolve()),
        name=args.out.name,
        exist_ok=True,
        **extra,
    )

    weights_dir = args.out / "weights"
    last = weights_dir / "last.pt"
    if not last.is_file():
        raise SystemExit(f"Training tidak menghasilkan checkpoint last.pt: {last}")
    final = args.out / "deploy.pt"
    shutil.copy2(last, final)
    print(f"Checkpoint deployment (salinan last.pt): {final}")
    print("Jangan klaim metrik training in-sample sebagai generalisasi. Uji webcam/video berikutnya.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
