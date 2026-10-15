"""Audit prediksi 2-fold: galat per kelas + indikasi leakage lintas fold.

Untuk tiap run: model fold_X memprediksi SEMUA gambar di fold_Y (held-out).
Hasil:
    - akurasi klasifikasi kotak (prediksi vs label), tabel salah -> benar
    - recall per kelas (5 terburuk)
    - leakage: jarak dHash minimum tiap gambar val ke seluruh train fold
        dist <= 6   -> seharusnya TIDAK ADA (grup sudah dipaksa satu fold)
        dist 7..16  -> indikasi augmentasi lolos klaster (kemungkinan bocor)
    - montage galat buat cek visual -> runs/audit/galat.jpg
    - laporan JSON -> runs/audit/audit_report.json

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/audit_predictions.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inventory import dhash  # scripts/ ada di path saat dijalankan langsung

LEAK_HARD = 6    # <= ini: pelanggaran aturan grup
LEAK_SOFT = 16   # 7..16: kandidat augmentasi bocor


def dhash_array(paths: list[str]) -> np.ndarray:
    return np.array([dhash(Path(p)) & ((1 << 64) - 1) for p in paths],
                    dtype=np.uint64)


def popcount(x: np.ndarray) -> np.ndarray:
    """Hamming weight per elemen (menerima uint64 atau byte-view uint8 terakhir)."""
    if x.dtype != np.uint8:
        x = x.view(np.uint8).reshape(*x.shape, 8)
    LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return LUT[x].sum(axis=-1)


def min_hamming_to(val: np.ndarray, train: np.ndarray,
                   chunk: int = 512) -> np.ndarray:
    """Jarak dHash minimum tiap elemen `val` ke semua elemen `train`."""
    train_u8 = train.view(np.uint8).reshape(len(train), 8)
    out = np.empty(len(val), dtype=np.uint64)
    for i in range(0, len(val), chunk):
        block = val[i:i + chunk].view(np.uint8).reshape(-1, 1, 8)
        xor = block ^ train_u8[None, :, :]          # (c, n, 8)
        out[i:i + chunk] = popcount(xor).min(axis=1)
    return out


def predict_all(model_path: Path, paths: list[str], device: str) -> list[str]:
    """Prediksi label huruf per gambar ('' bila tak ada deteksi)."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    names = model.names
    out: list[str] = []
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        results = model.predict(batch, imgsz=640, device=device, conf=0.25,
                                verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append("")
                continue
            cls_ids = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()
            best = max(range(len(cls_ids)), key=lambda k: confs[k])
            out.append(str(names[cls_ids[best]]))
        print(f"  prediksi {min(i + 64, len(paths))}/{len(paths)}",
              flush=True)
    return out


def make_error_montage(items: list[tuple[str, str, str]], out: Path,
                       n: int = 48) -> None:
    """Grid galat: path, truth, pred."""
    picks = items[:n]
    if not picks:
        return
    cols = 6
    tile = 160
    rows = (len(picks) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 26)), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (path, truth, pred) in enumerate(picks):
        try:
            with Image.open(path) as im:
                thumb = im.convert("RGB").resize((tile, tile))
        except Exception:
            continue
        x, y = (i % cols) * tile, (i // cols) * (tile + 26)
        canvas.paste(thumb, (x, y))
        draw.text((x + 2, y + tile + 2), f"benar:{truth}", fill="lightgreen")
        draw.text((x + 2, y + tile + 13), f"pred :{pred}", fill="red")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=85)


def audit_run(run: int, folds: dict, preds: list[str]) -> dict:
    train_paths = folds[f"fold_{run}"]
    val_paths = folds[f"fold_{3 - run}"]

    # galat: kelas asli dari file label YOLO (id -> huruf via names)
    names_fold = folds["_names"]
    errors: list[tuple[str, str, str]] = []
    cls_total: Counter[str] = Counter()
    cls_ok: Counter[str] = Counter()
    conf: Counter[tuple[str, str]] = Counter()
    for path, pred in zip(val_paths, preds):
        lab = Path(path).parents[1] / "labels" / (Path(path).stem + ".txt")
        lines = [l for l in lab.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if not lines:
            continue  # background: tak bisa dinilai
        truth = names_fold.get(lines[0].split()[0], "?")
        cls_total[truth] += 1
        if pred == truth:
            cls_ok[truth] += 1
        else:
            errors.append((path, truth, pred))
            conf[(truth, pred)] += 1

    # leakage (dHash): val -> train
    all_paths = sorted(set(train_paths) | set(val_paths))
    hashes = dhash_array(all_paths)
    idx = {p: i for i, p in enumerate(all_paths)}
    train_h = hashes[[idx[p] for p in train_paths]]
    val_h = hashes[[idx[p] for p in val_paths]]
    mind = min_hamming_to(val_h, train_h)
    hard = [(val_paths[i], int(mind[i])) for i in range(len(mind))
            if mind[i] <= LEAK_HARD]
    soft = [(val_paths[i], int(mind[i])) for i in range(len(mind))
            if LEAK_HARD < mind[i] <= LEAK_SOFT]

    total = sum(cls_total.values())
    ok = sum(cls_ok.values())
    return {
        "run": run,
        "train": len(train_paths),
        "val": len(val_paths),
        "akurasi_kotak": ok / total if total else 0.0,
        "total_dinilai": total,
        "galat_teratas": [{"benar": t, "pred": p, "jumlah": n}
                          for (t, p), n in conf.most_common(20)],
        "recall_per_kelas": {
            c: round(cls_ok[c] / cls_total[c], 4)
            for c in sorted(cls_total)},
        "leak_keras": hard,
        "leak_lembut": soft,
        "_errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=Path, default=Path("datasets/sibi/folds.json"))
    ap.add_argument("--max-error-montage", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    import yaml as _yaml
    names = _yaml.safe_load(Path("datasets/sibi/run1.yaml").read_text(
        encoding="utf-8"))["names"]
    folds["_names"] = {str(k): v for k, v in names.items()}

    out_dir = Path("runs/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"names": folds["_names"]}

    for run in (1, 2):
        candidates = [
            Path(f"runs/detect/runs/detect/fold_{run}/weights/best.pt"),
            Path(f"runs/detect/fold_{run}/weights/best.pt"),
            Path(f"runs/fold_{run}/weights/best.pt"),
        ]
        best = next((p for p in candidates if p.is_file()), None)
        if best is None:
            print(f"Model tidak ada, dicoba: "
                  f"{', '.join(str(p) for p in candidates)}", file=sys.stderr)
            return 1
        val_paths = folds[f"fold_{3 - run}"]
        print(f"=== RUN {run}: model fold_{run} vs val fold_{3 - run} ===")
        preds = predict_all(best, val_paths, "0")
        res = audit_run(run, folds, preds)

        print(f"  akurasi kotak: {res['akurasi_kotak']:.4f} "
              f"({res['total_dinilai']} gambar dinilai)")
        print(f"  leakage keras (<= {LEAK_HARD}): {len(res['leak_keras'])}")
        print(f"  leakage lembut ({LEAK_HARD + 1}-{LEAK_SOFT}): "
              f"{len(res['leak_lembut'])}")
        if res["galat_teratas"]:
            top = ", ".join(f"{g['benar']}->{g['pred']} x{g['jumlah']}"
                            for g in res["galat_teratas"][:8])
            print(f"  galat teratas: {top}")

        errs = sorted(res.pop("_errors"), key=lambda e: e[0])
        random.Random(args.seed).shuffle(errs)
        make_error_montage(errs, out_dir / f"galat_fold_{run}.jpg",
                           args.max_error_montage)
        report[f"run_{run}"] = res

    (out_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Laporan: {out_dir / 'audit_report.json'}")
    print(f"Montage galat: {out_dir / 'galat_fold_1.jpg'} dan "
          f"{out_dir / 'galat_fold_2.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
