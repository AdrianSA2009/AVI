"""Inventaris dataset mentah: statistik per kelas/sumber, deteksi duplikat, montage audit.

Asumsi layout: data/raw/<sumber>/<kelas>/*.{jpg,...} (folder per kelas).
File langsung di bawah sumber masuk kelas "_tanpa_kelas" dan ditandai.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/inventory.py
    python scripts/inventory.py --raw data/raw --out data/processed/audit --min-per-fold 20 --montage 8

Output:
    <out>/inventory.json          laporan lengkap
    <out>/<sumber>/<kelas>.jpg    montage sampel acak per kelas
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DUP_MAX_DIST = 6


@dataclass
class FileStat:
    path: str
    cls: str
    kind: str  # image | video
    ok: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    duration: float = 0.0
    error: str = ""
    dhash: int = -1


@dataclass
class SourceReport:
    name: str
    stats: list[FileStat] = field(default_factory=list)


def dhash(path: Path) -> int:
    """64-bit difference hash. -1 bila gagal dibaca."""
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            px = list(small.tobytes())  # mode L: 1 byte per piksel
        bits = 0
        for row in range(8):
            for col in range(8):
                left = px[row * 9 + col]
                right = px[row * 9 + col + 1]
                bits = (bits << 1) | (1 if left > right else 0)
        return bits
    except Exception:
        return -1


def stat_image(path: Path, cls: str) -> FileStat:
    rel = str(path)
    try:
        with Image.open(path) as im:
            im.verify()  # deteksi file korup
        with Image.open(path) as im:
            w, h = im.size
        return FileStat(rel, cls, "image", True, w, h, dhash=dhash(path))
    except Exception as exc:
        return FileStat(rel, cls, "image", False, error=str(exc))


def stat_video(path: Path, cls: str) -> FileStat:
    rel = str(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return FileStat(rel, cls, "video", False, error="tidak bisa dibuka")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok = frames > 0 and w > 0
        dur = frames / fps if fps > 0 else 0.0
        return FileStat(rel, cls, "video", ok, w, h, fps, frames, dur,
                        "" if ok else "frame/resolusi nol")
    finally:
        cap.release()


def yolo_names(source_dir: Path) -> list[str]:
    """Baca 'names' dari data.yaml YOLO (khusus sumber berformat Roboflow)."""
    for yaml_path in list(source_dir.glob("*.yaml")) + list(source_dir.glob("*.yml")):
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):
                return [str(v) for _, v in sorted(names.items())]
            if isinstance(names, list):
                return [str(v) for v in names]
        except Exception:
            continue
    return []


def yolo_image_class(image: Path, names: list[str]) -> str:
    """Kelas gambar YOLO dari file label sebelah (images/ <-> labels/)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return "_tanpa_label"
    label = Path(*parts).with_suffix(".txt")
    if not label.is_file():
        return "_tanpa_label"
    try:
        ids = sorted({int(line.split()[0])
                      for line in label.read_text(encoding="utf-8").splitlines()
                      if line.strip()})
    except Exception:
        return "_label_rusak"
    if not ids:
        return "_label_kosong"
    if names and max(ids) < len(names):
        return "+".join(names[i] for i in ids)
    return "+".join(f"id{i}" for i in ids)


def scan_source(source_dir: Path) -> SourceReport:
    """Kelompokkan file.

    Kelas = nama folder induk langsung (layout <sumber>/<kelas>/file.jpg,
    termasuk bersarang dan <split>/<kelas>/).
    Khusus gambar berformat YOLO (<split>/images/*.jpg + labels/*.txt),
    kelas diambil dari file label + data.yaml.
    """
    report = SourceReport(source_dir.name)
    names = yolo_names(source_dir)
    for p in sorted(source_dir.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMG_EXT | VID_EXT:
            continue
        if ext in IMG_EXT and "images" in p.parts:
            cls = yolo_image_class(p, names)
        else:
            cls = p.parent.name
        if ext in IMG_EXT:
            report.stats.append(stat_image(p, cls))
        else:
            report.stats.append(stat_video(p, cls))
    return report


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicates(stats: list[FileStat]) -> list[dict]:
    """Kelompokkan gambar dHash mirip (duplikat & near-duplicate).

    Kandidat pakai LSH 7 band x 9 bit (pigeonhole: hamming <= 6 pasti
    punya minimal 1 band identik) supaya bukan O(n^2) penuh.
    """
    usable = sorted((s for s in stats
                     if s.kind == "image" and s.ok and s.dhash >= 0),
                    key=lambda s: s.path)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, s in enumerate(usable):
        for band in range(7):
            buckets[(band, (s.dhash >> (9 * band)) & 0x1FF)].append(idx)

    candidates: set[tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        for i, a in enumerate(idxs):
            for b in idxs[i + 1:]:
                candidates.add((min(a, b), max(a, b)))

    parent = list(range(len(usable)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in candidates:
        if hamming(usable[a].dhash, usable[b].dhash) <= DUP_MAX_DIST:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[str]] = defaultdict(list)
    for idx, s in enumerate(usable):
        groups[find(idx)].append(s.path)
    return sorted(({"files": fs} for fs in groups.values() if len(fs) > 1),
                  key=lambda g: -len(g["files"]))


def make_montage(stats: list[FileStat], out: Path, n: int, seed: int) -> bool:
    """Grid sampel acak gambar per sumber (semua kelas dalam satu grid per sumber)."""
    rng = random.Random(seed)
    images = [s for s in stats if s.kind == "image" and s.ok]
    if not images:
        return False
    by_cls: dict[str, list[FileStat]] = defaultdict(list)
    for s in images:
        by_cls[s.cls].append(s)

    tiles: list[tuple[str, Path]] = []
    for cls in sorted(by_cls):
        picks = rng.sample(by_cls[cls], min(n, len(by_cls[cls])))
        tiles.extend((cls, Path(s.path)) for s in picks)

    cols = 8
    tile = 160
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 14)), "black")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, (cls, p) in enumerate(tiles):
        try:
            with Image.open(p) as im:
                thumb = im.convert("RGB").resize((tile, tile))
        except Exception:
            continue
        x, y = (i % cols) * tile, (i // cols) * (tile + 14)
        canvas.paste(thumb, (x, y))
        draw.text((x + 2, y + tile + 1), cls[:22], fill="white")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=85)
    return True


def summarize(report: SourceReport, min_per_fold: int) -> dict:
    cls_counts: Counter[str] = Counter(s.cls for s in report.stats if s.ok)
    per_kind = Counter(s.kind for s in report.stats if s.ok)
    corrupt = [s.path for s in report.stats if not s.ok]
    res_img = Counter((s.width, s.height) for s in report.stats
                      if s.ok and s.kind == "image")
    durations = [s.duration for s in report.stats if s.ok and s.kind == "video"]
    warn = [c for c, n in cls_counts.items() if n < 2 * min_per_fold]
    return {
        "sumber": report.name,
        "total_file": len(report.stats),
        "gambar": per_kind.get("image", 0),
        "video": per_kind.get("video", 0),
        "kelas": dict(sorted(cls_counts.items())),
        "jumlah_kelas": len(cls_counts),
        "resolusi_gambar_umum": [
            {"w": w, "h": h, "jumlah": n}
            for (w, h), n in res_img.most_common(5)
        ],
        "durasi_video_detik": {
            "min": round(min(durations), 2) if durations else None,
            "max": round(max(durations), 2) if durations else None,
            "rata2": round(sum(durations) / len(durations), 2) if durations else None,
        },
        "file_korup": corrupt,
        "duplikat": find_duplicates(report.stats),
        "peringatan_kelas_sedikit": {
            "ambang": f"< {2 * min_per_fold} total (~{min_per_fold} per fold)",
            "kelas": sorted(warn),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/audit"))
    ap.add_argument("--min-per-fold", type=int, default=20,
                    help="peringatan bila jumlah kelas < 2x nilai ini (2 fold)")
    ap.add_argument("--montage", type=int, default=8,
                    help="sampel acak per kelas untuk montage (0 = matikan)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.raw.is_dir():
        print(f"Folder tidak ada: {args.raw}  (buat dulu / unduh dataset ke sana)",
              file=sys.stderr)
        return 1

    sources = [d for d in sorted(args.raw.iterdir()) if d.is_dir()]
    if not sources:
        print(f"Tidak ada sumber di {args.raw}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    full: dict[str, dict] = {}
    for src in sources:
        report = scan_source(src)
        info = summarize(report, args.min_per_fold)
        full[src.name] = info
        if args.montage:
            make_montage(report.stats, args.out / f"{src.name}.jpg",
                         args.montage, args.seed)

        print(f"\n=== {src.name} ===")
        print(f"  gambar={info['gambar']} video={info['video']} "
              f"kelas={info['jumlah_kelas']} korup={len(info['file_korup'])} "
              f"duplikat_grup={len(info['duplikat'])}")
        if info["peringatan_kelas_sedikit"]["kelas"]:
            print(f"  PERINGATAN kelas sedikit ({info['peringatan_kelas_sedikit']['ambang']}): "
                  f"{', '.join(info['peringatan_kelas_sedikit']['kelas'])}")

    out_json = args.out / "inventory.json"
    out_json.write_text(json.dumps(full, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nLaporan: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
