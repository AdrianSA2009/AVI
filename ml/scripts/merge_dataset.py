"""Satukan data mentah dari banyak sumber + deduplikasi → manifest untuk 2-fold.

Baca : data/raw/<sumber>/<kelas>/*  (gambar; video dilewati, untuk Jalur B)
Tulis: data/processed/<kelas>/<sumber>__<n>_<nama>.<ext>  (salinan, asli tidak diubah)
       data/processed/manifest.json   input scripts/make_folds.py
       data/processed/dropped.txt     file yang dibuang (duplikat persis)

Deduplikasi:
- dHash sama (Hamming 0)  -> duplikat persis, BUANG, sisakan satu.
- dHash mirip (Hamming <= 6) -> near-duplicate, TETAP, tapi dijadikan satu
  grup supaya tidak terpecah antar fold (anti leakage).

Kelas = nama folder sumber (raw). Pemetaan ke kelas baku di labels.yaml
dilakukan belakangan lewat field 'kelas_baku' bila sudah ada pemetaan.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/merge_dataset.py
    python scripts/merge_dataset.py --raw data/raw --out data/processed --include-videos
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from inventory import (  # noqa: E402  (scripts/ ada di path saat dijalankan langsung)
    DUP_MAX_DIST,
    IMG_EXT,
    VID_EXT,
    FileStat,
    find_duplicates,
    scan_source,
    stat_image,
    stat_video,
)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_images(stats: list[FileStat]) -> tuple[list[FileStat], list[FileStat]]:
    """Pisahkan gambar jadi (yang dipakai, yang dibuang = dHash identik)."""
    usable = sorted((s for s in stats if s.ok and s.kind == "image"),
                    key=lambda s: s.path)
    kept: list[FileStat] = []
    dropped: list[FileStat] = []
    for s in usable:
        twin = next((k for k in kept if k.dhash >= 0 and s.dhash >= 0
                     and hamming(k.dhash, s.dhash) == 0), None)
        (dropped if twin is not None else kept).append(s)
    return kept, dropped


def assign_groups(kept: list[FileStat], videos: list[FileStat]) -> dict[str, str]:
    """grup per path: video = dirinya sendiri; gambar = klaster near-duplicate."""
    groups: dict[str, str] = {}
    for v in videos:
        groups[v.path] = f"vid:{v.path}"
    # union-find di atas klaster yang sudah dihitung inventory (Hamming <= 6)
    for cluster in find_duplicates(kept):
        root = min(cluster["files"])
        for p in cluster["files"]:
            groups[p] = f"dup:{root}"
    for s in kept:
        groups.setdefault(s.path, f"one:{s.path}")
    return groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    ap.add_argument("--include-videos", action="store_true",
                    help="ikutkan video ke manifest (Jalur B); default gambar saja")
    args = ap.parse_args(argv)

    if not args.raw.is_dir():
        print(f"Folder tidak ada: {args.raw}", file=sys.stderr)
        return 1
    sources = [d for d in sorted(args.raw.iterdir()) if d.is_dir()]
    if not sources:
        print(f"Tidak ada sumber di {args.raw}", file=sys.stderr)
        return 1

    all_stats: list[FileStat] = []
    per_source = {}
    for src in sources:
        rep = scan_source(src)
        per_source[src.name] = Counter(s.cls for s in rep.stats if s.ok)
        all_stats.extend(rep.stats)

    images = [s for s in all_stats if s.kind == "image"]
    videos = [s for s in all_stats if s.kind == "video"]
    corrupt = [s.path for s in all_stats if not s.ok]
    kept, dropped = dedup_images(images)
    groups = assign_groups(kept, videos)

    # salinan bernama unik ke data/processed/<kelas>/
    manifest: list[dict[str, str]] = []
    counters: dict[str, int] = defaultdict(int)
    for s in kept:
        src_name = Path(s.path).parts[Path(s.path).parts.index("raw") + 1] \
            if "raw" in Path(s.path).parts else "unknown"
        counters[s.cls] += 1
        dest = args.out / s.cls / f"{src_name}__{counters[s.cls]:05d}{Path(s.path).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.path, dest)
        manifest.append({
            "path": dest.as_posix(),
            "kelas": s.cls,
            "grup": groups[s.path],
            "sumber": src_name,
        })

    if args.include_videos:
        for v in videos:
            if not v.ok:
                continue
            src_name = Path(v.path).parts[Path(v.path).parts.index("raw") + 1] \
                if "raw" in Path(v.path).parts else "unknown"
            manifest.append({
                "path": Path(v.path).as_posix(),
                "kelas": v.cls,
                "grup": groups[v.path],
                "sumber": src_name,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    (args.out / "dropped.txt").write_text(
        "\n".join(s.path for s in dropped) + ("\n" if dropped else ""),
        encoding="utf-8")

    # ringkasan
    near = sum(1 for c in find_duplicates(kept) if len(c["files"]) > 1)
    print(f"Sumber: {len(sources)} | masuk: {len(images)} gambar"
          + (f" + {len(videos)} video" if args.include_videos else ""))
    for name, counts in per_source.items():
        print(f"  {name}: {sum(counts.values())} file ok, {len(counts)} kelas")
    print(f"Buang duplikat persis: {len(dropped)} → {args.out / 'dropped.txt'}")
    print(f"Klaster near-duplicate: {near} (tetap dipakai, 1 grup = 1 fold)")
    print(f"File korup dilewati: {len(corrupt)}")
    for p in corrupt:
        print(f"  ! {p}")
    print(f"Manifest: {args.manifest} ({len(manifest)} item)")
    print("Langkah berikutnya: python scripts/make_folds.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
