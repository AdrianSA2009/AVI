"""Hitung data huruf S yang tersedia di raw tapi tidak masuk training.

ingest_detect.py sengaja hanya memakai varian *_no_augmentation untuk mencegah
leakage. Akibatnya varian with_augmentation (besar) tidak dipakai.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\count_unused_s.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    raw = Path("data/raw")
    root = (raw / "Labeled_Official_Indonesian_Sign_Language_(SIBI)_Alphabet"
            / "Labeled Official Indonesian Sign Language (SIBI) Alphabet")

    print("=== varian Labeled_Official ===")
    for variant in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [p for p in (variant / "images").rglob("*")
                  if p.suffix.lower() in IMG_EXT] if (variant / "images").is_dir() else []
        multi = "multi" in variant.name
        per_letter: Counter[str] = Counter()
        for p in images:
            if multi:
                m = re.search(r"__([A-Za-z])_", p.name)
            else:
                m = re.match(r"([A-Za-z])_", p.name)
            if m:
                per_letter[m.group(1).upper()] += 1
        used = "DIPAKAI" if "no_augmentation" in variant.name else "TIDAK dipakai ingest_detect"
        print(f"  {variant.name:46} {len(images):6} gambar  S={per_letter.get('S', 0):5}  [{used}]")

    print("\n=== sumber lain yang punya huruf S ===")
    for name in ("SIBI_Skripsi", "Terbaru_SIBI", "SIBI_DATASET",
                 "SIBI_Official_Indonesian_Sign_Language",
                 "Klasifikasi_Isyarat_Satu_Tangan_SIBI", "archive",
                 "SIBI_Indonesian_Sign_Language_Alphabet_Image_Datas"):
        src = raw / name
        if not src.is_dir():
            continue
        total = 0
        s_like = 0
        for p in src.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXT:
                continue
            total += 1
            nm = p.name.lower()
            if (p.parent.name.lower() in {"s", "alphabet_s", "sibi_s"}
                    or nm.startswith("s_") or "__s_" in nm
                    or nm.startswith("s-") or "-s-" in nm
                    or re.match(r"^s\d", nm)):
                s_like += 1
        print(f"  {name:52} {total:6} gambar  S~{s_like}")

    manifest = Path("data/processed/manifest_hands.json")
    if manifest.is_file():
        items = json.loads(manifest.read_text(encoding="utf-8"))
        s_items = [i for i in items if i["kelas"] == "S"]
        print(f"\ndataset training saat ini: S={len(s_items)} gambar")
        print("  per sumber:", dict(Counter(i["sumber"] for i in s_items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
