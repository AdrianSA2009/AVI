"""Buang entri background PALSU dari dataset box-tangan.

Entri background = label kosong yang dibuat karena MediaPipe gagal menemukan
tangan. Ternyata 100% di antaranya PUNYA box asli di dataset sumber, jadi
gambar itu benar-benar bertangan. Melabelinya kosong mengajari model untuk
TIDAK mendeteksi tangan (merusak recall).

Tulis: hapus images/ + labels/ entri tsb, perbarui manifest.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\drop_fake_background.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_hands.json"))
    ap.add_argument("--dataset", type=Path, default=Path("data/processed/detect_hands"))
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest tidak ada: {args.manifest}")

    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    background = [it for it in items if not it.get("kelas")]
    keep = [it for it in items if it.get("kelas")]

    removed = 0
    for it in background:
        image = Path(it["path"])
        label = args.dataset / "labels" / f"{image.stem}.txt"
        for path in (image, label):
            if path.is_file():
                path.unlink()
                removed += 1

    args.manifest.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "background_removed": len(background),
        "files_deleted": removed,
        "remaining_items": len(keep),
        "note": "Dataset kini hanya berisi gambar bertangan (label benar).",
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
