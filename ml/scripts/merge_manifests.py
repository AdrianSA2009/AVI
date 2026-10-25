"""Gabungkan dataset gambar box-tangan dengan frame video SIBI (in-domain).

Masalah: dataset gambar adalah close-up terang; video SIBI wide shot gelap
(tangan ~1.2% frame vs ~4.6%). Huruf seperti S, D, H, P, T gagal di video.

Baca : data/processed/manifest_hands.json      (gambar)
       data/processed/manifest_sibi_video.json (frame video)
Tulis: data/processed/manifest_combined.json

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts\\merge_manifests.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path,
                    default=Path("data/processed/manifest_hands.json"))
    ap.add_argument("--video", type=Path,
                    default=Path("data/processed/manifest_sibi_video.json"))
    ap.add_argument("--extra", nargs="*", type=Path, default=[],
                    help="manifest tambahan (mis. P berlabel dari raw)")
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/manifest_combined.json"))
    ap.add_argument("--repeat-image", type=int, default=1,
                    help="duplikasi entri gambar agar tidak didominasi frame video")
    ap.add_argument("--repeat-video", type=int, default=1,
                    help="duplikasi entri frame video agar bobotnya cukup untuk belajar "
                         "(video hanya 288-598 dari ~7500 item, mudah terabaikan)")
    args = ap.parse_args(argv)

    for path in (args.image, args.video, *args.extra):
        if not path.is_file():
            raise SystemExit(f"Manifest tidak ada: {path}")

    images = json.loads(args.image.read_text(encoding="utf-8"))
    videos = json.loads(args.video.read_text(encoding="utf-8"))
    extras = [item for path in args.extra
              for item in json.loads(path.read_text(encoding="utf-8"))]

    combined = []
    for _ in range(max(1, args.repeat_image)):
        combined.extend(images)
    for _ in range(max(1, args.repeat_video)):
        combined.extend(videos)
    combined.extend(extras)

    args.out.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "image_items": len(images),
        "image_items_after_repeat": len(images) * max(1, args.repeat_image),
        "video_items": len(videos),
        "video_items_after_repeat": len(videos) * max(1, args.repeat_video),
        "extra_items": len(extras),
        "total": len(combined),
        "video_share": round(len(videos) * max(1, args.repeat_video) / max(1, len(combined)), 3),
        "letters": len({it["kelas"] for it in combined if it.get("kelas")}),
        "groups": len({it["grup"] for it in combined}),
        "out": str(args.out),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
