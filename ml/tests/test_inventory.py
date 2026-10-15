"""Smoke test inventory.py: grouping kelas, deteksi korup, dan duplikat dHash."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inventory import find_duplicates, scan_source  # noqa: E402
from PIL import Image  # noqa: E402


def _save(path: Path, seed: int) -> None:
    """Gambar noise deterministik — seed sama = piksel sama = dHash sama."""
    import random
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (64, 64),
                    bytes(rng.randrange(256) for _ in range(64 * 64 * 3))).save(path)


def test_scan_and_duplicates(tmp_path: Path) -> None:
    src = tmp_path / "sumber1"
    _save(src / "A" / "a1.png", seed=1)
    _save(src / "A" / "a2.png", seed=1)            # identik -> duplikat
    _save(src / "B" / "b1.png", seed=2)             # beda konten -> bukan duplikat
    (src / "A" / "broken.png").write_bytes(b"bukan gambar")
    (src / "readme.txt").write_text("abaikan")

    report = scan_source(src)
    assert report.name == "sumber1"

    ok = [s for s in report.stats if s.ok]
    bad = [s for s in report.stats if not s.ok]
    assert {s.cls for s in ok} == {"A", "B"}
    assert len(bad) == 1 and bad[0].path.endswith("broken.png")
    assert sum(1 for s in report.stats if s.kind == "image") == 4

    dups = find_duplicates(report.stats)
    assert len(dups) == 1
    assert len(dups[0]["files"]) == 2
