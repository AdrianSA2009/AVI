"""Smoke test merge_dataset.py: dedup persis, near-dup satu grup, manifest valid."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_dataset import main  # noqa: E402
from PIL import Image  # noqa: E402


def _save(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (64, 64),
                    bytes(rng.randrange(256) for _ in range(64 * 64 * 3))).save(path)


def test_merge_dedup_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "sumberA"
    _save(raw / "A" / "a1.png", seed=1)
    _save(raw / "A" / "a1_copy.png", seed=1)   # dHash sama -> dibuang
    _save(raw / "A" / "a2.png", seed=2)
    _save(raw / "B" / "b1.png", seed=3)
    (raw / "B" / "bad.png").write_bytes(b"bukan gambar")

    out = tmp_path / "processed"
    manifest = out / "manifest.json"
    rc = main(["--raw", str(tmp_path / "raw"), "--out", str(out),
               "--manifest", str(manifest)])
    assert rc == 0

    items = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [it["path"] for it in items]
    # duplikat persis dibuang: tersisa 3 gambar valid
    assert len(paths) == 3
    assert {it["kelas"] for it in items} == {"A", "B"}
    assert all(Path(p).is_file() for p in paths)

    # semua item punya grup & sumber
    assert all(it["grup"] and it["sumber"] == "sumberA" for it in items)

    dropped = (out / "dropped.txt").read_text(encoding="utf-8").splitlines()
    assert len(dropped) == 1
