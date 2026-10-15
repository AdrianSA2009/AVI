"""Smoke test ingest_detect.py: remap id baku, filter J/Z & kelas sampah."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ingest_detect import baku_ids, collect_sumber, main, remap_label  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (120, 30, 200)).save(path)


def test_baku_ids_from_labels_yaml() -> None:
    ids = baku_ids(ROOT / "labels.yaml")
    assert ids["A"] == 0 and ids["Z"] == 25 and len(ids) == 26


def test_remap_label_filters() -> None:
    baku = {"A": 0, "B": 1}
    lab = ROOT / "_tmp_lab.txt"
    lab.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # id sumber tidak dipakai; kelas dari letter -> id baku
    assert remap_label(lab, "A", baku) == "0 0.5 0.5 0.2 0.2\n"
    assert remap_label(lab, "J", baku) is None      # dinamis
    assert remap_label(lab, "Sign-language", baku) is None
    assert remap_label(lab, "", baku) is None       # letter tak dikenal
    lab.write_text("", encoding="utf-8")
    assert remap_label(lab, "", baku) == ""          # background tetap masuk
    lab.unlink()


def test_main_skripsi_mini(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "SIBI_Skripsi"
    src.mkdir(parents=True, exist_ok=True)
    (src / "data.yaml").write_text(
        "names: ['A', 'B', 'J', 'Sign-language']\n", encoding="utf-8")
    for stem, cid in [("a_001", 0), ("b_001", 1), ("j_001", 2),
                      ("x_001", 3)]:
        _save(src / "train" / "images" / f"{stem}.jpg")
        lab_dir = src / "train" / "labels"
        lab_dir.mkdir(parents=True, exist_ok=True)
        (lab_dir / f"{stem}.txt").write_text(
            f"{cid} 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    out = tmp_path / "processed" / "detect"
    manifest = tmp_path / "processed" / "manifest_detect.json"
    rc = main(["--raw", str(tmp_path / "raw"), "--out", str(out),
               "--manifest", str(manifest),
               "--labels-yaml", str(ROOT / "labels.yaml")])
    assert rc == 0

    items = json.loads(manifest.read_text(encoding="utf-8"))
    # J (dinamis) dan Sign-language (sampah) dibuang
    assert sorted(it["kelas"] for it in items) == ["A", "B"]
    # label sudah pakai id baku (A=0, B=1)
    labs = {p.stem: p.read_text(encoding="utf-8").split()[0]
            for p in (out / "labels").glob("*.txt")}
    assert all(v in {"0", "1"} for v in labs.values())
    assert all(it["grup"] for it in items)


def test_collect_labeled_excludes_augmentation() -> None:
    raw = ROOT / "data" / "raw"
    if not (raw / "Labeled_Official_Indonesian_Sign_Language_(SIBI)_Alphabet").is_dir():
        import pytest
        pytest.skip("dataset belum diunduh")
    pairs = collect_sumber(raw, "Labeled_Official")
    assert pairs, "no_augmentation variants harus terbaca"
    letters = {p[2] for p in pairs}
    assert "J" not in letters and "Z" not in letters
    assert all("with_augmentation" not in str(p[0]) for p in pairs)
