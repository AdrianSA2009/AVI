"""Smoke test make_folds.py: tanpa bocor fold, kelas ada di 2 fold, jumlah seimbang."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_folds import check, load_manifest, report_lines, split_folds  # noqa: E402


def _manifest(tmp_path: Path) -> Path:
    # 6 grup per kelas, 5 item per grup -> 30 item/kelas, harus pecah 15/15
    items = []
    for kelas in ("A", "B"):
        for g in range(6):
            for i in range(5):
                items.append({
                    "path": f"data/processed/{kelas}/{kelas}_g{g}_{i}.jpg",
                    "kelas": kelas,
                    "grup": f"subjek{g}",
                })
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def test_split_balanced_and_leak_free(tmp_path: Path) -> None:
    items = load_manifest(_manifest(tmp_path))
    folds = split_folds(items, seed=42)

    assert check(items, folds) == []  # tidak ada grup/path bocor, kelas lengkap

    for paths in folds:
        assert len(paths) == 30  # 60 item / 2 fold
    assert {it["kelas"] for it in items
            if it["path"] in set(folds[0])} == {"A", "B"}

    lines = report_lines(items, folds)
    # 6 grup total (subjek0-5, masing-masing berisi kelas A+B) -> 3 grup per fold
    assert "fold_1: 30 item, 3 grup" in lines
    assert "fold_2: 30 item, 3 grup" in lines
    assert "  A: 15" in lines and "  B: 15" in lines


def test_group_never_splits(tmp_path: Path) -> None:
    items = load_manifest(_manifest(tmp_path))
    folds = split_folds(items, seed=0)
    group_fold = {}
    for i, paths in enumerate(folds):
        for p in paths:
            g = next(it["grup"] for it in items if it["path"] == p)
            assert g not in group_fold or group_fold[g] == i
            group_fold[g] = i
