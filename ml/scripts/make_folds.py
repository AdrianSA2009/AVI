"""Pembagi 2-Fold Cross-Validation berbasis grup (anti data leakage).

Input: manifest JSON — daftar item, kelas, dan grup (subjek/video/klaster near-duplicate).
    [{"path": "data/processed/A/a1.jpg", "kelas": "A", "grup": "subjek01"}, ...]

Kontrak: semua item dalam satu grup HARUS jatuh di satu fold (satu video/subjek).
Stratifikasi: greedy per kelas — grup besar diprioritaskan, fold diseimbangkan
per jumlah kelas.

Output:
    <out>                    folds.json: {"fold_1": [path...], "fold_2": [path...]}
    <out>.report.txt         jumlah per kelas per fold + hasil cek leakage

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/make_folds.py
    python scripts/make_folds.py --manifest data/processed/manifest.json --seed 42

Fold file-list dipakai train Jalur A (YOLO) dan Jalur B — split sama agar adil.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for it in items:
        for key in ("path", "kelas", "grup"):
            if key not in it:
                raise ValueError(f"manifest kurang field '{key}': {it}")
    return items


def split_folds(items: list[dict[str, str]], seed: int) -> tuple[list[str], list[str]]:
    """Kelompokkan per grup, lalu greedy seimbangkan jumlah per kelas antar fold."""
    rng = random.Random(seed)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for it in items:
        groups[it["grup"]][it["kelas"]] += 1

    # grup besar & multi-kelas diproses dulu (paling sulit ditampung)
    order = sorted(groups, key=lambda g: (-sum(groups[g].values()),
                                          -len(groups[g]), g))
    rng.shuffle(order)

    counts: list[dict[str, int]] = [defaultdict(int), defaultdict(int)]
    assign: dict[str, int] = {}
    for g in order:
        cls = groups[g]

        def imbalance(fold: int) -> tuple[int, int]:
            # selisih total + selisih per kelas yang kena grup ini
            other = 1 - fold
            diff_total = abs(sum(counts[fold].values()) + sum(cls.values())
                             - sum(counts[other].values()))
            diff_cls = sum(abs(counts[fold][c] + n - counts[other][c])
                           for c, n in cls.items())
            return diff_total, diff_cls

        fold = min(range(2), key=lambda f: (imbalance(f), f))
        assign[g] = fold
        for c, n in cls.items():
            counts[fold][c] += n

    fold_paths: tuple[list[str], list[str]] = ([], [])
    for it in items:
        fold_paths[assign[it["grup"]]].append(it["path"])
    return fold_paths


def check(items: list[dict[str, str]], folds: tuple[list[str], list[str]]) -> list[str]:
    """Kembalikan daftar masalah; kosong = lolos."""
    problems: list[str] = []
    path_fold: dict[str, int] = {}
    for i, paths in enumerate(folds):
        for p in paths:
            if p in path_fold and path_fold[p] != i:
                problems.append(f"PATH BOCOR di 2 fold: {p}")
            path_fold[p] = i

    group_fold: dict[str, int] = {}
    by_path = {it["path"]: it for it in items}
    for i, paths in enumerate(folds):
        for p in paths:
            g = by_path[p]["grup"]
            if g in group_fold and group_fold[g] != i:
                problems.append(f"GRUP BOCOR antar fold: {g}")
            group_fold[g] = i

    cls_folds: list[set[str]] = [set(), set()]
    for i, paths in enumerate(folds):
        cls_folds[i] = {by_path[p]["kelas"] for p in paths
                        if by_path[p]["kelas"]}  # '' = background, bukan kelas
    for c in sorted(cls_folds[0] | cls_folds[1]):
        if c not in cls_folds[0] or c not in cls_folds[1]:
            problems.append(f"kelas '{c}' tidak ada di kedua fold")
    return problems


def report_lines(items: list[dict[str, str]],
                 folds: tuple[list[str], list[str]]) -> list[str]:
    by_path = {it["path"]: it for it in items}
    lines: list[str] = []
    for i, paths in enumerate(folds, start=1):
        per_cls: dict[str, int] = defaultdict(int)
        groups: set[str] = set()
        for p in paths:
            per_cls[by_path[p]["kelas"]] += 1
            groups.add(by_path[p]["grup"])
        lines.append(f"fold_{i}: {len(paths)} item, {len(groups)} grup")
        for c in sorted(per_cls):
            lines.append(f"  {c}: {per_cls[c]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("datasets/sibi/folds.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"Manifest belum ada: {args.manifest}\n"
              "Isi dulu lewat langkah satukan/deduplikasi (start.md langkah 7).",
              file=sys.stderr)
        return 1

    items = load_manifest(args.manifest)
    if not items:
        print("Manifest kosong", file=sys.stderr)
        return 1

    folds = split_folds(items, args.seed)
    problems = check(items, folds)

    lines = report_lines(items, folds)
    if problems:
        lines.append("CEK LEAKAGE / KELAS: GAGAL")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("CEK LEAKAGE / KELAS: lolos")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"seed": args.seed, "manifest": str(args.manifest),
         "fold_1": folds[0], "fold_2": folds[1]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".report.txt").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFold: {args.out}")

    if problems:
        return 2
    print("CEK LEAKAGE / KELAS: lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
