"""Gabungkan sumber ber-bounding box (Jalur A deteksi) ke format YOLO seragam.

Sumber (baru "ber-box"):
    SIBI_Skripsi            data.yaml names A-Z
    Terbaru_SIBI            data.yaml names (buang kelas 'Sign-language')
    Labeled_Official        HANYA variant *_no_augmentation (cegah leakage);
                            kelas dari NAMA FILE (mono: 'A_..', multi: 'bg_01__A_..'),
                            urutan id = alfabet minus J,Z

Kelas dikecualikan (Jalur B, sesuai labels.yaml): J, Z.
Gambar label kosong: dipakai sebagai background (file .txt kosong).

Output:
    data/processed/detect/images/<sumber>__<n>.<ext>
    data/processed/detect/labels/<sumber>__<n>.txt   (class id = id baku labels.yaml)
    data/processed/manifest_detect.json              input scripts/make_folds.py
    (grup = klaster near-duplicate dHash, lintas sumber)

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/ingest_detect.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from inventory import FileStat, find_duplicates, stat_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDE_KELAS = {"J", "Z"}          # gestur dinamis -> Jalur B
EXCLUDE_SOURCE_KELAS = {"Sign-language"}  # kelas sampah sumber

# nama sumber -> folder di data/raw (bila beda)
FOLDER = {
    "Labeled_Official":
        "Labeled_Official_Indonesian_Sign_Language_(SIBI)_Alphabet",
}


def baku_ids(labels_yaml: Path) -> dict[str, int]:
    data = yaml.safe_load(labels_yaml.read_text(encoding="utf-8"))
    out = {}
    for k in data["kelas"]:
        if k["kategori"] == "huruf":
            out[str(k["nama"]).upper()] = int(k["id"])
    return out


def collect_sumber(raw: Path, name: str) -> list[tuple[Path, Path, str]]:
    """Kembalikan daftar (gambar, label, kelas_huruf) untuk satu sumber."""
    pairs: list[tuple[Path, Path, str]] = []
    src = raw / FOLDER.get(name, name)

    if name in ("SIBI_Skripsi", "Terbaru_SIBI"):
        names = None
        for y in src.glob("*.yaml"):
            names = [str(x) for x in yaml.safe_load(
                y.read_text(encoding="utf-8"))["names"]]
        if names is None:
            raise FileNotFoundError(f"data.yaml tidak ada di {src}")
        for img in src.rglob("*"):
            if img.suffix.lower() not in IMG_EXT or img.parent.name != "images":
                continue
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            letter = ""  # ambil dari yaml names via isi label
            if lab.is_file():
                ids = [int(l.split()[0]) for l in
                       lab.read_text().splitlines() if l.strip()]
                if ids and max(ids) < len(names):
                    letter = names[ids[0]]
            pairs.append((img, lab, letter))
        return pairs

    if name == "Labeled_Official":
        root = src / "Labeled Official Indonesian Sign Language (SIBI) Alphabet"
        for variant in sorted(root.iterdir()):
            if not variant.is_dir() or "no_augmentation" not in variant.name:
                continue
            multi = "multi" in variant.name
            for img in (variant / "images").rglob("*"):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                if multi:
                    m = re.search(r"__([A-Za-z])_", img.name)
                else:
                    m = re.match(r"([A-Za-z])_", img.name)
                letter = m.group(1).upper() if m else ""
                lab = variant / "labels" / img.parent.name / (img.stem + ".txt")
                pairs.append((img, lab, letter))
        return pairs

    raise ValueError(f"sumber tidak dikenal: {name}")


def remap_label(lab: Path, letter: str, baku: dict[str, int]) -> str | None:
    """Isi ulang label dengan id baku. None = gambar dibuang."""
    if letter and letter in EXCLUDE_KELAS | EXCLUDE_SOURCE_KELAS:
        return None
    if not lab.is_file():
        return None  # tidak ada file label sama sekali
    raw_lines = [l for l in lab.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    if not raw_lines:
        return ""  # background / label kosong: tetap dipakai
    if not letter:
        return None  # tak bisa dipastikan kelasnya
    if letter not in baku:
        return None
    out = []
    for line in raw_lines:
        parts = line.split()
        out.append(" ".join([str(baku[letter])] + parts[1:]))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/detect"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/manifest_detect.json"))
    ap.add_argument("--labels-yaml", type=Path, default=Path("labels.yaml"))
    args = ap.parse_args(argv)

    baku_all = baku_ids(args.labels_yaml)
    # id kontigu 0..N-1 untuk YOLO (J,Z dikecualikan, urut alfabet)
    baku = {letter: i for i, letter in
            enumerate(sorted(k for k in baku_all if k not in EXCLUDE_KELAS))}
    sources = ["SIBI_Skripsi", "Terbaru_SIBI", "Labeled_Official"]
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    stats_out: list[FileStat] = []
    counters: Counter[str] = Counter()
    buang: Counter[str] = Counter()

    for name in sources:
        try:
            pairs = collect_sumber(args.raw, name)
        except FileNotFoundError as exc:
            print(f"LEWAT {name}: {exc}", file=sys.stderr)
            continue
        for img, lab, letter in pairs:
            new_label = remap_label(lab, letter, baku)
            if new_label is None:
                buang[name] += 1
                continue
            counters[name] += 1
            stem = f"{name}__{counters[name]:05d}"
            dest_img = args.out / "images" / f"{stem}{img.suffix.lower()}"
            shutil.copy2(img, dest_img)
            (args.out / "labels" / f"{stem}.txt").write_text(new_label,
                                                             encoding="utf-8")
            st = stat_image(dest_img, letter)
            stats_out.append(st)
            manifest.append({
                "path": dest_img.as_posix(),
                "kelas": letter,
                "grup": "",  # diisi setelah klaster duplikat
                "sumber": name,
            })

    # grup = klaster near-duplicate (lintas sumber, anti leakage)
    clusters = find_duplicates(stats_out)
    group_of: dict[str, str] = {}
    for cl in clusters:
        root = min(cl["files"])
        for p in cl["files"]:
            group_of[Path(p).as_posix()] = f"dup:{root}"
    for it in manifest:
        it["grup"] = group_of.get(it["path"], f"one:{it['path']}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    per_kelas: Counter[str] = Counter(it["kelas"] for it in manifest)
    print(f"Masuk: {len(manifest)} gambar ber-box "
          f"({sum(counters.values())} awal, buang {sum(buang.values())})")
    for name in sources:
        print(f"  {name}: {counters[name]} masuk, {buang[name]} dibuang")
    print(f"Kelas: {len(per_kelas)} -> {', '.join(sorted(per_kelas))}")
    print(f"Grup duplikat: {len(clusters)}")
    kosong = sum(1 for p in (args.out / 'labels').glob('*.txt')
                 if not p.read_text(encoding='utf-8').strip())
    print(f"Label kosong (background): {kosong}")
    kurang = sorted(c for c, n in per_kelas.items() if n < 40)
    if kurang:
        print(f"PERINGATAN kelas < 40: {', '.join(kurang)}")
    print(f"Manifest: {args.manifest}")
    print("Langkah berikutnya: python scripts/make_folds.py "
          "--manifest data/processed/manifest_detect.json "
          "--out datasets/sibi/folds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
