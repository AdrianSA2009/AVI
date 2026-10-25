"""Latih baseline eksploratif klasifikasi klip video SIBI dengan PyTorch.

Sumber: data/raw/Indonesian Sign Language System (SIBI) Dataset/
        sibi-dataset-dib-example-face-blurred/{alphabet,number,affix}/
Kelas: kategori:nama (mis. alphabet:H, number:10, affix:ber).
Folder sentence/ dikecualikan: isinya label ekspresi wajah, bukan kosakata isyarat.

Peringatan data: satu signer dan umumnya satu klip/kelas. Semua klip dipakai
untuk training; validasi independen dan akurasi generalisasi TIDAK dapat dihitung.
Frame dari video sama tidak dipisah ke train/test. Loss training bukan ukuran
akurasi; perlu video tambahan dari signer lain untuk evaluasi yang sah.

Output:
    runs/video_sibi/best.pt       checkpoint dengan loss training terendah
    runs/video_sibi/classes.json  indeks label
    runs/video_sibi/report.json   konfigurasi dan loss training (bukan akurasi)

Jalankan dari C:\\AVI\\ml:
    python scripts\\train_video.py
    python scripts\\train_video.py --epochs 30 --batch 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SOURCE = Path(
    "data/raw/Indonesian Sign Language System (SIBI) Dataset/"
    "sibi-dataset-dib-example-face-blurred"
)
CATEGORIES = ("alphabet", "number", "affix")
FRAMES = 8
SIZE = 96


class VideoDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], *, windows: bool = False,
                 stride: int = 4) -> None:
        self.clips: list[torch.Tensor] = []
        self.labels: list[int] = []
        for path, label in records:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Video tidak bisa dibuka: {path}")
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            cap.release()
            if not frames:
                raise RuntimeError(f"Video tidak punya frame yang bisa dibaca: {path}")

            if windows and len(frames) >= FRAMES:
                starts = list(range(0, len(frames) - FRAMES + 1, stride))
                last_start = len(frames) - FRAMES
                if starts[-1] != last_start:
                    starts.append(last_start)
                clips = [frames[start:start + FRAMES] for start in starts]
            else:
                positions = np.linspace(0, len(frames) - 1, FRAMES, dtype=int)
                clips = [[frames[int(i)] for i in positions]]

            for images in clips:
                tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
                self.clips.append(tensor)
                self.labels.append(label)
        mode = "jendela klip" if windows else "klip"
        print(f"Decode sekali selesai: {len(self.clips)} {mode} disimpan di RAM")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clips[index].float().div_(255.0), torch.tensor(
            self.labels[index], dtype=torch.long)


class ClipClassifier(nn.Module):
    def __init__(self, classes: int, temporal: bool = False) -> None:
        super().__init__()
        self.temporal = temporal
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal_head = (nn.GRU(96, 96, batch_first=True, bidirectional=True)
                              if temporal else None)
        head_input = 192 if temporal else 96
        self.head = nn.Sequential(nn.Linear(head_input, 96), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(96, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        features = self.features(x.reshape(batch * frames, channels, height, width))
        sequence = features.flatten(1).reshape(batch, frames, -1)
        if self.temporal_head is not None:
            sequence, _ = self.temporal_head(sequence)
            features = sequence.mean(dim=1)
        else:
            features = sequence.mean(dim=1)
        return self.head(features)


def load_records(source: Path, categories: tuple[str, ...] = CATEGORIES
                 ) -> tuple[list[tuple[Path, int]], list[str]]:
    labels: list[str] = []
    paths_by_label: dict[str, list[Path]] = {}
    for category in categories:
        category_dir = source / category
        if not category_dir.is_dir():
            continue
        for cls_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            label = f"{category}:{cls_dir.name.lower()}"
            videos = sorted(p for p in cls_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in
                            {".mp4", ".mov", ".avi", ".mkv", ".webm"})
            if videos:
                labels.append(label)
                paths_by_label[label] = videos
    labels = sorted(labels)
    records = [(path, idx) for idx, label in enumerate(labels)
               for path in paths_by_label[label]]
    return records, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=Path("runs/video_sibi"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=8,
                        help="jarak frame antar jendela training; FRAMES=8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES,
                        default=list(CATEGORIES),
                        help="kategori yang dilatih (default: alphabet number affix)")
    parser.add_argument("--temporal", action="store_true",
                        help="gunakan GRU bidirectional; klasifikasi memanfaatkan urutan frame")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA tidak aktif; cek PyTorch +cu128")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    records, labels = load_records(args.source, tuple(args.categories))
    if not records:
        raise SystemExit(f"Video tidak ditemukan di {args.source}")
    missing = [str(p) for p, _ in records if not p.is_file()]
    if missing:
        raise SystemExit(f"File tidak ada: {missing[0]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.json").write_text(
        json.dumps({i: name for i, name in enumerate(labels)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    dataset = VideoDataset(records, windows=True, stride=args.window_stride)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=True)
    model = ClipClassifier(len(labels), temporal=args.temporal).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    best_loss = float("inf")
    losses: list[float] = []

    print(f"Video: {len(records)} | kelas: {len(labels)} | frame/klip: {FRAMES}")
    print("PERINGATAN: semua video dipakai training; loss bukan akurasi validasi.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for clips, targets in loader:
            clips = clips.to("cuda:0", non_blocking=True)
            targets = targets.to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clips)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(targets)
        epoch_loss = total_loss / len(dataset)
        losses.append(epoch_loss)
        print(f"epoch {epoch:03d}/{args.epochs} loss={epoch_loss:.4f}", flush=True)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"state_dict": model.state_dict(), "classes": labels,
                        "frames": FRAMES, "size": SIZE,
                        "temporal": args.temporal}, args.out / "best.pt")

    (args.out / "report.json").write_text(json.dumps({
        "source": str(args.source), "videos": len(records),
        "classes": len(labels), "class_names": labels,
        "signer_count_known": 1, "validation": None,
        "temporal_model": args.temporal,
        "best_training_loss": best_loss, "training_losses": losses,
        "caveat": "Training-only loss; single signer and one clip/class "
                  "do not support independent accuracy/generalization claims.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Checkpoint: {args.out / 'best.pt'}")
    print(f"Loss report: {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
