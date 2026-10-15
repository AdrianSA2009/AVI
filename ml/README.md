# ml — Prototipe Translasi Isyarat SIBI ↔ Bahasa Indonesia

Struktur folder:

```
ml/
├── AGENTS.md          # konteks & aturan asisten AI
├── configs/           # config training (yaml): hyperparameter, path dataset
├── data/
│   ├── raw/           # data mentah (video/foto sumber, audio)
│   ├── processed/     # data hasil preprocessing (frame, augmentasi)
│   └── samples/       # sampel gestur baru dari pengguna (menunggu verifikasi Admin)
├── datasets/
│   └── sibi/          # dataset YOLO (train/val/test + dataset.yaml)
├── docs/              # laporan evaluasi, diagram
├── exports/           # model hasil export (ONNX) untuk VPS
├── notebooks/         # eksplorasi data (opsional)
├── runs/              # output training YOLO (runs/detect/...) — tidak di-commit
├── scripts/           # skrip sekali jalan: prepare_data, split, evaluasi
├── src/               # modul Python reusable (inference, preprocessing, tts/stt)
├── tests/             # pytest
└── weights/           # pretrained (.pt) & best.pt — tidak di-commit
```

## Metode evaluasi: 2-Fold Cross-Validation

Dataset dibagi jadi 2 fold seimbang (50/50). Training dijalankan 2 kali, tiap fold bergantian jadi data uji:

| Run | Train | Test |
|-----|-------|------|
| 1 | fold_1 | fold_2 |
| 2 | fold_2 | fold_1 |

Folder split ada di `datasets/sibi/{fold_1,fold_2}/`.

- Data validasi untuk early stopping/tuning diambil dari **train** fold (mis. 10-20%, stratifikas per kelas).
- Hasil akhir = **rata-rata metrik dari 2 run** (akurasi, mAP, precision, recall, F1), plus std dev sebagai penanda kestabilan.
- Keuntungan buat prototipe: semua data pernah jadi uji, cocok bila dataset kecil.

**Aturan anti data leakage (wajib):**
- Fold dibagi berdasarkan **subjek/video sumber**, bukan per-frame — semua frame dari satu video/kamera harus jatuh di satu fold saja.
- Stratifikasi per kelas supaya distribusi gestur seimbang antar fold.

## Pipeline data (2-Fold CV)

```powershell
# 1. unduh dataset ke data/raw/<sumber>/<kelas>/ (lihat docs/datasets.md)
python scripts\inventory.py        # statistik, duplikat, montage audit
python scripts\ingest_detect.py    # Jalur A: gabung sumber ber-box -> 24 huruf
python scripts\make_folds.py --manifest data/processed/manifest_detect.json `
    --out datasets/sibi/folds.json # 2-fold berbasis grup + cek leakage
python scripts\train_yolo.py       # YOLO11n, 2 run, rata-rata +/- std
```

Laporan per langkah: `data/processed/audit/`, `data/processed/dropped.txt`,
`datasets/sibi/folds.report.txt`.

## Aturan singkat

- Training lokal di RTX 3070 Laptop (8 GB VRAM), device `cuda:0`, AMP aktif.
- Cek GPU di awal tiap script training: `torch.cuda.is_available()` + `get_device_name(0)`.
- Jangan menimpa PyTorch CUDA: verifikasi `torch.__version__` berakhiran `+cu128`.
- OOM → turunkan `batch`/`imgsz` dulu, baru ganti varian model (n → s).
- Dataset & weight tidak di-commit (lihat `.gitignore` root).
