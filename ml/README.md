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

### Dataset box-tangan (memperbaiki box yang mencakup wajah)

Box pada `data/processed/detect` berasal dari sumber yang beberapa di antaranya
mengurung badan/wajah (median luas 0.39-0.48 dari frame). Pipeline box-tangan
menggantinya dengan union landmark tangan:

```powershell
# 2. ganti box dengan landmark tangan MediaPipe; gambar tanpa tangan dibuang
python scripts\relabel_hands.py --padding 0.15 --conf 0.3
#    -> data/processed/detect_hands/  (median luas box turun ke ~0.11)

python scripts\make_folds.py --manifest data/processed/manifest_hands.json `
    --out datasets/sibi/hands_folds.json

# 3. training + resume + gabung metrik
python scripts\train_yolo.py --folds datasets/sibi/hands_folds.json `
    --manifest data/processed/manifest_hands.json --prefix hands_ `
    --metrics-out runs/metrics_cv_hands_fold1.json --run-folds 1
python scripts\train_yolo.py --folds datasets/sibi/hands_folds.json `
    --manifest data/processed/manifest_hands.json --prefix hands_ `
    --run-folds 1 --resume          # lanjut dari weights/last.pt
python scripts\merge_hand_metrics.py

# 4. uji model box-tangan pada frame video SIBI (overlay + akurasi top-1)
python scripts\check_video_hands.py --weights runs/hands_fold_1/weights/best.pt
```

Butuh model MediaPipe: `weights/hand_landmarker.task` (tidak di-commit).
Gambar yang gagal dideteksi MediaPipe DIBUANG dari dataset. Jangan melabelinya
sebagai background: gambar itu punya box asli, jadi label kosong justru
mengajari model untuk tidak mendeteksi tangan.

False positive wajah ditekan di **inferensi** dengan gerbang tangan MediaPipe
(`scripts/verify_hand_gate.py`): 109 -> 6 dari 150 gambar tanpa tangan (turun 94.5%),
sementara 96% gambar bertangan tetap lolos. Aktif secara default di
`scripts/realtime_test.py`; matikan dengan `--no-hand-gate`.

Pemilihan model diuji lewat `scripts/compare_realtime_models.py` (akurasi video
dengan gerbang aktif): `hands_fold_1` 0.708 mengalahkan `handsc_fold_1` 0.486,
jadi `hands_fold_1/weights/best.pt` dipakai lebih dulu.

### Huruf S (dan D/H/P/T) gagal di video

Tangan di video SIBI hanya ~1.2% frame, sedangkan dataset training ~4.6%
(wide shot vs close-up). MediaPipe menaruh bbox tangan S dan M di posisi hampir
identik (pusat_y 0.45 vs 0.44), jadi S memang ambigu untuk detektor.

Solusi terukur: **ensemble voting** beberapa model (satu suara per anggota).

```powershell
python scripts\realtime_test.py --ensemble `
  runs\hands_fold_1\weights\best.pt:640:crop `
  runs\handsc_fold_1\weights\best.pt:1280:full `
  runs\comb6_fold_1\weights\best.pt:640:crop
```

| Konfigurasi | Akurasi 24 huruf video | Huruf S |
|---|---|---|
| `hands_fold_1` 640 + crop (tunggal) | 0.792 | gagal |
| `handsc_fold_1` 1280 tanpa crop | 0.667 | benar |
| **ensemble voting** | 0.875 (21/24) | gagal (N:12 vs S:2 per frame) |

Sisa yang masih gagal: P (dibaca G), S (dibaca N), T (dibaca X). Huruf S masih
belum beres: dengan sampling 10 frame merata, voting memberi N 12 suara dan
S hanya 2. Keberhasilan S di uji awal 21/24 hanya karena sampling jarang
(6 frame) kebetulan mengambil frame yang menguntungkan, jadi angka itu rapuh.

Biaya ensemble: 3 model per frame, FPS turun ~3x, dan ia hanya menambah
D/M/N/R dibanding tunggal sambil kehilangan P/S/T.

Akar masalah S bukan tuning: model yang dilatih pada 25 frame video S sendiri
tetap memprediksi M/N, dan geometri MediaPipe menaruh bbox S dan M di posisi
hampir identik (pusat_y 0.45 vs 0.44). Perbaikan yang benar adalah menambah
data S dari penanda/kamera lain; dataset SIBI hanya punya satu video per huruf.

### Tambahan huruf P, J, Z

P foto sudah ditambah dari `SIBI_DATASET` dan `Klasifikasi_Isyarat_Satu_Tangan_SIBI`.
Box dibuat oleh MediaPipe; foto tanpa tangan terdeteksi dibuang. Hasil:

- 127 gambar P tambahan lolos box/dedupe.
- P pada raw `SIBI_DATASET/valid`: 18/20 benar; `test`: 13/15 benar
  (sebelum tambahan: 0/20 dan 0/15).
- Video P tidak punya box per-frame; model augmented fold 2 diuji pada video
  alfabet SIBI dan mendapat 22/24 huruf top-1 total, tetapi metrik P-per-frame
  perlu laporan terpisah untuk kesimpulan khusus P.

J/Z **tidak masuk detector statis**. Labels.yaml menandai J sebagai gerak
pergelangan dan Z sebagai gerak zigzag. Foto J/Z tidak menyimpan urutan gerak.
Klip dipisah di `data/video_train_sibi/alphabet/{j,z}` dan prototipe temporal
tersimpan di `runs/video_sibi_jz_temporal/best.pt` (GRU bidirectional, 8 frame).
Training window resubstitution 46/46 benar hanya menunjukkan model menghafal
2 klip; bukan validasi atau bukti generalisasi. Model video lama pada test
signer lain hanya 3.85% akurasi. Perlu klip tambahan dari signer/kamera lain
sebelum mengaktifkan J/Z di aplikasi.

Dataset dan manifest P tambahan:
- `data/processed/p_extra/`
- `data/processed/manifest_p_extra.json`
- `data/processed/manifest_augvideo_p.json`
- `datasets/sibi/aug_p_folds.json` (cek leakage lolos)

Checkpoint detector hasil fine-tune P:
- `runs/augp_fold_1/weights/best.pt`
- `runs/augp_fold_2/weights/best.pt`
- deployment full-data 24 kelas A-I,K-Y,P: `runs/augp_deploy/deploy.pt`

### Eksperimen J/Z statis dan temporal

`archive.zip` dihapus atas permintaan pengguna. Isinya 2 dataset gambar
`SIBI_datasets_LEMLITBANG_SIBI_R_90.10_RAW` (20 foto/kelas) dan `..._V02`
(46-50 foto/kelas), termasuk J/Z; bukan klip gerakan. Foto ini CC BY 4.0
menurut sumber unduhan yang didokumentasikan dalam sesi; hanya mendukung pose
statis, tidak untuk mendeteksi gerak huruf J/Z.

Eksperimen pose-foto static J/Z terdahulu bersifat terpisah dan bukan bagian
model deployment. Skrip khusus static J/Z sudah dihapus; catatan eksperimen
berikut dipertahankan sebagai riwayat saja. Holdout foto `SIBI_DATASET/test`
pernah memberi fold 1 J 23/23, Z 23/24; fold 2 J 23/23, Z 24/24. Ini uji
**pose foto**, bukan evaluasi gerakan atau test signer/video independen.

Eksperimen temporal J/Z dahulu memakai satu klip per huruf. Model menghafal
klip training dan tidak punya validasi signer independen; jangan deploy sebagai
pengenal J/Z umum.

Model deployment tetap 24 kelas statis (J/Z tidak termasuk):
`runs/augp_deploy/deploy.pt`.

Eksperimen fusion foto static J/Z dahulu memberi hasil terbatas pada holdout
foto (valid 60.5%, test 58.7%) dan tidak mengenali gerakan. Skrip fusion sudah
dihapus; angka tersebut hanya catatan eksperimen, bukan kemampuan deployment.
Deployment tetap `runs/augp_deploy/deploy.pt` dengan 24 kelas statis.

Laporan per langkah: `data/processed/audit/`, `data/processed/dropped.txt`,
`datasets/sibi/folds.report.txt`, `outputs/relabel_hands.csv`. 

## Aturan singkat

- Training lokal di RTX 3070 Laptop (8 GB VRAM), device `cuda:0`, AMP aktif.
- Cek GPU di awal tiap script training: `torch.cuda.is_available()` + `get_device_name(0)`.
- Jangan menimpa PyTorch CUDA: verifikasi `torch.__version__` berakhiran `+cu128`.
- OOM → turunkan `batch`/`imgsz` dulu, baru ganti varian model (n → s).
- Dataset & weight tidak di-commit (lihat `.gitignore` root).
