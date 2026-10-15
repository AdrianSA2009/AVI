# Prototipe Translasi Dua Arah SIBI

Monorepo proyek PBLIF3-PD07, Politeknik Negeri Batam. Tujuan jangka panjangnya adalah translasi dua arah antara SIBI dan Bahasa Indonesia. Implementasi saat ini masih berupa prototipe deteksi alfabet SIBI statis melalui kamera; belum menerjemahkan rangkaian gestur menjadi kata atau kalimat.

## Status Implementasi

- `ml/scripts/realtime_test.py` menjalankan kamera lokal dan menampilkan hasil deteksi realtime.
- Model ONNX tersedia di `ml/exports/fold_1_best.onnx` dan `ml/exports/fold_2_best.onnx`. Model fold tersebut merupakan hasil training yang disediakan terpisah dari aplikasi.
- Kelas model mencakup 24 huruf statis: A-I dan K-Y. J dan Z tidak termasuk.
- Frontend masih halaman starter Next.js. Backend FastAPI saat ini baru menyediakan `GET /health`.
- Input mikrofon, input teks, dan translasi kalimat belum diimplementasikan.

## Uji Kamera

Di PowerShell, dari folder `ml`:

```powershell
.\.venv\Scripts\python.exe scripts\realtime_test.py --onnx
```

Command tersebut memakai `fold_1_best.onnx` dan ONNX Runtime CPU. Untuk mencoba fold 2:

```powershell
.\.venv\Scripts\python.exe scripts\realtime_test.py --onnx --model exports\fold_2_best.onnx
```

Jalankan di komputer yang memiliki kamera dan desktop/display. Tekan `Q` untuk keluar atau `S` untuk menyimpan screenshot. Untuk mengubah sensitivitas deteksi, gunakan `--conf`, misalnya `--conf 0.2`; nilai lebih rendah dapat meningkatkan deteksi sekaligus menambah risiko salah deteksi.

## Dataset dan Training

Data sumber masuk ke `ml/data/raw`. Pipeline yang tersedia mengolah sumber yang formatnya didukung menjadi `ml/data/processed`, membagi grup data ke fold, lalu menjalankan training. `ml/datasets/sibi` menyimpan artefak split dan konfigurasi YOLO, bukan lokasi umum untuk menaruh data mentah.

Lihat [panduan ML](ml/README.md) untuk struktur dataset dan command pipeline. Training membutuhkan lingkungan Python dan CUDA yang sesuai; webcam ONNX tidak membutuhkan GPU.

## Layanan Kerangka

Docker Compose menyediakan PostgreSQL, backend FastAPI, dan frontend Next.js untuk pengembangan:

1. Salin `.env.example` menjadi `.env`, lalu isi konfigurasi lokal.
2. Jalankan `docker compose up --build`.
3. Frontend tersedia di `http://localhost:3000`; health check backend di `http://localhost:8000/health`.

Menjalankan layanan Compose belum mengaktifkan deteksi kamera atau fitur translasi. Uji kamera saat ini dijalankan langsung dari skrip ML di komputer lokal.

Jangan commit `.env`, dataset, atau bobot/model hasil training.