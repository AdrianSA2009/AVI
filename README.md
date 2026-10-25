# Prototipe Translasi Dua Arah SIBI

Monorepo PBLIF3-PD07, Politeknik Negeri Batam. Tahap pertama menyediakan
kerangka Next.js, FastAPI, dan PostgreSQL yang dijalankan dengan Docker Compose.

## Struktur saat ini

<<<<<<< Updated upstream
```text
.
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── Dockerfile
│   └── package.json
├── ml_models/
├── .env.example
└── docker-compose.yml
=======
- `ml/scripts/realtime_test.py` menjalankan kamera lokal dan menampilkan hasil deteksi realtime.
- Model ONNX tersedia di `ml/exports/fold_1_best.onnx` dan `ml/exports/fold_2_best.onnx`. Model fold tersebut merupakan hasil training yang disediakan terpisah dari aplikasi.
- Model deployment utama tetap 24 kelas A-I dan K-Y di `ml/runs/augp_deploy/deploy.pt`.
- Dashboard pengguna isyarat memakai kamera browser dan mengirim frame ke FastAPI `POST /api/v1/predict`; backend memuat model eksperimen 26 kelas `ml/runs/static26_deploy/deploy.pt`.
- Backend menyediakan `GET /health` dan `GET /docs` untuk dokumentasi API.
- Input mikrofon, input teks, dan translasi kalimat belum diimplementasikan.

## Uji Kamera

Di PowerShell, dari folder `ml`:

```powershell
.\.venv\Scripts\python.exe scripts\realtime_test.py --onnx
>>>>>>> Stashed changes
```

## Menjalankan tahap 1

1. Salin `.env.example` menjadi `.env`.
2. Ganti `POSTGRES_PASSWORD` dan `JWT_SECRET_KEY` dengan nilai acak lokal.
3. Jalankan `docker compose up --build`.
4. Buka frontend di `http://localhost:3000`; backend health check tersedia di
   `http://localhost:8000/health` dan dokumentasi API di
   `http://localhost:8000/docs`.

<<<<<<< Updated upstream
Variabel konfigurasi sensitif dibaca dari `.env`; jangan commit file `.env`.
Folder `ml_models/` disediakan sebagai lokasi model YOLO lokal dan model tidak
perlu tersedia untuk menjalankan kerangka tahap pertama.
=======
Jalankan di komputer yang memiliki kamera dan desktop/display. Tekan `Q` untuk keluar atau `S` untuk menyimpan screenshot. Untuk mengubah sensitivitas deteksi, gunakan `--conf`, misalnya `--conf 0.2`; nilai lebih rendah dapat meningkatkan deteksi sekaligus menambah risiko salah deteksi.

## Dataset dan Training

Data sumber masuk ke `ml/data/raw`. Pipeline yang tersedia mengolah sumber yang formatnya didukung menjadi `ml/data/processed`, membagi grup data ke fold, lalu menjalankan training. `ml/datasets/sibi` menyimpan artefak split dan konfigurasi YOLO, bukan lokasi umum untuk menaruh data mentah.

Lihat [panduan ML](ml/README.md) untuk struktur dataset dan command pipeline. Training membutuhkan lingkungan Python dan CUDA yang sesuai; webcam ONNX tidak membutuhkan GPU.

## Layanan Kerangka

Docker Compose menyediakan PostgreSQL, backend FastAPI, dan frontend Next.js untuk pengembangan:

1. Salin `.env.example` menjadi `.env`, lalu isi konfigurasi lokal.
2. Jalankan `docker compose up --build`.
3. Frontend tersedia di `http://localhost:3000`; health check backend di `http://localhost:8000/health`.

Layanan Compose menghubungkan webcam dashboard ke model 26 kelas. Pastikan `ml/runs/static26_deploy/deploy.pt` tersedia sebelum build; Compose memasangnya read-only di container backend. Beri izin kamera pada browser (localhost atau HTTPS). Prediksi hanya berupa pose alfabet statis, tanpa penerjemahan kata/kalimat.
Jangan commit `.env`, dataset, atau bobot/model hasil training.
>>>>>>> Stashed changes
