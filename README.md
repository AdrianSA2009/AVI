# Prototipe Translasi Dua Arah SIBI

Monorepo PBLIF3-PD07, Politeknik Negeri Batam. Tahap pertama menyediakan
kerangka Next.js, FastAPI, dan PostgreSQL yang dijalankan dengan Docker Compose.

## Struktur saat ini

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
```

## Menjalankan tahap 1

1. Salin `.env.example` menjadi `.env`.
2. Ganti `POSTGRES_PASSWORD` dan `JWT_SECRET_KEY` dengan nilai acak lokal.
3. Jalankan `docker compose up --build`.
4. Buka frontend di `http://localhost:3000`; backend health check tersedia di
   `http://localhost:8000/health` dan dokumentasi API di
   `http://localhost:8000/docs`.

Variabel konfigurasi sensitif dibaca dari `.env`; jangan commit file `.env`.
Folder `ml_models/` disediakan sebagai lokasi model YOLO lokal dan model tidak
perlu tersedia untuk menjalankan kerangka tahap pertama.