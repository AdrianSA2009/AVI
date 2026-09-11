## Automated Visual Inspection

Fondasi sistem translasi dua arah bahasa isyarat dan bahasa Indonesia.

Stack awal:

- Backend: FastAPI, Python 3.10+, dan `uv`
- Frontend: Next.js 16, React 19, TypeScript, dan `pnpm`

## Backend

Backend API menggunakan FastAPI dan berjalan pada `http://127.0.0.1:8000`.

Setup dan jalankan dari root project:

```powershell
uv sync
uv run automated-visual-inspection
```

Dokumentasi interaktif tersedia di `http://127.0.0.1:8000/docs`.

Endpoint awal:

- `GET /` - informasi API
- `GET /api/v1/health` - status kesehatan service

## Frontend

Frontend Next.js berada di folder `frontend/`.

```powershell
cd frontend
pnpm install
Copy-Item .env.example .env.local
pnpm dev
```

Buka `http://localhost:3000`. Halaman utama menjalankan request ke backend dan
menampilkan status koneksi. URL API dapat diubah melalui
`NEXT_PUBLIC_API_URL` di `.env.local`.

Struktur utama:

```text
src/automated_visual_inspection/
	api/routes/       # Endpoint FastAPI
	api/models/       # Schema request/response
	services/         # Business logic
frontend/
	app/              # Next.js App Router
	components/       # Komponen UI
	lib/              # API client dan utilitas
	pages/            # Placeholder untuk kompatibilitas/migrasi
	assets/           # Aset aplikasi
```
