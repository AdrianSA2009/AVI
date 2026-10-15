# Frontend - Ruang Isyarat

Antarmuka Next.js untuk prototipe translasi dua arah SIBI dan Bahasa Indonesia, mengikuti tiga jenis pengguna pada RPP: pengguna bahasa isyarat, pengguna umum, dan admin.

## Menjalankan

Jalankan perintah dari folder `frontend/`.

```powershell
npm ci
npm run dev
```

Buka [http://localhost:3000](http://localhost:3000). Perintah pemeriksaan dan produksi:

```powershell
npm run lint
npm run build
npm start
```

## Halaman

| URL | Tampilan |
| --- | --- |
| `/` | Halaman masuk |
| `/login` | Pratinjau form masuk |
| `/register` | Pratinjau registrasi pengguna isyarat atau pengguna umum |
| `/dashboard/sign-language` | Kerangka kamera, hasil teks, visualisasi gestur, dan pengajuan sampel |
| `/dashboard/general` | Kerangka input teks/suara, visualisasi gestur, dan keluaran audio |
| `/dashboard/admin` | Kerangka statistik, verifikasi sampel, pustaka gestur, dan log |

Dashboard saat ini dapat dibuka langsung sebagai pratinjau. Registrasi tidak menyediakan pilihan admin; akun admin dibuat oleh pengelola sistem.

## Struktur

```text
app/
  (auth)/login/                 Route masuk
  (auth)/register/              Route registrasi
  dashboard/sign-language/      Dashboard pengguna isyarat
  dashboard/general/            Dashboard pengguna umum
  dashboard/admin/              Dashboard admin
components/                     Shell dan komponen bersama
features/auth/                  Tampilan masuk dan registrasi
features/dashboard/             Tampilan per dashboard
hooks/                           Custom React hooks
lib/                             Utilitas frontend
services/                        Integrasi layanan/API
types/                           Tipe TypeScript bersama
```

`app/(auth)` adalah route group Next.js dan tidak menjadi bagian URL. Implementasi halaman dipisahkan dari file route agar layout dan fitur dapat digunakan kembali.

## Status Fitur

- Form masuk dan registrasi memiliki validasi browser serta pesan status. Data akun belum dikirim atau disimpan karena backend autentikasi belum tersedia.
- Dashboard peran adalah kerangka UI; akses belum dibatasi oleh autentikasi atau RBAC.
- Backend FastAPI saat ini hanya menyediakan `GET /health`. Kamera browser, inferensi web, STT/TTS, avatar, unggah/verifikasi sampel, statistik, dan log belum terhubung.
- Inferensi webcam ONNX yang tersedia dijalankan sebagai skrip ML lokal, bukan melalui frontend.
- Model aktif saat ini mengenali 24 alfabet statis: A-I dan K-Y. J dan Z memerlukan gerakan dan tidak termasuk.

## Palet Tombol

- Primary: `#0F766E`; hover: `#115E59`.
- Secondary: `#134E4A`; secondary lembut: `#1F625D`.
- Aksen suara: `#C2410C`; hover: `#9A3412`; aksen terang: `#FDBA74`.

Token dan aturan visual berada di `app/globals.css`.