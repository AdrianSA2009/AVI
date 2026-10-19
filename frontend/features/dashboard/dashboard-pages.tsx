import Link from "next/link";
import { AudioLines, Camera, Hand, Mic, Play, Upload, Volume2 } from "lucide-react";
import { AppShell } from "@/components/app-shell";

function DashboardHeading({ eyebrow, title, description }: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <header className="dashboard-page-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
    </header>
  );
}

function PanelHeading({ label, title, status }: {
  label: string;
  title: string;
  status?: string;
}) {
  return (
    <div className="panel-heading">
      <div>
        <p className="eyebrow">{label}</p>
        <h2>{title}</h2>
      </div>
      {status && <span className="panel-status">{status}</span>}
    </div>
  );
}

export function SignLanguageDashboard() {
  return (
    <AppShell role="sign" active="/dashboard/sign-language">
      <div className="dashboard-page">
        <DashboardHeading
          eyebrow="DASHBOARD PENGGUNA ISYARAT"
          title="Ruang translasi"
          description="Terjemahkan gestur SIBI dan ikuti percakapan dari satu ruang."
        />

        <div className="sign-workspace">
          <div className="sign-main-column">
            <section className="sign-camera-section" aria-labelledby="sign-camera-title">
              <div className="camera-stage">
                <div className="camera-stage-topline">
                  <span><i className="status-dot is-pending" /> Kamera belum terhubung</span>
                  <span className="detection-badge">MENUNGGU GESTUR</span>
                </div>
                <div className="gesture-frame" aria-hidden="true">
                  <span className="gesture-frame-label">AREA DETEKSI</span>
                  <span className="gesture-frame-mark">+</span>
                </div>
                <div className="camera-stage-bottomline">
                  <span id="sign-camera-title">Pratinjau kamera</span>
                  <button className="stage-control" type="button" disabled>Kamera web belum tersedia</button>
                </div>
              </div>
            </section>
              <section className="translation-result" aria-labelledby="sign-output-title">
                <p className="eyebrow">HASIL TERJEMAHAN</p>
                <p className="result-context">Teks Bahasa Indonesia akan muncul setelah gestur diproses.</p>
                <h2 id="sign-output-title">Belum ada hasil</h2>
                <div className="result-actions"><button type="button" disabled>Salin teks</button><button type="button" disabled>Hapus</button></div>
              </section>
            </div>

          <aside className="conversation-rail">
            <section className="partner-section" id="avatar" aria-labelledby="partner-title">
              <PanelHeading label="DARI LAWAN BICARA" title="Lawan bicara" />
              <div className="partner-video" id="partner-title" aria-label="Video lawan bicara belum tersedia">
                <span className="video-play" aria-hidden="true">▶</span>
              </div>
              <div className="video-controls">
                <button type="button" disabled>Putar ulang</button>
                <span>Belum ada video</span>
                <div className="playback-rates"><button type="button" disabled>0,5×</button><button type="button" disabled>1×</button></div>
              </div>
              <div className="transcript-preview"><span>DARI SUARA</span><p>Transkrip lawan bicara akan tampil di sini.</p></div>
            </section>
            <section className="conversation-history" aria-labelledby="sign-history-title">
              <PanelHeading label="PERCAKAPAN" title="Riwayat percakapan" />
              <p id="sign-history-title">Belum ada percakapan.</p>
            </section>
          </aside>
        </div>

        <section className="sample-strip" id="sample" aria-labelledby="sample-title">
          <div><p className="eyebrow">KONTRIBUSI DATA</p><h2 id="sample-title">Bantu menyempurnakan gestur SIBI</h2></div>
          <button className="secondary-button" type="button" disabled>Lampirkan foto atau rekaman</button>
          <Link className="inline-link" href="/dashboard/sign-language/sample">Buka halaman kirim sampel</Link>
        </section>
      </div>
    </AppShell>
  );
}

export function GeneralDashboard() {
  return (
    <AppShell role="general" active="/dashboard/general">
      <div className="dashboard-page general-page">
        <div className="general-page-toolbar">
          <h1>Bicara atau ketik</h1>
          <div className="mode-switch" role="tablist" aria-label="Mode input">
            <button className="is-active" type="button" role="tab" aria-selected="true" disabled>Suara</button>
            <button type="button" role="tab" aria-selected="false" disabled>Teks</button>
          </div>
        </div>
        <div className="general-screen-grid">
          <div className="general-conversation-input">
            <section className="voice-capture" aria-label="Input suara">
              <div className="voice-capture-topline"><span><i className="status-dot is-pending" /> Mikrofon belum tersedia</span></div>
              <div className="voice-capture-center"><AudioLines aria-hidden="true" /><button type="button" disabled aria-label="Mikrofon belum tersedia"><Mic aria-hidden="true" /></button><AudioLines aria-hidden="true" /></div>
              <button className="voice-stop-button" type="button" disabled>Matikan mikrofon</button>
            </section>
            <section className="transcription-section" aria-labelledby="transcription-title">
              <div className="transcription-label"><label htmlFor="transcription-text" id="transcription-title">Hasil transkripsi, bisa diedit</label></div>
              <textarea className="text-field transcription-field" id="transcription-text" value="Selamat siang, ada yang bisa dibantu?" readOnly />
              <div className="transcription-actions"><button className="primary-button" type="button" disabled>Tampilkan dalam isyarat</button><button className="text-button" type="button" disabled>Hapus</button></div>
            </section>
            <section className="received-message" aria-labelledby="received-message-title">
              <div><span>Dari pengguna isyarat</span></div>
              <h2 id="received-message-title">Saya makan</h2>
              <div><button className="outline-icon-button" type="button" disabled><Volume2 aria-hidden="true" /> Putar suara</button><span>Suara otomatis: tidak tersedia</span></div>
            </section>
          </div>
          <aside className="general-response" aria-label="Respons isyarat dan percakapan">
            <section className="sign-visualization" aria-labelledby="visualization-title">
              <h2 id="visualization-title">Visualisasi isyarat SIBI</h2>
              <div className="sign-video-placeholder"><button type="button" disabled aria-label="Visualisasi belum tersedia"><Play aria-hidden="true" /></button></div>
              <p className="gloss-label">Urutan gloss dari kalimat</p>
              <div className="gloss-words"><span>SELAMAT</span><span className="is-current">SIANG</span><span>ADA</span><span>BANTU</span></div>
              <div className="sign-video-controls"><button type="button" disabled>Putar ulang</button><div><button type="button" disabled>0,5×</button><button className="is-current" type="button" disabled>1×</button></div></div>
            </section>
            <section className="general-history" aria-labelledby="general-history-title">
              <h2 id="general-history-title">Riwayat percakapan</h2>
              <div className="history-message from-sign"><span>Dari isyarat</span><p>Halo, nama saya Adrian.</p></div>
              <div className="history-message from-voice"><span>Dari suara</span><p>Senang bertemu dengan Anda.</p></div>
            </section>
          </aside>
        </div>
      </div>
    </AppShell>
  );
}

const usageMetrics = [
  { label: "Total translasi", value: "1.284" },
  { label: "Gestur baru masuk", value: "23" },
  { label: "Pengguna aktif", value: "86" },
  { label: "Rata-rata latensi", value: "420 ms" },
];

const dailyUsage = [
  { day: "Sen", value: "142" },
  { day: "Sel", value: "168" },
  { day: "Rab", value: "191" },
  { day: "Kam", value: "176" },
  { day: "Jum", value: "214" },
  { day: "Sab", value: "158" },
  { day: "Min", value: "235" },
];

const sampleRows = [
  { phrase: "terima kasih", sender: "Nama A", date: "28/09" },
  { phrase: "tolong", sender: "Nama B", date: "27/09" },
  { phrase: "maaf", sender: "Nama C", date: "27/09" },
  { phrase: "selamat pagi", sender: "Nama D", date: "26/09" },
];

const libraryItems = [
  { letter: "A", samples: 42 }, { letter: "B", samples: 38 },
  { letter: "C", samples: 40 }, { letter: "D", samples: 35 },
  { letter: "E", samples: 42 }, { letter: "F", samples: 38 },
  { letter: "G", samples: 40 }, { letter: "H", samples: 35 },
];

const activityRows = [
  { time: "10.42", user: "Nama A", direction: "Isyarat ke teks", input: "gestur", result: "“B”", confidence: "0,94", latency: "380 ms", status: "Berhasil" },
  { time: "10.41", user: "Nama B", direction: "Suara ke isyarat", input: "suara", result: "“siang”", confidence: "0,88", latency: "510 ms", status: "Berhasil" },
  { time: "10.39", user: "Nama A", direction: "Isyarat ke teks", input: "gestur", result: "-", confidence: "0,31", latency: "350 ms", status: "Tidak dikenali" },
  { time: "10.36", user: "Nama C", direction: "Suara ke isyarat", input: "teks", result: "“tolong”", confidence: "0,90", latency: "470 ms", status: "Berhasil" },
  { time: "10.34", user: "Nama B", direction: "Suara ke isyarat", input: "suara", result: "“selamat pagi”", confidence: "0,92", latency: "450 ms", status: "Berhasil" },
  { time: "10.31", user: "Nama A", direction: "Isyarat ke teks", input: "gestur", result: "“A”", confidence: "0,97", latency: "360 ms", status: "Berhasil" },
];

function AdminTitle({ title, control }: { title: string; control?: React.ReactNode }) {
  return (
    <header className="admin-titlebar">
      <h1>{title}</h1>
      {control && <div className="admin-title-actions">{control}</div>}
    </header>
  );
}

export function AdminDashboard() {
  return (
    <AppShell role="admin" active="/dashboard/admin">
      <div className="admin-screen admin-summary-screen">
        <AdminTitle
          title="Ringkasan penggunaan"
          control={<select className="admin-period" aria-label="Periode ringkasan" disabled defaultValue="7"> <option value="7">7 hari terakhir</option></select>}
        />
        <section className="summary-metrics" aria-label="Metrik penggunaan contoh">
          {usageMetrics.map((metric) => <article className="summary-metric" key={metric.label}><span>{metric.label}</span><strong>{metric.value}</strong></article>)}
        </section>
        <div className="summary-chart-grid">
          <section className="summary-section" aria-labelledby="daily-chart-title">
            <h2 id="daily-chart-title">Frekuensi translasi per hari</h2>
            <svg className="daily-chart" viewBox="0 0 520 190" role="img" aria-label="Grafik contoh frekuensi translasi selama tujuh hari">
              <g className="chart-grid-lines"><path d="M0 30H520M0 80H520M0 130H520M0 180H520" /></g>
              <path className="daily-chart-line" d="M36 142L112 114L188 88L264 103L340 62L416 124L492 40" />
              <g className="daily-chart-points"><circle cx="36" cy="142" r="5"/><circle cx="112" cy="114" r="5"/><circle cx="188" cy="88" r="5"/><circle cx="264" cy="103" r="5"/><circle cx="340" cy="62" r="5"/><circle cx="416" cy="124" r="5"/><circle cx="492" cy="40" r="5"/></g>
            </svg>
            <div className="chart-day-labels">{dailyUsage.map((item) => <div key={item.day}><span>{item.day}</span><strong>{item.value}</strong></div>)}</div>
          </section>
          <section className="summary-section direction-chart-section" aria-labelledby="direction-chart-title">
            <h2 id="direction-chart-title">Aktivitas menurut arah translasi</h2>
            <div className="direction-bars">
              <div className="direction-bar-item"><strong>744</strong><span className="direction-bar is-teal"/><small>Isyarat ke teks</small></div>
              <div className="direction-bar-item"><strong>540</strong><span className="direction-bar is-orange"/><small>Suara ke isyarat</small></div>
            </div>
          </section>
          <section className="summary-section demographic-section" aria-labelledby="demographic-title">
            <h2 id="demographic-title">Demografi pengguna</h2>
            <div className="demographic-chart-row">
              <div className="demographic-donut" role="img" aria-label="Contoh demografi: 34 pengguna isyarat, 52 pengguna umum" />
              <div className="demographic-legend"><p><i className="legend-dot is-teal"/>Pengguna isyarat: <strong>34</strong></p><p><i className="legend-dot is-orange"/>Pengguna umum: <strong>52</strong></p><small>86 pengguna aktif</small></div>
            </div>
          </section>
          <section className="summary-section pending-samples-section" aria-labelledby="pending-samples-title">
            <h2 id="pending-samples-title">Kiriman gestur menunggu</h2>
            <table className="compact-admin-table">
              <thead><tr><th>Usulan arti</th><th>Waktu</th><th>Aksi</th></tr></thead>
              <tbody>{sampleRows.slice(0, 3).map((item) => <tr key={item.phrase}><td>{item.phrase}</td><td>{item.date === "28/09" ? "2 jam lalu" : item.date === "27/09" ? "kemarin" : "2 hari lalu"}</td><td><Link className="review-link" href="/dashboard/admin/samples">Tinjau</Link></td></tr>)}</tbody>
            </table>
          </section>
        </div>
      </div>
    </AppShell>
  );
}

export function SignGestureSubmission() {
  return (
    <AppShell role="sign" active="/dashboard/sign-language/sample">
      <div className="dashboard-page sign-upload-page">
        <div className="sign-upload-grid">
          <section className="sign-upload-main" aria-labelledby="upload-title">
            <div className="sign-upload-title"><h1 id="upload-title">Kirim gestur baru</h1></div>
            <p className="upload-intro">Lampirkan foto atau rekaman gestur yang belum dikenali. Admin akan memeriksanya sebelum masuk ke pustaka.</p>
            <div className="upload-preview"><Hand aria-hidden="true" /><span>Pratinjau foto atau rekaman</span></div>
            <div className="upload-actions"><button className="outline-icon-button" type="button" disabled><Upload aria-hidden="true" /> Unggah berkas</button><button className="outline-icon-button" type="button" disabled><Camera aria-hidden="true" /> Ambil dari kamera</button></div>
            <label className="field-label" htmlFor="gesture-meaning">Arti gestur</label>
            <input className="text-field" id="gesture-meaning" placeholder="Contoh: terima kasih" disabled />
            <label className="field-label" htmlFor="gesture-category">Kategori</label>
            <select className="text-field" id="gesture-category" disabled defaultValue=""><option value="">Pilih kategori</option></select>
            <label className="field-label" htmlFor="gesture-description">Catatan untuk admin (opsional)</label>
            <textarea className="text-field upload-description" id="gesture-description" placeholder="Deskripsi posisi tangan" disabled />
            <div className="upload-submit-actions"><button className="primary-button" type="button" disabled>Kirim ke admin</button><Link className="text-button" href="/dashboard/sign-language">Batal</Link></div>
          </section>
          <aside className="my-submissions" aria-labelledby="my-submissions-title">
            <h2 id="my-submissions-title">Kiriman saya</h2>
            <div className="my-submissions-head"><span>Berkas</span><span>Arti</span><span>Status</span></div>
            <div className="my-submission-row"><span className="gesture-thumb"><Hand aria-hidden="true" /></span><span>terima kasih</span><span className="submission-status is-pending">Menunggu</span></div>
            <div className="my-submission-row"><span className="gesture-thumb"><Hand aria-hidden="true" /></span><span>tolong</span><span className="submission-status is-approved">Disetujui</span></div>
            <div className="my-submission-row"><span className="gesture-thumb"><Hand aria-hidden="true" /></span><span>maaf</span><span className="submission-status is-rejected">Ditolak</span></div>
            <div className="rejection-note"><span>Alasan penolakan untuk “maaf”</span><p>Gerakan tangan kurang jelas di rekaman. Kirim ulang dengan cahaya lebih terang.</p></div>
          </aside>
        </div>
      </div>
    </AppShell>
  );
}

export function AdminSamplesPage() {
  return (
    <AppShell role="admin" active="/dashboard/admin/samples">
      <div className="admin-screen verification-screen">
        <AdminTitle title="Verifikasi gestur" />
        <div className="verification-workspace">
          <section className="verification-queue" aria-label="Daftar sampel contoh">
            <div className="verification-tabs" role="tablist" aria-label="Status sampel">
              <button type="button" role="tab" aria-selected="true" disabled>Menunggu (7)</button><button type="button" role="tab" aria-selected="false" disabled>Disetujui</button><button type="button" role="tab" aria-selected="false" disabled>Ditolak</button>
            </div>
            <div className="verification-list-head"><span>Berkas</span><span>Usulan arti</span><span>Pengirim</span><span>Tanggal</span></div>
            <div className="verification-list">{sampleRows.map((item, index) => <button className={`verification-row${index === 0 ? " is-selected" : ""}`} type="button" disabled key={item.phrase}><span className="gesture-thumb"><Hand aria-hidden="true" /></span><span>{item.phrase}</span><span>{item.sender}</span><span>{item.date}</span></button>)}</div>
            <div className="pagination-controls"><button type="button" disabled>‹</button><button className="is-current" type="button" disabled>1</button><button type="button" disabled>2</button><button type="button" disabled>›</button></div>
          </section>
          <section className="verification-detail" aria-labelledby="verification-detail-title">
            <div className="detail-title-row"><h2 id="verification-detail-title">Detail kiriman</h2></div>
            <div className="gesture-preview"><Hand aria-hidden="true" /><small>Foto atau rekaman gestur</small></div>
            <label className="field-label" htmlFor="sample-meaning">Arti gestur</label><input className="text-field" id="sample-meaning" value="terima kasih" readOnly />
            <label className="field-label" htmlFor="sample-category">Kategori</label><select className="text-field" id="sample-category" disabled defaultValue="kosakata"><option value="kosakata">Kosakata</option></select>
            <label className="field-label" htmlFor="sample-sender-note">Catatan pengirim</label><textarea className="text-field verification-note" id="sample-sender-note" value="Tangan kanan menyentuh dagu, lalu bergerak ke depan." readOnly />
            <label className="field-label" htmlFor="sample-rejection-reason">Alasan bila ditolak</label><textarea className="text-field verification-rejection" id="sample-rejection-reason" placeholder="Wajib diisi bila kiriman ditolak" disabled />
            <div className="verification-actions"><button className="primary-button" type="button" disabled>Setujui dan tambah ke pustaka</button><button className="reject-button" type="button" disabled>Tolak</button></div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}

export function AdminLibraryPage() {
  return (
    <AppShell role="admin" active="/dashboard/admin/library">
      <div className="admin-screen library-screen">
        <AdminTitle title="Pustaka gestur" control={<div className="library-title-control"><input className="library-search" aria-label="Cari gestur" placeholder="⌕  Cari gestur" disabled /><button className="primary-button library-add-button" type="button" disabled>＋ Tambah gestur</button></div>} />
        <div className="library-tabs" role="tablist" aria-label="Kategori gestur"><button type="button" role="tab" aria-selected="true" disabled>Alfabet</button><button type="button" role="tab" aria-selected="false" disabled>Angka</button><button type="button" role="tab" aria-selected="false" disabled>Kosakata</button><button type="button" role="tab" aria-selected="false" disabled>Media avatar</button></div>
        <section className="library-grid" aria-label="Gestur contoh">{libraryItems.map((item) => <article className="library-item" key={item.letter}><div className="library-letter">{item.letter}</div><h2>Huruf {item.letter}</h2><p>{item.samples} sampel</p><div><button type="button" disabled>Ubah</button><button type="button" disabled>Hapus</button></div></article>)}</section>
        <div className="pagination-controls library-pagination"><button type="button" disabled>‹</button><button className="is-current" type="button" disabled>1</button><button type="button" disabled>2</button><button type="button" disabled>›</button></div>
        <section className="library-editor" aria-labelledby="library-editor-title"><h2 id="library-editor-title">Tambah atau ubah gestur</h2><div className="library-editor-fields"><label>Label<input className="text-field" placeholder="Contoh: A" disabled /></label><label>Kategori<select className="text-field" disabled defaultValue="alfabet"><option value="alfabet">Alfabet</option></select></label><label>Media avatar atau video<input className="text-field" type="text" placeholder="Pilih berkas" disabled /></label></div><div className="library-editor-actions"><button className="primary-button" type="button" disabled>Simpan</button><button className="text-button" type="button" disabled>Batal</button></div></section>
      </div>
    </AppShell>
  );
}

export function AdminLogsPage() {
  return (
    <AppShell role="admin" active="/dashboard/admin/logs">
      <div className="admin-screen logs-screen">
        <AdminTitle title="Log translasi" />
        <section className="log-metrics" aria-label="Metrik log contoh"><article><span>FPS rata-rata</span><strong>18</strong></article><article><span>Latensi rata-rata</span><strong>420 ms</strong></article><article><span>Akurasi deteksi</span><strong>91%</strong></article><article><span>Galat WebSocket</span><strong>0,8%</strong></article></section>
        <div className="log-filters"><label><span className="visually-hidden">Tanggal</span><select disabled defaultValue=""> <option value="">Tanggal</option></select></label><label><span className="visually-hidden">Arah</span><select disabled defaultValue=""> <option value="">Arah</option></select></label><label><span className="visually-hidden">Status</span><select disabled defaultValue=""><option value="">Status</option></select></label><button className="export-button" type="button" disabled>⇩ Ekspor CSV</button></div>
        <div className="log-table-wrap"><table className="log-table"><thead><tr><th>Waktu</th><th>Pengguna</th><th>Arah</th><th>Masukan</th><th>Hasil</th><th>Keyakinan</th><th>Latensi</th><th>Status</th></tr></thead><tbody>{activityRows.map((item, index) => <tr key={`${item.time}-${item.user}`}><td>{item.time}</td><td>{item.user}</td><td><span className={`direction-dot ${index % 2 ? "is-orange" : "is-teal"}`} />{item.direction}</td><td>{item.input}</td><td>{item.result}</td><td>{item.confidence}</td><td>{item.latency}</td><td><span className={`log-status${item.status === "Berhasil" ? " is-success" : " is-error"}`}>{item.status}</span></td></tr>)}</tbody></table></div>
        <div className="pagination-controls log-pagination"><button type="button" disabled>‹</button><button className="is-current" type="button" disabled>1</button><button type="button" disabled>2</button><button type="button" disabled>›</button></div>
      </div>
    </AppShell>
  );
}