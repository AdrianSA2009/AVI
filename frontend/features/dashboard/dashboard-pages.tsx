import { AppShell } from "@/components/app-shell";

function PageHeading({ eyebrow, title, description }: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <header className="page-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      <span className="scope-tag"><span className="status-dot is-pending" /> MODE PRATINJAU</span>
    </header>
  );
}

function PanelHeading({ index, label, title, status }: {
  index: string;
  label: string;
  title: string;
  status?: string;
}) {
  return (
    <div className="panel-heading">
      <span className="section-index">{index}</span>
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
        <PageHeading
          eyebrow="DASHBOARD PENGGUNA ISYARAT"
          title="Ruang translasi isyarat"
          description="Pantau input kamera, hasil pengenalan, dan media dari lawan bicara."
        />

        <div className="dashboard-grid sign-grid">
          <section className="panel camera-panel" aria-labelledby="sign-camera-title">
            <PanelHeading index="01" label="INPUT GESTUR" title="Kamera" status="BELUM TERHUBUNG" />
            <div className="camera-placeholder">
              <div className="viewfinder" aria-hidden="true"><i /><i /><i /><i /><span className="viewfinder-cross" /></div>
              <strong id="sign-camera-title">Pratinjau kamera</strong>
              <p>Akses kamera dan inferensi real-time belum tersedia di web.</p>
              <div className="camera-status-list">
                <span><i className="status-dot is-pending" /> Kamera aktif</span>
                <span><i className="status-dot is-pending" /> Sedang memproses</span>
                <span><i className="status-dot is-pending" /> Gestur terdeteksi</span>
              </div>
            </div>
          </section>

          <section className="panel translation-panel" aria-labelledby="sign-output-title">
            <PanelHeading index="02" label="HASIL PENGENALAN" title="Teks Bahasa Indonesia" status="MENUNGGU INPUT" />
            <div className="empty-output" id="sign-output-title">
              <span className="empty-output-mark" aria-hidden="true">Aa</span>
              <strong>Belum ada hasil translasi</strong>
              <p>Hasil teks akan muncul setelah gestur diproses oleh model.</p>
            </div>
            <div className="panel-footnote">MODEL WEB DAN PENERJEMAHAN KALIMAT BELUM TERINTEGRASI</div>
          </section>
        </div>

        <div className="dashboard-grid secondary-grid">
          <section className="panel media-panel" id="avatar" aria-labelledby="avatar-title">
            <PanelHeading index="03" label="DARI LAWAN BICARA" title="Visualisasi gestur SIBI" status="MEDIA BELUM TERSEDIA" />
            <div className="media-placeholder" id="avatar-title">
              <span className="media-play" aria-hidden="true">▶</span>
              <div><strong>Video atau avatar SIBI</strong><p>Visualisasi dari input teks atau suara akan ditampilkan di sini.</p></div>
            </div>
          </section>

          <section className="panel sample-panel" id="sample" aria-labelledby="sample-title">
            <PanelHeading index="04" label="KONTRIBUSI DATA" title="Kirim sampel gestur" status="BELUM AKTIF" />
            <p id="sample-title">Sampel baru dapat diajukan untuk ditinjau admin setelah layanan unggah tersedia.</p>
            <button className="secondary-button" type="button" disabled>Lampirkan foto atau rekaman</button>
          </section>
        </div>
      </div>
    </AppShell>
  );
}

export function GeneralDashboard() {
  return (
    <AppShell role="general" active="/dashboard/general">
      <div className="dashboard-page">
        <PageHeading
          eyebrow="DASHBOARD PENGGUNA UMUM"
          title="Ruang percakapan"
          description="Kirim teks atau suara Bahasa Indonesia untuk divisualisasikan sebagai gestur SIBI."
        />

        <div className="dashboard-grid general-grid">
          <section className="panel text-input-panel" id="text" aria-labelledby="text-input-title">
            <PanelHeading index="01" label="INPUT TEKS" title="Tulis pesan" status="BELUM TERKIRIM" />
            <label className="field-label" htmlFor="conversation-text">Pesan Bahasa Indonesia</label>
            <textarea className="text-field message-field" id="conversation-text" placeholder="Ketik pesan yang ingin disampaikan..." />
            <div className="form-actions">
              <span className="field-hint">Penerjemahan teks ke gestur belum terhubung.</span>
              <button className="primary-button compact-button" type="button" disabled>Kirim teks</button>
            </div>
          </section>

          <section className="panel voice-panel" id="voice" aria-labelledby="voice-title">
            <PanelHeading index="02" label="INPUT SUARA" title="Ucapan Bahasa Indonesia" status="MIKROFON NONAKTIF" />
            <div className="voice-control">
              <button className="mic-button" type="button" disabled aria-label="Mikrofon belum tersedia">●</button>
              <div><strong id="voice-title">Mulai bicara</strong><p>Speech-to-Text belum tersedia di prototipe web.</p></div>
            </div>
            <div className="transcript-state"><span className="status-dot is-pending" /> Menunggu input suara</div>
          </section>
        </div>

        <div className="dashboard-grid secondary-grid">
          <section className="panel media-panel general-media" aria-labelledby="general-avatar-title">
            <PanelHeading index="03" label="OUTPUT GESTUR" title="Visualisasi SIBI" status="MENUNGGU PESAN" />
            <div className="media-placeholder" id="general-avatar-title">
              <span className="media-play" aria-hidden="true">▶</span>
              <div><strong>Video atau avatar gestur</strong><p>Visualisasi hasil konversi akan tampil setelah layanan translasi tersedia.</p></div>
            </div>
          </section>

          <section className="panel speech-panel" aria-labelledby="speech-title">
            <PanelHeading index="04" label="OUTPUT SUARA" title="Audio Bahasa Indonesia" status="BELUM AKTIF" />
            <div className="speech-empty" id="speech-title">
              <span className="sound-bars" aria-hidden="true"><i /><i /><i /><i /><i /></span>
              <p>Hasil Text-to-Speech dari lawan bicara akan tersedia di sini.</p>
              <button className="secondary-button" type="button" disabled>Putar audio</button>
            </div>
          </section>
        </div>
      </div>
    </AppShell>
  );
}

const adminMetrics = [
  { label: "Aktivitas translasi", value: "--", note: "Data belum tersedia" },
  { label: "Sampel menunggu", value: "--", note: "Belum terhubung" },
  { label: "Pengguna terdaftar", value: "--", note: "Belum terhubung" },
];

export function AdminDashboard() {
  return (
    <AppShell role="admin" active="/dashboard/admin">
      <div className="dashboard-page">
        <PageHeading
          eyebrow="DASHBOARD ADMIN"
          title="Pusat pengelolaan"
          description="Pantau aktivitas, tinjau sampel gestur, dan kelola pustaka SIBI."
        />

        <section aria-labelledby="admin-metrics-title">
          <PanelHeading index="01" label="RINGKASAN SISTEM" title="Statistik aktivitas" status="MENUNGGU DATA API" />
          <div className="metric-grid" id="admin-metrics-title">
            {adminMetrics.map((metric) => (
              <article className="panel metric-panel" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                <small>{metric.note}</small>
              </article>
            ))}
          </div>
        </section>

        <section className="panel admin-section" id="samples" aria-labelledby="samples-title">
          <div className="admin-section-heading">
            <PanelHeading index="02" label="MODERASI DATA" title="Verifikasi sampel gestur" status="0 MENUNGGU" />
            <span className="admin-count">ANTRIAN</span>
          </div>
          <div className="admin-empty" id="samples-title">
            <strong>Belum ada sampel masuk</strong>
            <p>Kiriman pengguna akan muncul di sini untuk ditinjau dan disetujui.</p>
          </div>
        </section>

        <div className="dashboard-grid admin-bottom-grid">
          <section className="panel admin-section" id="library" aria-labelledby="library-title">
            <PanelHeading index="03" label="DATA GESTUR" title="Pustaka SIBI" status="BELUM TERHUBUNG" />
            <p className="section-description" id="library-title">Pengelolaan kelas gestur dan media avatar belum terhubung ke database.</p>
          </section>
          <section className="panel admin-section" id="logs" aria-labelledby="logs-title">
            <PanelHeading index="04" label="KEANDALAN SISTEM" title="Log aktivitas" status="KOSONG" />
            <p className="section-description" id="logs-title">Log proses translasi akan ditampilkan setelah API tersedia.</p>
          </section>
        </div>
      </div>
    </AppShell>
  );
}