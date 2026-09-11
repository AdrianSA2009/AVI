import { BackendStatus } from "@/components/backend-status";

export default function Home() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">TRANSLASI DUA ARAH</p>
        <h1>SignBridge</h1>
        <p className="intro">
          Fondasi aplikasi translasi bahasa isyarat dan bahasa Indonesia,
          disiapkan untuk computer vision dan speech processing.
        </p>
        <BackendStatus />
      </section>
      <section className="stack" aria-label="Project foundation">
        <article className="panel">
          <span className="panelIndex">01</span>
          <h2>FastAPI backend</h2>
          <p>API modular dengan routes, models, dan services untuk dikembangkan.</p>
        </article>
        <article className="panel">
          <span className="panelIndex">02</span>
          <h2>Next.js frontend</h2>
          <p>App Router dan API client siap menjadi antarmuka translasi.</p>
        </article>
        <article className="panel panelAccent">
          <span className="panelIndex">03</span>
          <h2>Ready to integrate</h2>
          <p>Health check ini membuktikan request frontend-backend berjalan.</p>
        </article>
      </section>
    </main>
  );
}
