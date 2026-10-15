import type { ReactNode } from "react";
import Link from "next/link";

type DashboardRole = "sign" | "general" | "admin";

const roleDetails: Record<DashboardRole, {
  label: string;
  title: string;
  links: { href: string; label: string }[];
}> = {
  sign: {
    label: "Pengguna isyarat",
    title: "Ruang isyarat",
    links: [
      { href: "/dashboard/sign-language", label: "Translasi isyarat" },
      { href: "/dashboard/sign-language#avatar", label: "Isyarat lawan bicara" },
      { href: "/dashboard/sign-language#sample", label: "Kirim sampel" },
    ],
  },
  general: {
    label: "Pengguna umum",
    title: "Ruang percakapan",
    links: [
      { href: "/dashboard/general", label: "Percakapan" },
      { href: "/dashboard/general#text", label: "Teks Indonesia" },
      { href: "/dashboard/general#voice", label: "Suara" },
    ],
  },
  admin: {
    label: "Administrator",
    title: "Pusat pengelolaan",
    links: [
      { href: "/dashboard/admin", label: "Ringkasan" },
      { href: "/dashboard/admin#samples", label: "Verifikasi sampel" },
      { href: "/dashboard/admin#library", label: "Pustaka gestur" },
      { href: "/dashboard/admin#logs", label: "Log aktivitas" },
    ],
  },
};

export function AppShell({
  children,
  role,
  active,
}: {
  children: ReactNode;
  role: DashboardRole;
  active: string;
}) {
  const details = roleDetails[role];

  return (
    <div className="dashboard-layout">
      <aside className="dashboard-sidebar">
        <Link className="brand" href="/login" aria-label="Ruang Isyarat, halaman masuk">
          <span className="brand-mark" aria-hidden="true">R</span>
          <span className="brand-copy">
            <strong>Ruang Isyarat</strong>
            <span>LAB SIBI</span>
          </span>
        </Link>

        <div className="sidebar-label">{details.title}</div>
        <nav className="dashboard-nav" aria-label={`Navigasi ${details.label}`}>
          {details.links.map((item) => (
            <Link
              aria-current={active === item.href ? "page" : undefined}
              className={`dashboard-nav-link${active === item.href ? " is-active" : ""}`}
              href={item.href}
              key={item.href}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <span className="status-dot is-pending" aria-hidden="true" />
          <span>MODE PRATINJAU</span>
        </div>
      </aside>

      <div className="dashboard-main">
        <header className="dashboard-topbar">
          <div className="dashboard-identity">
            <span className="breadcrumb">RUANG ISYARAT <span>/</span> {details.label.toUpperCase()}</span>
            <span className="role-badge">PRATINJAU</span>
          </div>
          <Link className="logout-link" href="/login">Keluar</Link>
        </header>
        <main className="dashboard-content">{children}</main>
      </div>
    </div>
  );
}