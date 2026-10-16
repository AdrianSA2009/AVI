import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft, BadgeCheck, BookOpen, LayoutDashboard, List, type LucideIcon } from "lucide-react";

type DashboardRole = "sign" | "general" | "admin";
type AdminLink = { href: string; label: string; icon: LucideIcon };

const roleDetails: Record<DashboardRole, {
  label: string;
  links: AdminLink[];
  action?: { href: string; label: string };
}> = {
  sign: {
    label: "Pengguna isyarat",
    links: [],
    action: { href: "/dashboard/sign-language/sample", label: "Kirim gestur baru" },
  },
  general: {
    label: "Pengguna umum",
    links: [],
  },
  admin: {
    label: "Admin",
    links: [
      { href: "/dashboard/admin", label: "Dashboard", icon: LayoutDashboard },
      { href: "/dashboard/admin/samples", label: "Verifikasi gestur", icon: BadgeCheck },
      { href: "/dashboard/admin/library", label: "Pustaka gestur", icon: BookOpen },
      { href: "/dashboard/admin/logs", label: "Log translasi", icon: List },
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
  const isAdmin = role === "admin";

  return (
    <div className={`dashboard-layout${isAdmin ? " is-admin-layout" : ""}`}>
      {isAdmin && (
        <aside className="dashboard-sidebar">
          <Link className="brand admin-sidebar-brand" href="/dashboard/admin" aria-label="Translasi SIBI, dashboard admin">
            <span className="dashboard-brand-mark" aria-hidden="true"><i /><i /></span>
            <span className="brand-copy">
              <strong>Translasi SIBI</strong>
            </span>
          </Link>
          <nav className="admin-nav" aria-label="Navigasi administrator">
            {details.links.map((item) => (
              <Link
                aria-current={active === item.href ? "page" : undefined}
                className={`admin-nav-link${active === item.href ? " is-active" : ""}`}
                href={item.href}
                key={item.href}
              >
                <item.icon className="admin-nav-icon" aria-hidden="true" />
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="admin-sidebar-account">
            <div><span className="admin-account-avatar">A</span><span>Admin</span></div>
            <Link href="/login"><span aria-hidden="true">↪</span> Keluar</Link>
          </div>
        </aside>
      )}
      <div className="dashboard-main">
        {!isAdmin && (
          <header className="dashboard-topbar">
            <Link className="brand dashboard-brand" href={role === "sign" ? "/dashboard/sign-language" : "/dashboard/general"} aria-label="Translasi SIBI, dashboard">
              <span className="dashboard-brand-mark" aria-hidden="true"><i /><i /></span>
              <span className="brand-copy">
                <strong>Translasi SIBI</strong>
              </span>
            </Link>
            <div className="dashboard-header-actions">
              {details.action && active !== details.action.href && <Link className="header-action" href={details.action.href}>{details.action.label}</Link>}
              {role === "sign" && active === details.action?.href && <Link className="header-back-link" href="/dashboard/sign-language"><ArrowLeft aria-hidden="true" /> Kembali ke dashboard</Link>}
              <div className="dashboard-user">
                <span className="dashboard-avatar" aria-hidden="true">N</span>
                <span className="dashboard-user-name">Nama Pengguna</span>
              </div>
            </div>
          </header>
        )}
        <main className="dashboard-content">{children}</main>
      </div>
    </div>
  );
}