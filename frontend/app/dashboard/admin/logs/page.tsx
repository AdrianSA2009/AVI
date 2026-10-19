import type { Metadata } from "next";
import { AdminLogsPage } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Log Aktivitas | Translasi SIBI" };

export default function AdminLogsRoute() {
  return <AdminLogsPage />;
}