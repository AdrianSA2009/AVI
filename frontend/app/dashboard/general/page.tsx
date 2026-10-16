import type { Metadata } from "next";
import { GeneralDashboard } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Dashboard Pengguna Umum | Ruang Isyarat" };

export default function GeneralPage() {
  return <GeneralDashboard />;
}