import type { Metadata } from "next";
import { AdminDashboard } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Dashboard Admin | Ruang Isyarat" };

export default function AdminPage() {
  return <AdminDashboard />;
}