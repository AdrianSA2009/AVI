import type { Metadata } from "next";
import { AdminSamplesPage } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Verifikasi Sampel | Translasi SIBI" };

export default function AdminSamplesRoute() {
  return <AdminSamplesPage />;
}