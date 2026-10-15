import type { Metadata } from "next";
import { SignLanguageDashboard } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Dashboard Pengguna Isyarat | Ruang Isyarat" };

export default function SignLanguagePage() {
  return <SignLanguageDashboard />;
}