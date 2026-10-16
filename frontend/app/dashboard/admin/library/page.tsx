import type { Metadata } from "next";
import { AdminLibraryPage } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Pustaka SIBI | Translasi SIBI" };

export default function AdminLibraryRoute() {
  return <AdminLibraryPage />;
}