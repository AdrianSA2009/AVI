import type { Metadata } from "next";
import { AuthPage } from "@/features/auth/auth-page";

export const metadata: Metadata = { title: "Masuk | Ruang Isyarat" };

export default function LoginPage() {
  return <AuthPage mode="login" />;
}