import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SIBI Translator",
  description: "Prototipe translasi dua arah SIBI dan Bahasa Indonesia.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}