import type { Metadata } from "next";
import { SignGestureSubmission } from "@/features/dashboard/dashboard-pages";

export const metadata: Metadata = { title: "Kirim Gestur | Translasi SIBI" };

export default function SignGestureSubmissionPage() {
  return <SignGestureSubmission />;
}