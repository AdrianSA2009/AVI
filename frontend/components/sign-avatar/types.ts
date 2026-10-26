import type { Ref } from "react";

/** Quaternion rotasi LOKAL tulang: [x, y, z, w] */
export type Quat = [number, number, number, number];

export interface PoseEntry {
  /** key = nama tulang tanpa prefix Mixamo, mis. "RightHandIndex1" */
  bones: Record<string, Quat>;
  /** waktu transisi menuju pose ini (detik) */
  duration?: number;
}

/** Key = huruf / gloss ("A", "SAYA"). Key khusus "_REST" = pose istirahat (opsional). */
export type PoseLibrary = Record<string, PoseEntry>;

export interface PlayOptions {
  /** 0.5 – 2 (default: prop `speed`) */
  speed?: number;
  /** jeda tahan setelah tiap pose, ms (default: prop `gapMs`) */
  gapMs?: number;
  /** kembali ke pose istirahat di akhir (default: true) */
  returnToRest?: boolean;
  /** kata/gloss yang tak ada posenya dieja per huruf (default: prop) */
  fingerspellFallback?: boolean;
}

export interface SignAvatarHandle {
  /** Resolve `true` jika selesai, `false` jika dibatalkan/gagal dimulai. */
  playSequence(tokens: string[], opts?: PlayOptions): Promise<boolean>;
  /** Hentikan dan kembalikan ke pose istirahat. */
  stop(): void;
  isPlaying(): boolean;
}

export interface SignAvatarProps {
  /** Array gloss (dipakai apa adanya) atau string (dieja per huruf: "SAYA" → S,A,Y,A) */
  gloss?: string | string[];
  /** 0.5 – 2, berlaku untuk pemutaran berikutnya */
  speed?: number;
  /** jeda antar pose (ms) */
  gapMs?: number;
  /** putar otomatis saat `gloss` berubah (default true) */
  autoPlay?: boolean;
  paused?: boolean;
  onFinish?: () => void;
  /** dipanggil untuk setiap peringatan (pose hilang, tulang tak ditemukan, dll.) */
  onWarning?: (message: string) => void;

  modelUrl?: string; // default "/models/avatar.glb"
  height?: number;
  startBone?: string;
  bodyPoseRef?: import("react").MutableRefObject<import("./pose-retarget").BodyPose | null>;
  /** Data pose langsung (prioritas di atas posesUrl). Gunakan referensi stabil / useMemo. */
  poses?: PoseLibrary;
  posesUrl?: string; // default "/poses/sibi-poses.json"
  /** Peta nama di JSON → nama node di model (untuk VRM / rig non-Mixamo) */
  boneNameMap?: Record<string, string>;
  restKey?: string; // default "_REST"
  fingerspellFallback?: boolean;

  debug?: boolean;
  background?: string;
  className?: string;
  /** Ref imperatif (pakai nama `apiRef` agar aman lewat next/dynamic) */
  apiRef?: Ref<SignAvatarHandle>;
}
