import type { PoseLibrary, Quat } from "./types";

const isFiniteNum = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);

/** Validasi quaternion: 4 angka terhingga, lalu dinormalisasi. */
function parseQuat(v: unknown): Quat | null {
  if (!Array.isArray(v) || v.length !== 4 || !v.every(isFiniteNum)) return null;
  const [x, y, z, w] = v as number[];
  const len = Math.hypot(x, y, z, w);
  if (len < 1e-6) return null;
  return [x / len, y / len, z / len, w / len];
}

/**
 * Ubah JSON mentah menjadi PoseLibrary yang aman dipakai.
 * Data yang rusak dilewati dan dilaporkan lewat `warnings` (tidak melempar error).
 */
export function parsePoseLibrary(raw: unknown): { poses: PoseLibrary; warnings: string[] } {
  const poses: PoseLibrary = {};
  const warnings: string[] = [];

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { poses, warnings: ["Format JSON pose tidak valid: harus berupa objek { \"A\": {...} }."] };
  }

  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    const entry = value as { bones?: unknown; duration?: unknown } | null;
    if (!entry || typeof entry !== "object" || !entry.bones || typeof entry.bones !== "object") {
      if (!key.startsWith("_")) warnings.push(`Pose "${key}" dilewati: field "bones" tidak ada.`);
      continue;
    }

    const bones: PoseLibrary[string]["bones"] = {};
    for (const [bone, q] of Object.entries(entry.bones as Record<string, unknown>)) {
      const parsed = parseQuat(q);
      if (parsed) bones[bone] = parsed;
      else warnings.push(`Pose "${key}": quaternion tulang "${bone}" tidak valid, dilewati.`);
    }

    poses[key] = {
      bones,
      duration: isFiniteNum(entry.duration) && entry.duration > 0 ? entry.duration : undefined,
    };
  }

  return { poses, warnings };
}
