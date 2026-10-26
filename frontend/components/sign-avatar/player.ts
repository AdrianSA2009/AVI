import * as THREE from "three";
import type { PoseLibrary, Quat } from "./types";
import type { Rig } from "./rig";

const DEFAULT_POSE_DURATION = 0.6; // detik; dipakai hanya jika JSON tidak punya "duration"
const REST_DURATION = 0.6; // detik; transisi kembali ke pose istirahat

export interface PlayConfig {
  speed: number;
  gapMs: number;
  returnToRest: boolean;
  fingerspell: boolean;
}

interface Segment {
  bones: Map<string, THREE.Quaternion>; // target; tulang lain → pose istirahat
  dur: number; // detik (sudah dibagi speed)
  hold: number; // detik tahan setelah tercapai
}

const toQuat = (q: Quat) => new THREE.Quaternion(q[0], q[1], q[2], q[3]).normalize();

/** Pemutar sequence non-React; dipanggil tiap frame lewat update(dt). */
export class SequencePlayer {
  private lib: PoseLibrary = {};
  private restKey = "_REST";
  private segs: Segment[] = [];
  private idx = -1;
  private t = 0;
  private moving = true;
  private from = new Map<string, THREE.Quaternion>();
  private to = new Map<string, THREE.Quaternion>();
  private touched = new Set<string>(); // tulang yang pernah digerakkan
  private warnedBones = new Set<string>();
  private warnedRest = false;
  private done?: (completed: boolean) => void;

  constructor(
    private rig: Rig,
    private onWarn: (msg: string) => void,
    private startBone = "RightHand"
  ) {}

  configure(lib: PoseLibrary, restKey: string) {
    this.lib = lib;
    this.restKey = restKey;
  }

  get playing() {
    return this.idx >= 0;
  }

  /** Rotasi istirahat sebuah tulang: dari pose `_REST` jika ada, jika tidak bind pose model. */
  private rest(name: string): THREE.Quaternion {
    const r = this.lib[this.restKey]?.bones[name];
    return r ? toQuat(r) : this.rig.bindQuat(name) ?? new THREE.Quaternion();
  }

  /** Cari key pose untuk satu token; kembalikan [] + peringatan jika tak ada. */
  private resolveToken(token: string, spell: boolean): string[] {
    const up = token.toUpperCase();
    if (this.lib[token]) return [token];
    if (this.lib[up]) return [up];

    if (spell && up.length > 1) {
      const letters = [...up];
      const missing = [...new Set(letters.filter((l) => !this.lib[l]))];
      if (missing.length) {
        this.onWarn(`Kata "${token}": pose huruf ${missing.join(", ")} belum ada, huruf itu dilewati.`);
      }
      return letters.filter((l) => this.lib[l]);
    }
    this.onWarn(`Pose "${token}" belum ada di JSON, dilewati.`);
    return [];
  }

  /** Mulai memutar. Resolve `true` jika selesai, `false` jika dibatalkan. */
  play(tokens: string[], cfg: PlayConfig): Promise<boolean> {
    this.finish(false); // batalkan pemutaran sebelumnya, pose terakhir dipertahankan

    const speed = THREE.MathUtils.clamp(cfg.speed, 0.5, 2);
    const hold = Math.max(0, cfg.gapMs) / 1000 / speed;
    const segs: Segment[] = [];

    for (const token of tokens) {
      for (const key of this.resolveToken(token, cfg.fingerspell)) {
        const entry = this.lib[key];
        const bones = new Map<string, THREE.Quaternion>();
        for (const [name, q] of Object.entries(entry.bones)) bones.set(name, toQuat(q));
        segs.push({ bones, dur: (entry.duration ?? DEFAULT_POSE_DURATION) / speed, hold });
      }
    }

    if (segs.length === 0) return Promise.resolve(true); // tidak ada yang bisa dimainkan

    if (!this.lib[this.restKey] && !this.warnedRest) {
      this.warnedRest = true;
      this.onWarn(`Pose istirahat "${this.restKey}" belum ada; memakai pose bawaan model (biasanya T-pose).`);
    }
    if (cfg.returnToRest) {
      const rest = new Map<string, THREE.Quaternion>();
      for (const name of this.touched) rest.set(name, this.rest(name));
      for (const seg of segs) for (const name of seg.bones.keys()) rest.set(name, this.rest(name));
      segs.push({ bones: rest, dur: REST_DURATION / speed, hold: 0 });
    }

    this.segs = segs;
    this.idx = 0;
    this.t = 0;
    this.moving = true;
    this.begin();
    return new Promise((resolve) => (this.done = resolve));
  }

  /** Hentikan; opsional langsung snap ke pose istirahat. */
  stop(toRest = false) {
    this.finish(false);
    if (!toRest) return;
    for (const name of this.touched) this.rig.get(name)?.quaternion.copy(this.rest(name));
  }

  /** Dipanggil tiap frame (dt dalam detik). */
  update(dt: number) {
    const seg = this.segs[this.idx];
    if (!seg) return;
    this.t += dt;

    if (this.moving) {
      const k = seg.dur > 0 ? Math.min(this.t / seg.dur, 1) : 1;
      const e = k * k * (3 - 2 * k); // smoothstep
      for (const [name, target] of this.to) {
        this.rig.get(name)?.quaternion.copy(this.from.get(name)!).slerp(target, e);
      }
      if (k >= 1) {
        this.moving = false;
        this.t = 0;
      }
    } else if (this.t >= seg.hold) {
      this.idx++;
      this.t = 0;
      this.moving = true;
      if (this.idx >= this.segs.length) this.finish(true);
      else this.begin();
    }
  }

  /** Siapkan from/to untuk segmen aktif. */
  private begin() {
    const seg = this.segs[this.idx];
    const names = new Set([...this.touched, this.startBone, ...seg.bones.keys()]);
    this.from.clear();
    this.to.clear();

    for (const name of names) {
      const bone = this.rig.get(name);
      if (!bone) {
        if (!this.warnedBones.has(name)) {
          this.warnedBones.add(name);
          this.onWarn(`Tulang "${name}" tidak ditemukan di model; cek boneNameMap.`);
        }
        continue;
      }
      this.from.set(name, bone.quaternion.clone());
      this.to.set(name, seg.bones.get(name) ?? this.rest(name));
      this.touched.add(name);
    }
  }

  private finish(completed: boolean) {
    const done = this.done;
    this.done = undefined;
    this.segs = [];
    this.idx = -1;
    done?.(completed);
  }
}
