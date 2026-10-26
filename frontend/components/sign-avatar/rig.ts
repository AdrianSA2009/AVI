import * as THREE from "three";

// Mixamo: "mixamorigRightHandIndex1" (GLTFLoader membuang ":" dari "mixamorig:...")
const MIXAMO_PREFIX = /^mixamorig[:_]?/i;

/**
 * Pembungkus skeleton: cari tulang berdasarkan nama JSON ("RightHandIndex1")
 * dan simpan bind pose (rotasi lokal awal) tiap tulang.
 */
export class Rig {
  private byName = new Map<string, THREE.Object3D>();
  private bind = new Map<THREE.Object3D, THREE.Quaternion>();

  constructor(root: THREE.Object3D, private nameMap: Record<string, string> = {}) {
    root.traverse((o) => {
      if (!(o as THREE.Bone).isBone) return;
      this.byName.set(o.name, o);
      this.byName.set(o.name.replace(MIXAMO_PREFIX, ""), o);
      this.bind.set(o, o.quaternion.clone());
    });
  }

  get(name: string): THREE.Object3D | undefined {
    return this.byName.get(this.nameMap[name] ?? name);
  }

  has(name: string): boolean {
    return !!this.get(name);
  }

  /** Rotasi lokal tulang pada pose bawaan model (jangan dimutasi). */
  bindQuat(name: string): THREE.Quaternion | undefined {
    const bone = this.get(name);
    return bone ? this.bind.get(bone) : undefined;
  }
}

// Tulang yang ditampilkan di panel debug (nama gaya Mixamo tanpa prefix)
const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"].flatMap((f) =>
  [1, 2, 3].map((i) => `Hand${f}${i}`)
);

export const DEBUG_GROUPS: { label: string; bones: string[] }[] = (
  [
    ["Right", "kanan"],
    ["Left", "kiri"],
  ] as const
).flatMap(([side, label]) => [
  {
    label: `Jari ${label}`,
    bones: FINGERS.map((f) => side + f),
  },
  {
    label: `Lengan ${label}`,
    bones: ["Shoulder", "Arm", "ForeArm", "Hand"].map((b) => side + b),
  },
]);
