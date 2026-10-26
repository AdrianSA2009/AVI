import * as THREE from "three";

type ArmSide = "Right" | "Left";
type Landmark = { x: number; y: number; z: number };

const CANDIDATES: Record<ArmSide, { upper: string[]; fore: string[] }> = {
  Right: { upper: ["RightArm", "RightUpperArm"], fore: ["RightForeArm", "RightLowerArm"] },
  Left: { upper: ["LeftArm", "LeftUpperArm"], fore: ["LeftForeArm", "LeftLowerArm"] },
};

export function detectArmBones(root: THREE.Object3D, side: ArmSide) {
  const nodes = new Map<string, THREE.Object3D>();
  root.traverse((node) => {
    if ((node as THREE.Bone).isBone) nodes.set(node.name.replace(/^mixamorig[:_]?/i, ""), node);
  });
  const upper = CANDIDATES[side].upper.map((name) => nodes.get(name)).find(Boolean);
  const fore = CANDIDATES[side].fore.map((name) => nodes.get(name)).find(Boolean);
  return upper && fore ? { upper, fore } : null;
}

function point(p: Landmark): THREE.Vector3 { return new THREE.Vector3(p.x, -p.y, -p.z); }

function localDirection(bone: THREE.Object3D, worldDirection: THREE.Vector3): THREE.Vector3 {
  const parentRotation = new THREE.Quaternion();
  bone.parent?.getWorldQuaternion(parentRotation);
  return worldDirection.clone().applyQuaternion(parentRotation.invert()).normalize();
}

export function retargetArm(root: THREE.Object3D, landmarks: Landmark[], side: ArmSide = "Right"): void {
  if (landmarks.length < 17) return;
  const arm = detectArmBones(root, side);
  if (!arm) return;
  const shoulder = point(landmarks[side === "Right" ? 12 : 11]);
  const elbow = point(landmarks[side === "Right" ? 14 : 13]);
  const wrist = point(landmarks[side === "Right" ? 16 : 15]);
  const worldUp = new THREE.Vector3(0, 1, 0);
  arm.upper.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(worldUp, localDirection(arm.upper, elbow.sub(shoulder))));
  arm.upper.updateWorldMatrix(true, false);
  arm.fore.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(worldUp, localDirection(arm.fore, wrist.sub(elbow))));
}
