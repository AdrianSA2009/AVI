import * as THREE from "three";

export type BodyPosePoint = { x: number; y: number; z: number; visibility?: number };
export type BodyPose = { landmarks: BodyPosePoint[]; sourceAspect: number; at: number };

type BonePair = { upper: string; fore: string };

export function findArmBones(root: THREE.Object3D, side: "Right" | "Left"): BonePair | null {
  const nodes = new Set<string>();
  root.traverse((node) => {
    if ((node as THREE.Bone).isBone) nodes.add(node.name.replace(/^mixamorig[:_]?/i, ""));
  });
  const upper = [`${side}Arm`, `${side}UpperArm`].find((name) => nodes.has(name));
  const fore = [`${side}ForeArm`, `${side}LowerArm`].find((name) => nodes.has(name));
  return upper && fore ? { upper, fore } : null;
}

const point = (p: BodyPosePoint) => new THREE.Vector3(p.x, -p.y, -p.z);

function rotationFromTo(from: THREE.Vector3, to: THREE.Vector3): THREE.Quaternion {
  return new THREE.Quaternion().setFromUnitVectors(from.clone().normalize(), to.clone().normalize());
}

export function retargetArm(
  root: THREE.Object3D,
  pose: BodyPose,
  side: "Right" | "Left",
  blend = 1,
): { upper: THREE.Object3D; fore: THREE.Object3D } | null {
  if (pose.landmarks.length < 17) return null;
  const bones = findArmBones(root, side);
  if (!bones) return null;
  const byName = new Map<string, THREE.Object3D>();
  root.traverse((node) => byName.set(node.name.replace(/^mixamorig[:_]?/i, ""), node));
  const upper = byName.get(bones.upper);
  const fore = byName.get(bones.fore);
  if (!upper || !fore) return null;

  const [shoulderIndex, elbowIndex, wristIndex] = side === "Right" ? [12, 14, 16] : [11, 13, 15];
  const shoulderPoint = pose.landmarks[shoulderIndex];
  const elbowPoint = pose.landmarks[elbowIndex];
  const wristPoint = pose.landmarks[wristIndex];
  if ([shoulderPoint, elbowPoint, wristPoint].some((p) => (p.visibility ?? 1) < 0.45)) return null;

  const aspect = Math.max(pose.sourceAspect, 0.25);
  const toAvatar = (p: BodyPosePoint) => new THREE.Vector3(
    (0.5 - p.x) * aspect * 2.0,
    (0.5 - p.y) * 2.0,
    Math.max(-0.55, Math.min(0.55, -p.z * 1.6)),
  );
  const shoulder = toAvatar(shoulderPoint);
  const elbow = toAvatar(elbowPoint);
  const wrist = toAvatar(wristPoint);
  if (elbow.distanceTo(shoulder) < 0.08 || wrist.distanceTo(elbow) < 0.06) return null;

  const avatarWorldUp = new THREE.Vector3(0, 1, 0);
  const upperParentRotation = new THREE.Quaternion();
  upper.parent?.getWorldQuaternion(upperParentRotation);
  const upperTarget = rotationFromTo(avatarWorldUp, elbow.clone().sub(shoulder).applyQuaternion(upperParentRotation.invert()));
  upper.quaternion.slerp(upperTarget, blend);

  upper.updateWorldMatrix(true, false);
  const foreParentRotation = new THREE.Quaternion();
  fore.parent?.getWorldQuaternion(foreParentRotation);
  const foreTarget = rotationFromTo(avatarWorldUp, wrist.clone().sub(elbow).applyQuaternion(foreParentRotation.invert()));
  fore.quaternion.slerp(foreTarget, blend);
  return { upper, fore };
}
