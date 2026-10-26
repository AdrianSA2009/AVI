import * as THREE from "three";
import type { BodyPose, BodyPosePoint } from "./pose-retarget";

type ArmBones = { upper: THREE.Object3D; fore: THREE.Object3D };

function pointInAvatarSpace(p: BodyPosePoint, aspect: number): THREE.Vector3 {
  return new THREE.Vector3((0.5 - p.x) * aspect, 0.5 - p.y, -p.z * 0.5);
}

function worldDirectionToLocal(bone: THREE.Object3D, direction: THREE.Vector3): THREE.Vector3 {
  const parentRotation = new THREE.Quaternion();
  bone.parent?.getWorldQuaternion(parentRotation);
  return direction.clone().applyQuaternion(parentRotation.invert()).normalize();
}

export function applyRetargetedArmPose(bones: ArmBones, pose: BodyPose, avatarWorldUp: THREE.Vector3): boolean {
  const [shoulderIndex, elbowIndex, wristIndex] = [12, 14, 16];
  const shoulder = pose.landmarks[shoulderIndex];
  const elbow = pose.landmarks[elbowIndex];
  const wrist = pose.landmarks[wristIndex];
  if (!shoulder || !elbow || !wrist || [shoulder, elbow, wrist].some((p) => (p.visibility ?? 1) < 0.5)) return false;

  const aspect = Math.max(0.25, Math.min(3, pose.sourceAspect));
  const shoulderPoint = pointInAvatarSpace(shoulder, aspect);
  const elbowPoint = pointInAvatarSpace(elbow, aspect);
  const wristPoint = pointInAvatarSpace(wrist, aspect);
  const upperDirection = elbowPoint.sub(shoulderPoint);
  const foreDirection = wristPoint.sub(elbowPoint);
  if (upperDirection.length() < 0.05 || foreDirection.length() < 0.05) return false;

  const upperTarget = new THREE.Quaternion().setFromUnitVectors(
    avatarWorldUp,
    worldDirectionToLocal(bones.upper, upperDirection),
  );
  bones.upper.quaternion.copy(upperTarget);
  bones.upper.updateWorldMatrix(true, false);

  const foreTarget = new THREE.Quaternion().setFromUnitVectors(
    avatarWorldUp,
    worldDirectionToLocal(bones.fore, foreDirection),
  );
  bones.fore.quaternion.copy(foreTarget);
  return true;
}
