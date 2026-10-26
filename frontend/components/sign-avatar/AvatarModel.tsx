"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { Rig } from "./rig";
import { retargetArm, type BodyPose } from "./pose-retarget";
import type { MutableRefObject } from "react";
import { SequencePlayer } from "./player";

const AVATAR_HEIGHT = 1.7; // model dinormalisasi ke tinggi ini (meter), kaki di y=0

interface Props {
  url: string;
  startBone?: string;
  bodyPoseRef?: MutableRefObject<BodyPose | null>;
  nameMap?: Record<string, string>;
  /** true saat mode debug: player tidak menimpa slider */
  paused: boolean;
  onWarn: (msg: string) => void;
  onReady: (rig: Rig | null, player: SequencePlayer | null) => void;
}

export function AvatarModel({ url, nameMap, paused, onWarn, onReady, startBone = "RightHand", bodyPoseRef }: Props) {
  const gltf = useGLTF(url, false); // false = tanpa Draco (tidak butuh CDN)

  const warnRef = useRef(onWarn);
  const readyRef = useRef(onReady);
  const pausedRef = useRef(paused);
  const armBase = useRef<{ upper: THREE.Quaternion; fore: THREE.Quaternion } | null>(null);
  useEffect(() => { warnRef.current = onWarn; readyRef.current = onReady; pausedRef.current = paused; }, [onWarn, onReady, paused]);

  const mapKey = JSON.stringify(nameMap ?? {});

  const built = useMemo(() => {
    const callbacks = { warn: onWarn, ready: onReady };
    // clone agar cache useGLTF tidak ikut termutasi (penting untuk rest pose)
    const root = clone(gltf.scene) as THREE.Object3D;
    root.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) o.frustumCulled = false; // mesh skinned bisa terpotong saat dipose
    });

    const box = new THREE.Box3().setFromObject(root);
    const scale = AVATAR_HEIGHT / (box.max.y - box.min.y || 1);

    const rig = new Rig(root, JSON.parse(mapKey));
    const player = new SequencePlayer(rig, (m) => callbacks.warn(m), startBone);

    return { root, rig, player, scale, offsetY: -box.min.y * scale };
  }, [gltf.scene, mapKey, onWarn, onReady, startBone]);

  useEffect(() => {
    readyRef.current(built.rig, built.player);
    return () => {
      built.player.stop();
      readyRef.current(null, null);
    };
  }, [built]);

  useFrame((_, dt) => {
    if (!pausedRef.current) built.player.update(Math.min(dt, 0.1));
    const pose = bodyPoseRef?.current;
    const fresh = !!pose && performance.now() - pose.at < 800;
    if (!fresh) return;
    const bones = retargetArm(built.root, pose, "Right", 0.18);
    if (!bones) return;
    armBase.current = { upper: bones.upper.quaternion.clone(), fore: bones.fore.quaternion.clone() };
  });

  return (
    <group scale={built.scale} position={[0, built.offsetY, 0]}>
      <primitive object={built.root} />
    </group>
  );
}
