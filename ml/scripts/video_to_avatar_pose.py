"""Create experimental Mixamo bone poses from one labeled SIBI video.

This heuristic prototype maps MediaPipe hand landmarks to hand/finger joint
rotations. It is not a validated motion-capture/IK retargeter.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGERS = {
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}
BONE_NAMES = [f"RightHand{finger}{joint}" for finger in (*FINGERS, "Thumb") for joint in range(1, 4)]
ARM_BONES = ["RightArm", "RightForeArm", "RightHand"]


def flexion(a, b, c) -> float:
    first = (a.x - b.x, a.y - b.y, a.z - b.z)
    second = (c.x - b.x, c.y - b.y, c.z - b.z)
    lengths = math.sqrt(sum(v * v for v in first)) * math.sqrt(sum(v * v for v in second))
    if lengths < 1e-8:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(first, second)) / lengths))
    return max(0.0, min(math.radians(110), math.pi - math.acos(cosine)))


def quaternion_x(angle: float) -> list[float]:
    return [math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]


def estimate_pose(video: Path, model_path: Path, letter: str, sample_fps: float) -> dict:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        detector.close()
        raise SystemExit(f"Video tidak bisa dibuka: {video}")

    samples: list[dict[str, float]] = []
    arm_samples: list[dict[str, list[float]]] = []
    palm_samples: list[float] = []
    frame_index = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(source_fps / max(sample_fps, 0.1)))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % interval:
                continue
            result = detector.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            ))
            if not result.hand_landmarks:
                continue
            hand = result.hand_landmarks[0]
            values: dict[str, float] = {}
            wrist, index_mcp, pinky_mcp, middle_mcp = hand[0], hand[5], hand[17], hand[9]
            across = (index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y,
                      index_mcp.z - pinky_mcp.z)
            across_len = math.sqrt(sum(v * v for v in across)) or 1.0
            vertical = (middle_mcp.x - wrist.x, middle_mcp.y - wrist.y,
                        middle_mcp.z - wrist.z)
            dot = sum(a * b for a, b in zip(vertical, across)) / across_len
            vertical = tuple(v - dot * a / across_len for v, a in zip(vertical, across))
            vertical_len = math.sqrt(sum(v * v for v in vertical)) or 1.0
            palm_samples.append(math.atan2(vertical[2] / vertical_len, vertical[1] / vertical_len))
            elbow = (wrist.x + (index_mcp.x - wrist.x) * 0.42,
                     wrist.y + (index_mcp.y - wrist.y) * 0.42,
                     wrist.z + (index_mcp.z - wrist.z) * 0.42)
            shoulder = (wrist.x + (index_mcp.x - wrist.x) * 0.85,
                        wrist.y + (index_mcp.y - wrist.y) * 0.85,
                        wrist.z + (index_mcp.z - wrist.z) * 0.85)
            direction = lambda target, origin: [target[i] - origin[i] for i in range(3)]
            def unit(v):
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                return [x / norm for x in v]
            def align_z(v):
                source = [0.0, 0.0, 1.0]
                target = unit(v)
                dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(source, target))))
                cross = [source[1]*target[2]-source[2]*target[1], source[2]*target[0]-source[0]*target[2], source[0]*target[1]-source[1]*target[0]]
                norm = math.sqrt(sum(x*x for x in cross))
                if norm < 1e-8:
                    return [1.0, 0.0, 0.0, 0.0] if dot < 0 else [0.0, 0.0, 0.0, 1.0]
                axis = [x / norm for x in cross]
                half = math.acos(dot) / 2
                return [axis[0]*math.sin(half), axis[1]*math.sin(half), axis[2]*math.sin(half), math.cos(half)]
            arm_samples.append({
                "RightArm": align_z(direction(elbow, shoulder)),
                "RightForeArm": align_z(direction((wrist.x, wrist.y, wrist.z), elbow)),
            })
            for finger, (mcp, pip, dip, tip) in FINGERS.items():
                pip_bend = flexion(hand[mcp], hand[pip], hand[dip])
                dip_bend = flexion(hand[pip], hand[dip], hand[tip])
                values[finger + "1"] = pip_bend * 0.55
                values[finger + "2"] = (pip_bend * 0.35 + dip_bend * 0.15)
                values[finger + "3"] = dip_bend * 0.35
            samples.append(values)
    finally:
        capture.release()
        detector.close()

    if not samples:
        raise SystemExit("Tidak ada tangan terlacak pada video")
    bones = {}
    for finger in FINGERS:
        for joint in range(1, 4):
            name = f"RightHand{finger}{joint}"
            angle = sorted(s[finger + str(joint)] for s in samples)[len(samples) // 2]
            bones[name] = quaternion_x(angle)
    palm_angle = sorted(palm_samples)[len(palm_samples) // 2]
    bones["RightHand"] = quaternion_x(palm_angle)
    return {
        "_REST": {"bones": {name: [0, 0, 0, 1] for name in BONE_NAMES}},
        letter.upper(): {"bones": bones, "duration": 0.3},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--letter", required=True, choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    parser.add_argument("--landmarker", type=Path, default=Path("weights/hand_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("../frontend/public/poses/prototype-a.json"))
    parser.add_argument("--fps", type=float, default=5.0, help="hand samples per second")
    args = parser.parse_args()
    pose = estimate_pose(args.video, args.landmarker, args.letter, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")
    print(f"Prototype pose written: {args.output}; experimental, review visually before use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
