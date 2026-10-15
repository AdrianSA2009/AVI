"""Uji realtime: webcam -> YOLO -> overlay status + FPS, di jendela OpenCV.

Indikator status (sesuai dokumen proyek):
    Kamera Aktif            kamera terbuka, belum ada frame dinilai
    Sedang Memproses        model jalan / hasil belum stabil (smoothing)
    Gestur Terdeteksi: X    hasil stabil N frame terakhir
    Input Tidak Terdeteksi  tidak ada kotak di atas ambang confidence

Smoothing: voting mayoritas dari 7 frame terakhir - teks tidak kedip-kedip.

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/realtime_test.py
    python scripts/realtime_test.py --camera 1 --conf 0.5 --onnx
    (tombol: S = simpan screenshot ke data/webcam_test/, Q = keluar)

Jalankan di mesin ber-kamera (laptop). Butuh display (bukan SSH/headless).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMGSZ = 640


def letterbox_frame(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(IMGSZ / width, IMGSZ / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_x = (IMGSZ - resized_width) // 2
    pad_y = (IMGSZ - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height),
                         interpolation=cv2.INTER_LINEAR)
    boxed = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    boxed[pad_y:pad_y + resized_height,
          pad_x:pad_x + resized_width] = resized
    return boxed, scale, pad_x, pad_y


def find_best_pt(run: int) -> Path:
    candidates = [
        Path(f"runs/detect/runs/detect/fold_{run}/weights/best.pt"),
        Path(f"runs/detect/fold_{run}/weights/best.pt"),
        Path(f"runs/fold_{run}/weights/best.pt"),
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise SystemExit(f"Model tidak ada (fold_{run})")
    return found


def make_predictor(args: argparse.Namespace):
    """Fungsi predict(frame BGR) -> l
    
    
    
    ist[dict(x1,y1,x2,y2,cls,conf)]."""
    if args.onnx:
        import ast
        import onnxruntime as ort
        onnx_path = Path(args.model).with_suffix(".onnx") \
            if args.model else Path("exports/fold_1_best.onnx")
        if not onnx_path.is_file():
            raise SystemExit(f"ONNX tidak ada: {onnx_path} "
                             "(jalankan scripts/export_onnx.py dulu)")
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        model_names = ast.literal_eval(
            sess.get_modelmeta().custom_metadata_map["names"])
        inp = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        def predict(frame: np.ndarray) -> list[dict]:
            boxed, scale, pad_x, pad_y = letterbox_frame(frame)
            arr = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).astype(np.float32)
            arr /= 255.0
            x = arr.transpose(2, 0, 1)[None]
            y = sess.run([out_name], {inp: x})[0][0]      # (nc+4, N)
            y = y.T                                       # (N, nc+4)
            grouped: dict[int, list[tuple[list[int], float]]] = {}
            frame_height, frame_width = frame.shape[:2]
            for row in y:
                cls = row[4:]
                cid = int(cls.argmax())
                conf = float(row[4 + cid])
                if conf < args.conf:
                    continue
                cx, cy, w, h = row[:4]
                x1 = round((cx - w / 2 - pad_x) / scale)
                y1 = round((cy - h / 2 - pad_y) / scale)
                x2 = round((cx + w / 2 - pad_x) / scale)
                y2 = round((cy + h / 2 - pad_y) / scale)
                x1 = max(0, min(frame_width - 1, x1))
                y1 = max(0, min(frame_height - 1, y1))
                x2 = max(0, min(frame_width - 1, x2))
                y2 = max(0, min(frame_height - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                box = [x1, y1, x2 - x1, y2 - y1]
                grouped.setdefault(cid, []).append((box, conf))

            detections = []
            for cid, candidates in grouped.items():
                boxes = [candidate[0] for candidate in candidates]
                scores = [candidate[1] for candidate in candidates]
                keep = cv2.dnn.NMSBoxes(boxes, scores, args.conf, args.iou)
                for index in np.asarray(keep).reshape(-1):
                    box, score = candidates[int(index)]
                    x1, y1, width, height = box
                    detections.append({
                        "x1": x1, "y1": y1,
                        "x2": x1 + width, "y2": y1 + height,
                        "cls": model_names[cid], "conf": score,
                    })
            return detections

        return predict

    # default: PyTorch GPU
    from ultralytics import YOLO
    model = YOLO(str(args.model or find_best_pt(1)))

    def predict(frame: np.ndarray) -> list[dict]:
        r = model.predict(frame, imgsz=IMGSZ, conf=args.conf, verbose=False)[0]
        return [{"x1": int(b.xyxy[0][0]), "y1": int(b.xyxy[0][1]),
                 "x2": int(b.xyxy[0][2]), "y2": int(b.xyxy[0][3]),
                 "cls": model.names[int(c)], "conf": float(cf)}
                for b, c, cf in zip(r.boxes, r.boxes.cls, r.boxes.conf)]

    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45,
                    help="ambang NMS untuk menghapus kotak duplikat")
    ap.add_argument("--model", type=Path, default=None,
                    help="path .pt (default best.pt fold_1)")
    ap.add_argument("--onnx", action="store_true",
                    help="pakai ONNX CPU (uji jalur VPS) alih-alih torch GPU")
    ap.add_argument("--smooth", type=int, default=5,
                    help="jumlah frame voting (3 = responsif, 15 = stabil)")
    ap.add_argument("--save-dir", type=Path, default=Path("data/webcam_test"))
    args = ap.parse_args()

    predict = make_predictor(args)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} tidak bisa dibuka")
    if args.onnx is False:
        import torchQ   
        if not torch.cuda.is_available():
            print("CUDA tidak ada — jalankan dengan --onnx bila tanpa GPU")

    history: deque[str | None] = deque(maxlen=args.smooth)
    fps, t_prev, warmup = 0.0, time.perf_counter(), 0
    saved = 0
    print("Tombol: Q = keluar | S = screenshot ke "
          f"{args.save_dir} (untuk set uji webcam)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal baca frame")
            break
        t0 = time.perf_counter()
        dets = predict(frame)
        warmup += 1

        status = "Kamera Aktif"
        if warmup and dets:
            status = "Sedang Memproses"
        best = max(dets, key=lambda d: d["conf"]) if dets else None
        history.append(best["cls"] if best else None)
        if best:
            for d in dets:
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{d['cls']} {d['conf']:.2f}",
                            (d["x1"], max(16, d["y1"] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:
            status = "Gestur belum terbaca"

        votes = Counter(label for label in history if label is not None)
        stable, n = votes.most_common(1)[0] if votes else ("", 0)
        recognized = (n >= max(2, len(history) // 2 + 1)
                      and len(history) >= 3)
        if recognized:
            status = f"Gestur Terdeteksi: {stable} ({n}/{len(history)})"

        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1 / dt if dt else 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (14, 20, 24), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        if recognized:
            headline, color = f"GESTUR: {stable}", (80, 255, 100)
        elif best:
            headline, color = f"MEMBACA: {best['cls']}", (80, 220, 255)
        else:
            headline, color = "GESTUR BELUM TERBACA", (80, 150, 255)
        cv2.putText(frame, headline, (16, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3, cv2.LINE_AA)
        confidence = f"Keyakinan: {best['conf']:.0%}" if best else ""
        cv2.putText(frame, f"{confidence}   FPS: {fps:.1f}", (18, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245),
                    2, cv2.LINE_AA)

        cv2.imshow("Uji Realtime SIBI (Q=keluar, S=screenshot)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            args.save_dir.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            cv2.imwrite(str(args.save_dir / name), frame)
            saved += 1
            print(f"  screenshot: {args.save_dir / name}")

    cap.release()
    cv2.destroyAllWindows()
    if saved:
        print(f"{saved} screenshot tersimpan di {args.save_dir} "
              "-> beri label folder kelas untuk evaluasi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
