"""Export model deteksi ke ONNX + benchmark latency torch vs ONNX.

Output:
    exports/<fold>_best.onnx       model ONNX (siap untuk VPS/onnxruntime)
    runs/onnx_report.json          hasil export + benchmark + parity

Parity = selisih output mentah torch vs ONNX pada input sama (sanity check;
harus kecil, mis. < 1e-3).

Pemakaian (PowerShell, dari C:\\AVI\\ml):
    python scripts/export_onnx.py
    python scripts/export_onnx.py --iters 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMGSZ = 640


def find_best_pt(run: int) -> Path:
    candidates = [
        Path(f"runs/detect/runs/detect/fold_{run}/weights/best.pt"),
        Path(f"runs/detect/fold_{run}/weights/best.pt"),
        Path(f"runs/fold_{run}/weights/best.pt"),
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise SystemExit("Model tidak ada, dicoba: "
                         + ", ".join(str(p) for p in candidates))
    return found


def sample_tensor() -> tuple[np.ndarray, Path]:
    """Satu gambar nyata dari dataset, NCHW float32 0..1 (640x640)."""
    root = Path("data/processed/detect/images")
    img = next(p for p in sorted(root.iterdir())
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    with Image.open(img) as im:
        arr = np.asarray(im.convert("RGB").resize((IMGSZ, IMGSZ)),
                         dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], img


def export_onnx(pt: Path, out_dir: Path, imgsz: int) -> Path:
    from ultralytics import YOLO
    path = YOLO(str(pt)).export(format="onnx", imgsz=imgsz, device="cpu",
                                opset=13, simplify=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pt.parent.parent.name}_{pt.stem}.onnx"
    shutil.copy2(path, dest)
    return dest


def raw_torch(pt: Path, tensor: np.ndarray) -> np.ndarray:
    """Output mentah graph torch (setara grafik ONNX)."""
    import torch
    from ultralytics import YOLO
    net = YOLO(str(pt)).model
    net.eval()
    with torch.inference_mode():
        out = net(torch.from_numpy(tensor))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out[0].cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)


def bench_torch(pt: Path, tensor: np.ndarray, iters: int) -> dict:
    import torch
    from ultralytics import YOLO
    net = YOLO(str(pt)).model
    net.eval()
    net.to("cuda:0")
    x = torch.from_numpy(tensor).to("cuda:0")
    with torch.inference_mode():
        for _ in range(10):
            net(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return {"ms_avg": round(statistics.mean(times), 2),
            "ms_min": round(min(times), 2),
            "fps": round(1000 / statistics.mean(times)),
            "device": torch.cuda.get_device_name(0)}


def bench_onnx(onnx_path: Path, tensor: np.ndarray, iters: int,
               ref: np.ndarray) -> tuple[dict, float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    for _ in range(10):
        out = sess.run(None, {inp: tensor})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = sess.run(None, {inp: tensor})
        times.append((time.perf_counter() - t0) * 1000)
    parity = float(np.abs(np.asarray(out[0]) - ref).max())
    return ({"ms_avg": round(statistics.mean(times), 2),
             "ms_min": round(min(times), 2),
             "fps": round(1000 / statistics.mean(times)),
             "providers": sess.get_providers()}), parity


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None,
                    help="path .pt (default: best.pt fold_1 & fold_2)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args(argv)

    pts = [args.model] if args.model else [find_best_pt(1), find_best_pt(2)]
    tensor, img = sample_tensor()
    report: dict = {"imgsz": args.imgsz, "iters": args.iters,
                    "sample": str(img), "models": []}

    for pt in pts:
        print(f"=== {pt} ===")
        ref = raw_torch(pt, tensor)
        onnx_path = export_onnx(pt, Path("exports"), args.imgsz)
        print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

        torch_b = bench_torch(pt, tensor, args.iters)
        print(f"  torch  {torch_b['device']}: {torch_b['ms_avg']} ms "
              f"({torch_b['fps']} FPS)")

        onnx_b, parity = bench_onnx(onnx_path, tensor, args.iters, ref)
        print(f"  onnx   {onnx_b['providers'][0]}: {onnx_b['ms_avg']} ms "
              f"({onnx_b['fps']} FPS) | parity max|diff|={parity:.2e}")

        report["models"].append({
            "pt": str(pt), "onnx": str(onnx_path),
            "onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
            "torch_gpu": torch_b, "onnx_cpu": onnx_b,
            "parity_max_abs_diff": parity,
        })

    Path("runs").mkdir(exist_ok=True)
    (Path("runs") / "onnx_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nLaporan: runs/onnx_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
