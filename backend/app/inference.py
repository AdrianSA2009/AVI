import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(os.environ.get(
    "SIBI_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "ml" / "runs" / "static26_deploy" / "deploy.pt"),
))


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model A-Z belum tersedia di backend")
    return YOLO(str(MODEL_PATH))


def predict_frame(contents: bytes, conf: float = 0.35) -> list[dict[str, str | float | list[float]]]:
    try:
        with Image.open(__import__("io").BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Gunakan gambar JPEG, PNG, atau WebP")
            if image.width * image.height > 16_000_000:
                raise HTTPException(status_code=413, detail="Resolusi gambar terlalu besar")
            frame = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Berkas bukan gambar yang valid") from exc

    result = load_model().predict(frame, conf=conf, verbose=False, device="cpu")[0]
    return [
        {
            "label": str(result.names[int(box.cls[0])]),
            "confidence": float(box.conf[0]),
            "box": [float(value) for value in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]
