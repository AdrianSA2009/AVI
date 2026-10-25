from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from ultralytics import YOLO

MODEL_PATH = Path("/app/ml/runs/static26_deploy/deploy.pt") if Path("/app").is_dir() else Path(__file__).resolve().parents[2] / "ml" / "runs" / "static26_deploy" / "deploy.pt"


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"Model tidak ditemukan: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def predict_frame(image: np.ndarray, conf: float) -> list[dict[str, str | float | list[float]]]:
    result = load_model().predict(image, conf=conf, verbose=False)[0]
    predictions = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        predictions.append({
            "label": str(result.names[class_id]),
            "confidence": float(box.conf[0]),
            "box": [float(value) for value in box.xyxy[0].tolist()],
        })
    return predictions
