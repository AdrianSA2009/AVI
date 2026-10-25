from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.inference import predict_frame

router = APIRouter(prefix="/api/v1")


@router.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    conf: Annotated[float, Form(ge=0, le=1)] = 0.25,
) -> dict[str, list[dict[str, str | float | list[float]]]]:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Gunakan gambar JPEG, PNG, atau WebP")

    contents = await image.read(8 * 1024 * 1024 + 1)
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 8 MiB")
    frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Berkas bukan gambar yang valid")

    return {"predictions": predict_frame(frame, conf)}
