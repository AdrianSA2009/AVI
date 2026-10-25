from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.inference import load_model, predict_frame

router = APIRouter(prefix="/api/v1")


@router.get("/model", tags=["inference"])
async def model_info() -> dict[str, object]:
    model = load_model()
    return {"classes": list(model.names.values()), "count": len(model.names)}


@router.post("/predict", tags=["inference"])
async def predict(image: Annotated[UploadFile, File()]) -> dict[str, list[dict[str, str | float | list[float]]]]:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Gunakan gambar JPEG, PNG, atau WebP")
    contents = await image.read(4 * 1024 * 1024 + 1)
    if len(contents) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 4 MiB")
    return {"predictions": predict_frame(contents)}
