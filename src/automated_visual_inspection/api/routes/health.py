from fastapi import APIRouter

from automated_visual_inspection.api.models.health import HealthResponse
from automated_visual_inspection.services.health_service import get_health_status


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return get_health_status()