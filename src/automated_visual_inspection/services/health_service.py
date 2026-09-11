from automated_visual_inspection.api.models.health import HealthResponse


def get_health_status() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="automated-visual-inspection-api",
        version="0.1.0",
    )