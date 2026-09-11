from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from automated_visual_inspection.api.routes.health import router as health_router


app = FastAPI(
    title="Automated Visual Inspection API",
    version="0.1.0",
    description="Backend API for automated visual inspection workflows.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "name": "Automated Visual Inspection API",
        "status": "running",
    }


app.include_router(health_router, prefix="/api/v1")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "automated_visual_inspection.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()