from fastapi import FastAPI

app = FastAPI(
    title="SIBI Translator API",
    description="API prototipe translasi dua arah SIBI dan Bahasa Indonesia.",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}