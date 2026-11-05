from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, metrics, predictions

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app = FastAPI(
    title="Glaucoma Fundus Research Demo",
    description="Local academic prototype; not for diagnosis or care decisions.",
)
app.include_router(health.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    built_index = FRONTEND_DIST / "index.html"
    if built_index.is_file():
        return FileResponse(built_index)
    return FileResponse(STATIC_INDEX)
