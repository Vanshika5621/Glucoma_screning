from fastapi import APIRouter

from app.schemas.api import HealthResponse
from app.services.inference_service import model_status

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available, message = model_status()
    return HealthResponse(
        status="ok",
        predictions_available=available,
        model_message=message,
    )
