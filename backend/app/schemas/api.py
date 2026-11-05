from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    predictions_available: bool
    model_message: str


class ImageValidationResponse(BaseModel):
    format: str
    width: int
    height: int
    message: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    disclaimer: str
    model_name: str
    explanation_method: str
    explanation_note: str
    explanation_error: str
    original_image: str | None = None
    gradcam_heatmap: str | None = None
    gradcam_overlay: str | None = None


class MetricsResponse(BaseModel):
    available: bool
    message: str
    metrics: dict[str, object] | None = None
