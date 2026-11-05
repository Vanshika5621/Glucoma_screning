from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.api import ImageValidationResponse, PredictionResponse
from app.services.image_service import read_validated_image
from app.services.inference_service import predict_image

router = APIRouter(tags=["images"])


@router.post("/images/validate", response_model=ImageValidationResponse)
async def validate_uploaded_image(file: UploadFile = File(...)) -> ImageValidationResponse:
    _, image_format, width, height = await read_validated_image(file)
    return ImageValidationResponse(
        format=image_format,
        width=width,
        height=height,
        message=(
            "The image passed file-format checks. Fundus content is not verified; "
            "no prediction was made and the image was not retained."
        ),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    content, _, _, _ = await read_validated_image(file)
    try:
        result = await run_in_threadpool(predict_image, content)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PredictionResponse(**result)
