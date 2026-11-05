import json
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import metrics_path
from app.schemas.api import MetricsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["evaluation"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse | JSONResponse:
    report_path = metrics_path()
    if not report_path.is_file():
        return MetricsResponse(
            available=False,
            message="No evaluation report exists yet. Train and evaluate a model first.",
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read evaluation report at %s", report_path)
        return JSONResponse(
            status_code=500,
            content={"detail": "The evaluation report could not be read."},
        )
    if not isinstance(report, dict) or not report:
        return JSONResponse(
            status_code=500,
            content={"detail": "The evaluation report has an invalid format."},
        )

    return MetricsResponse(
        available=True,
        message="Metrics from the latest local evaluation report.",
        metrics=report,
    )
