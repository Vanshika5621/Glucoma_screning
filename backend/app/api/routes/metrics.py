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

    confusion = report.get("confusion_matrix")
    if (
        isinstance(confusion, list)
        and len(confusion) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in confusion)
        and all(isinstance(value, (int, float)) for row in confusion for value in row)
    ):
        normal_count = int(sum(confusion[0]))
        glaucoma_count = int(sum(confusion[1]))
        evaluation_samples = normal_count + glaucoma_count
        report.setdefault(
            "class_counts",
            {"normal": normal_count, "glaucoma": glaucoma_count},
        )
        report.setdefault("positive_samples", glaucoma_count)
        if evaluation_samples:
            report.setdefault(
                "normal_baseline_accuracy",
                normal_count / evaluation_samples,
            )
    threshold_was_missing = "threshold" not in report
    report.setdefault(
        "evaluation_samples",
        report.get("test_samples", 0),
    )
    report.setdefault("threshold", 0.5)
    if "threshold_source" not in report and threshold_was_missing:
        report["threshold_source"] = (
            "legacy report fallback (0.5); not validation-selected"
        )

    return MetricsResponse(
        available=True,
        message="Metrics from the latest local evaluation report.",
        metrics=report,
    )
