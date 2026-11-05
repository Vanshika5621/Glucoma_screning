import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def model_path() -> Path:
    configured = os.getenv("GLAUCOMA_MODEL_PATH", "models/best_model.pt")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metrics_path() -> Path:
    configured = os.getenv("GLAUCOMA_METRICS_PATH", "reports/metrics.json")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def max_upload_bytes() -> int:
    limit = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    if limit < 1:
        raise ValueError("MAX_UPLOAD_BYTES must be a positive integer.")
    return limit
