from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app

client = TestClient(app)


def png_image() -> bytes:
    image_buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(60, 80, 90)).save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def test_homepage_serves_upload_ui_with_safety_notice() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="root"' in response.text or "Fundus image research demo" in response.text


def test_health_reports_model_unavailable(monkeypatch) -> None:
    from app.api.routes import health

    monkeypatch.setattr(
        health,
        "model_status",
        lambda: (False, "No trained research model is configured."),
    )
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["predictions_available"] is False
    assert "No trained" in response.json()["model_message"]


def test_cors_allows_configured_frontend_origin() -> None:
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_allow_unconfigured_frontend_origin() -> None:
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_metrics_endpoint_reports_unavailable_without_fake_values(
    monkeypatch,
    tmp_path,
) -> None:
    from app.api.routes import metrics

    monkeypatch.setattr(metrics, "metrics_path", lambda: tmp_path / "missing.json")
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["metrics"] is None


def test_metrics_endpoint_reports_invalid_json(monkeypatch, tmp_path) -> None:
    from app.api.routes import metrics

    invalid_report = tmp_path / "metrics.json"
    invalid_report.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(metrics, "metrics_path", lambda: invalid_report)

    response = client.get("/api/metrics")

    assert response.status_code == 500
    assert response.json() == {"detail": "The evaluation report could not be read."}


def test_metrics_endpoint_annotates_legacy_counts_without_inventing_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    from app.api.routes import metrics

    report_path = tmp_path / "metrics.json"
    report_path.write_text(
        '{"model_name":"resnet50","test_samples":10,'
        '"confusion_matrix":[[8,1],[1,0]]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(metrics, "metrics_path", lambda: report_path)

    response = client.get("/api/metrics")

    assert response.status_code == 200
    result = response.json()["metrics"]
    assert result["class_counts"] == {"normal": 9, "glaucoma": 1}
    assert result["normal_baseline_accuracy"] == 0.9
    assert result["threshold"] == 0.5
    assert "legacy report fallback" in result["threshold_source"]
    assert result.get("pr_auc") is None


def test_image_validation_accepts_png_without_claiming_fundus_verification() -> None:
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.png", png_image(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["format"] == "PNG"
    assert response.json()["width"] == 64
    assert "Fundus content is not verified" in response.json()["message"]
    assert "image was not retained" in response.json()["message"]


def test_image_validation_uses_content_not_filename_or_mime_type(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    response = client.post(
        "/api/images/validate",
        files={"file": ("../../misleading.jpg", png_image(), "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json()["format"] == "PNG"
    assert "../../misleading.jpg" not in response.text
    assert list(tmp_path.iterdir()) == []


def test_image_validation_rejects_empty_upload() -> None:
    response = client.post(
        "/api/images/validate",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert "uploaded image is empty" in response.json()["detail"]


def test_image_validation_rejects_non_image_payload() -> None:
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "not a valid supported image" in response.json()["detail"]


def test_prediction_rejects_invalid_image_before_inference(monkeypatch) -> None:
    from app.api.routes import predictions

    called = False

    def fake_predict(_content: bytes) -> dict[str, str | float]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(predictions, "predict_image", fake_predict)
    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert "not a valid supported image" in response.json()["detail"]
    assert not called


@pytest.mark.parametrize("image_format", ["BMP", "GIF", "TIFF", "WEBP"])
def test_image_validation_accepts_common_raster_formats(image_format: str) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (16, 16)).save(image_buffer, format=image_format)
    response = client.post(
        "/api/images/validate",
        files={
            "file": (
                f"sample.{image_format.lower()}",
                image_buffer.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["format"] == image_format


def test_image_validation_rejects_unsupported_decodable_format() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (16, 16)).save(image_buffer, format="ICO")
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.ico", image_buffer.getvalue(), "image/x-icon")},
    )

    assert response.status_code == 415
    assert "Supported image formats are" in response.json()["detail"]


def test_image_validation_rejects_excessive_pixel_count(monkeypatch) -> None:
    from app.services import image_service

    monkeypatch.setattr(image_service, "MAX_IMAGE_PIXELS", 100)
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.png", png_image(), "image/png")},
    )

    assert response.status_code == 413
    assert "dimensions are too large" in response.json()["detail"]


def test_image_validation_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4")
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.png", b"12345", "image/png")},
    )

    assert response.status_code == 413
    assert "configured upload size limit" in response.json()["detail"]


def test_prediction_endpoint_never_returns_placeholder_result(monkeypatch) -> None:
    from app.services import inference_service

    monkeypatch.setattr(inference_service, "_load_model", lambda: None)
    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", png_image(), "image/png")},
    )

    assert response.status_code == 503
    assert "No compatible trained research model is configured" in response.json()["detail"]
    assert "image was not retained" in response.json()["detail"]


def test_prediction_response_includes_gradcam_visualizations(monkeypatch) -> None:
    from app.api.routes import predictions

    monkeypatch.setattr(
        predictions,
        "predict_image",
        lambda _content: {
            "prediction": "glaucoma",
            "confidence": 0.8,
            "glaucoma_probability": 0.8,
            "threshold": 0.5,
            "threshold_source": "validation-selected checkpoint threshold",
            "disclaimer": "Research only.",
            "model_name": "resnet50",
            "explanation_method": "Grad-CAM",
            "explanation_note": "Approximate explanation.",
            "explanation_error": "",
            "original_image": "data:image/png;base64,original",
            "gradcam_heatmap": "data:image/png;base64,heatmap",
            "gradcam_overlay": "data:image/png;base64,overlay",
        },
    )

    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", png_image(), "image/png")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "glaucoma"
    assert result["explanation_method"] == "Grad-CAM"
    assert result["gradcam_heatmap"].endswith("heatmap")
    assert result["gradcam_overlay"].endswith("overlay")
