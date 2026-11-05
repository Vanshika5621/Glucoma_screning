from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

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
    assert "not a valid JPEG or PNG" in response.json()["detail"]


def test_image_validation_rejects_unsupported_format() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (16, 16)).save(image_buffer, format="BMP")
    response = client.post(
        "/api/images/validate",
        files={"file": ("sample.bmp", image_buffer.getvalue(), "image/bmp")},
    )

    assert response.status_code == 415
    assert "Only JPEG and PNG" in response.json()["detail"]


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
