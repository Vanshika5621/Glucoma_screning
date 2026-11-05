from io import BytesIO

import torch
from PIL import Image
from torch import nn

from app.services import inference_service


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Sequential(TinyBottleneck())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.layer4(image)
        score = features.mean(dim=(1, 2, 3))
        return torch.stack((-score, score), dim=1)


class TinyBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.conv3.weight, 1.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv3(image)


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_prediction_returns_class_specific_gradcam_images(monkeypatch) -> None:
    model = TinyClassifier().eval()
    monkeypatch.setattr(
        inference_service,
        "_load_model",
        lambda: (model, torch, "resnet50", torch.device("cpu")),
    )

    result = inference_service.predict_image(image_bytes())

    assert result["prediction"] == "glaucoma"
    assert result["explanation_method"] == "Grad-CAM"
    assert result["explanation_error"] == ""
    assert result["original_image"].startswith("data:image/png;base64,")
    assert result["gradcam_heatmap"].startswith("data:image/png;base64,")
    assert result["gradcam_overlay"].startswith("data:image/png;base64,")


def test_prediction_survives_unsupported_gradcam_layer(monkeypatch) -> None:
    model = TinyClassifier().eval()
    monkeypatch.setattr(
        inference_service,
        "_load_model",
        lambda: (model, torch, "efficientnet_b0", torch.device("cpu")),
    )

    result = inference_service.predict_image(image_bytes())

    assert result["prediction"] == "glaucoma"
    assert "visualization is unavailable" in result["explanation_error"]
    assert "gradcam_overlay" not in result
