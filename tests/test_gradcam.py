import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.glaucoma_ml.explainability.gradcam import (
    align_heatmap_to_original,
    generate_gradcam,
    render_gradcam_images,
    target_layer_for,
)


class TinyResNet(nn.Module):
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


def test_gradcam_generates_normalized_map_for_target_class() -> None:
    model = TinyResNet().eval()
    image = torch.ones((1, 3, 16, 16), dtype=torch.float32)

    heatmap = generate_gradcam(
        model,
        image,
        target_layer_for(model, "resnet50"),
        class_index=1,
    )

    assert heatmap.shape == (16, 16)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0
    assert heatmap.max() == 1


def test_gradcam_visuals_align_to_original_image_and_encode_as_png() -> None:
    original = Image.new("RGB", (40, 20), color=(100, 120, 140))
    model_heatmap = np.ones((224, 224), dtype=np.float32)
    aligned = align_heatmap_to_original(model_heatmap, original.size)

    visuals = render_gradcam_images(original, aligned)

    assert aligned.shape == (20, 40)
    assert set(visuals) == {
        "original_image",
        "gradcam_heatmap",
        "gradcam_overlay",
    }
    for data_url in visuals.values():
        assert data_url.startswith("data:image/png;base64,")
        encoded = data_url.split(",", maxsplit=1)[1]
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))
        assert decoded.size == original.size
