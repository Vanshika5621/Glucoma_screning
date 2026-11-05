from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from src.glaucoma_ml.models.classifier import Architecture


def target_layer_for(model: nn.Module, architecture: Architecture) -> nn.Module:
    if architecture == "resnet50":
        try:
            block = model.layer4[-1]
            return block.conv3
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "The loaded model does not expose the expected ResNet50 layer4 "
                "bottleneck conv3."
            ) from error
    if architecture == "efficientnet_b0":
        try:
            return model.features[-1]
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "The loaded model does not expose the expected EfficientNet-B0 features."
            ) from error
    raise ValueError(f"Grad-CAM is unsupported for architecture: {architecture}")


def generate_gradcam(
    model: nn.Module,
    input_tensor: Tensor,
    target_layer: nn.Module,
    class_index: int,
) -> np.ndarray:
    activations: list[Tensor] = []

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        activations.append(output)

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or not 0 <= class_index < logits.shape[1]:
                raise ValueError("The model output does not contain the target class.")
            if not activations:
                raise RuntimeError("The Grad-CAM target layer did not run.")
            activation = activations[-1]
            gradients = torch.autograd.grad(
                logits[:, class_index].sum(),
                activation,
                retain_graph=False,
                create_graph=False,
            )[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = torch_functional.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            maximum = cam.max()
            if not torch.isfinite(maximum) or float(maximum.item()) <= 1e-12:
                raise ValueError("Grad-CAM produced no meaningful activation map.")
            cam = cam / maximum
            return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()


def align_heatmap_to_original(
    heatmap: np.ndarray,
    original_size: tuple[int, int],
    model_input_size: int = 224,
) -> np.ndarray:
    original_width, original_height = original_size
    if original_width < 1 or original_height < 1:
        raise ValueError("Original image dimensions must be positive.")
    scale = model_input_size / max(original_width, original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (model_input_size - resized_width) // 2
    top = (model_input_size - resized_height) // 2
    content = heatmap[
        top : top + resized_height,
        left : left + resized_width,
    ]
    if content.size == 0:
        raise ValueError("Grad-CAM map does not align with the preprocessed image.")
    aligned = cv2.resize(
        content,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(aligned, 0.0, 1.0)


def encode_png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_gradcam_images(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> dict[str, str]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between zero and one.")
    if heatmap.shape != (original.height, original.width):
        raise ValueError("Grad-CAM map must match the displayed image dimensions.")
    normalized = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_image = Image.fromarray(heatmap_rgb)
    overlay = Image.blend(original.convert("RGB"), heatmap_image, alpha=alpha)
    return {
        "original_image": encode_png_data_url(original.convert("RGB")),
        "gradcam_heatmap": encode_png_data_url(heatmap_image),
        "gradcam_overlay": encode_png_data_url(overlay),
    }
