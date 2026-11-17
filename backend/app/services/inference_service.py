from functools import lru_cache
from io import BytesIO
import logging
import pickle
from threading import Lock

import cv2

from app.core.config import model_path

logger = logging.getLogger(__name__)
_inference_lock = Lock()
DISCLAIMER = (
    "Experimental research output only. This is not a diagnosis and must not "
    "be used for medical decisions."
)
EXPLANATION_NOTE = (
    "Grad-CAM is an approximate visualization of image regions that influenced "
    "this model's score. Highlighted regions do not prove glaucoma or disease."
)


@lru_cache(maxsize=1)
def _load_model() -> tuple[object, object, str, object, float, str] | None:
    checkpoint_path = model_path()
    if not checkpoint_path.is_file():
        return None

    try:
        import torch

        from src.glaucoma_ml.data.preprocessing import PREPROCESSING_VERSION
        from src.glaucoma_ml.models.classifier import Architecture, build_model

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        architecture = checkpoint["architecture"]
        if architecture not in ("resnet50", "efficientnet_b0"):
            raise ValueError("Checkpoint declares an unsupported architecture.")
        if checkpoint.get("class_order") != ["normal", "glaucoma"]:
            raise ValueError("Checkpoint class order is missing or incompatible.")
        if checkpoint.get("image_size") != 224:
            raise ValueError("Checkpoint image size is missing or incompatible.")
        if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Checkpoint preprocessing is missing or incompatible.")
        stored_threshold = checkpoint.get("threshold")
        if stored_threshold is None:
            threshold = 0.5
            threshold_source = "legacy checkpoint fallback (0.5)"
        elif (
            isinstance(stored_threshold, bool)
            or not isinstance(stored_threshold, (int, float))
            or not 0 <= stored_threshold <= 1
        ):
            raise ValueError("Checkpoint threshold must be between zero and one.")
        else:
            threshold = float(stored_threshold)
            threshold_source = "validation-selected checkpoint threshold"
        typed_architecture: Architecture = architecture
        model = build_model(typed_architecture, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model.to(device)
        model.eval()
        return model, torch, typed_architecture, device, threshold, threshold_source
    except (
        ImportError,
        IndexError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        logger.exception("Could not load research checkpoint from %s", checkpoint_path)
        return None


def model_status() -> tuple[bool, str]:
    if not model_path().is_file():
        return False, "No trained research model is configured."
    loaded = _load_model()
    if loaded is None:
        return False, "The configured model could not be loaded."
    return True, f"{loaded[2]} research model is available on {loaded[3]}."


def predict_image(image_bytes: bytes) -> dict[str, str | float]:
    with _inference_lock:
        return _predict_image_locked(image_bytes)


def _predict_image_locked(image_bytes: bytes) -> dict[str, str | float]:
    loaded = _load_model()
    if loaded is None:
        raise RuntimeError(
            "No compatible trained research model is configured. No prediction "
            "was made and the image was not retained."
        )

    model, torch, architecture, device, threshold, threshold_source = loaded
    from src.glaucoma_ml.data.preprocessing import (
        build_image_transform,
        decode_and_resize,
    )

    from PIL import Image, ImageOps

    image = decode_and_resize(image_bytes)
    with Image.open(BytesIO(image_bytes)) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    tensor = build_image_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    glaucoma_probability = float(probabilities[1].item())
    predicted_index = int(glaucoma_probability >= threshold)
    classes = ("normal", "glaucoma")
    result: dict[str, str | float] = {
        "prediction": classes[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "glaucoma_probability": glaucoma_probability,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "disclaimer": DISCLAIMER,
        "model_name": architecture,
        "explanation_method": "Grad-CAM",
        "explanation_note": EXPLANATION_NOTE,
    }
    try:
        from src.glaucoma_ml.explainability.gradcam import (
            align_heatmap_to_original,
            generate_gradcam,
            render_gradcam_images,
            target_layer_for,
        )

        model.eval()
        heatmap = generate_gradcam(
            model,
            tensor,
            target_layer_for(model, architecture),
            predicted_index,
        )
        original.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        aligned_heatmap = align_heatmap_to_original(
            heatmap,
            original.size,
        )
        result.update(render_gradcam_images(original, aligned_heatmap))
        result["explanation_error"] = ""
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error):
        logger.exception("Could not generate Grad-CAM visualization.")
        result["explanation_error"] = (
            "Prediction completed, but its Grad-CAM visualization is unavailable."
        )
    return result


def clear_model_cache() -> None:
    _load_model.cache_clear()
