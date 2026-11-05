from io import BytesIO

import numpy as np
from PIL import Image

from src.glaucoma_ml.data.preprocessing import (
    PREPROCESSING_VERSION,
    build_image_transform,
    decode_and_resize,
)


def image_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def test_decode_and_resize_preserves_aspect_ratio_with_center_padding() -> None:
    image = decode_and_resize(image_bytes(160, 80))
    pixels = np.asarray(image)

    assert image.size == (224, 224)
    assert tuple(pixels[20, 112]) == (0, 0, 0)
    assert tuple(pixels[112, 112]) == (100, 150, 200)
    assert PREPROCESSING_VERSION == "rgb-letterbox-imagenet-v1"


def test_shared_inference_transform_returns_normalized_rgb_tensor() -> None:
    image = decode_and_resize(image_bytes(120, 120))
    transform = build_image_transform()

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert first.dtype == second.dtype
    assert (first == second).all()


def test_training_transform_keeps_expected_tensor_shape() -> None:
    image = decode_and_resize(image_bytes(120, 120))

    tensor = build_image_transform(training=True)(image)

    assert tensor.shape == (3, 224, 224)
