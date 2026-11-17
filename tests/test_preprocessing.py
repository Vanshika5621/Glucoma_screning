from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.glaucoma_ml.data.dataset import FundusDataset
from src.glaucoma_ml.data.preprocessing import (
    IMAGE_MEAN,
    IMAGE_STD,
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


def test_decode_and_resize_converts_grayscale_and_rgba_to_rgb() -> None:
    for mode in ("L", "RGBA"):
        buffer = BytesIO()
        Image.new(mode, (80, 40), color=128).save(buffer, format="PNG")

        image = decode_and_resize(buffer.getvalue())

        assert image.mode == "RGB"
        assert image.size == (224, 224)
        assert np.asarray(image).shape == (224, 224, 3)


def test_shared_inference_transform_returns_normalized_rgb_tensor() -> None:
    image = decode_and_resize(image_bytes(120, 120))
    transform = build_image_transform()

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert first.dtype == second.dtype
    assert (first == second).all()


def test_inference_transform_applies_imagenet_channel_normalization() -> None:
    image = Image.new("RGB", (224, 224), color=(100, 150, 200))

    tensor = build_image_transform()(image)
    expected = np.asarray(
        [
            (channel / 255 - mean) / std
            for channel, mean, std in zip(
                (100, 150, 200),
                IMAGE_MEAN,
                IMAGE_STD,
            )
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(tensor[:, 112, 112].numpy(), expected, rtol=1e-6)


def test_training_transform_keeps_expected_tensor_shape() -> None:
    image = decode_and_resize(image_bytes(120, 120))

    tensor = build_image_transform(training=True)(image)

    assert tensor.shape == (3, 224, 224)


def test_validation_dataset_matches_shared_prediction_preprocessing(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "fundus.png"
    Image.new("RGB", (160, 80), color=(100, 150, 200)).save(image_path)
    frame = pd.DataFrame(
        [{"image_path": str(image_path), "label": "normal"}]
    )
    dataset_tensor, label = FundusDataset(frame, training=False)[0]
    expected_tensor = build_image_transform()(
        decode_and_resize(image_path.read_bytes())
    )

    assert label == 0
    assert dataset_tensor.shape == (3, 224, 224)
    assert (dataset_tensor == expected_tensor).all()


def test_invalid_image_bytes_are_rejected() -> None:
    with pytest.raises(ValueError, match="Could not decode retinal image"):
        decode_and_resize(b"not an image")
