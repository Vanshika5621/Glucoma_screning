from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

PREPROCESSING_VERSION = "rgb-letterbox-imagenet-v1"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def decode_and_resize(image_bytes: bytes, image_size: int = 224) -> Image.Image:
    if image_size < 1:
        raise ValueError("Image size must be a positive integer.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Could not decode retinal image.") from error

    height, width = rgb_array.shape[:2]
    scale = image_size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        rgb_array,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return Image.fromarray(canvas)


def build_image_transform(training: bool = False) -> transforms.Compose:
    operations: list[transforms.Transform] = []
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.05,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return transforms.Compose(operations)
