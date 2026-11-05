from io import BytesIO
import warnings

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import max_upload_bytes

MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}


async def read_validated_image(file: UploadFile) -> tuple[bytes, str, int, int]:
    content = await file.read(max_upload_bytes() + 1)
    if len(content) > max_upload_bytes():
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the configured upload size limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image_format = image.format
                if (
                    not isinstance(image_format, str)
                    or image_format not in ALLOWED_IMAGE_FORMATS
                ):
                    raise HTTPException(
                        status_code=415,
                        detail="Only JPEG and PNG images are supported.",
                    )
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions are too large to process safely.",
                    )
                image.verify()
    except HTTPException:
        raise
    except Image.DecompressionBombError as error:
        raise HTTPException(
            status_code=413,
            detail="Image dimensions are too large to process safely.",
        ) from error
    except (
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="The file is not a valid JPEG or PNG image.",
        ) from error

    return content, image_format, width, height
