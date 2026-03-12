from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from gui_detector_api.errors import InvalidUploadError, PayloadTooLargeError

SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@dataclass
class LoadedImage:
    image: Image.Image
    filename: str | None
    content_type: str
    size_bytes: int


async def load_image_from_upload(upload: UploadFile, max_size_bytes: int) -> LoadedImage:
    content_type = (upload.content_type or "").lower()
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise InvalidUploadError("Only PNG and JPEG uploads are supported.", status_code=415)

    payload = await upload.read()
    if not payload:
        raise InvalidUploadError("Uploaded image is empty.")
    if len(payload) > max_size_bytes:
        raise PayloadTooLargeError(f"Upload exceeds the {max_size_bytes} byte limit.")

    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except UnidentifiedImageError as exc:
        raise InvalidUploadError("Uploaded file is not a valid image.") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    return LoadedImage(
        image=image,
        filename=upload.filename,
        content_type=content_type,
        size_bytes=len(payload),
    )
