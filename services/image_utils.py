from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_supported_image(file_name: str) -> bool:
    """Check if the uploaded filename has an allowed image extension."""
    if not file_name:
        return False
    return os.path.splitext(file_name)[1].lower() in ALLOWED_EXTENSIONS


def ensure_rgb(image: Any) -> Image.Image:
    """Convert an image to RGB so the model pipeline receives a consistent format."""
    if image is None:
        raise ValueError("No image was provided.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    if not isinstance(image, Image.Image):
        raise ValueError("Unsupported image type.")
    return image.convert("RGB")


def load_image(file_obj: Any) -> Image.Image:
    """Open an uploaded image or any file-like object and return a PIL image."""
    if file_obj is None:
        raise ValueError("No image uploaded.")

    try:
        if hasattr(file_obj, "read") and hasattr(file_obj, "name"):
            file_obj.seek(0)
            image = Image.open(file_obj)
            image.load()
            return ensure_rgb(image)

        if isinstance(file_obj, (bytes, bytearray)):
            image = Image.open(__import__("io").BytesIO(file_obj))
            image.load()
            return ensure_rgb(image)

        if isinstance(file_obj, str):
            if not os.path.exists(file_obj):
                raise ValueError("Image path does not exist.")
            with Image.open(file_obj) as image:
                image.load()
                return ensure_rgb(image)

        if hasattr(file_obj, "getvalue"):
            data = file_obj.getvalue()
            image = Image.open(__import__("io").BytesIO(data))
            image.load()
            return ensure_rgb(image)

        raise ValueError("Could not read the uploaded image.")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The image could not be opened. Please upload a valid JPG or PNG image.") from exc


def basic_image_quality_check(image: Any) -> dict:
    """Run a light-weight quality check before model inference."""
    try:
        rgb_image = ensure_rgb(image)
    except ValueError:
        return {"is_valid": False, "message": "The image is missing or invalid."}

    width, height = rgb_image.size
    if width < 100 or height < 100:
        return {
            "is_valid": False,
            "message": "The image is too small. Please upload a close-up image of the affected leaf.",
        }

    gray = np.array(rgb_image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()

    if brightness < 25:
        return {
            "is_valid": False,
            "message": "The image is very dark. Please upload a well-lit photo of the leaf.",
        }
    if brightness > 245:
        return {
            "is_valid": False,
            "message": "The image is very bright or washed out. Please upload a more balanced lighting setup.",
        }
    if contrast > 8 and blur_score < 60:
        return {
            "is_valid": False,
            "message": "The image is unclear. Please upload a well-lit close-up image of the affected leaf.",
        }

    return {
        "is_valid": True,
        "message": "Image quality looks acceptable for analysis.",
        "width": width,
        "height": height,
        "brightness": brightness,
        "blur_score": float(blur_score),
    }


def prepare_image_for_model(image: Any, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize and convert an image to a model-ready NumPy array."""
    rgb_image = ensure_rgb(image)
    processed = ImageOps.fit(rgb_image, target_size, method=Image.Resampling.BICUBIC)
    array = np.asarray(processed, dtype=np.float32) / 255.0
    return array


def get_image_metadata(image: Any) -> dict:
    """Return simple image metadata for display or validation."""
    rgb_image = ensure_rgb(image)
    return {
        "width": rgb_image.size[0],
        "height": rgb_image.size[1],
        "mode": rgb_image.mode,
    }
