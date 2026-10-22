"""Small UI utilities."""

from pathlib import Path
from PIL import Image


def load_catalog_image(crop_path: str, size: tuple[int, int] = (300, 300)):
    """Load a catalog crop image; returns None if missing/unreadable."""
    try:
        if not crop_path or not Path(crop_path).is_file():
            return None
        img = Image.open(crop_path).convert("RGB")
        img.thumbnail(size)
        return img
    except Exception:
        return None


def is_verified(match: dict) -> bool:
    return str(match.get("verification_status", "")).lower() == "verified" or str(
        match.get("data_origin", "")
    ).lower() in ("verified", "real")
