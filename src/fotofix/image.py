"""Image loading, proxy creation, and corrected-image saving."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from .config import JPEG_QUALITY, MAX_PROXY_DIM

# Suppress common PIL/Tiff warnings from scanners, and allow the large
# images high-resolution scans produce.
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", message="Image size.*exceeds limit")
Image.MAX_IMAGE_PIXELS = None

# EXIF orientation tag.
_ORIENTATION_TAG = 274


def get_proxy_image(image_path, max_dim: int = MAX_PROXY_DIM):
    """Small EXIF-normalized copy of an image, for fast analysis.

    Returns ``None`` when the file cannot be loaded.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            return img.copy()
    except Exception:
        return None


def _unique_path(directory: Path, filename: str) -> Path:
    """A non-colliding path in ``directory`` for ``filename``."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = os.path.splitext(filename)
    n = 2
    while True:
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def save_corrected_image(source_path, angle: int, backup_dir=None) -> None:
    """Physically rotate an image and rewrite it in place.

    ``angle`` is the clockwise rotation the detectors reported. Pillow
    rotates counter-clockwise, hence the negation.

    Unlike the original code, this preserves the rest of the EXIF data
    (dates, scanner tags) and only resets the orientation tag to 1, so
    a corrected file does not silently lose its metadata. When
    ``backup_dir`` is given, the untouched original is copied there
    first (never overwriting an existing backup). The rewrite goes
    through a temporary file in the same directory plus ``os.replace``,
    so a failed save cannot leave a truncated original behind.
    """
    source = Path(source_path)
    if backup_dir is not None:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, _unique_path(backup_dir, source.name))

    with Image.open(source) as img:
        fmt = img.format
        img = ImageOps.exif_transpose(img)
        if angle != 0:
            img = img.rotate(-angle, expand=True)
        exif = img.getexif()
        if exif:
            exif[_ORIENTATION_TAG] = 1

        save_kwargs: dict = {}
        if fmt == "JPEG":
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            save_kwargs.update(
                format="JPEG", quality=JPEG_QUALITY, subsampling=0
            )
            if exif:
                save_kwargs["exif"] = exif.tobytes()
        elif fmt:
            save_kwargs["format"] = fmt

        fd, tmp_name = tempfile.mkstemp(
            dir=str(source.parent), prefix=".fotofix-", suffix=source.suffix
        )
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                img.save(tmp_file, **save_kwargs)
            os.replace(tmp_name, source)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
