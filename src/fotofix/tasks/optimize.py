"""Image optimization: resize + re-encode as WebP or JPEG.

Whether a scan is treated as a document (smaller max dimension) or a
photo is decided by the OCR document-affinity heuristic. When Tesseract
is unavailable the image is treated as a photo and optimized anyway;
that fallback is reported in the result string.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

from ..config import (
    OPTIMIZE_MAX_DIM_DOC,
    OPTIMIZE_MAX_DIM_PHOTO,
    OPTIMIZE_QUALITY_JPEG,
    OPTIMIZE_QUALITY_WEBP,
    OPTIMIZE_TARGET_FORMAT,
)
from ..image import get_proxy_image


def optimize_image(
    source_path,
    target_dir=None,
    dry_run: bool = True,
    target_format: str | None = None,
    quality: int | None = None,
):
    """Optimize one image. Returns a one-line result string."""
    source = Path(source_path)
    filename = source.name
    base, _ext = os.path.splitext(filename)

    proxy = get_proxy_image(source)
    if proxy is None:
        return f"SKIP: {filename} (Load Error)"

    ocr_unavailable = False
    try:
        from ..vision.ocr import get_document_affinity

        doc_score = get_document_affinity(proxy)
    except RuntimeError:
        doc_score = 0.0
        ocr_unavailable = True
    is_doc = doc_score > 0.5
    type_label = "Document" if is_doc else "Photo"
    if ocr_unavailable:
        type_label += " (OCR unavailable)"

    fmt = (target_format or OPTIMIZE_TARGET_FORMAT).lower()
    if fmt not in ("webp", "jpeg"):
        return f"ERROR: Unknown target format {fmt!r} for {filename}"
    max_dim = OPTIMIZE_MAX_DIM_DOC if is_doc else OPTIMIZE_MAX_DIM_PHOTO
    if quality is None:
        quality = OPTIMIZE_QUALITY_WEBP if fmt == "webp" else OPTIMIZE_QUALITY_JPEG

    dest_ext = f".{fmt}"
    dest_filename = f"{base}_optimized{dest_ext}"
    if target_dir:
        dest_path = Path(target_dir) / dest_filename
    else:
        dest_path = source.parent / dest_filename

    if dry_run:
        return (
            f"DRY RUN: Would optimize {filename} as {type_label} -> "
            f"{dest_filename} (Max Dim: {max_dim}px)"
        )

    try:
        if target_dir:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            exif = img.getexif()
            if exif:
                exif[274] = 1
            if fmt == "webp":
                if img.mode == "P":
                    img = img.convert("RGBA")
                img.save(dest_path, "WEBP", quality=quality, lossless=False)
            else:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                kwargs: dict = {
                    "format": "JPEG",
                    "quality": quality,
                    "subsampling": 0,
                }
                if exif:
                    kwargs["exif"] = exif.tobytes()
                img.save(dest_path, **kwargs)

            old_size = os.path.getsize(source) / (1024 * 1024)
            new_size = os.path.getsize(dest_path) / (1024 * 1024)
            reduction = ((old_size - new_size) / old_size) * 100 if old_size > 0 else 0
            return (
                f"OPTIMIZED: {filename} ({type_label}) | {old_size:.1f}MB "
                f"-> {new_size:.1f}MB ({reduction:.0f}% smaller)"
            )
    except Exception as e:
        return f"ERROR: Failed to optimize {filename}: {e}"
