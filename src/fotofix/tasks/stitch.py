"""Front/back stitching: combine a front scan with its ``_b`` back scan.

The back scan is scaled proportionally so its shared edge matches the
front scan's edge; the front scan itself is never resampled. The
default direction is ``bottom`` (front on top, back below), which is
what the original code did — the original README's "side-by-side"
wording was wrong, and the direction is now explicit instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

from ..config import STITCH_BG_COLOR, STITCH_MARGIN, STITCH_QUALITY

DIRECTIONS = ("top", "bottom", "left", "right")


def stitch_front_back(
    front_path,
    back_path,
    output_path,
    dry_run: bool = True,
    direction: str = "bottom",
    margin: int = STITCH_MARGIN,
):
    """Stitch front + back. Returns a one-line result string."""
    direction = direction.lower()
    if direction not in DIRECTIONS:
        name = os.path.basename(str(front_path))
        return f"ERROR stitching {name}: unknown direction {direction!r}"

    if dry_run:
        return (
            f"DRY RUN: Would stitch {os.path.basename(str(front_path))} + "
            f"{os.path.basename(str(back_path))} -> "
            f"{os.path.basename(str(output_path))} ({direction})"
        )

    try:
        output_path = Path(output_path)
        if output_path.parent and str(output_path.parent) not in ("", "."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(front_path) as front_img, Image.open(back_path) as back_img:
            front_img = ImageOps.exif_transpose(front_img).convert("RGB")
            back_img = ImageOps.exif_transpose(back_img).convert("RGB")

            if direction in ("top", "bottom"):
                if back_img.width != front_img.width:
                    new_h = max(
                        1, round(back_img.height * front_img.width / back_img.width)
                    )
                    back_img = back_img.resize(
                        (front_img.width, new_h), Image.LANCZOS
                    )
                canvas = Image.new(
                    "RGB",
                    (front_img.width, front_img.height + back_img.height + margin),
                    STITCH_BG_COLOR,
                )
                if direction == "bottom":
                    canvas.paste(front_img, (0, 0))
                    canvas.paste(back_img, (0, front_img.height + margin))
                else:
                    canvas.paste(back_img, (0, 0))
                    canvas.paste(front_img, (0, back_img.height + margin))
            else:
                if back_img.height != front_img.height:
                    new_w = max(
                        1, round(back_img.width * front_img.height / back_img.height)
                    )
                    back_img = back_img.resize(
                        (new_w, front_img.height), Image.LANCZOS
                    )
                canvas = Image.new(
                    "RGB",
                    (front_img.width + back_img.width + margin, front_img.height),
                    STITCH_BG_COLOR,
                )
                if direction == "right":
                    canvas.paste(front_img, (0, 0))
                    canvas.paste(back_img, (front_img.width + margin, 0))
                else:
                    canvas.paste(back_img, (0, 0))
                    canvas.paste(front_img, (back_img.width + margin, 0))

            canvas.save(output_path, "JPEG", quality=STITCH_QUALITY, subsampling=0)
            return f"STITCHED: {os.path.basename(str(output_path))}"
    except Exception as e:
        return f"ERROR stitching {os.path.basename(str(front_path))}: {e}"
