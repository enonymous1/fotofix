"""Diagnostics for why a scan was (or was not) detected as text.

Replaces the former ``diagnose_misses.py`` / ``diagnose_text.py``
scripts, which hardcoded local drive paths and file lists. This takes
file paths as arguments and reports the same three OCR signals for
each: OSD orientation confidence, extracted-string length, and word
counts. Raises ``RuntimeError`` when Tesseract is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def diagnose_file(image_path, ocr_fn=None) -> str:
    """Return a diagnostic report for one image.

    ``ocr_fn`` is an injectable ``(image) -> (osd, text, data)`` used by
    tests; the CLI leaves it unset and uses Tesseract directly.
    """
    path = Path(image_path)
    lines = [f"\n--- Diagnosing: {path.name} ---"]
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((1500, 1500))
            if ocr_fn is None:
                from ..vision.ocr import _load_pytesseract

                pytesseract = _load_pytesseract()
                osd = pytesseract.image_to_osd(img, timeout=10)
                text = pytesseract.image_to_string(img, timeout=10).strip()
                data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT, timeout=10
                )
            else:
                osd, text, data = ocr_fn(img)

            try:
                conf = float(
                    osd.split("Orientation confidence:")[1].split("\n")[0].strip()
                )
                lines.append(f"OSD Orientation Confidence: {conf}")
            except (IndexError, ValueError):
                lines.append("OSD Orientation Confidence: unavailable")
            lines.append(f"Extracted String Length: {len(text)}")
            if text:
                lines.append(f"First 100 chars: {text[:100].replace(chr(10), ' ')}")
            words = [
                w
                for i, w in enumerate(data["text"])
                if int(data["conf"][i]) > 0 and len(w.strip()) > 1
            ]
            high_conf = [
                w
                for i, w in enumerate(data["text"])
                if int(data["conf"][i]) > 40 and len(w.strip()) > 1
            ]
            lines.append(f"Total Words (any confidence): {len(words)}")
            lines.append(f"High Confidence Words (>40): {len(high_conf)}")
    except RuntimeError:
        # Missing Tesseract is a setup error, not a per-image failure:
        # let it reach the CLI, which exits non-zero with the message.
        raise
    except Exception as e:
        lines.append(f"General error reading image: {e}")
    return "\n".join(lines)
