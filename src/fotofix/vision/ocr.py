"""Tesseract OCR helpers: orientation detection and text affinity.

``pytesseract`` and the Tesseract binary are imported / located lazily,
so importing this module never fails on a machine without them. Call
``tesseract_available()`` to preflight, and expect a ``RuntimeError``
with a readable message from the detectors when Tesseract is missing.
"""

from __future__ import annotations

import os
import shutil

from ..config import TESSERACT_PATH

_VISION_HINT = "Install Tesseract OCR and set TESSERACT_PATH if needed."


def _load_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for OCR-based detection. "
            "Install it with: pip install -e '.[vision]'. "
            + _VISION_HINT
        ) from exc
    cmd = TESSERACT_PATH
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
    elif cmd and not os.path.exists(cmd):
        raise RuntimeError(
            f"Tesseract binary not found at {cmd!r} and not on PATH. "
            + _VISION_HINT
        )
    return pytesseract


def tesseract_available() -> bool:
    """True when pytesseract imports and a Tesseract binary is reachable."""
    try:
        _load_pytesseract()
    except RuntimeError:
        return False
    return True


def _parse_osd(osd_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in osd_text.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip()] = val.strip()
    return result


def _osd_float(values: dict[str, str], key: str) -> float:
    try:
        return float(values.get(key, 0))
    except (TypeError, ValueError):
        return 0.0


def detect_orientation_tesseract(proxy_img):
    """Detect orientation with Tesseract OSD. Returns ``(angle, conf)``.

    Returns ``(0, 0)`` when OSD cannot confidently determine an
    orientation for this image; raises ``RuntimeError`` only when
    Tesseract itself is unavailable.
    """
    pytesseract = _load_pytesseract()
    try:
        osd = pytesseract.image_to_osd(proxy_img, timeout=20)
        values = _parse_osd(osd)
        angle = int(_osd_float(values, "Rotate"))
        conf = _osd_float(values, "Orientation confidence")
        script = values.get("Script", "Unknown")
        threshold = 1.0 if script == "Latin" else 2.0
        if conf >= threshold:
            return angle, conf
        return 0, 0
    except RuntimeError:
        raise
    except Exception:
        return 0, 0


def get_document_affinity(proxy_img) -> float:
    """Heuristic 0.0-1.0 for 'how much is this a text-heavy document?'.

    Uses Tesseract script/orientation confidence as a proxy, as in the
    original code. Raises ``RuntimeError`` when Tesseract is missing;
    callers that only use this as a hint (optimize) catch it and fall
    back to treating the image as a photo.
    """
    pytesseract = _load_pytesseract()
    try:
        osd = pytesseract.image_to_osd(proxy_img, config="--psm 0", timeout=20)
        values = _parse_osd(osd)
        script_conf = _osd_float(values, "Script confidence")
        orient_conf = _osd_float(values, "Orientation confidence")
        score = min(1.0, (script_conf / 20.0) + (orient_conf / 40.0))
        return score
    except RuntimeError:
        raise
    except Exception:
        return 0.0
