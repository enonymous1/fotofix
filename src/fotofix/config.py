"""Single source of configuration for fotofix.

Every tunable lives here and is read by the tasks, so the CLI, the
library code, and the tests never drift into separate hardcoded copies.
Environment overrides are honoured where a value is machine-specific
(Tesseract location, model location).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Tesseract -------------------------------------------------------------
# The Windows default install location. Override with TESSERACT_PATH on
# any platform, or put ``tesseract`` on PATH (then the env var may be
# left unset and the PATH lookup is used).
TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --- Orientation model -----------------------------------------------------
# Third-party ONNX orientation model, downloaded on first use into a
# per-user directory (never into the repository). The class-map
# assumption {0: 0, 1: 90, 2: 180, 3: 270} comes from the upstream model
# and is documented in vision/ai.py; verify it against known-rotated
# scans if the upstream model ever changes.
MODEL_URL = os.getenv(
    "FOTOFIX_MODEL_URL",
    "https://github.com/duartebarbosadev/deep-image-orientation-detection"
    "/releases/download/v2/orientation_model_v2_0.9882.onnx",
)
DEFAULT_MODEL_NAME = "orientation_model.onnx"
# Optional SHA-256 of the downloaded model, checked when set via
# FOTOFIX_MODEL_SHA256. Unset by default because no hash is published
# alongside the upstream release; set it once you have verified a copy.
MODEL_SHA256 = os.getenv("FOTOFIX_MODEL_SHA256", "")


def default_model_path() -> Path:
    """Where the orientation model lives, unless overridden."""
    override = os.getenv("FOTOFIX_MODEL_PATH")
    if override:
        return Path(override)
    model_dir = os.getenv("FOTOFIX_MODEL_DIR")
    if model_dir:
        return Path(model_dir) / DEFAULT_MODEL_NAME
    return Path.home() / ".fotofix" / "models" / DEFAULT_MODEL_NAME


# --- Processing ------------------------------------------------------------
DEFAULT_THREADS = 8
MAX_PROXY_DIM = 1500
JPEG_QUALITY = 95
# Only accept a non-zero AI rotation when the model is at least this
# confident; below it, fall through to OCR / face detection instead of
# rotating on a guess. A confident "already correct" (angle 0) needs
# 0.95, as in the original code.
AI_MIN_CONFIDENCE = 0.50
AI_CONFIDENT_ZERO = 0.95

# --- Optimization ----------------------------------------------------------
OPTIMIZE_MAX_DIM_PHOTO = 4096
OPTIMIZE_MAX_DIM_DOC = 2500
OPTIMIZE_QUALITY_JPEG = 85
OPTIMIZE_QUALITY_WEBP = 80
OPTIMIZE_TARGET_FORMAT = "webp"  # "webp" or "jpeg"

# --- Stitching ---------------------------------------------------------------
STITCH_MARGIN = 40  # Padding between front and back, in pixels.
STITCH_BG_COLOR = (255, 255, 255)  # White background.
STITCH_QUALITY = 92  # JPEG quality for stitched output.

# --- File support ------------------------------------------------------------
# One list, used everywhere files are collected. ``.tif`` is included:
# the original code accepted it in some commands and not others.
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
