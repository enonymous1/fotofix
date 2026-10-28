"""ONNX orientation classifier.

The model is a third-party release (see ``config.MODEL_URL``). It is
downloaded on first use into a per-user directory and never committed
to the repository. The class map ``{0: 0, 1: 90, 2: 180, 3: 270}`` is the
upstream model's convention; if the upstream model is ever replaced,
verify the map against scans rotated by known amounts before trusting
it.

Heavy dependencies (OpenCV, ONNX Runtime, requests) are imported
lazily so that ``fotofix --help`` and the Pillow-only commands work
without the ``fotofix[vision]`` extra installed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import MODEL_SHA256, MODEL_URL, default_model_path

_VISION_HINT = (
    "Install the vision dependencies with: pip install -e '.[vision]'"
)


class OrientationClassifier:
    def __init__(self, model_path=None, model_url: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "onnxruntime is required for AI orientation detection. "
                + _VISION_HINT
            ) from exc

        if model_path is None:
            model_path = default_model_path()
        self.model_path = Path(model_path)
        self._ensure_model_exists(model_url or MODEL_URL)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_map = {0: 0, 1: 90, 2: 180, 3: 270}

    def _ensure_model_exists(self, model_url: str) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            self._verify_checksum()
            return
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "requests is required to download the orientation model. "
                + _VISION_HINT
            ) from exc
        print(f"Downloading AI orientation model to {self.model_path}...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_suffix(".download")
        try:
            with requests.get(model_url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("Downloaded orientation model is empty.")
            os.replace(tmp_path, self.model_path)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        self._verify_checksum()
        print("Download complete.")

    def _verify_checksum(self) -> None:
        expected = MODEL_SHA256
        if not expected:
            return
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest != expected.lower():
            raise RuntimeError(
                "Orientation model checksum mismatch: the downloaded file "
                "does not match FOTOFIX_MODEL_SHA256. Delete the model "
                f"file at {self.model_path} and retry."
            )

    def preprocess(self, image, target_size: int = 384):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "opencv-python is required for AI orientation detection. "
                + _VISION_HINT
            ) from exc
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = np.array(image).copy()
        img = cv2.resize(
            img,
            (target_size + 32, target_size + 32),
            interpolation=cv2.INTER_LINEAR,
        )
        start = 32 // 2
        img = img[start : start + target_size, start : start + target_size]
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image):
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        return self.class_map.get(predicted_class, 0), confidence


__all__ = ["OrientationClassifier"]
