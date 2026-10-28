"""Face-count orientation detection (Haar cascade, OpenCV).

The cascade is loaded once and cached: the original code reloaded it
for every file. OpenCV is imported lazily so this module never breaks
``fotofix --help`` on a machine without it.
"""

from __future__ import annotations

import numpy as np

_CASCADE = None
_VISION_HINT = (
    "Install the vision dependencies with: pip install -e '.[vision]'"
)


def _get_cascade():
    global _CASCADE
    if _CASCADE is not None:
        return _CASCADE
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for face-based orientation "
            "detection. " + _VISION_HINT
        ) from exc
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
    _CASCADE = cascade
    return cascade


def detect_orientation_faces(proxy_img):
    """Return ``(best_angle, max_faces)`` over 0/90/180/270 degrees.

    Ties keep the earliest angle (0 first), as in the original code.
    Raises ``RuntimeError`` when OpenCV or the cascade is unavailable.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for face-based orientation "
            "detection. " + _VISION_HINT
        ) from exc
    img_cv = cv2.cvtColor(np.array(proxy_img), cv2.COLOR_RGB2BGR)
    face_cascade = _get_cascade()

    best_angle = 0
    max_faces = 0
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = img_cv
        elif angle == 90:
            rotated = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img_cv, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40)
        )
        num_faces = len(faces)
        if num_faces > max_faces:
            max_faces = num_faces
            best_angle = angle
    return best_angle, max_faces
