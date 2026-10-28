"""Orientation fixing and orientation-based sorting.

Detection order in ``auto`` mode: AI, then OCR, then faces. A detector
only claims a rotation when it is confident enough; otherwise the next
detector runs. In particular, a non-zero AI rotation is only accepted
at or above ``AI_MIN_CONFIDENCE`` — the original code accepted any
non-zero AI result, however uncertain.

``sort`` mode groups files by their scan group and moves the whole
group into ``Good`` / ``Bad Orientation`` under the target directory,
based on whether any non-backside file in the group needs rotating.
This is *orientation* sorting; text sorting lives in ``organize``.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from ..config import AI_CONFIDENT_ZERO, AI_MIN_CONFIDENCE
from ..grouping import is_backside
from ..image import get_proxy_image, save_corrected_image

_AI_CLASSIFIER = None
_AI_LOCK = threading.Lock()


def get_ai_classifier():
    """Process-wide singleton classifier, created on first use."""
    global _AI_CLASSIFIER
    if _AI_CLASSIFIER is None:
        with _AI_LOCK:
            if _AI_CLASSIFIER is None:
                from ..vision.ai import OrientationClassifier

                _AI_CLASSIFIER = OrientationClassifier()
    return _AI_CLASSIFIER


def analyze_single_file(source_path, mode: str = "auto"):
    """Return ``(angle, method, confidence)`` for one file."""
    proxy = get_proxy_image(source_path)
    if proxy is None:
        return 0, "Error", 0

    # 1. AI. In auto mode a missing AI backend falls through to the
    # other detectors; in pinned ai mode it is an error.
    if mode in ("auto", "ai"):
        try:
            classifier = get_ai_classifier()
            angle, conf = classifier.predict(proxy)
        except RuntimeError:
            if mode == "ai":
                raise
            angle, conf = 0, 0.0
        else:
            if angle != 0 and conf >= AI_MIN_CONFIDENCE:
                return angle, f"AI(conf={conf:.2f})", conf
            if angle == 0 and conf >= AI_CONFIDENT_ZERO:
                return 0, f"AI(conf={conf:.2f})", conf

    # 2. OCR
    if mode in ("auto", "text"):
        from ..vision.ocr import detect_orientation_tesseract

        try:
            angle, conf = detect_orientation_tesseract(proxy)
        except RuntimeError:
            if mode == "text":
                raise
            angle, conf = 0, 0
        else:
            if angle != 0:
                return angle, f"OCR(conf={conf})", conf

    # 3. Faces
    if mode in ("auto", "face"):
        from ..vision.faces import detect_orientation_faces

        try:
            angle, count = detect_orientation_faces(proxy)
        except RuntimeError:
            if mode == "face":
                raise
            angle, count = 0, 0
        else:
            if count >= 2:
                return angle, f"Faces(count={count})", count

    return 0, "Default", 0


def _unique_path(directory: Path, filename: str) -> Path:
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


def process_file_or_group(
    paths,
    dry_run: bool = True,
    mode: str = "auto",
    sort_mode: bool = False,
    target_dir=None,
    backup_dir=None,
):
    """Process one scan group. Returns a multi-line result string."""
    details = []
    for path in paths:
        path = Path(path)
        filename = path.name
        angle, method, conf = analyze_single_file(path, mode)
        if sort_mode and is_backside(filename):
            details.append(
                {
                    "path": path,
                    "angle": 0,
                    "method": f"Backside(analysis={angle})",
                    "is_trigger": False,
                }
            )
        else:
            details.append(
                {
                    "path": path,
                    "angle": angle,
                    "method": method,
                    "is_trigger": (angle != 0),
                }
            )

    group_angle_needed = any(d["is_trigger"] for d in details)
    category = "Bad Orientation" if group_angle_needed else "Good"
    results = []

    for d in details:
        path, angle, method = d["path"], d["angle"], d["method"]
        filename = path.name

        if sort_mode and target_dir:
            dest_dir = Path(target_dir) / category
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = _unique_path(dest_dir, filename)
                shutil.move(str(path), str(dest))
                results.append(f"SORTED: {filename} -> {category}")
            else:
                results.append(f"DRY RUN SORT: {filename} -> {category}")
        else:
            if angle != 0:
                if not dry_run:
                    try:
                        save_corrected_image(path, angle, backup_dir=backup_dir)
                        results.append(
                            f"FIXED: {filename} ({angle}deg via {method})"
                        )
                    except Exception as e:
                        results.append(f"ERROR FIXING {filename}: {e}")
                else:
                    results.append(
                        f"DRY RUN: {filename} would rotate {angle}deg via {method}"
                    )
            else:
                results.append(f"PASS: {filename} (Correct)")

    return "\n".join(results)
