"""Text-based organizing: sort scan groups into ``with_text`` / ``no_text``.

This is the former standalone ``organize.py``, consolidated into the
package. Two fixes over the original: it honours dry-run (the original
always moved files, with no ``--live`` flag at all), and it shares the
single grouping implementation and Tesseract configuration instead of
duplicating both.

A group is classified by its front scans only (non-``_b`` files); the
whole group moves together. This is *text* sorting; orientation
sorting is the ``sort`` command in ``fix.py``.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ..grouping import find_file_groups, parse_scan_name


def analyze_image_type(image_path):
    """Return ``(has_text, confidence, source)`` for one image.

    Raises ``RuntimeError`` when Tesseract is unavailable.
    """
    from ..vision.ocr import _load_pytesseract

    pytesseract = _load_pytesseract()
    try:
        with Image.open(image_path) as img:
            img.thumbnail((1500, 1500), Image.Resampling.LANCZOS)

            # 1. Orientation confidence (printed text).
            try:
                osd = pytesseract.image_to_osd(img, timeout=20)
                conf = 0.0
                for line in osd.split("\n"):
                    if "Orientation confidence:" in line:
                        conf = float(line.split(":", 1)[1].strip())
                        break
                if conf >= 0.5:
                    text_sample = pytesseract.image_to_string(img, timeout=15)
                    if len(text_sample.strip()) > 5:
                        return True, conf, f"OSD (conf={conf})"
            except Exception:
                pass

            # 2. String extraction (handwriting, small text, memos).
            full_text = pytesseract.image_to_string(img, timeout=20).strip()
            if len(full_text) > 20:
                return True, len(full_text), "TextLength"

            # 3. Word data (sparse text).
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT, timeout=20
            )
            words = [
                w
                for i, w in enumerate(data["text"])
                if int(data["conf"][i]) > 30 and len(w.strip()) > 1
            ]
            if len(words) > 3:
                return True, len(words), f"WordCount ({len(words)})"
    except Exception:
        pass
    return False, 0, None


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


def organize_files(source_dir, dry_run: bool = True, threads: int = 4):
    """Sort groups into ``with_text`` / ``no_text``. Returns result lines."""
    from ..config import SUPPORTED_EXTENSIONS

    source_dir = Path(source_dir).resolve()
    if source_dir.name in ("with_text", "no_text"):
        return []
    text_dir = source_dir / "with_text"
    no_text_dir = source_dir / "no_text"

    files = [
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    groups = find_file_groups(files)
    lines = [
        f"Organizing {len(files)} files into {len(groups)} photo sets..."
    ]

    def process_group(key, paths):
        front_candidates = [
            p for p in paths if parse_scan_name(p.name)[1] != "b"
        ] or list(paths)
        group_has_text = False
        best_source = ""
        for candidate in front_candidates:
            has_txt, _conf, src = analyze_image_type(candidate)
            if has_txt:
                group_has_text = True
                best_source = str(src)
                break
        target = text_dir if group_has_text else no_text_dir
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if not path.exists():
                continue
            if dry_run:
                continue
            dest = _unique_path(target, path.name)
            shutil.move(str(path), str(dest))
        label = "[TEXT ]" if group_has_text else "[PHOTO]"
        source = best_source if group_has_text else "No text"
        action = "DRY RUN: would move" if dry_run else "Moved"
        return f"Group {key} -> {label} ({source}) [{action} -> {target.name}]"

    lines.append(f"Processing with {threads} threads. Please wait...")
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {
            executor.submit(process_group, key, paths): key
            for key, paths in groups.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = f"Error in group {key}: {e}"
    for i, key in enumerate(sorted(results), 1):
        lines.append(f"[{i:2d}/{len(results):2d}] {results[key]}")
    lines.append(f"\nFinished organization of {source_dir}")
    return lines
