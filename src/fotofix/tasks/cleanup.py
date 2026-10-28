"""Cleanup of redundant FastFoto originals.

A base scan is only deleted when an ``_a`` scan with the *same file
extension* exists in the same group and both files are non-empty. The
original code deleted the base scan whenever any ``_a`` file existed,
regardless of format — but an ``_a`` scan is a distinct second front
scan, not proof the base scan is redundant, so the same-extension
guard and the dry-run default are the safety net. Backside (``_b``)
scans are always preserved. Nothing is deleted unless ``--live`` is
passed.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..grouping import find_file_groups, parse_scan_name


def clean_duplicates(directory, dry_run: bool = True, recursive: bool = False):
    """Scan for redundant originals. Returns a list of result lines."""
    from ..config import SUPPORTED_EXTENSIONS

    source_dir = Path(directory).resolve()
    if recursive:
        files = [
            p
            for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    lines: list[str] = []
    if not files:
        return [f"No images found in {source_dir}"]

    groups = find_file_groups(files)
    lines.append(f"Analyzing {len(groups)} photo groups in {source_dir}...")

    deleted_count = 0
    kept_count = 0
    for _key, paths in groups.items():
        by_suffix: dict[str, list[Path]] = {}
        for path in paths:
            suffix = parse_scan_name(path.name)[1]
            by_suffix.setdefault(suffix, []).append(path)
        if by_suffix.get("b"):
            kept_count += len(by_suffix["b"])
        for enhanced in by_suffix.get("a", []):
            ext = enhanced.suffix.lower()
            for original in by_suffix.get("", []):
                if original.suffix.lower() != ext:
                    continue
                if enhanced.stat().st_size == 0 or original.stat().st_size == 0:
                    continue
                if dry_run:
                    lines.append(
                        f"[DRY RUN] Would DELETE: {original.name} "
                        f"(Keeping {enhanced.name})"
                    )
                else:
                    try:
                        os.remove(original)
                        lines.append(
                            f"DELETED: {original.name} (Kept {enhanced.name})"
                        )
                    except Exception as e:
                        lines.append(f"ERROR deleting {original.name}: {e}")
                        continue
                deleted_count += 1

    lines.append("")
    lines.append("--- Cleanup Complete ---")
    lines.append(
        f"Total Originals {'would be ' if dry_run else ''}Deleted: {deleted_count}"
    )
    lines.append(f"Backside scans preserved: {kept_count}")
    if dry_run:
        lines.append("Run with --live to perform actual deletions.")
    return lines
