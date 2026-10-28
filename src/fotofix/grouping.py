"""Grouping of scan files into front/back sets.

Naming convention (Epson FastFoto-style, but the same rule applies to
any scans renamed to match)::

    <base>.jpg        primary front scan
    <base>_a.jpg      second front scan (the "enhanced" copy)
    <base>_b.jpg      back-of-photo scan

The grouping key is the base name, case-insensitively, so
``FastFoto_0452.jpg`` and ``fastfoto_0452_a.jpg`` group together. The
display base keeps the case of the first file seen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS

# A trailing ``_a`` / ``_b`` on the stem marks the second front / back.
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<suffix>[aAbB])$")


@dataclass
class PhotoGroup:
    """One base name and the scan files that belong to it."""

    key: str
    base: str
    paths: list[Path] = field(default_factory=list)

    @property
    def primary(self) -> Path | None:
        return _pick(self.paths, "")

    @property
    def secondary(self) -> Path | None:
        return _pick(self.paths, "a")

    @property
    def back(self) -> Path | None:
        return _pick(self.paths, "b")

    @property
    def front(self) -> Path | None:
        """Best front scan: the ``_a`` copy when present, else primary."""
        return self.secondary or self.primary


def parse_scan_name(filename: str) -> tuple[str, str]:
    """Split a filename into ``(base, suffix)``.

    ``suffix`` is ``""``, ``"a"``, or ``"b"`` (lowercased). Only a
    trailing ``_a``/``_b`` on the stem counts; ``my_a_photo.jpg`` keeps
    its full stem as the base.
    """
    stem = Path(filename).stem
    match = _SUFFIX_RE.match(stem)
    if match:
        return match.group("base"), match.group("suffix").lower()
    return stem, ""


def get_base_name(filename: str) -> str:
    """Display base name for a scan file, preserving its case."""
    return parse_scan_name(filename)[0]


def is_backside(filename: str) -> bool:
    """True when a filename represents a backside (``_b``) scan."""
    return parse_scan_name(filename)[1] == "b"


def _pick(paths: list[Path], suffix: str) -> Path | None:
    for path in paths:
        if parse_scan_name(path.name)[1] == suffix:
            return path
    return None


def find_file_groups(file_paths) -> dict[str, list[Path]]:
    """Group paths by base name.

    Returns ``{lowercased_base: [paths sorted by filename]}``. Accepts
    strings or Paths.
    """
    groups: dict[str, list[Path]] = {}
    for raw in file_paths:
        path = Path(raw)
        base = get_base_name(path.name)
        groups.setdefault(base.lower(), []).append(path)
    for paths in groups.values():
        paths.sort(key=lambda p: p.name.lower())
    return groups


def scan_directory(directory, *, recursive: bool = False) -> list[PhotoGroup]:
    """Group the supported images in a directory, sorted by base name."""
    directory = Path(directory)
    if recursive:
        entries = [p for p in directory.rglob("*") if p.is_file()]
    else:
        entries = [p for p in directory.iterdir() if p.is_file()]
    entries = [p for p in entries if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    entries.sort(key=lambda p: p.name.lower())
    grouped = find_file_groups(entries)
    result: list[PhotoGroup] = []
    for key in sorted(grouped):
        paths = grouped[key]
        base = get_base_name(paths[0].name)
        result.append(PhotoGroup(key=key, base=base, paths=paths))
    return result
