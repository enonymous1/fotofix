"""Command-line interface for fotofix.

Importing this module — and therefore ``fotofix --help`` — never
requires OpenCV, ONNX Runtime, or Tesseract. Those are loaded lazily
by the vision backends, and the commands that need them preflight
before processing a whole folder so one missing dependency fails fast
with a readable message instead of once per file.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .config import DEFAULT_THREADS, SUPPORTED_EXTENSIONS
from .grouping import scan_directory


def _collect_files(directory: Path, recursive: bool):
    if recursive:
        files = [p for p in directory.rglob("*") if p.is_file()]
    else:
        files = [p for p in directory.iterdir() if p.is_file()]
    files = [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort(key=lambda p: p.name.lower())
    return files


def _preflight_fix(mode: str) -> str | None:
    """Return an error message when no usable backend is available.

    A pinned mode requires its own backend. ``auto`` requires at least
    one backend and warns (once, on stderr) about the missing ones,
    since ``analyze_single_file`` falls through past unavailable
    backends in auto mode.
    """
    available: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def _check(name, fn):
        try:
            fn()
            available[name] = True
        except RuntimeError as e:
            available[name] = False
            errors[name] = str(e)

    if mode in ("auto", "ai"):
        from .tasks.fix import get_ai_classifier

        _check("ai", get_ai_classifier)
    if mode in ("auto", "text"):
        from .vision.ocr import tesseract_available

        available["text"] = tesseract_available()
        if not available["text"]:
            errors["text"] = "Tesseract OCR was not found."
    if mode in ("auto", "face"):
        from .vision.faces import _get_cascade

        _check("face", _get_cascade)

    if mode != "auto":
        engine = {"ai": "ai", "text": "text", "face": "face"}[mode]
        if not available.get(engine):
            return errors.get(engine, f"Backend for mode {mode!r} is unavailable.")
        return None
    if not any(available.values()):
        return (
            "No orientation backend is available (AI, OCR, and faces all "
            "unavailable). Install Tesseract OCR and/or "
            "pip install -e '.[vision]'."
        )
    for name, ok in available.items():
        if not ok:
            print(
                f"Warning: {name} backend unavailable ({errors.get(name, '')}); "
                "falling through to the remaining detectors.",
                file=sys.stderr,
            )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fotofix",
        description=(
            "Bulk photo orientation fixing, sorting, optimizing, and "
            "stitching for batch-scanned photos."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp, *, modes: bool = False):
        sp.add_argument(
            "--directory",
            default=".",
            help="Directory with photos to process (default: current directory)",
        )
        sp.add_argument(
            "--target",
            help="Target directory for output commands (defaults to --directory)",
        )
        sp.add_argument(
            "--live",
            action="store_true",
            help="Perform actual file operations (without this, dry run only)",
        )
        sp.add_argument(
            "--threads",
            type=int,
            default=DEFAULT_THREADS,
            help=f"Number of parallel workers (default: {DEFAULT_THREADS})",
        )
        sp.add_argument(
            "--recursive",
            action="store_true",
            help="Process subdirectories too (top level only by default)",
        )
        if modes:
            sp.add_argument(
                "--mode",
                choices=["auto", "ai", "text", "face"],
                default="auto",
                help="Orientation detection engine (default: auto)",
            )

    fix_p = sub.add_parser(
        "fix", help="Detect and correct image orientation in place"
    )
    add_common(fix_p, modes=True)
    fix_p.add_argument(
        "--backup-dir",
        help="Copy untouched originals here before overwriting (live mode)",
    )

    sort_p = sub.add_parser(
        "sort",
        help="Move scan groups into Good / Bad Orientation by rotation need",
    )
    add_common(sort_p, modes=True)

    org_p = sub.add_parser(
        "organize",
        help="Move scan groups into with_text / no_text by OCR text detection",
    )
    add_common(org_p)

    opt_p = sub.add_parser("optimize", help="Resize and re-encode images")
    add_common(opt_p)
    opt_p.add_argument(
        "--format",
        choices=["webp", "jpeg"],
        default=None,
        help="Output format (default: webp, from config)",
    )
    opt_p.add_argument(
        "--quality", type=int, default=None, help="Output quality (default: per-format)"
    )

    stitch_p = sub.add_parser(
        "stitch", help="Combine front and back scans into one image"
    )
    add_common(stitch_p)
    stitch_p.add_argument(
        "--direction",
        choices=["top", "bottom", "left", "right"],
        default="bottom",
        help="Where the back scan goes relative to the front (default: bottom)",
    )

    clean_p = sub.add_parser(
        "cleanup",
        help="Delete redundant base scans when a same-format _a scan exists",
    )
    add_common(clean_p)

    diag_p = sub.add_parser(
        "diagnose", help="Report OCR signals for one or more images"
    )
    diag_p.add_argument("files", nargs="+", help="Image files to diagnose")

    return parser


def _run_parallel(executor, jobs):
    """Run jobs in submission order, collecting per-job errors.

    Returns ``(messages, had_error)``. One bad file never aborts the
    batch, unlike the original code, where ``future.result()`` raised
    out of ``main`` on the first failure.
    """
    messages: list[str] = []
    had_error = False
    futures = [executor.submit(fn, *args) for fn, *args in jobs]
    for i, future in enumerate(futures):
        try:
            result = future.result()
        except Exception as e:  # noqa: BLE001 - report, don't crash
            messages.append(f"[{i + 1}/{len(futures)}] ERROR: {e}")
            had_error = True
        else:
            messages.append(f"[{i + 1}/{len(futures)}] {result}")
    return messages, had_error


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "diagnose":
        from .tasks.diagnose import diagnose_file

        had_error = False
        for path in args.files:
            try:
                print(diagnose_file(path))
            except RuntimeError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                had_error = True
        return 1 if had_error else 0

    source_dir = Path(args.directory).resolve()
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory.", file=sys.stderr)
        return 2
    target_dir = Path(args.target).resolve() if args.target else source_dir
    dry_run = not args.live

    if args.command in ("fix", "sort"):
        error = _preflight_fix(args.mode)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            return 2

    if args.command == "organize":
        from .vision.ocr import tesseract_available

        if not tesseract_available():
            print(
                "Error: Tesseract OCR is required for organize but was not "
                "found. Install Tesseract OCR and set TESSERACT_PATH if needed.",
                file=sys.stderr,
            )
            return 2

    if args.command == "optimize":
        from .vision.ocr import tesseract_available

        if not tesseract_available():
            print(
                "Warning: Tesseract OCR not found; images will be treated "
                "as photos for sizing. Install Tesseract for document "
                "detection.",
                file=sys.stderr,
            )

    files = _collect_files(source_dir, args.recursive)
    if not files and args.command != "cleanup":
        print(f"No valid image files found in {source_dir}")
        return 0

    if args.command == "optimize":
        from .tasks.optimize import optimize_image

        print(f"Starting OPTIMIZE in {source_dir}")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            jobs = [
                (optimize_image, f, target_dir, dry_run, args.format, args.quality)
                for f in files
            ]
            messages, had_error = _run_parallel(executor, jobs)
        for message in messages:
            print(message)
        print("\n--- Processing Complete ---")
        return 1 if had_error else 0

    if args.command == "cleanup":
        from .tasks.cleanup import clean_duplicates

        for line in clean_duplicates(source_dir, dry_run=dry_run, recursive=args.recursive):
            print(line)
        return 0

    if args.command == "organize":
        from .tasks.organize import organize_files

        for line in organize_files(source_dir, dry_run=dry_run, threads=args.threads):
            print(line)
        return 0

    groups = scan_directory(source_dir, recursive=args.recursive)

    if args.command == "stitch":
        from .tasks.stitch import stitch_front_back

        print(f"Starting STITCH in {source_dir}")
        jobs = []
        for group in groups:
            front, back = group.front, group.back
            if front and back:
                out_path = target_dir / f"{group.base}_stitched.jpg"
                jobs.append(
                    (
                        stitch_front_back,
                        front,
                        back,
                        out_path,
                        dry_run,
                        args.direction,
                    )
                )
        if not jobs:
            print("No front/back pairs found.")
            return 0
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            messages, had_error = _run_parallel(executor, jobs)
        for message in messages:
            print(message)
        print("\n--- Processing Complete ---")
        return 1 if had_error else 0

    # fix / sort
    from .tasks.fix import process_file_or_group

    print(f"Found {len(groups)} photo groups ({len(files)} total files) in {source_dir}")
    print(
        f"Action: {args.command.upper()} | Mode: {args.mode.upper()} | "
        f"Threads: {args.threads} | Live: {args.live}"
    )
    backup_dir = Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        jobs = [
            (
                process_file_or_group,
                group.paths,
                dry_run,
                args.mode,
                args.command == "sort",
                target_dir,
                backup_dir,
            )
            for group in groups
        ]
        messages, had_error = _run_parallel(executor, jobs)
    for i, message in enumerate(messages):
        if message:
            print(f"\n--- Group {i + 1}/{len(groups)} ---")
            print(message.split("] ", 1)[-1])
    print("\n--- Processing Complete ---")
    return 1 if had_error else 0


def main_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
