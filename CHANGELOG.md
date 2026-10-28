# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-10-28

Modernization of the April 2026 `ss-fotoFixing` codebase, delivered as a
squashed public export under the `fotofix` name. No earlier version was
released under this name.

### Added

- Installable `fotofix` package and `fotofix` CLI (`python -m fotofix`
  also works), with `sort`, `fix`, `organize`, `optimize`, `stitch`,
  `cleanup`, and `diagnose` subcommands.
- `organize` (text sorting) and `cleanup` (redundant-original removal)
  as first-class CLI commands; both honour dry-run, unlike the
  original standalone scripts.
- `diagnose` command replacing the hardcoded-path diagnose scripts.
- `--recursive`, `--backup-dir` (fix), `--direction` (stitch), and
  `--format` / `--quality` (optimize) options.
- Same-format guard in cleanup: a base scan is only deleted when a
  same-extension `_a` scan exists and both files are non-empty.
- EXIF preservation in `fix` and `optimize` (orientation tag reset to
  1, other tags kept), atomic rewrite in `fix`, and backup copies via
  `--backup-dir`.
- Proportional back-scan scaling in stitching, with an explicit
  `--direction` instead of the README's incorrect "side-by-side".
- Tests (pytest), ruff, and GitHub Actions CI (Ubuntu + Windows,
  Python 3.10-3.12). MIT LICENSE.

### Fixed

- `fotofix --help` no longer crashes when OpenCV / ONNX Runtime /
  Tesseract are not installed: heavy dependencies are imported lazily,
  and commands that need them fail fast with a readable message.
- `run.ps1` now passes the command and `--directory` correctly, checks
  for the virtual environment, and prompts for dry-run vs. live mode.
- Grouping now works for any `<base>` / `<base>_a` / `<base>_b`
  naming, not only `FastFoto_NNNN` names.
- Dry runs no longer create output directories. Detection itself
  still runs in a dry run (that is how it knows what would happen),
  so a first AI-mode run may still download the orientation model
  into your per-user model directory.
- One failing file no longer aborts a whole batch; failures are
  reported per group/file and the CLI exits non-zero.
- AI rotations now require a minimum confidence before rotating;
  low-confidence non-zero results fall through to OCR / faces.
- A single shared configuration and grouping implementation; the
  duplicated copies in the old standalone scripts are gone.
- Line endings normalized to LF with a `.gitattributes`.

### Known limitations

- Orientation detection needs Tesseract OCR and, for AI mode, the
  third-party ONNX model downloaded on first use. The AI class map
  and rotation-sign convention are the upstream model's; verify them
  against known-rotated scans if the model changes.
- Live `fix` rewrites files in place; use `--backup-dir` (or a dry
  run first) on irreplaceable scans. Live `cleanup` deletes files;
  it is dry-run by default for the same reason.
