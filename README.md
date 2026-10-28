# fotofix

Bulk orientation fixing, sorting, optimizing, and stitching for
batch-scanned photos — built for high-speed batch scanners (like the
Epson FastFoto series) that output front and back image pairs.

This is the bulk, automatic companion to a manual review pass: point
it at a folder of scans and let it straighten, sort, compress, or
stitch the whole folder. For picking the best front scan group by
group, with a human decision on each one, see
[photo-stitcher](https://github.com/enonymous1/photo-stitcher) instead.
It is not affiliated with or endorsed by Epson; it just works with the
naming the scanner produces.

## Naming convention

```text
FastFoto_0158.jpg      primary front scan
FastFoto_0158_a.jpg    second front scan
FastFoto_0158_b.jpg    back-of-photo scan
```

Any other scans renamed to `<base>` / `<base>_a` / `<base>_b` group
the same way. Files are grouped by base name; a group is processed
together.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/enonymous1/fotofix.git
cd fotofix
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
pip install -e ".[vision]"
```

The `[vision]` extra adds OpenCV, ONNX Runtime, and pytesseract. The
base install (Pillow only) still runs `stitch`, `cleanup`, and
`fotofix --help`; the detection commands need the extra. Orientation
detection also needs the Tesseract OCR binary itself: install it
(Windows: UB Mannheim build) and set `TESSERACT_PATH` if it is not on
your PATH. The AI orientation model is downloaded on first AI-mode use
into `~/.fotofix/models/`; override with `FOTOFIX_MODEL_PATH`.

Or without installing the command:

```bash
python -m fotofix <command> --directory "C:\path\to\photos"
```

On Windows, `run.ps1` wraps the same thing interactively (it prompts
for dry-run vs. live mode):

```powershell
.\run.ps1 -command fix -target_dir "C:\path\to\photos"
```

## Commands

Dry run is the default everywhere. Nothing is written, moved, or
deleted until you pass `--live`. A dry run still runs detection, so it
can tell you what *would* happen.

| Command | What it does |
| --- | --- |
| `fix` | Detect and correct orientation in place. `--mode auto` (default), `ai`, `text`, or `face`. `--backup-dir` copies untouched originals aside first. |
| `sort` | Move each group into `Good` / `Bad Orientation` under the target directory, based on whether any front scan needs rotating. |
| `organize` | Move each group into `with_text` / `no_text` based on OCR text detection of its front scans. |
| `optimize` | Resize and re-encode as WebP (default) or JPEG, with a smaller max dimension for text documents. Output is `<base>_optimized.webp`. |
| `stitch` | Combine each front + back pair into `<base>_stitched.jpg`. `--direction bottom` (default), `top`, `left`, or `right`. The back scan is scaled to match the shared edge; the front is never resampled. |
| `cleanup` | Delete a base scan only when a same-format `_a` scan exists and both files are non-empty. Back scans are always kept. |
| `diagnose` | Report OCR signals (OSD confidence, text length, word counts) for the given files. Takes file paths, not a directory. |

```bash
# See what would be fixed, without touching anything:
fotofix fix --directory "C:\path\to\photos"

# Actually fix, keeping backups:
fotofix fix --directory "C:\path\to\photos" --live --backup-dir "C:\backups"

# Stitch backs on the right, recursively:
fotofix stitch --directory "C:\path\to\photos" --live --direction right --recursive

# Diagnose why a scan was missed:
fotofix diagnose "C:\path\to\photos\FastFoto_0438_a.jpg"
```

Common options: `--directory`, `--target`, `--threads` (default 8),
`--recursive`, `--live`.

## Safety notes

- `fix` rewrites files in place and resets only the EXIF orientation
  tag; other EXIF data is preserved. Use `--backup-dir` on
  irreplaceable scans.
- `cleanup` deletes files. It only deletes a base scan when a
  same-extension `_a` scan exists, and it is dry-run by default.
- `optimize` writes new `_optimized` files and never overwrites the
  originals.
- One failing file is reported and the batch continues; the CLI exits
  non-zero if anything failed.

## How orientation detection works

In `auto` mode, each scan is checked in order: the ONNX model first
(non-zero rotations need at least 50% confidence; a confident "already
correct" needs 95%), then Tesseract OSD, then Haar face counts at four
rotations. The first confident detector wins. `--mode` pins one engine.
The AI class map and rotation-sign convention are the upstream
model's; verify them against known-rotated scans if the model changes.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

Tests use synthetic images and mocked vision backends, so they run
without Tesseract, OpenCV, or the ONNX model installed.

## Origin

This started with an Epson FastFoto scanner and a folder of sideways
scans. The April 2026 `ss-fotoFixing` scripts did the job on one
machine; this is that code consolidated into an installable package
with tests, a real CLI, and the destructive edges (always-moving
organizer, backup-less overwrites, a broken PowerShell wrapper) fixed.
See `CHANGELOG.md` for what changed.
