from pathlib import Path

import pytest
from PIL import Image

from fotofix.tasks import fix as fix_mod
from fotofix.tasks.cleanup import clean_duplicates
from fotofix.tasks.diagnose import diagnose_file
from fotofix.tasks.optimize import optimize_image
from fotofix.tasks.organize import organize_files
from fotofix.vision.ocr import tesseract_available


def _image(path, size=(200, 100), color="red", mode="RGB"):
    Image.new(mode, size, color).save(path)
    return path


class _FakeClassifier:
    def __init__(self, angle, conf):
        self._angle, self._conf = angle, conf

    def predict(self, _proxy):
        return self._angle, self._conf


def _patch_detectors(monkeypatch, *, ai=(0, 0.0), ocr=(0, 0), faces=(0, 0)):
    monkeypatch.setattr(
        fix_mod, "get_ai_classifier", lambda: _FakeClassifier(*ai)
    )
    monkeypatch.setattr(
        "fotofix.vision.ocr.detect_orientation_tesseract", lambda _p: ocr
    )
    monkeypatch.setattr(
        "fotofix.vision.faces.detect_orientation_faces", lambda _p: faces
    )


def test_analyze_ai_confident_wins(tmp_path, monkeypatch):
    _patch_detectors(monkeypatch, ai=(90, 0.9), ocr=(180, 5.0))
    src = _image(tmp_path / "a.jpg")
    assert fix_mod.analyze_single_file(src, "auto")[0] == 90


def test_analyze_ai_low_confidence_falls_through_to_ocr(tmp_path, monkeypatch):
    _patch_detectors(monkeypatch, ai=(90, 0.3), ocr=(180, 5.0))
    src = _image(tmp_path / "a.jpg")
    angle, method, _ = fix_mod.analyze_single_file(src, "auto")
    assert angle == 180
    assert method.startswith("OCR")


def test_analyze_ai_confident_zero_stops(tmp_path, monkeypatch):
    calls = []
    _patch_detectors(monkeypatch, ai=(0, 0.99), ocr=(180, 5.0))
    monkeypatch.setattr(
        "fotofix.vision.ocr.detect_orientation_tesseract",
        lambda _p: calls.append(1) or (180, 5.0),
    )
    src = _image(tmp_path / "a.jpg")
    assert fix_mod.analyze_single_file(src, "auto")[0] == 0
    assert calls == []


def test_analyze_falls_through_to_faces(tmp_path, monkeypatch):
    _patch_detectors(monkeypatch, ai=(0, 0.5), ocr=(0, 0), faces=(270, 3))
    src = _image(tmp_path / "a.jpg")
    angle, method, _ = fix_mod.analyze_single_file(src, "auto")
    assert angle == 270
    assert method.startswith("Faces")


def test_analyze_missing_ai_falls_through_in_auto(tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("no onnxruntime")

    monkeypatch.setattr(fix_mod, "get_ai_classifier", _boom)
    monkeypatch.setattr(
        "fotofix.vision.ocr.detect_orientation_tesseract", lambda _p: (90, 4.0)
    )
    src = _image(tmp_path / "a.jpg")
    assert fix_mod.analyze_single_file(src, "auto")[0] == 90


def test_analyze_missing_ai_raises_in_pinned_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fix_mod, "get_ai_classifier", lambda: (_ for _ in ()).throw(RuntimeError("no onnx"))
    )
    src = _image(tmp_path / "a.jpg")
    with pytest.raises(RuntimeError):
        fix_mod.analyze_single_file(src, "ai")


def test_analyze_unreadable_file(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    _patch_detectors(monkeypatch)
    assert fix_mod.analyze_single_file(bad, "auto") == (0, "Error", 0)


def _patch_analyze(monkeypatch, angle_map):
    def _fake(path, mode="auto"):
        name = Path(path).name
        if "_b." in name.lower():
            return 0, "Default", 0
        return angle_map.get(name, (0, "Default", 0))

    monkeypatch.setattr(fix_mod, "analyze_single_file", _fake)


def test_sort_dry_run_creates_no_directories(tmp_path, monkeypatch):
    front = _image(tmp_path / "FastFoto_0001_a.jpg")
    back = _image(tmp_path / "FastFoto_0001_b.jpg")
    _patch_analyze(monkeypatch, {"FastFoto_0001_a.jpg": (90, "OCR(conf=5)", 5)})
    result = fix_mod.process_file_or_group(
        [front, back], dry_run=True, sort_mode=True, target_dir=tmp_path
    )
    assert "DRY RUN SORT" in result
    assert not (tmp_path / "Good").exists()
    assert not (tmp_path / "Bad Orientation").exists()
    assert front.exists() and back.exists()


def test_sort_live_moves_group_and_avoids_collision(tmp_path, monkeypatch):
    front = _image(tmp_path / "FastFoto_0001_a.jpg")
    back = _image(tmp_path / "FastFoto_0001_b.jpg")
    dest = tmp_path / "Bad Orientation"
    dest.mkdir()
    (dest / "FastFoto_0001_a.jpg").write_bytes(b"existing")
    _patch_analyze(monkeypatch, {"FastFoto_0001_a.jpg": (90, "OCR(conf=5)", 5)})
    fix_mod.process_file_or_group(
        [front, back], dry_run=False, sort_mode=True, target_dir=tmp_path
    )
    assert not front.exists()
    assert (dest / "FastFoto_0001_a_2.jpg").exists()
    assert (dest / "FastFoto_0001_b.jpg").exists()


def test_fix_live_rotates_and_reports_error(tmp_path, monkeypatch):
    src = _image(tmp_path / "a.jpg", size=(100, 50))
    _patch_analyze(monkeypatch, {"a.jpg": (90, "AI(conf=0.90)", 0.9)})
    result = fix_mod.process_file_or_group([src], dry_run=False)
    assert "FIXED" in result
    with Image.open(src) as img:
        assert img.size == (50, 100)

    src2 = _image(tmp_path / "b.jpg", size=(100, 50))
    _patch_analyze(monkeypatch, {"b.jpg": (90, "AI(conf=0.90)", 0.9)})
    monkeypatch.setattr(
        fix_mod, "save_corrected_image", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    result = fix_mod.process_file_or_group([src2], dry_run=False)
    assert "ERROR FIXING" in result


def test_optimize_dry_run_and_live(tmp_path, monkeypatch):
    src = _image(tmp_path / "photo.jpg", size=(2000, 1000))
    monkeypatch.setattr(
        "fotofix.vision.ocr.get_document_affinity", lambda _p: 0.1
    )
    out_dir = tmp_path / "out"
    result = optimize_image(src, out_dir, dry_run=True)
    assert result.startswith("DRY RUN")
    assert not out_dir.exists()
    result = optimize_image(src, out_dir, dry_run=False, target_format="webp")
    assert result.startswith("OPTIMIZED")
    out = out_dir / "photo_optimized.webp"
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "WEBP"


def test_optimize_document_uses_smaller_max_dim(tmp_path, monkeypatch):
    src = _image(tmp_path / "doc.jpg", size=(3000, 2000))
    monkeypatch.setattr(
        "fotofix.vision.ocr.get_document_affinity", lambda _p: 0.9
    )
    result = optimize_image(src, tmp_path, dry_run=True)
    assert "2500px" in result
    assert "Document" in result


def test_optimize_rgba_to_jpeg_and_ocr_fallback(tmp_path, monkeypatch):
    src = _image(tmp_path / "alpha.png", size=(100, 100), color=(255, 0, 0, 128), mode="RGBA")

    def _boom(_p):
        raise RuntimeError("no tesseract")

    monkeypatch.setattr("fotofix.vision.ocr.get_document_affinity", _boom)
    result = optimize_image(src, tmp_path, dry_run=False, target_format="jpeg")
    assert "OCR unavailable" in result
    assert result.startswith("OPTIMIZED")
    with Image.open(tmp_path / "alpha_optimized.jpeg") as img:
        assert img.mode == "RGB"


def test_cleanup_dry_run_live_and_extension_guard(tmp_path):
    base = _image(tmp_path / "FastFoto_0001.jpg")
    enhanced = _image(tmp_path / "FastFoto_0001_a.jpg")
    back = _image(tmp_path / "FastFoto_0001_b.jpg")
    lines = clean_duplicates(tmp_path, dry_run=True)
    assert any("Would DELETE" in line for line in lines)
    assert base.exists()
    lines = clean_duplicates(tmp_path, dry_run=False)
    assert not base.exists()
    assert enhanced.exists() and back.exists()
    assert any("Backside scans preserved: 1" in line for line in lines)

    # Different extension: no deletion.
    base2 = _image(tmp_path / "FastFoto_0002.jpg")
    _image(tmp_path / "FastFoto_0002_a.tiff")
    lines = clean_duplicates(tmp_path, dry_run=False)
    assert base2.exists()


def test_cleanup_skips_empty_enhanced(tmp_path):
    base = _image(tmp_path / "FastFoto_0003.jpg")
    (tmp_path / "FastFoto_0003_a.jpg").write_bytes(b"")
    lines = clean_duplicates(tmp_path, dry_run=False)
    assert base.exists()
    assert not any("DELETED" in line for line in lines)


def test_organize_dry_run_and_live(tmp_path, monkeypatch):
    front = _image(tmp_path / "doc_a.jpg")
    _image(tmp_path / "doc_b.jpg")
    _image(tmp_path / "pic.jpg")
    monkeypatch.setattr(
        "fotofix.tasks.organize.analyze_image_type",
        lambda p: (True, 1.0, "stub") if "doc" in Path(p).name else (False, 0, None),
    )
    lines = organize_files(tmp_path, dry_run=True)
    assert any("Organizing" in line for line in lines)
    assert front.exists()
    assert not (tmp_path / "with_text").exists()
    organize_files(tmp_path, dry_run=False)
    assert (tmp_path / "with_text" / "doc_a.jpg").exists()
    assert (tmp_path / "with_text" / "doc_b.jpg").exists()
    assert (tmp_path / "no_text" / "pic.jpg").exists()


def test_diagnose_file_with_stub(tmp_path):
    src = _image(tmp_path / "a.jpg")

    def _stub(_img):
        return (
            "Orientation confidence: 5.0\nScript confidence: 3.0",
            "hello world",
            {"text": ["hello", "world"], "conf": ["50", "60"]},
        )

    report = diagnose_file(src, ocr_fn=_stub)
    assert "OSD Orientation Confidence: 5.0" in report
    assert "Extracted String Length: 11" in report
    assert "High Confidence Words (>40): 2" in report


def test_diagnose_file_bad_image(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    report = diagnose_file(bad, ocr_fn=lambda _img: ("", "", {"text": [], "conf": []}))
    assert "General error" in report


@pytest.mark.skipif(
    tesseract_available(), reason="Tesseract is installed here"
)
def test_diagnose_file_raises_without_tesseract(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (10, 10), "white").save(path)
    with pytest.raises(RuntimeError, match="pytesseract|Tesseract"):
        diagnose_file(path)
