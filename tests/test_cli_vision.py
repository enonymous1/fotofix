import importlib.util

import pytest
from PIL import Image

from fotofix.cli import build_parser, main
from fotofix.vision.ocr import _osd_float, _parse_osd, tesseract_available


def test_osd_parsing():
    values = _parse_osd("Rotate: 90\nOrientation confidence: 4.5\nScript: Latin")
    assert values["Rotate"] == "90"
    assert _osd_float(values, "Orientation confidence") == 4.5
    assert _osd_float(values, "Missing") == 0.0
    assert _osd_float({"Rotate": "not-a-number"}, "Rotate") == 0.0


def test_tesseract_available_is_bool():
    assert isinstance(tesseract_available(), bool)


@pytest.mark.skipif(
    tesseract_available(), reason="Tesseract is installed here"
)
def test_ocr_detector_raises_without_tesseract(tmp_path):
    from fotofix.vision.ocr import detect_orientation_tesseract

    img = Image.new("RGB", (10, 10), "white")
    with pytest.raises(RuntimeError, match="pytesseract|Tesseract"):
        detect_orientation_tesseract(img)


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is not None, reason="OpenCV is installed here"
)
def test_faces_detector_raises_without_opencv():
    from fotofix.vision.faces import detect_orientation_faces

    img = Image.new("RGB", (10, 10), "white")
    with pytest.raises(RuntimeError, match="opencv"):
        detect_orientation_faces(img)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="ONNX Runtime is installed here",
)
def test_ai_classifier_raises_without_onnxruntime(tmp_path):
    from fotofix.vision.ai import OrientationClassifier

    with pytest.raises(RuntimeError, match="onnxruntime"):
        OrientationClassifier(model_path=tmp_path / "model.onnx")


def test_cli_help_exits_zero_without_heavy_deps(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "fotofix" in capsys.readouterr().out


def test_parser_rejects_bad_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bogus"])


def test_cli_stitch_end_to_end(tmp_path, capsys):
    front = tmp_path / "FastFoto_0001_a.jpg"
    back = tmp_path / "FastFoto_0001_b.jpg"
    Image.new("RGB", (200, 100), "red").save(front)
    Image.new("RGB", (100, 200), "blue").save(back)
    assert main(["stitch", "--directory", str(tmp_path), "--live"]) == 0
    out = tmp_path / "FastFoto_0001_stitched.jpg"
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (200, 540)


def test_cli_fix_dry_run_with_mocked_backends(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (100, 50), "red").save(src)
    monkeypatch.setattr("fotofix.cli._preflight_fix", lambda mode: None)
    monkeypatch.setattr(
        "fotofix.tasks.fix.analyze_single_file",
        lambda path, mode="auto": (0, "Default", 0),
    )
    assert main(["fix", "--directory", str(tmp_path)]) == 0
    assert src.exists()
    assert "PASS" in capsys.readouterr().out


def test_cli_missing_directory_returns_2(tmp_path):
    assert main(["stitch", "--directory", str(tmp_path / "nope"), "--live"]) == 2


def test_run_ps1_passes_command_and_directory():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "run.ps1").read_text()
    assert "--directory" in text
    assert "-m fotofix" in text
    assert "$command" in text
    assert ".venv" in text
