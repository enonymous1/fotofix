from PIL import Image

from fotofix.tasks.stitch import stitch_front_back


def _image(path, size, color):
    Image.new("RGB", size, color).save(path)
    return path


def test_stitch_bottom_scales_back(tmp_path):
    front = _image(tmp_path / "f.jpg", (200, 100), "red")
    back = _image(tmp_path / "b.jpg", (100, 200), "blue")
    out = tmp_path / "out.jpg"
    result = stitch_front_back(
        front, back, out, dry_run=False, direction="bottom", margin=0
    )
    assert result.startswith("STITCHED")
    with Image.open(out) as img:
        # Back scaled 100x200 -> 200x400; canvas 200 x (100+400).
        assert img.size == (200, 500)


def test_stitch_directions(tmp_path):
    front = _image(tmp_path / "f.jpg", (200, 100), "red")
    back = _image(tmp_path / "b.jpg", (100, 200), "blue")
    cases = {
        "top": (200, 540),
        "bottom": (200, 540),
        "left": (290, 100),
        "right": (290, 100),
    }
    for direction, expected in cases.items():
        out = tmp_path / f"out_{direction}.jpg"
        result = stitch_front_back(front, back, out, dry_run=False, direction=direction)
        assert result.startswith("STITCHED"), result
        with Image.open(out) as img:
            assert img.size == expected


def test_stitch_dry_run_creates_nothing(tmp_path):
    front = _image(tmp_path / "f.jpg", (100, 100), "red")
    back = _image(tmp_path / "b.jpg", (100, 100), "blue")
    out = tmp_path / "nested" / "out.jpg"
    result = stitch_front_back(front, back, out, dry_run=True)
    assert result.startswith("DRY RUN")
    assert not out.exists()
    assert not (tmp_path / "nested").exists()


def test_stitch_unknown_direction(tmp_path):
    front = _image(tmp_path / "f.jpg", (100, 100), "red")
    back = _image(tmp_path / "b.jpg", (100, 100), "blue")
    result = stitch_front_back(front, back, tmp_path / "o.jpg", dry_run=False, direction="diagonal")
    assert result.startswith("ERROR")
