from PIL import Image

from fotofix.image import get_proxy_image, save_corrected_image


def _image(path, size=(100, 200), color="red"):
    Image.new("RGB", size, color).save(path)
    return path


def test_proxy_resizes_and_returns_none_for_bad_file(tmp_path):
    src = _image(tmp_path / "a.jpg", size=(2000, 1000))
    proxy = get_proxy_image(src, max_dim=500)
    assert proxy is not None
    assert max(proxy.size) <= 500
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert get_proxy_image(bad) is None


def test_proxy_applies_exif_orientation(tmp_path):
    src = tmp_path / "oriented.jpg"
    img = Image.new("RGB", (100, 200), "blue")
    exif = img.getexif()
    exif[274] = 6  # Rotate 90 CW to display
    img.save(src, exif=exif.tobytes())
    proxy = get_proxy_image(src, max_dim=1500)
    assert proxy.size == (200, 100)


def test_save_corrected_rotates_and_resets_orientation(tmp_path):
    src = _image(tmp_path / "a.jpg", size=(100, 50))
    save_corrected_image(src, 90)
    with Image.open(src) as img:
        assert img.size == (50, 100)
        assert img.getexif().get(274, 1) == 1


def test_save_corrected_preserves_other_exif_and_backs_up(tmp_path):
    src = tmp_path / "a.jpg"
    img = Image.new("RGB", (100, 200), "green")
    exif = img.getexif()
    exif[306] = "2026:04:29 12:00:00"
    exif[274] = 6
    img.save(src, exif=exif.tobytes())
    original = src.read_bytes()
    backup_dir = tmp_path / "backups"
    save_corrected_image(src, 0, backup_dir=backup_dir)
    backup = backup_dir / "a.jpg"
    assert backup.read_bytes() == original
    with Image.open(src) as img:
        assert img.getexif().get(306) == "2026:04:29 12:00:00"
        assert img.getexif().get(274) == 1
    assert not list(tmp_path.glob(".fotofix-*"))


def test_save_corrected_failed_save_leaves_original(tmp_path, monkeypatch):
    src = _image(tmp_path / "a.jpg", size=(100, 50))
    original = src.read_bytes()
    monkeypatch.setattr(
        Image.Image,
        "save",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    try:
        save_corrected_image(src, 90)
    except RuntimeError:
        pass
    assert src.read_bytes() == original
