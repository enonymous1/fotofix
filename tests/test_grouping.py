from fotofix.grouping import (
    find_file_groups,
    get_base_name,
    is_backside,
    parse_scan_name,
    scan_directory,
)


def make_files(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_parse_scan_name():
    assert parse_scan_name("FastFoto_0452.jpg") == ("FastFoto_0452", "")
    assert parse_scan_name("FastFoto_0452_a.jpg") == ("FastFoto_0452", "a")
    assert parse_scan_name("FastFoto_0452_B.JPG") == ("FastFoto_0452", "b")
    assert parse_scan_name("photo_a.jpg") == ("photo", "a")
    assert parse_scan_name("my_a_photo.jpg") == ("my_a_photo", "")
    assert parse_scan_name("scan_b2.jpg") == ("scan_b2", "")


def test_get_base_name_preserves_case():
    assert get_base_name("FastFoto_0452_a.jpg") == "FastFoto_0452"


def test_is_backside():
    assert is_backside("FastFoto_0452_b.jpg")
    assert not is_backside("FastFoto_0452_a.jpg")
    assert not is_backside("FastFoto_0452.jpg")


def test_find_file_groups_fastfoto_and_generic(tmp_path):
    make_files(
        tmp_path,
        "FastFoto_0452.jpg",
        "FastFoto_0452_a.jpg",
        "FastFoto_0452_b.jpg",
        "photo.jpg",
        "photo_a.jpg",
    )
    groups = find_file_groups([tmp_path / n for n in sorted(p.name for p in tmp_path.iterdir())])
    assert set(groups) == {"fastfoto_0452", "photo"}
    assert len(groups["fastfoto_0452"]) == 3
    assert len(groups["photo"]) == 2


def test_find_file_groups_case_insensitive(tmp_path):
    make_files(tmp_path, "FastFoto_0001.jpg", "fastfoto_0001_a.jpg")
    groups = find_file_groups(list(tmp_path.iterdir()))
    assert len(groups) == 1


def test_scan_directory_filters_and_sorts(tmp_path):
    make_files(tmp_path, "b.jpg", "a.jpg", "notes.txt", "A_b.jpg")
    groups = scan_directory(tmp_path)
    assert [g.base for g in groups] == ["a", "b"]
    assert groups[0].front.name == "a.jpg"
    assert groups[0].back.name == "A_b.jpg"


def test_scan_directory_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    make_files(tmp_path, "top.jpg")
    make_files(sub, "deep.jpg")
    assert len(scan_directory(tmp_path)) == 1
    assert len(scan_directory(tmp_path, recursive=True)) == 2


def test_group_front_prefers_secondary(tmp_path):
    make_files(tmp_path, "x.jpg", "x_a.jpg", "x_b.jpg")
    group = scan_directory(tmp_path)[0]
    assert group.primary.name == "x.jpg"
    assert group.secondary.name == "x_a.jpg"
    assert group.front.name == "x_a.jpg"
    assert group.back.name == "x_b.jpg"
