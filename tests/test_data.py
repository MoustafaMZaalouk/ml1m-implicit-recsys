import pytest

from recsys import data


def test_pins_look_sane():
    assert len(data.SHA256) == 64 and len(data.MD5) == 32
    assert data.URL.startswith("https://files.grouplens.org/")


def test_corrupt_download_is_rejected(tmp_path):
    bad = tmp_path / "ml-1m.zip"
    bad.write_bytes(b"not the dataset")
    with pytest.raises(RuntimeError):
        data.verify_zip(bad)


def test_real_download_matches_pin():
    z = data.DATA_DIR / "ml-1m.zip"
    if not z.exists():
        pytest.skip("not downloaded")
    data.verify_zip(z)


def test_dataset_is_gitignored():
    from pathlib import Path
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "data/" in ignore
