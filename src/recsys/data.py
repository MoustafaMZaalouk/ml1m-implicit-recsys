"""Download-at-build with checksum pin. The dataset is never committed."""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MD5 = "c4d9eecfca2ab87c1945afe126590906"  # published by GroupLens in ml-1m.zip.md5
N_RATINGS = 1_000_209
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(path: Path) -> None:
    got = sha256_of(path)
    if got != SHA256:
        raise RuntimeError(f"checksum mismatch for {path}: expected {SHA256}, got {got}")
    if md5_of(path) != MD5:
        raise RuntimeError(f"md5 mismatch for {path}")


def ensure_data(data_dir: Path = DATA_DIR) -> Path:
    """Return the path to ratings.dat, downloading and verifying if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "ml-1m.zip"
    ratings = data_dir / "ml-1m" / "ratings.dat"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    try:
        verify_zip(zpath)
    except RuntimeError:
        zpath.unlink()
        raise
    if not ratings.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data_dir)
    return ratings


def load_ratings(path: Path | None = None) -> pd.DataFrame:
    """Columns: user, item, rating, ts (raw MovieLens ids)."""
    path = path or ensure_data()
    df = pd.read_csv(path, sep="::", engine="python", names=["user", "item", "rating", "ts"],
                     dtype={"user": np.int64, "item": np.int64, "rating": np.int8, "ts": np.int64})
    return df
