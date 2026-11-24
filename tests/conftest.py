import numpy as np
import pandas as pd
import pytest

from recsys import data


def synthetic_ratings(n_users=30, n_items=40, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        k = int(rng.integers(5, 25))
        items = rng.choice(n_items, size=k, replace=False)
        ts = rng.integers(1000, 1010 if ties else 10**6, size=k)  # many tied timestamps when ties=True
        rows += [(1000 + u, 7 + int(i), int(t)) for i, t in zip(items, ts)]
    return pd.DataFrame(rows, columns=["user", "item", "ts"])


@pytest.fixture(scope="session")
def real_ratings():
    path = data.DATA_DIR / "ml-1m" / "ratings.dat"
    if not path.exists():
        pytest.skip("MovieLens not downloaded; run `make data`")
    data.ensure_data()
    return data.load_ratings(path)
