import numpy as np
import scipy.sparse as sp

from recsys import metrics, models


def block_matrix(n_users=60, n_items=40, seed=0):
    """Two user groups, each preferring one half of the catalog."""
    rng = np.random.default_rng(seed)
    M = np.zeros((n_users, n_items), dtype=np.float32)
    half = n_items // 2
    for u in range(n_users):
        pool = np.arange(half) if u % 2 == 0 else np.arange(half, n_items)
        M[u, rng.choice(pool, 12, replace=False)] = 1
    return sp.csr_matrix(M)


def block_hit_rate(model, R):
    users = np.arange(R.shape[0])
    s = model.score(users).copy()
    s[R.toarray() > 0] = -np.inf
    top = s.argmax(1)
    half = R.shape[1] // 2
    return float(np.mean((top < half) == (users % 2 == 0)))


def test_popularity_scores_are_counts():
    R = sp.csr_matrix(np.array([[1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32))
    assert models.Popularity().fit(R).score([0])[0].tolist() == [3, 1, 1]


def test_knn_cosine_values():
    R = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32))
    m = models.ItemKNN(k=None).fit(R)
    assert np.isclose(m.S[0, 1], 1.0) and m.S[0, 2] == 0 and m.S[0, 0] == 0


def test_models_learn_block_structure():
    R = block_matrix()
    assert block_hit_rate(models.ItemKNN(k=10).fit(R), R) == 1.0
    assert block_hit_rate(models.ALS(factors=4, iters=8).fit(R), R) >= 0.95
    tt = models.TwoTower(dim=8, hidden=16, dropout=0.0, epochs=40, batch=64, seed=0).fit(R)
    assert block_hit_rate(tt, R) >= 0.9


def test_random_is_near_chance():
    R = block_matrix(n_users=200, n_items=100)
    tg = np.array([int(np.flatnonzero(R[u].toarray()[0])[0]) for u in range(200)])
    ranks = metrics.ranks_of_targets(models.Random(0).fit(R), sp.csr_matrix(R.shape, dtype=np.float32), tg)
    assert 0.05 < metrics.recall_at(ranks, 10) < 0.2  # chance = 0.10


def test_seeded_models_are_deterministic():
    R = block_matrix()
    a = models.ALS(factors=4, iters=3, seed=2).fit(R).score(np.arange(5))
    b = models.ALS(factors=4, iters=3, seed=2).fit(R).score(np.arange(5))
    assert np.allclose(a, b)
    t1 = models.TwoTower(dim=8, hidden=16, epochs=2, batch=64, seed=1).fit(R).score(np.arange(5))
    t2 = models.TwoTower(dim=8, hidden=16, epochs=2, batch=64, seed=1).fit(R).score(np.arange(5))
    assert np.allclose(t1, t2)
