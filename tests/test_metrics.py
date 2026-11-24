import numpy as np
import scipy.sparse as sp

from recsys import metrics


class Fixed:
    def __init__(self, S):
        self.S = np.asarray(S, dtype=np.float32)

    def score(self, users):
        return self.S[users]


def test_rank_excludes_history_and_breaks_ties_by_index():
    S = [[5, 4, 3, 2, 1],
         [1, 1, 1, 1, 1]]
    hist = sp.csr_matrix(np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
    r = metrics.ranks_of_targets(Fixed(S), hist, np.array([3, 3]))
    # user0: item0 hidden, order is 1,2,3,4 -> item 3 is rank 3. user1: all tied, index order -> rank 4
    assert r.tolist() == [3, 4]


def test_topk_matches_rank_definition():
    rng = np.random.default_rng(0)
    S = rng.integers(0, 4, (20, 15)).astype(np.float32)  # many ties
    hist = sp.csr_matrix((rng.random((20, 15)) < 0.2).astype(np.float32))
    m = Fixed(S)
    top = metrics.topk_lists(m, hist, 15)
    for u in range(20):
        t = top[u, 4]
        assert metrics.ranks_of_targets(m, hist, np.full(20, t))[u] == 5 or hist[u, t] == 1
        assert not any(hist[u, i] for i in top[u, : 15 - int(hist[u].sum())])


def test_recall_ndcg_known_values():
    r = np.array([1, 2, 10, 11, 25])
    assert metrics.recall_at(r, 10) == 3 / 5
    assert metrics.recall_at(r, 20) == 4 / 5
    expect = (1 + 1 / np.log2(3) + 1 / np.log2(11)) / 5
    assert np.isclose(metrics.ndcg_at(r, 10), expect)


def test_coverage_gini_buckets():
    top = np.array([[0, 1], [0, 2], [0, 1]])
    assert metrics.coverage_at(top, 10, 2) == 0.3
    assert metrics.gini(np.array([1, 1, 1, 1])) == 0.0
    assert metrics.gini(np.array([0, 0, 0, 4])) > 0.7
    counts = np.arange(100)[::-1]  # item 0 most popular
    b = metrics.popularity_buckets(counts)
    assert (b == 0).sum() == 10 and (b == 1).sum() == 40 and (b == 2).sum() == 50 and b[0] == 0 and b[99] == 2


def test_bootstrap_ci_contains_mean():
    x = (np.random.default_rng(0).random(500) < 0.3).astype(float)
    lo, hi = metrics.bootstrap_ci(x)
    assert lo < x.mean() < hi
