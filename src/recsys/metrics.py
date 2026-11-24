"""Full-catalog ranking metrics. No sampled negatives anywhere in evaluation.

Each user has exactly one held-out item, so Recall@K equals HitRate@K.
Items in the user's history are excluded from the ranking. Ties are broken by lower
item index, deterministically.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

BATCH = 1000


def _masked_scores(model, users: np.ndarray, hist: sp.csr_matrix) -> np.ndarray:
    s = np.asarray(model.score(users), dtype=np.float32).copy()
    if not np.isfinite(s).all():
        raise FloatingPointError("non-finite scores")
    sub = hist[users]
    rows = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
    s[rows, sub.indices] = -np.inf
    return s


def ranks_of_targets(model, hist: sp.csr_matrix, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each user's held-out item among all non-history catalog items."""
    n = hist.shape[0]
    out = np.empty(n, dtype=np.int64)
    idx = np.arange(hist.shape[1])
    for a in range(0, n, BATCH):
        users = np.arange(a, min(a + BATCH, n))
        s = _masked_scores(model, users, hist)
        t = targets[users]
        ts = s[np.arange(len(users)), t][:, None]
        out[users] = 1 + (s > ts).sum(1) + ((s == ts) & (idx[None, :] < t[:, None])).sum(1)
    return out


def topk_lists(model, hist: sp.csr_matrix, k: int = 20) -> np.ndarray:
    n = hist.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for a in range(0, n, BATCH):
        users = np.arange(a, min(a + BATCH, n))
        s = _masked_scores(model, users, hist)
        out[users] = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return out


def recall_at(ranks: np.ndarray, k: int) -> float:
    return float((ranks <= k).mean())


def ndcg_at(ranks: np.ndarray, k: int) -> float:
    g = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return float(g.mean())


def per_user_ndcg(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    return np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)


def coverage_at(topk: np.ndarray, n_items: int, k: int) -> float:
    return float(len(np.unique(topk[:, :k])) / n_items)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def popularity_buckets(train_counts: np.ndarray) -> np.ndarray:
    """0=head (top 10% of catalog by popularity), 1=mid (next 40%), 2=tail (bottom 50%)."""
    order = np.lexsort((np.arange(len(train_counts)), -train_counts))
    b = np.empty(len(train_counts), dtype=np.int64)
    n = len(order)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    b[:] = 2
    b[pos < int(0.5 * n)] = 1
    b[pos < int(0.1 * n)] = 0
    return b


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """95% CI of mean(a-b) over users."""
    return bootstrap_ci(a - b, n_boot, seed)


def full_report(ranks: np.ndarray, topk: np.ndarray, train_counts: np.ndarray, targets: np.ndarray) -> dict:
    n_items = len(train_counts)
    buckets = popularity_buckets(train_counts)
    exposure = np.bincount(topk[:, :10].ravel(), minlength=n_items)
    tb = buckets[targets]
    rec10_by_bucket = {}
    for name, b in (("head", 0), ("mid", 1), ("tail", 2)):
        m = tb == b
        rec10_by_bucket[name] = {"n_users": int(m.sum()),
                                 "recall@10": float((ranks[m] <= 10).mean()) if m.any() else None}
    rec_buckets = buckets[topk[:, :10]]
    pop_pct = np.argsort(np.argsort(-train_counts, kind="stable"), kind="stable") / (n_items - 1)
    nd = per_user_ndcg(ranks, 10)
    return {
        "recall@10": recall_at(ranks, 10),
        "recall@20": recall_at(ranks, 20),
        "ndcg@10": ndcg_at(ranks, 10),
        "coverage@10": coverage_at(topk, n_items, 10),
        "coverage@20": coverage_at(topk, n_items, 20),
        "popularity_bias": {
            "head_share_of_top10_slots": float((rec_buckets == 0).mean()),
            "mid_share_of_top10_slots": float((rec_buckets == 1).mean()),
            "tail_share_of_top10_slots": float((rec_buckets == 2).mean()),
            "head_share_of_test_items": float((tb == 0).mean()),
            "mean_popularity_percentile_of_recs": float(pop_pct[topk[:, :10]].mean()),
            "mean_popularity_percentile_of_test_items": float(pop_pct[targets].mean()),
            "exposure_gini@10": gini(exposure),
            "recall@10_by_test_item_bucket": rec10_by_bucket,
        },
        "per_user": {
            "rank_percentiles": {str(q): float(np.percentile(ranks, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
            "share_ndcg@10_zero": float((nd == 0).mean()),
            "rank_histogram": {
                "labels": ["1-10", "11-20", "21-50", "51-100", "101-500", "501-1000", ">1000"],
                "counts": np.histogram(ranks, bins=[1, 11, 21, 51, 101, 501, 1001, n_items + 2])[0].tolist(),
            },
        },
    }
