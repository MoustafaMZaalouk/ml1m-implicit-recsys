import json

import pytest

from recsys import report


@pytest.fixture(scope="module")
def r():
    if not report.RES.exists():
        pytest.skip("results.json missing")
    return json.loads(report.RES.read_text())


def m(r, name, k):
    return r["final"][name]["aggregate_over_seeds"][k]["mean"]


def test_als_best_on_main_metrics(r):
    for k in ("recall@10", "recall@20", "ndcg@10"):
        assert m(r, "als", k) == max(m(r, n, k) for n in ("popularity", "item_knn_cosine", "als", "two_tower"))


def test_paired_claims(r):
    p = r["paired_ndcg@10_seed0"]
    assert p["als_minus_item_knn_cosine"]["ci95"][0] > 0
    assert p["two_tower_minus_als"]["ci95"][1] < 0
    lo, hi = p["two_tower_minus_item_knn_cosine"]["ci95"]
    assert lo < 0 < hi
    for n in ("item_knn_cosine", "als", "two_tower"):
        assert p[f"{n}_minus_popularity"]["ci95"][0] > 0


def test_head_dominates_and_tail_recall_low(r):
    for n in ("popularity", "item_knn_cosine", "als", "two_tower"):
        pb = r["final"][n]["runs"][0]["popularity_bias"]
        assert pb["head_share_of_top10_slots"] > 0.5
        assert pb["recall@10_by_test_item_bucket"]["tail"]["recall@10"] < 0.01
    rnd = r["final"]["random"]
    assert m(r, "random", "coverage@10") == max(m(r, n, "coverage@10") for n in r["final"])
    assert m(r, "random", "ndcg@10") == min(m(r, n, "ndcg@10") for n in r["final"])


def test_knn_median_rank_worse_than_popularity_but_recall_higher(r):
    med = lambda n: r["final"][n]["runs"][0]["per_user"]["rank_percentiles"]["50"]
    assert med("item_knn_cosine") > med("popularity")
    assert m(r, "item_knn_cosine", "recall@10") > m(r, "popularity", "recall@10")


def test_als_search_hit_low_alpha_edge(r):
    grid = r["tuning"]["als"]["grid"]
    assert r["tuning"]["als"]["best"]["alpha"] == min(g["alpha"] for g in grid)
