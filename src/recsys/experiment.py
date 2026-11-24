"""Run the full ladder: tune on validation, refit on train+val, evaluate once on test.

    python -m recsys.experiment            # writes results/results.json
"""
from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import data, metrics, models
from .split import leave_last_out

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEEDS = [0, 1, 2]
KNN_GRID = [dict(k=k, shrink=s) for k, s in itertools.product([3, 5, 10, 20, 50, 100, 300, None], [0.0, 10.0])]
ALS_GRID = [dict(factors=f, alpha=a, reg=r) for f, a, r in itertools.product([32, 64, 128], [0.5, 1.0, 2.0, 5.0], [10.0, 50.0])]
# Second-round configs added because the best first-round config sat on the grid edge (more factors, lower alpha/reg).
ALS_GRID += [dict(factors=f, alpha=a, reg=r) for f, a, r in
             [(128, 0.5, 3.0), (128, 1.0, 3.0), (128, 0.5, 1.0), (256, 0.5, 10.0), (256, 1.0, 10.0), (256, 0.5, 3.0)]]
ALS_ITERS = 15
TT_GRID = [dict(dim=64, hidden=128, dropout=0.2), dict(dim=64, hidden=256, dropout=0.5),
           dict(dim=32, hidden=128, dropout=0.2)]
TT_EPOCHS = 12


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def env_info():
    import scipy, pandas, torch
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pandas.__version__, "torch": torch.__version__,
            "cpu_count": __import__("os").cpu_count(), "torch_threads": 1,
            "device": "cpu"}


def main(quick: bool = False):
    t_start = time.time()
    df = data.load_ratings()
    assert len(df) == data.N_RATINGS
    sp = leave_last_out(df)
    R_tr, R_trv = sp.matrix(["train"]), sp.matrix(["train", "val"])
    tg_val, tg_test = sp.targets("val"), sp.targets("test")
    cold_test = int((np.asarray(R_trv.sum(0)).ravel()[tg_test] == 0).sum())
    out = {"dataset": {"url": data.URL, "sha256": data.SHA256, "n_users": sp.n_users, "n_items": sp.n_items,
                       "n_interactions": int(len(df)), "n_train": len(sp.train), "n_val": len(sp.val),
                       "n_test": len(sp.test), "test_items_never_seen_in_train_or_val": cold_test,
                       "density": len(df) / (sp.n_users * sp.n_items)},
           "env": env_info(), "seeds": SEEDS, "selection_metric": "validation NDCG@10 (history = train)",
           "tuning": {}, "final": {}, "timing_seconds": {}}

    def val_ndcg(model):
        return metrics.ndcg_at(metrics.ranks_of_targets(model, R_tr, tg_val), 10)

    knn_grid, als_grid, tt_grid = (KNN_GRID[:2], ALS_GRID[:1], TT_GRID[:1]) if quick else (KNN_GRID, ALS_GRID, TT_GRID)
    tt_epochs = 2 if quick else TT_EPOCHS

    # ---- every unit of work is memoised in a gitignored cache so an interrupted run resumes ----
    cache_path = RESULTS / ".cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not quick else {}

    def memo(key, fn):
        if key not in cache:
            t0 = time.time()
            val = fn()
            val["_secs"] = time.time() - t0
            cache[key] = val
            if not quick:
                RESULTS.mkdir(exist_ok=True)
                cache_path.write_text(json.dumps(cache))
        return cache[key]

    # ---- tuning on validation (seed 0) ----
    def tune_stage(name, grid, fit):
        rows = []
        for cfg in grid:
            key = f"tune_{name}_{json.dumps(cfg, sort_keys=True)}"
            row = memo(key, lambda: fit(cfg))
            rows.append({**cfg, **{k: v for k, v in row.items() if k != "_secs"}})
            log(name, {k: v for k, v in rows[-1].items() if k != "curve"})
        out["timing_seconds"][f"tune_{name}"] = sum(cache[f"tune_{name}_{json.dumps(c, sort_keys=True)}"]["_secs"] for c in grid)
        return rows

    tune = tune_stage("item_knn", knn_grid, lambda c: {"val_ndcg@10": val_ndcg(models.ItemKNN(**c).fit(R_tr))})
    best_knn = max(tune, key=lambda r: r["val_ndcg@10"])
    out["tuning"]["item_knn_cosine"] = {"grid": tune, "best": best_knn}

    tune = tune_stage("als", als_grid, lambda c: {"iters": ALS_ITERS, "val_ndcg@10": val_ndcg(models.ALS(**c, iters=ALS_ITERS, seed=0).fit(R_tr))})
    best_als = max(tune, key=lambda r: r["val_ndcg@10"])
    out["tuning"]["als"] = {"grid": tune, "best": best_als}

    def fit_tt(c):
        m = models.TwoTower(**c, epochs=tt_epochs, seed=0, eval_fn=val_ndcg).fit(R_tr)
        return {"best_epoch": m.best_epoch, "val_ndcg@10": m.best_eval, "curve": m.history}
    tune = tune_stage("two_tower", tt_grid, fit_tt)
    best_tt = max(tune, key=lambda r: r["val_ndcg@10"])
    out["tuning"]["two_tower"] = {"grid": tune, "best": {k: v for k, v in best_tt.items() if k != "curve"}, "epochs_max": tt_epochs}

    # ---- final: refit on train+val, evaluate on test ----
    counts = np.asarray(R_trv.sum(0)).ravel()
    bk = metrics.popularity_buckets(counts)
    out["final_buckets"] = {"head_items": int((bk == 0).sum()), "mid_items": int((bk == 1).sum()), "tail_items": int((bk == 2).sum())}
    per_user_ndcg = {}  # seed-0 per-user NDCG@10 for paired bootstrap

    def evaluate(name, make, seeds):
        runs = []
        for s in seeds:
            def run():
                m = make(s).fit(R_trv)
                ranks = metrics.ranks_of_targets(m, R_trv, tg_test)
                top = metrics.topk_lists(m, R_trv, 20)
                rep = metrics.full_report(ranks, top, counts, tg_test)
                rep["seed"] = s
                nd = metrics.per_user_ndcg(ranks, 10)
                rep["bootstrap95_seed0"] = {
                    "recall@10": list(metrics.bootstrap_ci((ranks <= 10).astype(float))),
                    "recall@20": list(metrics.bootstrap_ci((ranks <= 20).astype(float))),
                    "ndcg@10": list(metrics.bootstrap_ci(nd))}
                return {"rep": rep, "ndcg": nd.tolist()}
            res = memo(f"final_{name}_{s}_{json.dumps(sorted(best_cfg_key.get(name, {}).items()))}", run)
            rep = dict(res["rep"]); rep["fit_and_eval_seconds"] = res["_secs"]
            if s != seeds[0]:
                rep.pop("bootstrap95_seed0", None)
            else:
                per_user_ndcg[name] = np.array(res["ndcg"])
            runs.append(rep)
            log("final", name, s, round(rep["recall@10"], 4), round(rep["ndcg@10"], 4))
        keys = ["recall@10", "recall@20", "ndcg@10", "coverage@10", "coverage@20"]
        agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys}
        out["final"][name] = {"aggregate_over_seeds": agg, "n_seeds": len(seeds), "runs": runs}

    best_cfg_key = {"item_knn_cosine": {k: best_knn[k] for k in ("k", "shrink")},
                    "als": {k: best_als[k] for k in ("factors", "alpha", "reg")},
                    "two_tower": {k: best_tt[k] for k in ("dim", "hidden", "dropout", "best_epoch")}}
    seeds = SEEDS[:1] if quick else SEEDS
    evaluate("random", lambda s: models.Random(s), seeds)
    evaluate("popularity", lambda s: models.Popularity(), [0])
    evaluate("item_knn_cosine", lambda s: models.ItemKNN(k=best_knn["k"], shrink=best_knn["shrink"]), [0])
    evaluate("als", lambda s: models.ALS(factors=best_als["factors"], alpha=best_als["alpha"], reg=best_als["reg"], iters=ALS_ITERS, seed=s), seeds)
    tt_cfg = {k: best_tt[k] for k in ("dim", "hidden", "dropout")}
    evaluate("two_tower", lambda s: models.TwoTower(**tt_cfg, epochs=best_tt["best_epoch"], seed=s), seeds)

    names = ["popularity", "item_knn_cosine", "als", "two_tower"]
    paired = {}
    for a, b in itertools.combinations(names, 2):
        d = per_user_ndcg[b] - per_user_ndcg[a]
        lo, hi = metrics.bootstrap_ci(d)
        paired[f"{b}_minus_{a}"] = {"mean_diff_ndcg@10": float(d.mean()), "ci95": [lo, hi]}
    out["paired_ndcg@10_seed0"] = paired
    out["timing_seconds"]["last_resumed_session"] = time.time() - t_start
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / ("results_quick.json" if quick else "results.json")
    path.write_text(json.dumps(out, indent=1))
    log("wrote", path)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
