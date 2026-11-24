import numpy as np
import pandas as pd

from conftest import synthetic_ratings
from recsys.split import leave_last_out
from recsys import models


def _keys(df):
    return list(zip(df.ts.tolist(), df.item.tolist()))


def check_no_future_leakage(sp):
    tr = sp.train.groupby("user")
    for u in range(sp.n_users):
        t_tr = _keys(sp.train[sp.train.user == u])
        v = _keys(sp.val[sp.val.user == u])
        te = _keys(sp.test[sp.test.user == u])
        assert len(v) == 1 and len(te) == 1 and len(t_tr) >= 1
        # every train interaction strictly precedes val, which strictly precedes test
        assert max(t_tr) < v[0] < te[0]
        # timestamps are never later than the next split's timestamp (ties allowed only across split edges)
        assert max(k[0] for k in t_tr) <= v[0][0] <= te[0][0]


def test_synthetic_split_with_ties():
    sp = leave_last_out(synthetic_ratings(ties=True))
    check_no_future_leakage(sp)


def test_synthetic_split_without_ties():
    sp = leave_last_out(synthetic_ratings(ties=False, seed=3))
    check_no_future_leakage(sp)


def test_splits_are_disjoint_and_cover_everything():
    df = synthetic_ratings()
    sp = leave_last_out(df)
    pairs = [set(zip(p.user, p.item)) for p in (sp.train, sp.val, sp.test)]
    assert not (pairs[0] & pairs[1]) and not (pairs[0] & pairs[2]) and not (pairs[1] & pairs[2])
    assert sum(len(p) for p in pairs) == len(df.drop_duplicates(["user", "item"]))


def test_test_item_is_latest_even_if_input_is_shuffled():
    df = synthetic_ratings(ties=False)
    a = leave_last_out(df)
    b = leave_last_out(df.sample(frac=1, random_state=1))
    assert (a.test.values == b.test.values).all() and (a.train.values == b.train.values).all()
    last = df.sort_values(["user", "ts", "item"]).groupby("user").tail(1)
    assert set(zip(last.user, last.item)) == set(zip(a.user_ids[a.test.user], a.item_ids[a.test.item]))


def test_train_matrix_contains_no_val_or_test_pairs():
    sp = leave_last_out(synthetic_ratings())
    R = sp.matrix(["train"])
    for part in (sp.val, sp.test):
        assert R[part.user.values, part.item.values].sum() == 0
    Rv = sp.matrix(["train", "val"])
    assert Rv[sp.test.user.values, sp.test.item.values].sum() == 0
    assert Rv[sp.val.user.values, sp.val.item.values].sum() == len(sp.val)


def test_models_fit_on_train_are_invariant_to_val_and_test_content():
    """Replace every val/test item with a different item: models fit on train must not change."""
    df = synthetic_ratings(ties=False, seed=5)
    sp1 = leave_last_out(df)
    sp2 = leave_last_out(df)
    rng = np.random.default_rng(0)
    hist = sp2.matrix(["train"])
    for part in (sp2.val, sp2.test):
        part["item"] = rng.integers(0, sp2.n_items, len(part))
    R1, R2 = sp1.matrix(["train"]), sp2.matrix(["train"])
    assert (R1 != R2).nnz == 0
    users = np.arange(sp1.n_users)
    for mk in (lambda: models.Popularity(), lambda: models.ItemKNN(k=5),
               lambda: models.ALS(factors=4, iters=3, seed=1)):
        s1 = mk().fit(R1).score(users)
        s2 = mk().fit(R2).score(users)
        assert np.allclose(s1, s2)


def test_real_data_split_is_temporal_per_user(real_ratings):
    sp = leave_last_out(real_ratings)
    assert (sp.n_users, sp.n_items) == (6040, 3706)
    assert len(sp.train) + len(sp.val) + len(sp.test) == 1_000_209
    check = pd.concat([sp.train.assign(p=0), sp.val.assign(p=1), sp.test.assign(p=2)])
    g = check.groupby(["user", "p"]).ts.agg(["min", "max"]).unstack("p")
    assert (g[("max", 0)] <= g[("min", 1)]).all()
    assert (g[("max", 1)] <= g[("min", 2)]).all()
    check_no_future_leakage(sp)
