"""Temporal leave-last-interaction-out split per user.

For each user, interactions are ordered by (timestamp, item id) (item id is a deterministic
tie-break because MovieLens timestamps have one-second resolution and many ties).
  test       = the user's last interaction
  validation = the user's second-to-last interaction
  train      = everything earlier
Hyperparameters are chosen on validation using train as history. The final models are
refit on train+validation and scored on test with train+validation as history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Split:
    n_users: int
    n_items: int
    train: pd.DataFrame  # columns user, item, ts (contiguous indices)
    val: pd.DataFrame
    test: pd.DataFrame
    user_ids: np.ndarray  # index -> raw MovieLens id
    item_ids: np.ndarray

    def matrix(self, parts=("train",)) -> sp.csr_matrix:
        df = pd.concat([getattr(self, p) for p in parts])
        m = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.user.values, df.item.values)),
                          shape=(self.n_users, self.n_items))
        m.sum_duplicates()
        m.data[:] = 1.0
        return m

    def targets(self, part: str) -> np.ndarray:
        df = getattr(self, part).sort_values("user")
        assert len(df) == self.n_users and (df.user.values == np.arange(self.n_users)).all()
        return df.item.values.astype(np.int64)


def leave_last_out(df: pd.DataFrame, min_interactions: int = 3) -> Split:
    """df: raw columns user, item, ts (rating is ignored: every rating is an implicit positive)."""
    df = df[["user", "item", "ts"]].drop_duplicates(["user", "item"])
    counts = df.groupby("user").size()
    if (counts < min_interactions).any():
        raise ValueError("every user needs at least 3 interactions")
    user_ids = np.sort(df.user.unique())
    item_ids = np.sort(df.item.unique())  # full catalog (all items appearing in the file)
    d = pd.DataFrame({
        "user": np.searchsorted(user_ids, df.user.values),
        "item": np.searchsorted(item_ids, df.item.values),
        "ts": df.ts.values,
    })
    d = d.sort_values(["user", "ts", "item"], kind="mergesort").reset_index(drop=True)
    from_end = d.groupby("user").cumcount(ascending=False).values
    return Split(len(user_ids), len(item_ids),
                 d[from_end >= 2].reset_index(drop=True),
                 d[from_end == 1].reset_index(drop=True),
                 d[from_end == 0].reset_index(drop=True),
                 user_ids, item_ids)
