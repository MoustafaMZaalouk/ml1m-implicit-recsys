"""Model ladder. Every model exposes fit(hist_csr) and score(user_indices) -> (n, n_items).

fit() receives only the history matrix it is allowed to see (binary users x items).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Random:
    """Seeded random ranking, a sanity floor."""
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, R: sp.csr_matrix):
        self.n_items = R.shape[1]
        return self

    def score(self, users):
        out = np.empty((len(users), self.n_items), dtype=np.float32)
        for j, u in enumerate(users):
            out[j] = np.random.default_rng((self.seed, int(u))).random(self.n_items)
        return out


class Popularity:
    name = "popularity"

    def fit(self, R: sp.csr_matrix):
        self.counts = np.asarray(R.sum(0)).ravel().astype(np.float32)
        return self

    def score(self, users):
        return np.broadcast_to(self.counts, (len(users), len(self.counts)))


class ItemKNN:
    """Item-item cosine similarity on binary data, optional top-k neighbours and shrinkage."""
    name = "item_knn_cosine"

    def __init__(self, k: int | None = 100, shrink: float = 0.0):
        self.k, self.shrink = k, shrink

    def fit(self, R: sp.csr_matrix):
        self.R = R.tocsr().astype(np.float32)
        co = (self.R.T @ self.R).toarray().astype(np.float32)
        norm = np.sqrt(np.diag(co))
        denom = np.outer(norm, norm) + self.shrink
        S = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        np.fill_diagonal(S, 0.0)
        if self.k is not None and self.k < S.shape[0]:
            drop = np.argpartition(-S, self.k, axis=1)[:, self.k:]
            np.put_along_axis(S, drop, 0.0, axis=1)
        self.S = S
        return self

    def score(self, users):
        return np.asarray(self.R[users] @ self.S)


class ALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky 2008), exact per-row solves in numpy.

    Observed entries have preference 1 and confidence 1 + alpha; unobserved have preference 0
    and confidence 1.
    """
    name = "als"

    def __init__(self, factors=64, alpha=10.0, reg=10.0, iters=15, seed=0):
        self.f, self.alpha, self.reg, self.iters, self.seed = factors, alpha, reg, iters, seed

    def _solve_side(self, R: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        f = self.f
        YtY = Y.T @ Y + self.reg * np.eye(f, dtype=np.float64)
        X = np.zeros((R.shape[0], f), dtype=np.float64)
        for u in range(R.shape[0]):
            cols = R.indices[R.indptr[u]:R.indptr[u + 1]]
            if len(cols) == 0:
                continue
            Yu = Y[cols]
            A = YtY + self.alpha * (Yu.T @ Yu)
            b = (1.0 + self.alpha) * Yu.sum(0)
            X[u] = np.linalg.solve(A, b)
        return X

    def fit(self, R: sp.csr_matrix):
        R = R.tocsr()
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (R.shape[0], self.f))
        self.V = rng.normal(0, 0.01, (R.shape[1], self.f))
        for _ in range(self.iters):
            self.U = self._solve_side(R, self.V)
            self.V = self._solve_side(Rt, self.U)
        return self

    def score(self, users):
        return (self.U[users] @ self.V.T).astype(np.float32)


class TwoTower:
    """Two-tower MLP trained with a full-catalog softmax (CPU, PyTorch).

    user tower: user-id embedding -> MLP; item tower: item-id embedding -> MLP.
    Score = dot product. Training uses the full softmax over all items (no sampled negatives);
    one epoch visits every training (user, item) pair once.
    """
    name = "two_tower"

    def __init__(self, dim=64, hidden=128, dropout=0.2, lr=5e-3, weight_decay=1e-5,
                 batch=8192, epochs=12, seed=0, eval_fn=None):
        self.kw = dict(dim=dim, hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay,
                       batch=batch, epochs=epochs)
        self.seed, self.eval_fn = seed, eval_fn
        self.history = []  # per-epoch (epoch, train_loss, eval_score) when eval_fn is given

    def _build(self, n_users, n_items):
        import torch
        from torch import nn
        k = self.kw

        def tower(n):
            return nn.Sequential(nn.Embedding(n, k["dim"]), nn.Linear(k["dim"], k["hidden"]), nn.ReLU(),
                                 nn.Dropout(k["dropout"]), nn.Linear(k["hidden"], k["dim"]))
        self.ut, self.it = tower(n_users), tower(n_items)
        for t in (self.ut, self.it):
            nn.init.normal_(t[0].weight, 0, 0.1)
        self.n_items = n_items

    def _item_matrix(self):
        import torch
        return self.it(torch.arange(self.n_items))

    def fit(self, R: sp.csr_matrix):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        k = self.kw
        coo = R.tocoo()
        u_all = torch.from_numpy(coo.row.astype(np.int64))
        i_all = torch.from_numpy(coo.col.astype(np.int64))
        self._build(*R.shape)
        params = list(self.ut.parameters()) + list(self.it.parameters())
        opt = torch.optim.AdamW(params, lr=k["lr"], weight_decay=k["weight_decay"])
        gen = torch.Generator().manual_seed(self.seed)
        self.best_epoch, self.best_eval = None, -1.0
        for ep in range(1, k["epochs"] + 1):
            self.ut.train(); self.it.train()
            perm = torch.randperm(len(u_all), generator=gen)
            tot, nb = 0.0, 0
            for a in range(0, len(perm), k["batch"]):
                ix = perm[a:a + k["batch"]]
                ue = self.ut(u_all[ix])
                logits = ue @ self._item_matrix().T
                loss = F.cross_entropy(logits, i_all[ix])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            rec = {"epoch": ep, "train_loss": tot / nb}
            if self.eval_fn is not None:
                self._refresh()
                rec["eval"] = float(self.eval_fn(self))
                if rec["eval"] > self.best_eval:
                    self.best_eval, self.best_epoch = rec["eval"], ep
            self.history.append(rec)
        self._refresh()
        return self

    def _refresh(self):
        import torch
        self.ut.eval(); self.it.eval()
        with torch.no_grad():
            self.Vmat = self._item_matrix().numpy()
            self.Umat = self.ut(torch.arange(self.ut[0].num_embeddings)).numpy()

    def score(self, users):
        return (self.Umat[users] @ self.Vmat.T).astype(np.float32)
