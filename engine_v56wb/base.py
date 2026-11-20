from threadpoolctl import threadpool_limits
import os,time
import numpy as np
_FLOOR=1e-3
_SEED=20261020
_NET_TRACE=[]
from threadpoolctl import threadpool_limits
_THREADS=max(1,min(8,os.cpu_count() or 1))
threadpool_limits(limits=_THREADS)
def _lookup(values):
    table = {}
    for i, value in enumerate(values):
        table.setdefault(value, i)
    for i, value in enumerate(values):
        table.setdefault(str(value), i)
    return table

def _codes(series, values, unknown):
    """Option index per cell; NaN -> -1; a value outside the list -> unknown."""
    table = _lookup(values)
    raw = series.to_numpy(dtype=object)
    out = np.empty(len(raw), np.int64)
    for i, value in enumerate(raw):
        if value is None or (isinstance(value, float) and value != value):
            out[i] = -1
        else:
            hit = table.get(value)
            if hit is None:
                hit = table.get(str(value), unknown)
            out[i] = hit
    return out

def _dense(columns):
    """Dense integer id for each distinct combination of the given columns."""
    if len(columns) == 1:
        _, ids = np.unique(columns[0], return_inverse=True)
        return ids.astype(np.int64)
    stacked = np.stack(columns, 1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return ids.reshape(-1).astype(np.int64)

def _floored_logp(p, y, K):
    picked = p[np.arange(len(y)), y]
    return np.log(_FLOOR + (1.0 - K * _FLOOR) * picked)

def _nll(p, y, K):
    return -float(_floored_logp(p, y, K).mean())

def _normalize(p):
    p = np.clip(p, 0.0, None)
    s = p.sum(1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s

class _Tables:
    """Leave-fold-out count tables, shared across the items of one instrument."""

    def __init__(self, X, fold, folds):
        self.X, self.fold, self.F = X, fold, folds
        self.cache = {}

    def ids(self, features):
        key = tuple(features)
        if key not in self.cache:
            self.cache[key] = _dense([self.X[:, f] for f in key])
        return self.cache[key]

    def counts(self, ids, y, train, K, query=None):
        """Fold-excluded counts on query rows and full counts on every row."""
        nid = int(ids.max()) + 1
        fold = self.fold
        idt, yt, ft = ids[train], y[train], fold[train]
        total = np.bincount(idt * K + yt, minlength=nid * K).reshape(nid, K)
        per = np.bincount((ft * nid + idt) * K + yt,
                          minlength=self.F * nid * K).reshape(self.F, nid, K)
        if query is None:
            iq, fq = idt, ft
        else:
            iq, fq = ids[query], fold[query]
            if np.any((fq < 0) | (fq >= self.F)):
                raise ValueError('fold-excluded query needs visible folds')
        oof = total[iq] - per[fq, iq]
        return oof.astype(float), total[ids].astype(float)

def _table_chain(tables, chain, y, train, K, alpha, prior_oof, prior_all,
                 query=None):
    p_oof, p_all = prior_oof, prior_all
    for depth in range(1, len(chain) + 1):
        c_oof, c_all = tables.counts(tables.ids(chain[:depth]), y, train, K,
                                    query=query)
        p_oof = (c_oof + alpha * p_oof) / (c_oof.sum(1, keepdims=True) + alpha)
        p_all = (c_all + alpha * p_all) / (c_all.sum(1, keepdims=True) + alpha)
    return p_oof, p_all

def _safe_table_config(X, rows, n_given):
    """Choose a reusable chain and shrinkage using only visible GIVEN values."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows) or not n_given:
        return (), 10.0
    entropy = np.zeros(n_given)
    for feature in range(n_given):
        counts = np.bincount(X[rows, feature].astype(np.int64))
        p = counts[counts > 0] / len(rows)
        entropy[feature] = -np.sum(p * np.log(p))
    order = tuple(int(f) for f in np.argsort(-entropy, kind='stable')[:3]
                  if entropy[f] > 0)
    alpha = float(np.clip(np.sqrt(len(rows)) / 10., 1., 30.))
    return order, alpha

def _mi_drivers(X, y, rows, levels, count=3):
    """Rank GIVEN columns using labels from this fit partition only."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return ()
    target = y[rows].astype(np.int64)
    if np.any(target < 0):
        raise ValueError('MI fit rows contain an unknown target')
    classes = int(target.max()) + 1
    scores = np.zeros(X.shape[1], float)
    for f, width in enumerate(levels):
        contingency = np.bincount(X[rows, f] * classes + target,
                                  minlength=int(width) * classes).reshape(int(width), classes)
        joint = contingency / len(rows)
        px = joint.sum(1, keepdims=True)
        py = joint.sum(0, keepdims=True)
        valid = joint > 0
        scores[f] = np.sum(joint[valid] * np.log(joint[valid] /
                         (px * py)[valid]))
    return tuple(int(f) for f in np.argsort(-scores, kind='stable')[:count])

def _fit_table(tables, y, train, K, n_given, check_clock=None,
               fixed_config=None, oof_query=None, return_config=False):
    """Pick the item's feature chain and shrinkage on out-of-fold loss."""
    if check_clock is not None:check_clock()
    yt, ft, F = y[train], tables.fold[train], tables.F
    n = train.sum()
    total = np.bincount(yt, minlength=K).astype(float)
    per = np.stack([np.bincount(yt[ft == f], minlength=K) for f in range(F)])
    if fixed_config is not None:
        query = np.flatnonzero(train) if oof_query is None else np.asarray(oof_query, dtype=np.int64)
        qfold = tables.fold[query]
        if np.any((qfold < 0) | (qfold >= F)):
            raise ValueError('fold-excluded query needs visible folds')
        prior_oof = (total - per[qfold] + 1.0 / K) / (n - per[qfold].sum(1, keepdims=True) + 1.0)
        prior_all = np.broadcast_to((total + 1.0 / K) / (n + 1.0), (len(y), K))
        order, alpha = fixed_config
        p_oof, p_all = _table_chain(tables, order, y, train, K, alpha,
                                    prior_oof, prior_all, query=query)
        if check_clock is not None:check_clock()
        result = (p_oof, np.asarray(p_all), np.zeros(n_given))
        return (*result, fixed_config) if return_config else result
    if oof_query is not None:
        raise ValueError('fold-excluded query needs a safe fixed configuration')
    prior_oof = (total - per[ft] + 1.0 / K) / (n - per[ft].sum(1, keepdims=True) + 1.0)
    prior_all = np.broadcast_to((total + 1.0 / K) / (n + 1.0), (len(y), K))
    base = _nll(prior_oof, yt, K)

    gains = np.zeros(n_given)
    for f in range(n_given):
        if check_clock is not None:check_clock()
        p_oof, _ = _table_chain(tables, (f,), y, train, K, 4.0, prior_oof, prior_all)
        gains[f] = base - _nll(p_oof, yt, K)
    order = [int(f) for f in np.argsort(-gains) if gains[f] > 0][:3]

    best = (base, prior_oof, prior_all)
    best_config = ((), 10.0)
    if order:
        for alpha in (1.0, 3.0, 10.0, 30.0):
            if check_clock is not None:check_clock()
            p_oof, p_all = _table_chain(tables, tuple(order), y, train, K, alpha,
                                        prior_oof, prior_all)
            loss = _nll(p_oof, yt, K)
            if loss < best[0]:
                best = (loss, p_oof, p_all)
                best_config = (tuple(order), alpha)
    if check_clock is not None:check_clock()
    result = (best[1], np.asarray(best[2]), gains)
    return (*result, best_config) if return_config else result

def _design(X, levels, drivers, max_columns=600, support_rows=None,
            retain_crosses=False, given=None):
    """Sparse one-hot of every GIVEN plus drivers x every other GIVEN."""
    from scipy import sparse
    n, G = X.shape
    blocks, width = [], 0
    for f in range(G):
        blocks.append(X[:, f] + width)
        width += levels[f]
    single_width = width
    crossed = set()
    for a in drivers:
        for b in range(G):
            if b == a:
                continue
            pair = tuple(sorted((int(a), b)))
            if pair in crossed:
                continue
            crossed.add(pair)
            blocks.append(X[:, a] * levels[b] + X[:, b] + width)
            width += levels[a] * levels[b]
    cols = np.stack(blocks, 1)
    rows = np.repeat(np.arange(n), cols.shape[1])
    data = np.ones(rows.size)
    M = sparse.csr_matrix((data, (rows, cols.reshape(-1))), shape=(n, width))
    if retain_crosses:
        return M
    support = np.asarray((M if support_rows is None else M[support_rows]).sum(0)).reshape(-1)
    interactions = np.flatnonzero((np.arange(width) >= single_width) & (support >= 5))
    budget = max(0, max_columns - single_width)
    ranked = interactions[np.argsort(-support[interactions], kind='stable')[:budget]]
    keep = np.concatenate((np.arange(single_width), np.sort(ranked)))
    return M[:, keep].tocsr()

def _logit(D, y, rows, K, C):
    from sklearn.linear_model import LogisticRegression
    classes = np.unique(y[rows])
    if classes.size < 2:
        return None, classes
    model = LogisticRegression(C=C, max_iter=60, tol=1e-3)
    model.fit(D[rows], y[rows])
    return model, classes

def _logit_proba(model, classes, D, rows, K):
    out = np.zeros((rows.size, K))
    if model is None:
        out[:, classes[0]] = 1.0
    else:
        out[:, classes] = model.predict_proba(D[rows])
    return out

class _Heads:
    """Column layout of every item's softmax inside one shared output layer.
    Items with the same option count share a block stored option-major:
    option k of the block's J items fills J adjacent columns, so every
    softmax pass reads contiguous row segments."""

    def __init__(self, Ks):
        self.Ks = np.asarray(Ks)
        self.start = np.zeros(len(Ks), np.int64)
        self.stride = np.zeros(len(Ks), np.int64)
        self.blocks = []
        at = 0
        for K in np.unique(self.Ks):
            js = np.flatnonzero(self.Ks == K)
            self.start[js] = at + np.arange(js.size)
            self.stride[js] = js.size
            self.blocks.append((at, at + K * js.size, js.size, int(K)))
            at += K * js.size
        self.width = at

    def col(self, j, k):
        """Column of item j's option(s) k."""
        return self.start[j] + np.asarray(k) * self.stride[j]

    def cols(self, j):
        return self.col(j, np.arange(self.Ks[j]))

    def softmax(self, Z):
        # Reductions over a 2-6 wide option axis pay numpy's per-row
        # overhead; K passes over contiguous (n, J) option planes vectorize.
        for a, b, J, K in self.blocks:
            planes = [Z[:, a + k * J:a + (k + 1) * J] for k in range(K)]
            m = planes[0].copy()
            for p in planes[1:]:
                np.maximum(m, p, out=m)
            s = np.zeros_like(m)
            for p in planes:
                p -= m
                np.maximum(p, -80.0, out=p)
                np.exp(p, out=p)
                s += p
            np.reciprocal(s, out=s)
            for p in planes:
                p *= s
        return Z

def _carry_checkpoints(oof, out, tails, F):
    """Folds that stopped early repeat their last checkpoint in the later
    marks, so one slow fold no longer drops every fold to its depth."""
    usable = max((t[3] for t in tails), default=0)
    for val_idx, rest_idx, rest_pe, c in tails:
        for m in range(c, usable):
            oof[m][val_idx] = oof[c - 1][val_idx]
            out[m][rest_idx] += rest_pe / F
    return usable

def _finish_net(oof, out, heads, Ys, trains, Ks, usable):
    if usable == 0:
        return None
    n = oof.shape[1]
    results = []
    for j, (y, tr) in enumerate(zip(Ys, trains)):
        cols = heads.cols(j)
        rows = np.flatnonzero(tr)
        other = np.flatnonzero(~tr)
        losses = [_nll(oof[m][np.ix_(rows, cols)].astype(float), y[rows], heads.Ks[j])
                  for m in range(usable)]
        best = int(np.argmin(losses))
        full = np.zeros((n, heads.Ks[j]))
        full[other] = _normalize(out[best][np.ix_(other, cols)].astype(float))
        results.append((oof[best][np.ix_(rows, cols)].astype(float), full))
    return results

def _fit_net(A, Ys, trains, Ks, fold, steps, deadline,
             hidden=64, rate=0.01, decay=1e-4):
    """One-hidden-layer network shared by every item; each item keeps the
    checkpoint with its lowest out-of-fold loss. Returns per item the oof
    probabilities on its train rows and the fold-average on its other rows."""
    global _NET_TRACE
    _NET_TRACE=[]
    heads = _Heads(Ks)
    n, d = A.shape
    J = len(Ys)
    # Short CPU runs keep early checkpoints a few steps apart.
    marks = (sorted({s for s in (2, 4, 8, 12, 24, 36, 48, 64, 96, 128, steps)
                     if s <= steps}) if steps >= 48 else
             sorted({max(1, int(steps * portion)) for portion in
                     (0.03, 0.06, 0.125, 0.25, 0.5, 0.75, 1.0)}))
    # _finish_net scores every checkpoint of every item (1.8 s for 7 x 213 on
    # a 4-core CPU); the loop stops early enough for it to end by deadline.
    # UNICEF's 12 heads finish in milliseconds and keep the v9 timing.
    reserve = 0.0006 * J * len(marks) if J > 50 else 0.0
    eff_deadline = deadline - reserve
    step_guard = deadline - max(0.35, reserve)
    target = np.zeros((n, heads.width), np.float32)
    weight = np.zeros((n, J), np.float32)
    for j, (y, tr) in enumerate(zip(Ys, trains)):
        target[np.flatnonzero(tr), heads.col(j, y[tr])] = 1.0
        weight[tr, j] = 1.0
    col_item = np.zeros(heads.width, np.int64)
    # Initial output weights are drawn in the item-major column order and
    # moved to the option-major layout, so every option starts from its v9 value.
    item_major = np.zeros(heads.width, np.int64)
    at = 0
    for j in range(J):
        col_item[heads.cols(j)] = j
        item_major[heads.cols(j)] = at + np.arange(heads.Ks[j])
        at += heads.Ks[j]
    oof = np.zeros((len(marks), n, heads.width), np.float32)
    out = np.zeros((len(marks), n, heads.width), np.float32)
    F = int(fold.max()) + 1
    labelled = weight.max(1) > 0
    completed = []
    tails = []
    # Sparse one-hot inputs retain the same affine maps while avoiding dense
    # multiplication by zeros in the small questionnaire's CPU network.
    sparse_A = None
    if d >= 64 and np.count_nonzero(A) < .15 * A.size:
        from scipy.sparse import csr_matrix
        sparse_A = csr_matrix(A)

    for f in range(F):
        now = time.monotonic()
        if now >= eff_deadline:
            return None
        fold_stop = now + max(0.0, ((eff_deadline - now) / (F - f) - 0.5)
                             if J == 12 else (eff_deadline - now - 0.15) / (F - f))
        fit = labelled & (fold != f)
        val = labelled & (fold == f)
        other = ~labelled
        rng = np.random.default_rng(_SEED + 7 * f)
        ones = max(float(A[:100].sum(1).mean()), 1.0)
        W1 = (rng.standard_normal((d, hidden)) * np.sqrt(2.0 / ones)).astype(np.float32)
        b1 = np.zeros(hidden, np.float32)
        W2 = (rng.standard_normal((hidden, heads.width)) * 0.01).astype(np.float32)[:, item_major]
        freq = (target[fit].sum(0) + 0.5) / (weight[fit][:, col_item].sum(0) + 0.5 * heads.Ks[col_item])
        b2 = np.log(freq).astype(np.float32)
        params = [W1, b1, W2, b2]
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        Af, Tf, Wf = (A if sparse_A is None else sparse_A)[fit], target[fit], weight[fit][:, col_item]
        dense_weight = bool(np.all(Wf == 1.0))
        nf = float(fit.sum())
        Aeval = (A if sparse_A is None else sparse_A)[val | other]
        mark = 0
        for step in range(1, steps + 1):
            if (time.monotonic() >= fold_stop and mark > 0
                    and (J == 12 or step < marks[mark] - 2 or time.monotonic() >= step_guard)):
                break
            H = Af @ W1 + b1
            np.maximum(H, 0, out=H)
            P = heads.softmax(H @ W2 + b2)
            if dense_weight:
                P -= Tf
                P *= 1.0 / nf
                dZ = P
            else:
                dZ = (P * Wf - Tf) / nf
            gW2 = H.T @ dZ + decay * W2
            gb2 = dZ.sum(0)
            dH = dZ @ W2.T
            dH[H <= 0] = 0
            gW1 = Af.T @ dH + decay * W1
            gb1 = dH.sum(0)
            step_rate = rate * (0.5 * (1.0 + np.cos(np.pi * (step - 48) /
                         max(1, steps - 48))) if J > 50 and step > 48 else 1.0)
            for i, (p, g) in enumerate(zip(params, (gW1, gb1, gW2, gb2))):
                m1[i] = 0.9 * m1[i] + 0.1 * g
                m2[i] = 0.999 * m2[i] + 0.001 * g * g
                p -= step_rate * (m1[i] / (1 - 0.9 ** step)) / (
                    np.sqrt(m2[i] / (1 - 0.999 ** step)) + 1e-8)
            if step == marks[mark]:
                He = np.maximum(Aeval @ W1 + b1, 0)
                Pe = heads.softmax(He @ W2 + b2)
                idx = np.flatnonzero(val | other)
                is_val = val[idx]
                oof[mark][idx[is_val]] = Pe[is_val]
                out[mark][idx[~is_val]] += Pe[~is_val] / F
                last = (idx[is_val], idx[~is_val], Pe[~is_val].copy())
                mark += 1
                if time.monotonic() >= fold_stop:
                    break
        completed.append(mark)
        tails.append(last + (mark,))
        _NET_TRACE.append({"fold":f,"steps":step,"checkpoints":mark,"marks":marks[:mark]})
    return _finish_net(oof, out, heads, Ys, trains, Ks,
                       _carry_checkpoints(oof, out, tails, F))

def _pair_crosses(A, width, count=256):
    """Products of the ``width`` most frequent binary design columns, capped."""
    freq = A.mean(0)
    top = np.argsort(-np.minimum(freq, 1.0 - freq))[:width]
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    if not pairs:
        return np.zeros((len(A), 0), np.float32)
    left = np.array([a for a, _ in pairs])
    right = np.array([b for _, b in pairs])
    support = (A[:, left] * A[:, right]).mean(0)
    keep = np.argsort(-np.minimum(support, 1.0 - support))[:count]
    keep = keep[support[keep] > 0.002]
    return (A[:, left[keep]] * A[:, right[keep]]).astype(np.float32)


def _fit_net_gpu(A, Ys, trains, Ks, fold, steps, deadline,
                 hidden=256, rate=0.003, decay=1e-4, device="cuda",
                 categories=None,category_levels=None):
    """Cross-fit residual tabular network shared by every item's head.

    Input is the GIVEN one-hot design plus capped second-order crosses; two
    256-wide residual blocks (LayerNorm, ReLU, Dropout 0.15) with a cosine
    learning rate; each item keeps its lowest out-of-fold checkpoint.
    """
    import torch

    torch.set_float32_matmul_precision("high")
    device = torch.device(device)
    hidden = max(int(hidden), 256)
    A = np.concatenate((A, _pair_crosses(A, 40)), 1).astype(np.float32)
    if categories is None:
        categories=np.zeros((len(A),1),np.int64);category_levels=np.array([1])
    category_levels=np.asarray(category_levels,dtype=np.int64)
    offsets=np.r_[0,np.cumsum(category_levels)[:-1]]
    tokens=np.minimum(np.maximum(categories,0),category_levels-1)+offsets
    tokens=torch.as_tensor(tokens,dtype=torch.long,device=device)
    global _NET_TRACE
    _NET_TRACE=[]
    heads = _Heads(Ks)
    n, d = A.shape
    J = len(Ys)
    marks = sorted({max(1, int(steps * portion)) for portion in
                    (0.03, 0.06, 0.125, 0.25, 0.5, 0.75, 1.0)})
    target = np.zeros((n, heads.width), np.float32)
    weight = np.zeros((n, J), np.float32)
    for j, (y, tr) in enumerate(zip(Ys, trains)):
        target[np.flatnonzero(tr), heads.col(j, y[tr])] = 1.0
        weight[tr, j] = 1.0
    col_item = np.zeros(heads.width, np.int64)
    for j in range(J):
        col_item[heads.cols(j)] = j
    oof = np.zeros((len(marks), n, heads.width), np.float32)
    out = np.zeros((len(marks), n, heads.width), np.float32)
    F = int(fold.max()) + 1
    labelled = weight.max(1) > 0
    completed = []
    tails = []

    def probabilities(logits):
        blocks = []
        for a, b, count, K in heads.blocks:
            block = logits[:, a:b].reshape(-1, K, count)
            blocks.append(torch.softmax(block, dim=1).reshape(-1, b - a))
        return torch.cat(blocks, dim=1)

    for f in range(F):
        now = time.monotonic()
        if now >= deadline:
            return None
        fold_stop = now + max(0.0, (deadline - now) / (F - f) - 0.5)
        fit = labelled & (fold != f)
        val = labelled & (fold == f)
        other = ~labelled
        if not fit.any() or not val.any():
            return None
        torch.manual_seed(_SEED + 7 * f)
        # The temperature set is held out of gradient training, inside the
        # outer fit partition. Outer validation labels never tune temperature.
        eligible=np.flatnonzero(fit)
        rng=np.random.default_rng(_SEED+31*f)
        calib=np.zeros(n,bool)
        if len(eligible)>=50:calib[rng.choice(eligible,max(8,len(eligible)//10),replace=False)]=True
        fit=fit&~calib
        freq = (target[fit].sum(0) + 0.5) / (
            weight[fit][:, col_item].sum(0) + 0.5 * heads.Ks[col_item])
        nn = torch.nn
        model = nn.ModuleDict({
            "inp": nn.Linear(d, hidden),
            "embed":nn.Embedding(int(category_levels.sum()),12),
            "embed_proj":nn.Linear(len(category_levels)*12,hidden),
            "n1": nn.LayerNorm(hidden), "l1": nn.Linear(hidden, hidden),
            "n2": nn.LayerNorm(hidden), "l2": nn.Linear(hidden, hidden),
            "out_norm": nn.LayerNorm(hidden), "out": nn.Linear(hidden, heads.width),
            "drop": nn.Dropout(0.15),
        }).to(device)
        with torch.no_grad():
            model["out"].weight.mul_(0.1)
            model["out"].bias.copy_(torch.as_tensor(np.log(freq).astype(np.float32)))
        drop = model["drop"]

        def forward(x,cat):
            h = model["inp"](x)+model['embed_proj'](model['embed'](cat).flatten(1))
            h = h + drop(torch.nn.functional.silu(model["l1"](model["n1"](h))))
            h = h + drop(torch.nn.functional.silu(model["l2"](model["n2"](h))))
            return model["out"](torch.nn.functional.silu(model["out_norm"](h)))

        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        Af = torch.as_tensor(A[fit], device=device)
        Tf = torch.as_tensor(target[fit], device=device)
        Aeval = torch.as_tensor(A[val | other], device=device)
        Acal=torch.as_tensor(A[calib],device=device)
        Tcal=torch.as_tensor(target[calib],device=device)
        nf = float(fit.sum())
        idx = np.flatnonzero(val | other)
        is_val = val[idx]
        mark = 0
        for step in range(1, steps + 1):
            if time.monotonic() >= fold_stop and mark > 0:break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            P = probabilities(forward(Af,tokens[fit]))
            loss = -(Tf * torch.log(P.clamp_min(1e-9))).sum() / nf
            loss.backward()
            optimizer.step()
            schedule.step()
            if step == marks[mark]:
                model.eval()
                with torch.no_grad():
                    temperature=1.
                    if calib.any():
                        logits=forward(Acal,tokens[calib])
                        candidates=(.7,.85,1.,1.2,1.5,2.)
                        losses=[float(-(Tcal*probabilities(logits/t).clamp_min(1e-9).log()).sum()) for t in candidates]
                        temperature=candidates[int(np.argmin(losses))]
                    Pe = probabilities(forward(Aeval,tokens[val|other])/temperature).cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                oof[mark][idx[is_val]] = Pe[is_val]
                out[mark][idx[~is_val]] += Pe[~is_val] / F
                last = (idx[is_val], idx[~is_val], Pe[~is_val].copy())
                mark += 1
                if time.monotonic() >= fold_stop:
                    break
        completed.append(mark)
        tails.append(last + (mark,))
        _NET_TRACE.append({"fold":f,"steps":step,"checkpoints":mark,
                           "temperature":temperature,"calibration_rows":int(calib.sum()),
                           "embedding_columns":len(category_levels)})
    return _finish_net(oof, out, heads, Ys, trains, Ks,
                       _carry_checkpoints(oof, out, tails, F))

def _rb_distill(D, work, names, fold, deadline, rank=24, ridge=10.0, mix=0.5):
    """Rao-Blackwellized whole-survey soft targets distilled onto GIVEN design.

    Inside each outer training partition only: the one-hot answers of every
    item form H; its top ``rank`` principal directions reconstruct each item's
    block from the *other* items' answers (the item's own block is removed
    from the scores).  That leave-own-block-out reconstruction, mixed with the
    item's one-hot, is a low-variance soft target.  One ridge solve maps the
    design to every item's log soft target at once; validation and hidden
    rows only see the design times those weights.
    """
    names = [k for k in names if k in work]
    if not names:
        return {}
    widths = np.array([int(work[k]['K']) for k in names])
    starts = np.concatenate(([0], np.cumsum(widths)[:-1]))
    W = int(widths.sum())
    n = D.shape[0]
    labelled = np.ones(n, bool)
    for k in names:
        labelled &= work[k]['train']
    names = [k for k in names if np.array_equal(work[k]['train'], labelled)]
    if not names:
        return {}
    widths = np.array([int(work[k]['K']) for k in names])
    starts = np.concatenate(([0], np.cumsum(widths)[:-1]))
    W = int(widths.sum())
    rows = np.flatnonzero(labelled)
    if len(rows) < 50:
        return {}
    Y = np.stack([work[k]['y'] for k in names], 1)
    folds = fold[rows]
    F = int(folds.max()) + 1
    rest = np.flatnonzero(~labelled)
    oof = np.zeros((len(rows), W), np.float32)
    hidden = np.zeros((len(rest), W), np.float32)
    D = D.tocsr()
    for f in range(F):
        if time.monotonic() >= deadline:
            return {}
        tr = rows[folds != f]
        va = rows[folds == f]
        if not len(tr) or not len(va):
            return {}
        H = np.zeros((len(tr), W), np.float32)
        for j, k in enumerate(names):
            H[np.arange(len(tr)), starts[j] + Y[tr, j]] = 1.0
        mu = H.mean(0)
        Hc = H - mu
        _, vecs = np.linalg.eigh((Hc.T @ Hc).astype(np.float64))
        V = vecs[:, -min(rank, W - 1):].astype(np.float32)
        Z = Hc @ V
        T = np.empty_like(H)
        for j in range(len(names)):
            a, b = starts[j], starts[j] + widths[j]
            z = Z - Hc[:, a:b] @ V[a:b]
            q = np.clip(z @ V[a:b].T + mu[a:b], 0.0, 1.0) + 1e-6
            q /= q.sum(1, keepdims=True)
            T[:, a:b] = np.log(mix * H[:, a:b] + (1.0 - mix) * q + 1e-3)
        Dt = D[tr]
        G = (Dt.T @ Dt).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += ridge
        weights = np.linalg.solve(G, np.asarray(Dt.T @ T, dtype=np.float64)).astype(np.float32)
        oof[folds == f] = np.asarray(D[va] @ weights)
        if len(rest):
            hidden += np.asarray(D[rest] @ weights) / F
    out = {}
    for j, k in enumerate(names):
        a, b = starts[j], starts[j] + widths[j]
        o = _softmax(oof[:, a:b].astype(float))
        full = np.zeros((n, widths[j]))
        if len(rest):
            full[rest] = _softmax(hidden[:, a:b].astype(float))
        out[k] = (o, full)
    return out


def _stack(oofs, rests, y, K, deadline=float('inf')):
    """Mixture weights then a temperature, fitted on out-of-fold loss."""
    from scipy.optimize import minimize
    M = len(oofs)
    c = 1.0 - K * _FLOOR
    picked = np.stack([o[np.arange(len(y)), y] for o in oofs], 1)
    if M == 1:
        w = np.ones(1)
    else:
        components = _FLOOR + c * picked.astype(np.float64)
        def objective(weights):
            q = components @ weights
            return -np.log(q).mean(), -(components.T @ (1.0 / q)) / len(y)
        def kkt_gap(weights):
            gain = (components.T @ (1.0 / (components @ weights))) / len(y)
            return float(gain.max() - weights @ gain)
        def em(weights, steps):
            weights = weights.copy()
            for _ in range(steps):
                if time.monotonic() >= deadline:break
                gain = (components.T @ (1.0 / (components @ weights))) / len(y)
                weights *= gain
                weights /= weights.sum()
            return weights
        if time.monotonic() >= deadline:
            w_em = np.zeros(M)
            w_em[int(np.argmin(-np.log(components).mean(0)))] = 1.
        else:
            w_em = em(np.full(M, 1.0 / M), 200)
        w = w_em
        if time.monotonic() < deadline:
            from .chain_features import BudgetExpired,check_clock
            def timed_objective(weights):
                check_clock(deadline)
                return objective(weights)
            try:
                result = minimize(timed_objective, w_em, method='SLSQP', jac=True,
                                  bounds=[(0.0, 1.0)] * M,
                                  constraints={'type': 'eq', 'fun': lambda weights: weights.sum() - 1.0,
                                               'jac': lambda weights: np.ones(M)},
                                  options={'maxiter': 200, 'ftol': 1e-9})
            except BudgetExpired:
                pass  # The completed EM iterate is already a valid simplex.
            else:
                if np.isfinite(result.x).all() and np.maximum(result.x, 0.).sum() > 0:
                    candidate = np.maximum(result.x, 0.0)
                    candidate /= candidate.sum()
                    if objective(candidate)[0] < objective(w)[0]:w = candidate
        if time.monotonic() < deadline and kkt_gap(w) > 1e-7:
            candidate = em(np.maximum(w, 1e-12) / np.maximum(w, 1e-12).sum(), 500)
            if objective(candidate)[0] < objective(w)[0]:w = candidate
        # Nearly identical arms make EM arbitrarily slow. Exact pairwise
        # transfers close the remaining simplex KKT gap without a line-search
        # status deciding whether an otherwise valid prediction can be used.
        q = np.clip(components @ w, 1e-12, 1.0)
        for _ in range(32):
            if time.monotonic() >= deadline:break
            gain = (components.T @ (1.0 / q)) / len(y)
            if gain.max() - w @ gain <= 1e-7:break
            receiver = int(np.argmax(gain))
            donors = np.flatnonzero(w > 0.)
            donor = int(donors[np.argmin(gain[donors])])
            if donor == receiver or w[donor] <= 1e-12:break
            delta = components[:,receiver] - components[:,donor]
            lo, hi = 0., w[donor]
            if np.mean(delta / (q + hi * delta)) >= 0.:
                amount = hi
            else:
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if np.mean(delta / (q + mid * delta)) > 0.:lo = mid
                    else:hi = mid
                amount = (lo + hi) / 2
            if amount <= 1e-12:break
            w[donor] = max(0.0, w[donor] - amount)
            w[receiver] += amount
            s = w.sum()
            w = w / s if s > 0 else np.full(M, 1.0 / M)
            q = np.clip(components @ w, 1e-12, 1.0)
    mix_oof = sum(wi * o for wi, o in zip(w, oofs))
    mix_rest = sum(wi * r for wi, r in zip(w, rests))
    best_t, best_loss = 1.0, _nll(mix_oof, y, K)
    log_oof = np.log(np.clip(mix_oof, 1e-12, None))
    for t in (0.8, 0.9, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0):
        if time.monotonic() >= deadline:break
        loss = _nll(_softmax(log_oof * t), y, K)
        if loss < best_loss:
            best_t, best_loss = t, loss
    if best_t != 1.0:
        mix_rest = _softmax(np.log(np.clip(mix_rest, 1e-12, None)) * best_t)
        mix_oof = _softmax(log_oof * best_t)
    return mix_rest, mix_oof, w, best_t, best_loss

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


_CROSS_TRACE={}

def _cross_target(X, index, work, deadline, n_raw_given=None):
    """Nested cross-fit of residual ridge over every target's OOF table probabilities."""
    global _CROSS_TRACE
    _CROSS_TRACE={"completed":0,"admitted":0}
    visible = index.visible
    if len(visible) < 120 or not work or time.monotonic() >= deadline:
        return None
    if n_raw_given is None:n_raw_given=X.shape[1]
    names = list(work)
    if any(not np.all(work[name]["train"][visible]) for name in names):
        return None
    n = len(X)
    width = sum(work[name]["K"] for name in names)
    blocks, offset = {}, 0
    for name in names:
        K = work[name]["K"]
        blocks[name] = (offset, offset + K)
        offset += K
    labels = np.zeros((len(visible), width), np.float32)
    for name in names:
        a, _ = blocks[name]
        labels[np.arange(len(visible)), a + work[name]["y"][visible]] = 1.0
    position = np.full(n, -1, np.int32)
    position[visible] = np.arange(len(visible))
    outer = index.fold[visible]
    meta_oof = np.zeros((len(visible), width), np.float32)

    def correction(features, desired):
        gram = features.T @ features
        gram.flat[::width + 1] += max(12.0, 0.01 * len(features))
        rhs = features.T @ (desired - features)
        coef = np.linalg.solve(gram, rhs)
        # The correction uses other questions; the target's own first-stage
        # distribution stays in the uncorrected probability block.
        for a, b in blocks.values():
            coef[a:b, a:b] = 0.0
        return coef

    def normalize_blocks(matrix, reserve):
        for a, b in blocks.values():
            part = np.clip(matrix[:, a:b], 0.0, None)
            sums = part.sum(1, keepdims=True)
            bad = sums[:, 0] <= 0
            sums[bad] = 1.0
            part /= sums
            part[bad] = reserve[bad, a:b]
            matrix[:, a:b] = part
        return matrix

    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        train_rows = visible[outer != f]
        val_rows = visible[outer == f]
        if len(train_rows) < 80 or not len(val_rows):
            return None
        train = np.zeros(n, bool)
        train[train_rows] = True
        inner = np.zeros(n, np.int64)
        inner[train_rows] = index.fold[train_rows] % 2
        tables = _Tables(X, inner, 2)
        fit_features = np.zeros((len(train_rows), width), np.float32)
        val_features = np.zeros((len(val_rows), width), np.float32)
        for name in names:
            if time.monotonic() >= deadline:
                return None
            job = work[name]
            a, b = blocks[name]
            p_oof, p_all, _ = _fit_table(tables, job["y"], train,
                                         job["K"], n_raw_given, fixed_config=(job.get("table_config") if len(names)>100 else None))
            fit_features[:, a:b] = p_oof
            val_features[:, a:b] = p_all[val_rows]
        coef = correction(fit_features, labels[position[train_rows]])
        meta_oof[position[val_rows]] = normalize_blocks(
            val_features + val_features @ coef, val_features)

    if time.monotonic() >= deadline:
        return None
    fit_features = np.zeros((len(visible), width), np.float32)
    rest = np.flatnonzero(position < 0)
    query_features = np.zeros((len(rest), width), np.float32)
    for name in names:
        a, b = blocks[name]
        fit_features[:, a:b] = work[name]["oof"][0]
        query_features[:, a:b] = work[name]["all"][0][rest]
    coef = correction(fit_features, labels)
    meta_query = normalize_blocks(query_features + query_features @ coef,
                                  query_features)
    _CROSS_TRACE["completed"]=len(names)
    results = {}
    for name in names:
        a, b = blocks[name]
        rows = np.flatnonzero(work[name]["train"])
        base = work[name]["oof"][0]
        candidate = meta_oof[position[rows], a:b]
        y = work[name]["y"][rows]
        K = b - a
        gain = _nll(base, y, K) - _nll(candidate, y, K)
        half = index.fold[rows] % 2 == 0
        if (gain <= (0.0002 if len(names)>100 else 0.0005) or not half.any() or half.all() or
            _nll(candidate[half], y[half], K) >= _nll(base[half], y[half], K) or
            _nll(candidate[~half], y[~half], K) >= _nll(base[~half], y[~half], K)):
            continue
        full = np.zeros((n, K))
        full[rest] = meta_query[:, a:b]
        results[name] = (candidate, full, gain)
    _CROSS_TRACE["admitted"]=len(results)
    return results


def _legacy_stacked_cross_target(work, fold, deadline=float('inf'), workers=1):
    """Cross-fit logit residuals from fold-selected peer probability blocks.

    Stage-one OOF predictions are reused. Internal selection scores are tested
    separately on the masked cohort. Each outer fold selects its experts and
    probability-weighted contingency MI using only its fitting rows.

    Visible rows are sorted by fold, so a fold's fitting rows are the total
    minus one contiguous segment: the peer Gram, the MI contingency table and
    every target's raw cross products are summed once per fold segment and
    combined by subtraction. Columns whose selected experts differ in a fold
    are recomputed on that fold's fitting rows only. Standardization and the
    centered logit and residual blocks are applied algebraically afterwards,
    so each fold solves the same ridge system as an explicit construction.
    Every peer block is a probability simplex, so the Gram has an exact null
    space; it is accumulated in float64 so G+alpha*I stays positive definite.
    """
    with threadpool_limits(limits=1):
        from collections import Counter
        from scipy.linalg.lapack import dtrtri
        started=time.monotonic()
        hard_stop=min(deadline,started+7.5)
        trace={'eligible':0,'completed':0,'admitted':0,'alphas':{},
               'selection_oof_role':'model_selection','folds':5,
               'patched_columns':0,'patched_best':0}
        if not work:return {},trace
        names=[name for name,j in work.items() if j['train'].sum()>=60]
        trace['eligible']=len(names)
        if not names:return {},trace
        n=len(next(iter(work.values()))['y'])
        common=np.logical_and.reduce([work[name]['train'] for name in names])
        rows=np.flatnonzero(common);query=np.flatnonzero(~common)
        if len(rows)<60 or any(not np.array_equal(work[name]['train'],common) for name in names):
            trace['reason']='joint_visible_rows_required';return {},trace
        split=fold[rows];blocks={};offset=0;starts=[]
        for name in names:
            starts.append(offset);blocks[name]=(offset,offset+work[name]['K']);offset+=work[name]['K']
        block_cols=[np.arange(*blocks[name]) for name in names]
        completed={name:0 for name in names}
        trace['folds_completed']=completed
        n_v=len(rows);n_q=len(query)
        order=np.argsort(split,kind='stable')
        bounds=np.searchsorted(split[order],np.arange(6))
        if np.any(np.diff(bounds)==0) or np.any(np.diff(bounds)==n_v):
            trace['preparation_seconds']=time.monotonic()-started
            return {},trace
        segments=[slice(int(bounds[f]),int(bounds[f+1])) for f in range(5)]
        onehot=np.zeros((n_v,5));onehot[np.arange(n_v),split]=1.
        # Sorted visible rows: Ys one-hot targets, Xs shifted peer probabilities.
        Ys=np.zeros((n_v,offset),order="F");chosen=[]
        for name in names:
            job=work[name];a,b=blocks[name];K=job['K'];y=job['y'][rows]
            Ys[np.arange(n_v),a+y[order]]=1.
            flat=np.arange(n_v)*K+y
            picked=np.stack([np.ascontiguousarray(o).reshape(-1)[flat] for o in job['oof']])
            sums=-np.log(_FLOOR+(1.0-K*_FLOOR)*picked)@onehot
            chosen.append(np.argsort(sums.sum(1)[:,None]-sums,axis=0,kind='stable')[:3].T)
        sets=[[tuple(sorted(int(k) for k in c[f])) for f in range(5)] for c in chosen]
        bests=[[int(c[f][0]) for f in range(5)] for c in chosen]
        modal=[Counter(s).most_common(1)[0][0] for s in sets]
        modal_best=[Counter(s).most_common(1)[0][0] for s in bests]
        averaged={}
        def average(i,arms):
            if (i,arms) not in averaged:
                job=work[names[i]]
                averaged[(i,arms)]=((sum(job['oof'][k] for k in arms)/len(arms))[order].astype(np.float64),
                                    (sum(job['all'][k][query] for k in arms)/len(arms)).astype(np.float64))
            return averaged[(i,arms)]
        Xs=np.empty((n_v,offset),order='F');Xq=np.empty((n_q,offset),order='F')
        for i,name in enumerate(names):
            a,b=blocks[name];Xs[:,a:b],Xq[:,a:b]=average(i,modal[i])
        shift=Xs.mean(0)
        Xs-=shift;Xq-=shift
        patches=[]
        for f in range(5):
            cols=[];zv=[];zq=[]
            for i,name in enumerate(names):
                if sets[i][f]==modal[i]:continue
                o,p=average(i,sets[i][f]);a,b=blocks[name]
                cols.append(block_cols[i]);zv.append(o-shift[a:b]);zq.append(p-shift[a:b])
            if cols:
                patches.append((np.concatenate(cols),np.asfortranarray(np.column_stack(zv)),
                                np.asfortranarray(np.column_stack(zq))))
                trace['patched_columns']+=len(patches[-1][0])
            else:patches.append(None)
        def outside(A,f):
            # The two contiguous row ranges that make up fold f's fitting rows.
            s=segments[f];return A[:s.start],A[s.stop:]
        grams=[];col_sums=[];joints_raw=[];label_sums=[]
        for s in segments:
            if time.monotonic()>=hard_stop:break
            X=Xs[s];grams.append(X.T@X);col_sums.append(X.sum(0))
            joints_raw.append(X.T@Ys[s]);label_sums.append(Ys[s].sum(0))
        fold_states=[]
        if len(grams)==5:
            total_G=sum(grams);total_S=sum(col_sums);total_J=sum(joints_raw);total_Y=sum(label_sums)
            for f in range(5):
                if time.monotonic()>=hard_stop:break
                n_tr=n_v-(segments[f].stop-segments[f].start)
                G=total_G-grams[f];S=total_S-col_sums[f];J=total_J-joints_raw[f];ny=total_Y-label_sums[f]
                patch=patches[f];where=None
                if patch is not None:
                    C,Zv,Zq=patch
                    (z0,z1),(x0,x1),(y0,y1)=outside(Zv,f),outside(Xs,f),outside(Ys,f)
                    Q=z0.T@x0+z1.T@x1
                    G[C,:]=Q;G[:,C]=Q.T;G[np.ix_(C,C)]=z0.T@z0+z1.T@z1
                    S[C]=z0.sum(0)+z1.sum(0)
                    J[C,:]=z0.T@y0+z1.T@y1
                    where=np.full(offset,-1,np.int64);where[C]=np.arange(len(C))
                mu=S/n_tr;sd=np.sqrt(np.maximum(np.diag(G)/n_tr-mu*mu,0.));sd[sd<1e-5]=1.
                G-=n_tr*np.outer(mu,mu);G/=np.outer(sd,sd)
                fold_states.append({'n_train':n_tr,'G':G,'S':S,'mu':mu,'sd':sd,'patch':patch,'where':where,
                    'J':J,'joint':(J+np.outer(shift,ny))/n_tr,'prior':ny/n_tr,
                    'peer_prior':mu+shift})
        trace['preparation_seconds']=time.monotonic()-started
        candidates={name:np.zeros((n_v,work[name]['K']),np.float32) for name in names}
        tests={name:np.zeros((n_q,work[name]['K']),np.float32) for name in names}
        if len(fold_states)==5:
            joints=np.stack([st['joint'] for st in fold_states])
            priors=np.stack([st['prior'] for st in fold_states])
            peer_priors=np.stack([st['peer_prior'] for st in fold_states])
            n_tr=np.array([st['n_train'] for st in fold_states],np.float64)
        wb_like=60<=len(names)<=90
        # UNHCR's 18 collinear peers picked the 36 cap in 98% of folds.
        alphas=(np.array((16.,64.,256.)) if wb_like else
                np.array((4.,12.,36.,108.,324.)) if len(names)>100 else np.array((4.,12.,36.)))
        peer_cap=8 if wb_like else (18 if len(names)>100 else 24)
        def solve(target_index,name):
            if time.monotonic()>=hard_stop:return None
            job=work[name];K=job['K'];a,b=blocks[name];patched=0
            # Peer MI for all five folds at once.
            counts=joints[:,:,a:b];independent=peer_priors[:,:,None]*priors[:,None,a:b]
            information=np.where(counts>0,counts*np.log(np.maximum(counts,1e-12)/np.maximum(independent,1e-12)),0).sum(2)
            scores=np.add.reduceat(information,starts,axis=1);scores[:,target_index]=-np.inf
            selected=np.argsort(-scores,axis=1,kind='stable')[:,:min(peer_cap,len(names)-1)]
            peers=[np.concatenate([block_cols[i] for i in selected[f]]+[np.empty(0,np.int64)]) for f in range(5)]
            union=np.unique(np.concatenate(peers))
            XU=Xs[:,union];XqU=Xq[:,union]
            M=max(len(p) for p in peers)+K+1
            # Raw moments of [own logits, own probabilities, labels] per fold segment.
            moments={}
            def raw(best):
                if best not in moments:
                    own=job['oof'][best][order];own_q=job['all'][best][query]
                    lg=np.log(np.clip(own,1e-5,1.));lg-=lg.mean(1,keepdims=True)
                    lq=np.log(np.clip(own_q,1e-5,1.));lq-=lq.mean(1,keepdims=True)
                    T=np.column_stack((lg,own,Ys[:,a:b]))
                    T2=T[:,:2*K]
                    cross=[XU[s].T@T2[s] for s in segments]
                    second=[T[s].T@T[s] for s in segments];first=[T[s].sum(0) for s in segments]
                    moments[best]=(lg,lq,T,cross,sum(cross),second,sum(second),first,sum(first))
                return moments[best]
            gram=np.zeros((5,M,M));r=np.zeros((5,M,K));norm=np.zeros(5);held=[]
            for f,st in enumerate(fold_states):
                best=bests[target_index][f]
                if best!=modal_best[target_index]:patched+=1
                lg,lq,T,cross,cross_total,second,second_total,first,first_total=raw(best)
                nt=st['n_train']
                R=cross_total-cross[f];Mo=second_total-second[f];Su=first_total-first[f]
                patch=st['patch']
                if patch is not None:
                    u_idx=np.flatnonzero(st["where"][union]>=0)
                    if len(u_idx):
                        z_idx=st["where"][union[u_idx]]
                        z0,z1=outside(patch[1][:,z_idx],f);t0,t1=outside(T[:,:2*K],f)
                        R[u_idx]=z0.T@t0+z1.T@t1
                lm=Su[:K]/nt;ls=np.sqrt(np.maximum(np.diag(Mo)[:K]/nt-lm*lm,0.));ls[ls<1e-5]=1.
                scale=np.maximum((Su[K:2*K]-np.diag(Mo)[K:2*K])/nt,.05)
                res_sum=(Su[2*K:]-Su[K:2*K])/scale
                LL=(Mo[:K,:K]-nt*np.outer(lm,lm))/np.outer(ls,ls)
                LR=((Mo[:K,2*K:]-Mo[:K,K:2*K])-nt*np.outer(lm,(Su[2*K:]-Su[K:2*K])/nt))/np.outer(ls,scale)
                res_norm=float(((np.diag(Mo)[2*K:]-2*np.diag(Mo[2*K:,K:2*K])+np.diag(Mo)[K:2*K])/scale**2).sum())
                mu_u,sd_u,S_u=st['mu'][union],st['sd'][union],st['S'][union]
                PL=((R[:,:K]-np.outer(S_u,lm))/ls)/sd_u[:,None]
                PR=((st['J'][union,a:b]-R[:,K:2*K])/scale-np.outer(mu_u,res_sum))/sd_u[:,None]
                pc=peers[f];pos=np.searchsorted(union,pc);m=len(pc)
                g=gram[f]
                g[:m,:m]=st['G'].take(pc,0).take(pc,1)
                g[:m,m:m+K]=PL[pos];g[m:m+K,:m]=PL[pos].T;g[m:m+K,m:m+K]=LL
                g[m+K,m+K]=nt
                r[f,:m]=PR[pos];r[f,m:m+K]=LR;r[f,m+K]=res_sum
                norm[f]=res_norm
                held.append((pc,pos,m,lm,ls))
            # GCV over alpha from one Cholesky factor per alpha: the solve gives the
            # residual energy, tr((G+aI)^-1) gives dof=M-a*tr. Zero padding adds
            # nothing to either.
            factors=np.linalg.cholesky(gram[:,None]+alphas[None,:,None,None]*np.eye(M))
            coefs=np.empty((5,len(alphas),M,K));traces=np.empty((5,len(alphas)))
            for f in range(5):
                for i in range(len(alphas)):
                    inverse,_=dtrtri(factors[f,i],lower=1)
                    traces[f,i]=np.square(inverse).sum()
                    coefs[f,i]=inverse.T@(inverse@r[f])
            residuals=norm[:,None]-(r[:,None]*coefs).sum((2,3))-alphas*np.square(coefs).sum((2,3))
            dof=M-alphas*traces
            gcv_score=residuals/np.maximum(1.,n_tr[:,None]-dof)**2
            # Each fold uses only its selected peer columns. Apply its patch
            # once, then reuse the validation matrix for both ridge scales.
            va_folds=[]
            for f,st in enumerate(fold_states):
                pc,pos,m,lm,ls=held[f];s=segments[f]
                va=XU[s][:,pos].copy()
                if st['patch'] is not None:
                    cols=np.flatnonzero(st['where'][pc]>=0)
                    if len(cols):
                        z_idx=st['where'][pc[cols]]
                        va[:,cols]=st['patch'][1][s][:,z_idx]
                va_folds.append(va)
            def predict_part(picks,test=False):
                out=np.zeros((n_q if test else n_v,K),np.float32)
                for f,st in enumerate(fold_states):
                    pc,pos,m,lm,ls=held[f];lg,lq=raw(bests[target_index][f])[:2]
                    coef=coefs[f,picks[f]]
                    peer_coef,own_coef,bias=coef[:m],coef[m:m+K],coef[m+K]
                    weights=peer_coef/st['sd'][pc][:,None]
                    offset_term=bias-(st['mu'][pc]/st['sd'][pc])@peer_coef
                    s=segments[f]
                    if test:
                        te=XqU[:,pos]
                        if st['patch'] is not None:
                            cols=np.flatnonzero(st['where'][pc]>=0)
                            if len(cols):
                                z_idx=st['where'][pc[cols]]
                                te[:,cols]=st['patch'][2][:,z_idx]
                        linear=te@weights
                    else:
                        linear=va_folds[f]@weights
                    logits_held=lq if test else lg[s]
                    pred=_softmax(logits_held+linear+((logits_held-lm)/ls)@own_coef+offset_term)
                    if test:out+=pred/5.
                    else:out[order[s]]=pred
                return out
            half=split%2==0;y=job['y'][rows];baseline=job['oof'][0]
            def admitted(o):
                return (half.any() and not half.all() and _nll(o,y,K)<job['base']-5e-4 and
                        all(_nll(o[h],y[h],K)<_nll(baseline[h],y[h],K) for h in (half,~half)))
            scale=None
            if len(names)>100:
                picks_lo=np.argmin(gcv_score[:,:3],axis=1)
                picks_hi=1+np.argmin(gcv_score[:,1:],axis=1)
                o_lo=predict_part(picks_lo)
                o_hi=predict_part(picks_hi)
                o,picks,scale=o_lo,picks_lo,'lo'
                lo_ok=admitted(o_lo)
                for trial,tp,label in ((o_hi,picks_hi,'hi'),
                                       (.5*o_lo+.5*o_hi,picks_hi,'mix')):
                    if not admitted(trial):continue
                    if lo_ok and not all(_nll(trial[h],y[h],K)<_nll(o_lo[h],y[h],K)-2.5e-3 for h in (half,~half)):continue
                    if lo_ok and job.get('route_parent') and any(
                            _nll(trial[split==f],y[split==f],K)>=_nll(o_lo[split==f],y[split==f],K)
                            for f in range(5)):continue
                    if scale!='lo' and _nll(trial,y,K)>=_nll(o,y,K):continue
                    o,picks,scale=trial,tp,label
            else:
                picks=np.argmin(gcv_score,axis=1);o=predict_part(picks)
            candidates[name]=o
            tests[name]=((.5*predict_part(picks_lo,True)+.5*predict_part(picks_hi,True)
                         if scale=='mix' else predict_part(picks,True)) if admitted(o) else None)
            return [str(float(alphas[p])) for p in picks],patched,scale
        if len(fold_states)==5:
            from concurrent.futures import ThreadPoolExecutor
            def collect(name,outcome):
                if outcome is None:return
                picked,patched,scale=outcome
                for alpha in picked:trace['alphas'][alpha]=trace['alphas'].get(alpha,0)+1
                if scale is not None:
                    scales=trace.setdefault('scales',{});scales[scale]=scales.get(scale,0)+1
                trace['patched_best']+=patched;completed[name]=5
            if workers>1:
                with ThreadPoolExecutor(max_workers=workers, initializer=lambda: threadpool_limits(limits=1, user_api="openmp")) as executor:
                    for name,outcome in zip(names,executor.map(lambda pair:solve(*pair),enumerate(names))):
                        collect(name,outcome)
            else:
                for target_index,name in enumerate(names):
                    collect(name,solve(target_index,name))
        found={}
        for name in names:
            if completed[name]!=5:continue
            trace['completed']+=1
            job=work[name];y=job['y'][rows];K=job['K'];o=candidates[name]
            loss=_nll(o,y,K);half=split%2==0;baseline=job['oof'][0]
            if (loss>=job['base']-5e-4 or not half.any() or half.all() or
                any(_nll(o[h],y[h],K)>=_nll(baseline[h],y[h],K) for h in (half,~half))):continue
            full=np.zeros((n,K));full[query]=tests[name]
            found[name]=(o,full,job['base']-loss)
        trace['admitted']=len(found);trace['seconds']=time.monotonic()-started
        trace['folds_completed']=completed
        return found,trace


def stacked_cross_target(work, fold, deadline=float('inf'), workers=1, bank=None):
    """Nested cross-target residuals; all inputs exclude the outer labels."""
    from .chain_features import BudgetExpired, check_clock
    from scipy.linalg import solve
    trace = {'eligible': len(work), 'completed': 0, 'admitted': 0,
             'feature_scope': 'outer_partition', 'folds': 5}
    if bank is None: return {}, trace
    bank.deadline = deadline
    names = list(work)
    found = {}
    for name in names:
        job = work[name]; rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train']); K = job['K']
        if len(rows) < 60: continue
        o = np.zeros((len(rows), K), np.float32)
        p = np.zeros_like(job['all'][0])
        try:
            for f in [*np.unique(fold[rows]), None]:
                check_clock(deadline)
                fit = rows if f is None else rows[fold[rows] != f]
                query = rest if f is None else rows[fold[rows] == f]
                yy = job['y'][fit]
                target = np.eye(K)[yy]
                # Ranking sees only fit labels; select a small peer context.
                ranking = []
                for other in names:
                    if other == name: continue
                    peer = work[other]; labels = peer['y'][fit]
                    if np.any(labels < 0): continue
                    nk = peer['K']
                    counts = np.bincount(labels*K+yy, minlength=nk*K).reshape(nk,K).astype(float)
                    joint = counts / len(fit)
                    independent = joint.sum(1)[:,None] * joint.sum(0)[None,:]
                    nz = joint > 0
                    mi = float(np.sum(joint[nz] * np.log(joint[nz]/independent[nz])))
                    ranking.append((mi - (nk-1)*(K-1)/(2*len(fit)), other))
                peers = [other for _,other in sorted(ranking, reverse=True)[:3]]
                cache = bank.partition(fit)
                own = bank.table_probability(name, fit, cache)[0]
                blocks = [bank.table_probability(a, fit, cache)[0] for a in peers]
                Z = np.column_stack(blocks + [own])
                mu = Z[fit].mean(0); sd = Z[fit].std(0); sd[sd < .01] = 1.
                Zf = (Z[fit]-mu)/sd; Zq = (Z[query]-mu)/sd
                residual = target-own[fit]
                gram = Zf.T@Zf
                coef = solve(gram + 36.*np.eye(gram.shape[0]), Zf.T@residual, assume_a='pos')
                pred = _normalize(np.maximum(own[query] + Zq@coef, .001))
                if f is None: p[query] = pred
                else: o[fold[rows] == f] = pred
        except BudgetExpired:
            break
        trace['completed'] += 1
        gain = _nll(job['oof'][0], job['y'][rows], K)-_nll(o, job['y'][rows], K)
        found[name] = (o,p,float(gain))
        trace['admitted'] += 1
    return found, trace
