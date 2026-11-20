"""Latent-class model over every column, read back through GIVEN only.

One discrete respondent type q (Q classes) explains all answers at once:
P(row) = sum_q pi_q prod_c theta_c[q, x_c].  EM fits pi and theta on a
fold's training rows using every observed GIVEN and PREDICT column; a
hidden row then gets P(q | GIVEN) by Bayes on its GIVEN columns alone and
P(item) = P(q | GIVEN) @ theta_item.  The GIVEN columns are conditionally
independent given the type, which is exactly how a shared respondent
trait reaches demographics and answers alike, so the read-back pools the
evidence of all 213 items instead of re-learning GIVEN -> answer per item.

fit() returns {name: (oof on job.train rows, prediction for ALL rows)}:
OOF comes from the model whose training fold excludes the row; hidden
rows get the mean of the five fold models.  Labels outside a fold's
training rows never enter its EM.  deadline uses time.monotonic().
"""
import time

import numpy as np
from scipy import sparse

Q = 20
ITERATIONS = 40
ALPHA = 2.0
TEMPERS = (1.0, 0.7, 0.45)
GRACEFUL_ITERATIONS = 10


class Expired(TimeoutError):
    pass


def _clock(deadline):
    if time.monotonic() >= deadline:
        raise Expired('latent class deadline reached')


def _columns(Xg, levels, work, names):
    """Column blocks: (offset, width) per GIVEN column then per item."""
    blocks, at = [], 0
    for w in levels:
        blocks.append((at, int(w)))
        at += int(w)
    for name in names:
        blocks.append((at, int(work[name]['K'])))
        at += int(work[name]['K'])
    return blocks, at


def _onehot(rows, Xg, work, names, blocks, width, label_rows):
    """Sparse one-hot over GIVEN for ``rows`` plus items observed in label_rows."""
    n = len(rows)
    r, c = [], []
    G = Xg.shape[1]
    for g in range(G):
        r.append(np.arange(n))
        c.append(blocks[g][0] + Xg[rows, g])
    for i, name in enumerate(names):
        y = work[name]['y'][rows]
        ok = label_rows & (y >= 0)
        r.append(np.flatnonzero(ok))
        c.append(blocks[G + i][0] + y[ok])
    r = np.concatenate(r)
    c = np.concatenate(c)
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, width))


def _init(Z, rng, Q=Q):
    """Soft quantile bins of the leading answer component."""
    from sklearn.decomposition import TruncatedSVD
    Zc = Z.astype(np.float64)
    mean = np.asarray(Zc.mean(0)).ravel()
    score = TruncatedSVD(n_components=1, n_iter=3, random_state=int(rng.integers(1 << 30))).fit(Zc).components_[0]
    s = Zc @ score - mean @ score
    rank = np.argsort(np.argsort(s)) / max(len(s) - 1, 1)
    centres = (np.arange(Q) + 0.5) / Q
    logits = -((rank[:, None] - centres[None, :]) ** 2) / (2 * (0.6 / Q) ** 2)
    logits += 0.05 * rng.standard_normal(logits.shape)
    logits -= logits.max(1, keepdims=True)
    R = np.exp(logits)
    return R / R.sum(1, keepdims=True)


def _posterior(Z, pi, logtheta):
    L = np.asarray(Z @ logtheta.T) + np.log(pi)[None, :]
    L -= L.max(1, keepdims=True)
    R = np.exp(L)
    return R / R.sum(1, keepdims=True)


def _fit_fold(Xg, work, names, blocks, width, train_rows, deadline, seed, Q=Q, graceful=False):
    rng = np.random.default_rng(seed)
    Z = _onehot(train_rows, Xg, work, names, blocks, width, np.ones(len(train_rows), bool))
    counts = np.asarray(Z.sum(0)).ravel()
    prior = np.empty(width)
    for off, w in blocks:
        prior[off:off + w] = (counts[off:off + w] + 0.5) / (counts[off:off + w].sum() + 0.5 * w)
    R = _init(Z, rng, Q)
    ZT = Z.T.tocsr()
    for it in range(ITERATIONS):
        if graceful and it >= GRACEFUL_ITERATIONS and time.monotonic() >= deadline:
            break
        _clock(deadline)
        N = np.asarray(ZT @ R).T
        theta = np.empty_like(N)
        for off, w in blocks:
            cnt = N[:, off:off + w]
            theta[:, off:off + w] = (cnt + ALPHA * prior[off:off + w]) / (cnt.sum(1, keepdims=True) + ALPHA)
        pi = (R.sum(0) + 1.0) / (len(R) + Q)
        logtheta = np.log(np.maximum(theta, 1e-12))
        R = _posterior(Z, pi, logtheta)
    return pi, theta, logtheta


def _given_posterior(Xg, rows, pi, logtheta, blocks, temper):
    """P(q | GIVEN) with the GIVEN log-likelihood raised to ``temper``.

    Naive Bayes counts correlated demographics as independent evidence;
    a power below one undoes that overconfidence where the data have it."""
    L = np.zeros((len(rows), len(pi)))
    for g in range(Xg.shape[1]):
        off, _ = blocks[g]
        L += logtheta[:, off + Xg[rows, g]].T
    L = temper * L + np.log(pi)[None, :]
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def _nll(p, y, K):
    return float(-np.log(1e-3 + (1 - K * 1e-3) * p[np.arange(len(y)), y] / np.maximum(p.sum(1), 1e-300)).mean())


def fit(Xg, levels, work, names, fold, deadline, workers=1, q=Q, seed=20261106, graceful=False):
    """graceful: a fold past the deadline after GRACEFUL_ITERATIONS EM steps
    keeps its current parameters instead of dropping the whole arm."""
    """Five fold-isolated EM fits; returns {name: (oof, all)} and per-item tempers."""
    n = len(Xg)
    blocks, width = _columns(Xg, levels, work, names)
    G = Xg.shape[1]
    labelled = np.zeros(n, bool)
    for name in names:
        labelled |= work[name]['train']
    F = int(fold.max()) + 1
    T = len(TEMPERS)
    outs = {name: (np.zeros((T, n, work[name]['K'])), np.zeros((T, n, work[name]['K']))) for name in names}
    hidden = np.flatnonzero(~labelled)

    def one(f):
        train_rows = np.flatnonzero(labelled & (fold != f) & (fold >= 0))
        val_rows = np.flatnonzero(labelled & (fold == f))
        pi, theta, logtheta = _fit_fold(Xg, work, names, blocks, width, train_rows, deadline, seed + f, q, graceful)
        Pv = [_given_posterior(Xg, val_rows, pi, logtheta, blocks, b) for b in TEMPERS]
        Ph = [_given_posterior(Xg, hidden, pi, logtheta, blocks, b) for b in TEMPERS]
        return f, val_rows, Pv, Ph, theta

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, F)) as ex:
            results = list(ex.map(one, range(F)))
    else:
        results = [one(f) for f in range(F)]
    for f, val_rows, Pv, Ph, theta in results:
        for i, name in enumerate(names):
            off, w = blocks[G + i]
            t = theta[:, off:off + w]
            o, a = outs[name]
            for b in range(T):
                o[b, val_rows] = Pv[b] @ t
                a[b, hidden] += (Ph[b] @ t) / F
    found, tempers = {}, {}
    for name in names:
        o, a = outs[name]
        tr = work[name]['train']
        y, K = work[name]['y'][tr], work[name]['K']
        b = int(np.argmin([_nll(o[k, tr], y, K) for k in range(T)]))
        tempers[name] = TEMPERS[b]
        a[b, tr] = o[b, tr]
        found[name] = (o[b, tr], a[b])
    return found, tempers


#  inside a class the answer distribution also depends on the stratum.
# After the pooled EM, STRAT_ITERATIONS more EM steps re-estimate every item's
# emissions per stratum, shrunk to the pooled class emissions with STRAT_TAU
# pseudo-counts; GIVEN emissions stay pooled.  Each entry of STRATA is one
# stratification; all (scale, fold) tasks share one pool, and every scale x
# stratification fit is mixed per item with weights learned on the other
# folds' OOF into one arm.
STRATA = (('wave',), ('wave', 's1_2_country_asylum_lbl'))
STRAT_TAU = 10.0
STRAT_ITERATIONS = 8
WEIGHT_ITERS = 60
#  later items read the class posterior back through GIVEN plus the
# stacked root predictions, sum_y p_hat log theta per root, at power beta.
BETAS = (0.0, 0.05, 0.12, 0.25, 0.5)


def _strata_codes(Xg, levels, given, columns):
    idx = [given.index(c) for c in columns if c in given]
    if not idx:
        return None
    s = np.zeros(len(Xg), np.int64)
    for j in idx:
        s = s * int(levels[j]) + Xg[:, j]
    used, s = np.unique(s, return_inverse=True)
    return s.astype(np.int64), len(used)


def _block_normalize(cnt, starts, sizes, add):
    """(cnt + add) / per-block sums along the last axis."""
    num = cnt + add
    tot = np.add.reduceat(num, starts, axis=-1)
    return num / np.repeat(tot, sizes, axis=-1)


def _stratify(Z, R, strata, G, blocks, prior, item_from, deadline, Q):
    """Stratum emissions continued from the pooled responsibilities R."""
    groups = [np.flatnonzero(strata == s) for s in range(G)]
    Zs = [Z[idx] for idx in groups]
    ZTs = [z.T.tocsr() for z in Zs]
    starts = np.array([b[0] for b in blocks])
    sizes = np.array([b[1] for b in blocks])
    I0 = blocks[item_from][0]
    m = Z.shape[0]
    local = None
    for it in range(STRAT_ITERATIONS):
        if it and time.monotonic() >= deadline:
            break
        counts = np.stack([np.asarray(zt @ R[idx]).T if len(idx) else np.zeros((Q, Z.shape[1])) for zt, idx in zip(ZTs, groups)])
        theta = _block_normalize(counts.sum(0), starts, sizes, ALPHA * prior[None, :])
        local = np.repeat(theta[None], G, 0)
        local[:, :, I0:] = _block_normalize(counts[:, :, I0:], starts[item_from:] - I0, sizes[item_from:], STRAT_TAU * theta[None, :, I0:])
        pi = (R.sum(0) + 1.0) / (m + Q)
        loglocal = np.log(np.maximum(local, 1e-12))
        R = np.empty_like(R)
        for s, idx in enumerate(groups):
            if len(idx):
                R[idx] = _posterior(Zs[s], pi, loglocal[s])
    if local is None:
        return None
    return pi, np.log(np.maximum(theta, 1e-12)), local[:, :, I0:].astype(np.float32)


def _fit_fold_full(Xg, work, names, blocks, width, train_rows, deadline, seed, Q, strata):
    """Pooled EM as in _fit_fold (same numbers), then one continuation per stratification."""
    rng = np.random.default_rng(seed)
    Z = _onehot(train_rows, Xg, work, names, blocks, width, np.ones(len(train_rows), bool))
    counts = np.asarray(Z.sum(0)).ravel()
    prior = np.empty(width)
    for off, w in blocks:
        prior[off:off + w] = (counts[off:off + w] + 0.5) / (counts[off:off + w].sum() + 0.5 * w)
    R = _init(Z, rng, Q)
    ZT = Z.T.tocsr()
    for it in range(ITERATIONS):
        if it >= GRACEFUL_ITERATIONS and time.monotonic() >= deadline:
            break
        _clock(deadline)
        N = np.asarray(ZT @ R).T
        theta = np.empty_like(N)
        for off, w in blocks:
            cnt = N[:, off:off + w]
            theta[:, off:off + w] = (cnt + ALPHA * prior[off:off + w]) / (cnt.sum(1, keepdims=True) + ALPHA)
        pi = (R.sum(0) + 1.0) / (len(R) + Q)
        logtheta = np.log(np.maximum(theta, 1e-12))
        R = _posterior(Z, pi, logtheta)
    strat = []
    for codes, G in strata:
        strat.append(_stratify(Z, R, codes[train_rows], G, blocks, prior, Xg.shape[1], deadline, Q) if time.monotonic() < deadline else None)
    return pi, theta, logtheta, strat


def _mixture_weights(likes):
    """Simplex weights maximizing the mean log of the mixed likelihood (EM)."""
    m, C = likes.shape
    w = np.full(C, 1.0 / C)
    lt = np.ascontiguousarray(likes.T)
    for _ in range(WEIGHT_ITERS):
        new = w * (lt @ (1.0 / np.maximum(likes @ w, 1e-300))) / m
        new /= new.sum()
        done = np.abs(new - w).max() < 1e-7
        w = new
        if done:
            break
    return w


def _floored_like(p, y, K):
    return 1e-3 + (1 - K * 1e-3) * p[np.arange(len(y)), y] / np.maximum(p.sum(1), 1e-300)


def _item_slices(work, names):
    at, out = 0, {}
    for name in names:
        out[name] = (at, at + int(work[name]['K']))
        at += int(work[name]['K'])
    return out, at


def _pick(o, a, work, names, sl, fold=None):
    """Per item, one choice along the leading axis of the (C, n, W) buffers.

    fold=None: the choice minimizing OOF NLL on every training row (as the
    temper choice of fit); otherwise each fold's rows take the choice that
    is best on the other folds.  Hidden rows take the all-rows choice."""
    found, picked = {}, {}
    for name in names:
        lo, hi = sl[name]
        tr = work[name]['train']
        rows = np.flatnonzero(tr)
        y, K = work[name]['y'][tr], work[name]['K']
        loss = -np.log(np.stack([_floored_like(o[c][rows, lo:hi].astype(np.float64), y, K) for c in range(len(o))]))
        c_all = int(np.argmin(loss.mean(1)))
        full = a[c_all][:, lo:hi].astype(np.float64)
        if fold is None:
            oof = o[c_all][rows, lo:hi].astype(np.float64)
        else:
            rf = fold[rows]
            oof = np.zeros((len(rows), K))
            for f in np.unique(rf):
                hit = rf == f
                oof[hit] = o[int(np.argmin(loss[:, ~hit].mean(1)))][rows[hit], lo:hi]
        full[tr] = oof
        found[name] = (oof, full)
        picked[name] = c_all
    return found, picked


def mix_scales(arms, work, names, fold):
    """Per item, fold-held-out simplex mixture of several (oof, all) arms."""
    found, weights = {}, {}
    for name in names:
        tr = work[name]['train']
        rf = fold[tr]
        y, K = work[name]['y'][tr], work[name]['K']
        cand = [a[name] for a in arms if name in a]
        if not cand:
            continue
        if len(cand) == 1:
            found[name] = cand[0]
            continue
        likes = np.column_stack([_floored_like(o, y, K) for o, _ in cand])
        oof = np.zeros_like(cand[0][0])
        for f in np.unique(rf):
            hit = rf == f
            w = _mixture_weights(likes[~hit])
            oof[hit] = sum(wk * o[hit] for wk, (o, _) in zip(w, cand))
        w = _mixture_weights(likes)
        full = sum(wk * a for wk, (_, a) in zip(w, cand))
        full[tr] = oof
        found[name] = (oof, full)
        weights[name] = [round(float(v), 3) for v in w]
    return found, weights


def fit_scales(Xg, levels, given, work, names, fold, deadline, workers=1, scales=((Q, 20261106),), strata=(), executor=None, items=None):
    """Every (scale, fold) EM in one pool; graceful deadline as fit(graceful=True).

    Returns dict(pooled=[found per scale], tempers=[...], strat=found or {},
    models={(scale index, fold): params} for the root read-back, info)."""
    began = time.monotonic()
    n = len(Xg)
    blocks, width = _columns(Xg, levels, work, names)
    G0 = Xg.shape[1]
    I0 = blocks[G0][0]
    sl, W = _item_slices(work, names)
    labelled = np.zeros(n, bool)
    for name in names:
        labelled |= work[name]['train']
    F = int(fold.max()) + 1
    T = len(TEMPERS)
    hidden = np.flatnonzero(~labelled)
    codes = [c for c in (_strata_codes(Xg, levels, given, s) for s in (strata or ())) if c is not None]
    use_zero_gate = bool(LCA_ZERO_GATE and items is not None and _pair_revisits(Xg, list(given[:G0]), items) is None)
    tasks = [(qi, f) for qi in range(len(scales)) for f in range(F)]

    def one(task):
        qi, f = task
        q, seed = scales[qi]
        train_rows = np.flatnonzero(labelled & (fold != f) & (fold >= 0))
        val_rows = np.flatnonzero(labelled & (fold == f))
        try:
            fit_work, zero_fold = _zero_gate_work(Xg, given, work, names, items, train_rows, enabled=True) if use_zero_gate else (work, {})
            pi, theta, logtheta, strat = _fit_fold_full(Xg, fit_work, names, blocks, width, train_rows, deadline, seed + f, q, codes)
            del fit_work
            _trim_heap()
        except Expired:
            return None
        return dict(pi=pi, theta=theta, logtheta=logtheta, strat=strat, val=val_rows, zero_fold=zero_fold)

    if executor is not None:
        models = dict(zip(tasks, executor.map(one, tasks)))
    elif workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(tasks))) as ex:
            models = dict(zip(tasks, ex.map(one, tasks)))
    else:
        models = {t: one(t) for t in tasks}
    def readback():
        em_seconds = time.monotonic() - began
        pooled, tempers, strat_arms = [], [], []
        for qi in range(len(scales)):
            fm = [models[(qi, f)] for f in range(F)]
            if any(m is None for m in fm):
                break
            o = np.zeros((T, n, W), np.float32)
            a = np.zeros((T, n, W), np.float32)
            for m in fm:
                t = m['theta'][:, I0:]
                for b in range(T):
                    o[b, m['val']] = _lca_fold(_given_posterior(Xg, m['val'], m['pi'], m['logtheta'], blocks, TEMPERS[b]) @ t, Xg, given, items, m['val'], sl, m['zero_fold'])
                    a[b, hidden] += _lca_fold(_given_posterior(Xg, hidden, m['pi'], m['logtheta'], blocks, TEMPERS[b]) @ t, Xg, given, items, hidden, sl, m['zero_fold']) / F
            found, tb = _pick(o, a, work, names, sl)
            pooled.append(found)
            tempers.append(tb)
            for k in range(len(codes)):
                if any(m['strat'][k] is None for m in fm):
                    continue
                s_codes = codes[k][0]
                o[:] = 0.0
                a[:] = 0.0
                for m in fm:
                    pi_s, lt_s, local = m['strat'][k]
                    for rows, dest, scale in ((m['val'], o, None), (hidden, a, 1.0 / F)):
                        st = s_codes[rows]
                        for b in range(T):
                            P = _given_posterior(Xg, rows, pi_s, lt_s, blocks, TEMPERS[b])
                            for s in np.unique(st):
                                hit = np.flatnonzero(st == s)
                                v = _lca_fold(P[hit] @ local[s], Xg, given, items, rows[hit], sl, m['zero_fold'])
                                if scale is None:
                                    dest[b, rows[hit]] = v
                                else:
                                    dest[b, rows[hit]] += scale * v
                strat_arms.append(_pick(o, a, work, names, sl)[0])
            for m in fm:
                m.pop('strat', None)
                m.pop('zero_fold', None)
            del o, a
            _trim_heap()
        n_strat = len(strat_arms)
        strat, weights = mix_scales(strat_arms, work, names, fold) if strat_arms else ({}, {})
        del strat_arms
        _trim_heap()
        info = dict(scales=len(pooled), strat_fits=n_strat, strata=[c[1] for c in codes], em_seconds=round(em_seconds, 2),
                    readback_seconds=round(time.monotonic() - began - em_seconds, 2),
                    strat_weight_mean=(np.mean(list(weights.values()), 0).round(3).tolist() if weights else None))
        return dict(pooled=pooled, tempers=tempers, strat=strat, models=models, blocks=blocks, names=list(names), F=F, hidden=hidden, info=info)
    return executor.submit(readback).result() if executor is not None else readback()

def root_readback(state, Xg, work, roots, targets, root_prob, fold, deadline):
    """: class posterior from GIVEN plus the stacked root predictions.

    root_prob[r] is n x K_r: OOF stack on the item's training rows, the
    hidden-row stack elsewhere; neither uses the target row's own labels.
    Per item and scale beta is chosen on the other folds' OOF; the scales
    are then mixed as in mix_scales.  beta = 0 is the GIVEN-only arm."""
    blocks, F, hidden, names = state['blocks'], state['F'], state['hidden'], state['names']
    G0 = Xg.shape[1]
    I0 = blocks[G0][0]
    col = {name: G0 + i for i, name in enumerate(names)}
    roots = [r for r in roots if r in root_prob and r in col]
    targets = [t for t in targets if t in col and t not in roots]
    sl, W = _item_slices(work, names)
    n = len(Xg)
    B = len(BETAS)
    arms = []
    for qi in range(len(state['pooled'])):
        temper = state['tempers'][qi]
        by_temper = [[t for t in targets if temper[t] == b] for b in range(len(TEMPERS))]
        o = np.zeros((B, n, W), np.float32)
        a = np.zeros((B, n, W), np.float32)
        for f in range(F):
            if time.monotonic() >= deadline:
                return {}, {}
            m = state['models'][(qi, f)]
            lt = m['logtheta']
            t_items = m['theta'][:, I0:]
            for rows, dest, scale in ((m['val'], o, None), (hidden, a, 1.0 / F)):
                Lg = np.zeros((len(rows), len(m['pi'])))
                for g in range(G0):
                    off, _ = blocks[g]
                    Lg += lt[:, off + Xg[rows, g]].T
                Lr = np.zeros_like(Lg)
                for r in roots:
                    off, w = blocks[col[r]]
                    Lr += root_prob[r][rows] @ lt[:, off:off + w].T
                logpi = np.log(m['pi'])[None, :]
                for b, group in enumerate(by_temper):
                    if not group:
                        continue
                    cols = np.concatenate([np.arange(*sl[t]) for t in group])
                    tb = t_items[:, cols]
                    for k, beta in enumerate(BETAS):
                        L = TEMPERS[b] * Lg + beta * Lr + logpi
                        L -= L.max(1, keepdims=True)
                        P = np.exp(L)
                        P /= P.sum(1, keepdims=True)
                        v = P @ tb
                        if scale is None:
                            dest[k][np.ix_(rows, cols)] = v
                        else:
                            dest[k][np.ix_(rows, cols)] += scale * v
        arms.append(_pick(o, a, work, targets, sl, fold)[0])
        del o, a
    return mix_scales(arms, work, targets, fold)


#  a standard-normal trait on a TRAIT_NODES quadrature grid; every
# column's answer logits are base + trait . loading, the smooth limit of many
# ordered classes and the official sandbox's own generator.  A gated item's
# skip is routed rather than emitted: its NA_GATED cells leave the trait
# likelihood (the parent already carries that evidence), and on the grid
# P(skip | t) = sum_v P(parent = v | t) s[v], with s[v] the fold's training
# skip rate given the parent's answer v; a GIVEN parent's s is read per row.
TRAIT_DIM = 1
TRAIT_NODES = {1: 41, 2: 15}
TRAIT_ITERS = 60
TRAIT_MIN_ITERS = 8
TRAIT_TOL = 2e-5
TRAIT_RIDGE = 0.25
TRAIT_TEMPERS = (1.0, 0.7)
#   reads the pooled fold models again for the twin copy posterior.
TRAIT_MODELS = {}


def _trait_grid(dim, nodes):
    axis = np.linspace(-4.0, 4.0, nodes)
    g = np.stack(np.meshgrid(*([axis] * dim), indexing='ij'), -1).reshape(-1, dim)
    lw = -0.5 * (g ** 2).sum(1)
    lw -= lw.max()
    return g, lw - np.log(np.exp(lw).sum())


def _block_lse(Zl, starts, widths):
    top = np.maximum.reduceat(Zl, starts, axis=1)
    s = np.add.reduceat(np.exp(Zl - np.repeat(top, widths, 1)), starts, axis=1)
    return np.repeat(top + np.log(s), widths, 1)


def _trait_layout(levels, given, work, names, items):
    """Trait blocks (a routed item drops its NA_GATED column) and each item's route:
    ('item', parent) for an earlier PREDICT parent, ('given', column) or None."""
    blocks, at = [], 0
    for w in levels:
        blocks.append((at, int(w)))
        at += int(w)
    seen, route = set(), {}
    for name in names:
        K = int(work[name]['K'])
        gate = items[name].get('gate')
        r = None
        if isinstance(gate, dict) and K > 2:
            p = gate.get('parent')
            if p in seen:
                r = ('item', p)
            elif p in given:
                r = ('given', given.index(p))
        route[name] = r
        w = K - 1 if r is not None else K
        blocks.append((at, w))
        at += w
        seen.add(name)
    return blocks, at, route


def _trait_onehot(rows, Xg, work, names, blocks, width, route):
    """Sparse one-hot plus the rows x blocks observed indicator."""
    n, G = len(rows), Xg.shape[1]
    r, c = [np.repeat(np.arange(n), G)], [(np.array([b[0] for b in blocks[:G]])[None, :] + Xg[rows]).ravel()]
    obs = np.zeros((n, len(blocks)), np.float32)
    obs[:, :G] = 1.0
    for i, name in enumerate(names):
        y = work[name]['y'][rows]
        ok = y >= 0
        if route[name] is not None:
            ok &= y < work[name]['K'] - 1
        hit = np.flatnonzero(ok)
        r.append(hit)
        c.append(blocks[G + i][0] + y[hit])
        obs[hit, G + i] = 1.0
    r, c = np.concatenate(r), np.concatenate(c)
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, width)), obs


def _fit_trait(Z, obs, blocks, dim, seed, deadline):
    """Nominal-response EM on a quadrature grid; returns (log emissions Q x D, log prior)."""
    m, D = Z.shape
    starts = np.array([b[0] for b in blocks])
    widths = np.array([b[1] for b in blocks])
    g, lw = _trait_grid(dim, TRAIT_NODES[dim])
    ZT = Z.T.tocsr()
    colsum = np.asarray(Z.sum(0)).ravel().astype(np.float64)
    B = np.log(_block_normalize(colsum + 1e-3, starts, widths, 0.0))
    A = np.zeros((dim, D))
    # Start the posterior around the leading principal scores of the indicators.
    rng = np.random.default_rng(seed)
    mean = colsum / m
    V = rng.normal(size=(D, dim))
    for _ in range(6):
        U = Z @ V - mean @ V
        V, _ = np.linalg.qr(ZT @ U - np.outer(mean, U.sum(0)))
    S = Z @ V - mean @ V
    S = (S - S.mean(0)) / np.maximum(S.std(0), 1e-9)
    R = -0.5 * ((S[:, None, :] - g[None]) ** 2).sum(-1) / 0.36 + lw[None]
    R = np.exp(R - R.max(1, keepdims=True))
    R /= R.sum(1, keepdims=True)
    X1 = np.column_stack([np.ones(len(g)), g])
    ridge = np.diag([1e-3] + [TRAIT_RIDGE] * dim)
    prev, log_t, it = -np.inf, None, 0
    for it in range(TRAIT_ITERS):
        if time.monotonic() >= deadline:
            if log_t is not None and it >= TRAIT_MIN_ITERS:
                break
            raise Expired('latent trait deadline reached')
        N = np.asarray(ZT @ R).T
        nb = np.repeat(R.T @ obs, widths, 1)
        for _ in range(2):
            Zl = B[None] + g @ A
            P = np.exp(Zl - _block_lse(Zl, starts, widths))
            Gr = N - nb * P
            W = nb * P * (1.0 - P) + 1e-6
            # One (dim + 1)-parameter Newton step per answer column.
            gr = X1.T @ Gr - np.vstack([np.zeros((1, D)), TRAIT_RIDGE * A])
            H = np.einsum('qa,qb,qd->dab', X1, X1, W) + ridge
            # The diagonal-category Hessian overshoots on rare answers; cap every step.
            step = np.clip(np.linalg.solve(H, gr.T[:, :, None])[:, :, 0], -1.0, 1.0)
            B = np.clip(B + step[:, 0], -25.0, 25.0)
            A = np.clip(A + step[:, 1:].T, -8.0, 8.0)
        Zl = B[None] + g @ A
        log_t = Zl - _block_lse(Zl, starts, widths)
        L = np.asarray(Z @ log_t.T) + lw[None]
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        # Capped Newton steps can dip the likelihood early; converge on its size.
        if it >= TRAIT_MIN_ITERS and abs(ll - prev) < TRAIT_TOL * m:
            break
        prev = ll
    return log_t, lw, it + 1, B, A, R


#  the same trait with a round x country intercept and a round x trait
# loading on every item column.  After the pooled fit, TRAIT_STRAT_ITERS more
# EM steps re-estimate the item columns per stratum (intercepts) and per round
# (loadings), each shrunk to the pooled fit by TRAIT_STRAT_TAU pseudo-rows that
# answer like the pooled model, spread over the grid as the stratum's own trait
# posterior; GIVEN columns stay pooled, so a stratum's trait shift still comes
# through its round and country columns.  Both arms come out of one fold fit.
TRAIT_STRAT = ('wave', 's1_2_country_asylum_lbl')
TRAIT_ROUND = ('wave',)
TRAIT_STRAT_TAU = 3.0
TRAIT_STRAT_ITERS = 40
TRAIT_STRAT_MIN_ITERS = 4
TRAIT_POOLED_SHARE = 0.65


def _fit_trait_strat(Z, obs, blocks, G, strata, S, rounds, B, A, R, lw, deadline):
    """Stratum item intercepts and round item loadings continued from the pooled
    trait fit (B, A, posterior R).  strata: stratum per training row; rounds:
    round index per stratum.  Returns (log GIVEN emissions Q x DG, log item
    emissions S x Q x DI, steps) or None when no step fits before the deadline."""
    m, D = Z.shape
    I0 = blocks[G][0]
    starts = np.array([b[0] for b in blocks])
    widths = np.array([b[1] for b in blocks])
    gs, gw = starts[:G], widths[:G]
    st, sw = starts[G:] - I0, widths[G:]
    g, _ = _trait_grid(A.shape[0], TRAIT_NODES[A.shape[0]])
    X1 = np.column_stack([np.ones(len(g)), g])
    ridge = np.diag([1e-3] + [TRAIT_RIDGE] * A.shape[0])
    groups = [np.flatnonzero(strata == s) for s in range(S)]
    Zg = [Z[idx][:, :I0].tocsr() for idx in groups]
    Zi = [Z[idx][:, I0:].tocsr() for idx in groups]
    ZgT = [z.T.tocsr() for z in Zg]
    ZiT = [z.T.tocsr() for z in Zi]
    obs_s = [obs[idx] for idx in groups]
    W = int(rounds.max()) + 1
    BG, AG = B[:I0].copy(), A[:, :I0].copy()
    BI0, AI0 = B[I0:].copy(), A[:, I0:].copy()
    ZlI0 = BI0[None] + g @ AI0
    PI0 = np.exp(ZlI0 - _block_lse(ZlI0, st, sw))
    BI = np.repeat(BI0[None], S, 0)
    AI = np.repeat(AI0[None], W, 0)
    Rs = [R[idx] for idx in groups]
    prev, out, it = -np.inf, None, 0
    for it in range(TRAIT_STRAT_ITERS):
        if time.monotonic() >= deadline:
            if out is None:
                return None
            break
        NG = np.zeros((len(g), I0))
        nbG = np.zeros((len(g), G))
        NI, nbI = [], []
        for s, idx in enumerate(groups):
            if not len(idx):
                NI.append(None)
                nbI.append(None)
                continue
            NG += np.asarray(ZgT[s] @ Rs[s]).T
            nb = Rs[s].T @ obs_s[s]
            nbG += nb[:, :G]
            r = Rs[s].sum(0) / len(idx)
            NI.append(np.asarray(ZiT[s] @ Rs[s]).T + TRAIT_STRAT_TAU * r[:, None] * PI0)
            nbI.append(np.repeat(nb[:, G:] + TRAIT_STRAT_TAU * r[:, None], sw, 1))
        nbG = np.repeat(nbG, gw, 1)
        for _ in range(2):
            Zl = BG[None] + g @ AG
            P = np.exp(Zl - _block_lse(Zl, gs, gw))
            Gr = NG - nbG * P
            Wt = nbG * P * (1.0 - P) + 1e-6
            gr = X1.T @ Gr - np.vstack([np.zeros((1, I0)), TRAIT_RIDGE * AG])
            H = np.einsum('qa,qb,qd->dab', X1, X1, Wt) + ridge
            step = np.clip(np.linalg.solve(H, gr.T[:, :, None])[:, :, 0], -1.0, 1.0)
            BG = np.clip(BG + step[:, 0], -25.0, 25.0)
            AG = np.clip(AG + step[:, 1:].T, -8.0, 8.0)
            # Item intercepts per stratum, then loadings per round on the new intercepts.
            gA = np.zeros_like(AI)
            hA = np.zeros_like(AI)
            for s in range(S):
                if NI[s] is None:
                    continue
                w = rounds[s]
                Zl = BI[s][None] + g @ AI[w]
                P = np.exp(Zl - _block_lse(Zl, st, sw))
                Gr = NI[s] - nbI[s] * P
                Wt = nbI[s] * P * (1.0 - P) + 1e-6
                BI[s] = np.clip(BI[s] + np.clip(Gr.sum(0) / (Wt.sum(0) + 1e-3), -1.0, 1.0), -25.0, 25.0)
                Zl = BI[s][None] + g @ AI[w]
                P = np.exp(Zl - _block_lse(Zl, st, sw))
                Gr = NI[s] - nbI[s] * P
                Wt = nbI[s] * P * (1.0 - P) + 1e-6
                gA[w] += g.T @ Gr
                hA[w] += (g ** 2).T @ Wt
            AI = np.clip(AI + np.clip((gA - TRAIT_RIDGE * (AI - AI0[None])) / (hA + TRAIT_RIDGE), -1.0, 1.0), -8.0, 8.0)
        Zl = BG[None] + g @ AG
        logG = Zl - _block_lse(Zl, gs, gw)
        logI = np.empty((S, len(g), D - I0))
        for s in range(S):
            Zl = BI[s][None] + g @ AI[rounds[s]]
            logI[s] = Zl - _block_lse(Zl, st, sw)
        ll = 0.0
        for s, idx in enumerate(groups):
            if not len(idx):
                continue
            L = np.asarray(Zg[s] @ logG.T) + np.asarray(Zi[s] @ logI[s].T) + lw[None]
            top = L.max(1, keepdims=True)
            E = np.exp(L - top)
            tot = E.sum(1, keepdims=True)
            Rs[s] = E / tot
            ll += float(np.log(tot).sum() + top.sum())
        out = (logG, logI)
        if it >= TRAIT_STRAT_MIN_ITERS and abs(ll - prev) < TRAIT_TOL * m:
            break
        prev = ll
    return out[0], out[1], it + 1


def _skip_rates(codes, skip, ok, size):
    """Per-code skip rate on the rows ok, shrunk to the overall rate by half a row."""
    base = skip[ok].mean() if ok.any() else 0.0
    hit = np.bincount(codes[ok], weights=skip[ok].astype(np.float64), minlength=size)
    tot = np.bincount(codes[ok], minlength=size)
    return (hit + 0.5 * base) / (tot + 0.5)


def _trait_emissions(log_t, blocks, route, work, names, Xg, train_rows, G):
    """Full-width answer distribution on the grid per item, routed in order.

    A GIVEN-routed item also returns its per-code skip rates for the read-back;
    its grid row uses the training-average skip so routed children see it."""
    E, given_skip = {}, {}
    for i, name in enumerate(names):
        off, w = blocks[G + i]
        nom = np.exp(log_t[:, off:off + w])
        r = route[name]
        if r is None:
            E[name] = nom
            continue
        y, K = work[name]['y'], int(work[name]['K'])
        yt = y[train_rows]
        skip = yt == K - 1
        if r[0] == 'item':
            yp = work[r[1]]['y'][train_rows]
            s = _skip_rates(np.maximum(yp, 0), skip, (yp >= 0) & (yt >= 0), int(work[r[1]]['K']))
            sk = E[r[1]] @ s
        else:
            codes = Xg[train_rows, r[1]]
            s = _skip_rates(codes, skip, yt >= 0, int(Xg[:, r[1]].max()) + 1)
            given_skip[name] = s
            sk = np.full(len(nom), float(skip[yt >= 0].mean()) if (yt >= 0).any() else 0.0)
        E[name] = np.column_stack([(1.0 - sk)[:, None] * nom, sk])
    return E, given_skip



#  round one writes "0" for a zero-on-skip item whose gate is closed
# while round two writes the skip.  Read those round-one cells as the skip, so
# the fit sees the gate instead of an answer, and fold the skip mass of the
# round-one read-back onto "0".
ZERO_GATE = False
LCA_ZERO_GATE = False
MULTIPARENT = False


def _zero_gate_work(Xg, given, work, names, items, rows, enabled=None):
    """work with round-one closed-gate "0" cells of zero-on-skip items set to the
    skip, plus {name: (zero code, cap)} for the read-back fold."""
    from .propagate_structural_gates import detect_zero_on_skip
    if not (ZERO_GATE if enabled is None else enabled) or 'wave' not in given or 'wave' not in items:
        return work, {}
    found = detect_zero_on_skip(items, list(given), Xg, work, rows)
    if not found:
        return work, {}
    wave = Xg[:, list(given).index('wave')]
    first = [str(v) for v in items['wave']['values']].index('1')
    out, fold = dict(work), {}
    for name in names:
        if name not in found:
            continue
        gate = items[name]['gate']
        p = gate['parent']
        vals = items[p]['values']
        allowed = [vals.index(v) for v in gate['observed_if'] if v in vals]
        if p in out:
            py = out[p]['y']
            known, opened = py >= 0, np.isin(py, allowed)
        elif p in given:
            code = Xg[:, list(given).index(p)]
            known, opened = np.ones(len(code), bool), np.isin(code, allowed)
        else:
            continue
        j = out[name]
        zero = [str(v) for v in items[name]['values']].index('0')
        hit = (wave == first) & known & ~opened & (j['y'] == zero)
        if hit.any():
            y = j['y'].copy()
            y[hit] = int(j['K']) - 1
            out[name] = dict(j, y=y)
        fold[name] = (zero, found[name])
    return out, fold


def _fold_zero(dest, wave_one, sl, zero_fold):
    """dest (T, n, W): move round-one skip mass above the cap onto "0"."""
    r = np.flatnonzero(wave_one)
    if not len(r):
        return
    for name, (zero, cap) in zero_fold.items():
        lo, hi = sl[name]
        for b in range(dest.shape[0]):
            v = dest[b][r, lo:hi]
            excess = np.clip(v[:, -1] - cap, 0.0, None)
            v[:, -1] -= excess
            v[:, zero] += excess
            dest[b][r, lo:hi] = v


#  the round-two side of a panel pair repeats its round-one twin on
# most root subtrees and keeps its own trait on the rest, so as one trait
# observation it mixes two respondents and counts the twin twice.  The trait
# EM (pooled and stratum continuation, skip rates included) fits on rows that
# are not the round-two side of an exact or one-band-drift pair; the OOF and
# hidden read-back still cover every row.
PAIR_PURE = False
PAIR_PURE_STRAT_ALL = True


def _pair_revisits(Xg, given, items):
    """Round-two rows of the exact and drift pairs, or None when the panel
    signature is absent or below the twin layer's edge floor."""
    if not PAIR_PURE or 'wave' not in given:
        return None
    from . import twin_identity as TI
    wcol = list(given).index('wave')
    pairs = TI.exact_pairs(Xg, Xg.shape[1], wcol)
    taken = Xg[:, wcol] > 1
    for a, b in pairs:
        taken[a] = taken[b] = True
    pairs = pairs + TI.drift_pairs(Xg, Xg.shape[1], wcol, list(given), {'items': items}, taken)
    if len(pairs) < TI.MIN_EDGES:
        return None
    drop = np.zeros(len(Xg), bool)
    drop[np.asarray(pairs, int).reshape(-1, 2)[:, 1]] = True
    return drop


def fit_trait(Xg, levels, given, work, names, items, fold, deadline, workers=5, dim=TRAIT_DIM, strat=False, executor=None):
    """Five fold-isolated trait fits read back through GIVEN; returns
    ({arm: {name: (oof, all)}}, info): 'latent_trait', plus 'latent_trait_strat'
    ( ) when strat is set and every fold's continuation fits in time."""
    began = time.monotonic()
    TRAIT_MODELS.clear()
    n, G = Xg.shape
    names = list(names)
    labels = work
    rows0 = np.flatnonzero(np.logical_or.reduce([work[k]['train'] for k in names]))
    work, zero_fold = _zero_gate_work(Xg, list(given[:G]), work, names, items, rows0)
    wave_one = (Xg[:, list(given[:G]).index('wave')] == [str(v) for v in items['wave']['values']].index('1')) if zero_fold else None
    blocks, width, route = _trait_layout(levels, list(given[:G]), work, names, items)
    sl, W = _item_slices(work, names)
    labelled = np.zeros(n, bool)
    for name in names:
        labelled |= work[name]['train']
    F = int(fold.max()) + 1
    hidden = np.flatnonzero(~labelled)
    T = len(TRAIT_TEMPERS)
    revisit = _pair_revisits(Xg, list(given[:G]), items)
    fit_mask = labelled if revisit is None else labelled & ~revisit

    sc = _strata_codes(Xg, levels, list(given[:G]), TRAIT_STRAT) if strat else None
    if sc is not None:
        codes, S = sc
        rounds = np.zeros(S, np.int64)
        rc = _strata_codes(Xg, levels, list(given[:G]), TRAIT_ROUND)
        if rc is not None:
            rounds[codes] = rc[0]

    def one(f):
        train_rows = np.flatnonzero(fit_mask & (fold != f) & (fold >= 0))
        Z, obs = _trait_onehot(train_rows, Xg, work, names, blocks, width, route)
        t0 = time.monotonic()
        # With the continuation on, the pooled EM leaves it the last share of the window.
        pooled_by = deadline if sc is None else began + TRAIT_POOLED_SHARE * (deadline - began)
        try:
            log_t, lw, iters, B, A, R = _fit_trait(Z, obs, blocks, dim, 20261192 + 97 * f, pooled_by)
        except Expired:
            return None
        E, gskip = _trait_emissions(log_t, blocks, route, work, names, Xg, train_rows, G)
        m = dict(log_t=log_t, lw=lw, E=E, gskip=gskip, val=np.flatnonzero(labelled & (fold == f)),
                 iters=iters, seconds=round(time.monotonic() - t0, 2))
        if MULTIPARENT:
            from . import latent_gate_routes
            m['joint_routes'] = latent_gate_routes.fit(Xg, list(given[:G]), work, names, items, train_rows, R, E, route)
        if sc is not None:
            t1 = time.monotonic()
            if revisit is not None and PAIR_PURE_STRAT_ALL:
                # The stratum continuation keeps every labelled row: a round's
                # intercepts and loadings need the round-two rows, and the
                # pooled columns it starts from are already the clean ones.
                train_rows = np.flatnonzero(labelled & (fold != f) & (fold >= 0))
                Z, obs = _trait_onehot(train_rows, Xg, work, names, blocks, width, route)
                L = np.asarray(Z @ log_t.T) + lw[None]
                R = np.exp(L - L.max(1, keepdims=True))
                R /= R.sum(1, keepdims=True)
                del L
            fitted = _fit_trait_strat(Z, obs, blocks, G, codes[train_rows], S, rounds, B, A, R, lw, deadline)
            if fitted is not None:
                logG, logI, m['strat_iters'] = fitted
                m['strat'] = (logG, [_trait_emissions(np.hstack([logG, logI[s]]), blocks, route, work, names, Xg, train_rows, G)[0] for s in range(S)])
            m['strat_seconds'] = round(time.monotonic() - t1, 2)
        del Z, obs, B, A, R
        return m

    if executor is not None:
        models = list(executor.map(one, range(F)))
    elif workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, F)) as ex:
            models = list(ex.map(one, range(F)))
    else:
        models = [one(f) for f in range(F)]
    def readback():
        info = dict(dim=dim, routed=sum(r is not None for r in route.values()), fit_seconds=round(time.monotonic() - began, 2),
                    iters=[m['iters'] if m else None for m in models], fold_seconds=[m['seconds'] if m else None for m in models])
        if revisit is not None:
            info['pair_pure_dropped'] = int((labelled & revisit).sum())
        if any(m is None for m in models):
            info['status'] = 'budget_expired'
            return {}, info
        o = np.zeros((T, n, W), np.float32)
        a = np.zeros((T, n, W), np.float32)
        for m in models:
            for rows, dest, scale in ((m['val'], o, None), (hidden, a, 1.0 / F)):
                L0 = np.zeros((len(rows), len(m['lw'])))
                for g in range(G):
                    off, _ = blocks[g]
                    L0 += m['log_t'][:, off + Xg[rows, g]].T
                for b, temper in enumerate(TRAIT_TEMPERS):
                    L = temper * L0 + m['lw'][None]
                    L -= L.max(1, keepdims=True)
                    P = np.exp(L)
                    P /= P.sum(1, keepdims=True)
                    for name in names:
                        lo, hi = sl[name]
                        if name in m['gskip']:
                            s = m['gskip'][name]
                            code = Xg[rows, route[name][1]]
                            s = np.where(code < len(s), s[np.minimum(code, len(s) - 1)], s.mean())[:, None]
                            nom = m['E'][name][:, :-1] / np.maximum(1.0 - m['E'][name][:, -1:], 1e-12)
                            v = np.column_stack([(1.0 - s) * (P @ nom), s])
                        else:
                            v = P @ m['E'][name]
                        if scale is None:
                            dest[b, rows, lo:hi] = v
                        else:
                            dest[b, rows, lo:hi] += scale * v
        if zero_fold:
            _fold_zero(o, wave_one, sl, zero_fold)
            _fold_zero(a, wave_one, sl, zero_fold)
        found, picked = _pick(o, a, labels, names, sl, fold)
        TRAIT_MODELS.update(models=[dict(log_t=m['log_t'], lw=m['lw'], E=m['E'], gskip=m['gskip']) for m in models],
                            blocks=blocks, route=route, names=names, G=G)
        info.update(status='complete', temper_mean=round(float(np.mean([TRAIT_TEMPERS[c] for c in picked.values()])), 3))
        arms = {'latent_trait': found}
        if sc is not None:
            # _pick has copied both selected outputs; release the pooled buffers
            # before allocating the stratum read-back buffers.
            del o, a, dest
            info.update(strata=S, strat_iters=[m.get('strat_iters') for m in models], strat_seconds=[m.get('strat_seconds') for m in models])
            if all('strat' in m for m in models):
                arms['latent_trait_strat'] = _trait_strat_read(models, Xg, blocks, route, names, sl, W, n, hidden, codes, G, labels, fold, (wave_one, zero_fold))
        if MULTIPARENT:
            from . import latent_gate_routes
            t_route = time.monotonic()
            args = (models, Xg, blocks, route, names, sl, W, hidden, G, labels, fold)
            arms['latent_trait_route'] = latent_gate_routes.read(*args, zero=(wave_one, zero_fold))
            if sc is not None and all('strat' in m for m in models):
                arms['latent_trait_strat_route'] = latent_gate_routes.read(*args, codes=codes, zero=(wave_one, zero_fold))
            info['joint_route_items'] = [len(m['joint_routes']) for m in models]
            info['joint_route_parents'] = {k:list(v['parents']) for k,v in models[0]['joint_routes'].items()}
            info['joint_route_seconds'] = round(time.monotonic() - t_route, 2)
        for m in models:
            if m is not None:
                m.pop('strat', None)
                m.pop('joint_routes', None)
        _trim_heap()
        info['seconds'] = round(time.monotonic() - began, 2)
        return arms, info
    return executor.submit(readback).result() if executor is not None else readback()


def _trait_strat_read(models, Xg, blocks, route, names, sl, W, n, hidden, codes, G, work, fold, zero=(None, {})):
    """Stratum read-back: P(t | GIVEN) under the continued GIVEN emissions, then
    each row's own stratum emissions; GIVEN-routed skips as in fit_trait."""
    T = len(TRAIT_TEMPERS)
    F = len(models)
    o = np.zeros((T, n, W), np.float32)
    a = np.zeros((T, n, W), np.float32)
    routed = [name for name in names if name in models[0]['gskip']]
    for m in models:
        logG, Es = m['strat']
        # One Q x W emission matrix per stratum; a GIVEN-routed item carries its
        # nominal part, the per-row skip is applied after the product.
        cat = []
        for E in Es:
            cols = []
            for name in names:
                e = E[name]
                if name in m['gskip']:
                    e = np.column_stack([e[:, :-1] / np.maximum(1.0 - e[:, -1:], 1e-12), np.zeros(len(e))])
                cols.append(e)
            cat.append(np.hstack(cols))
        for rows, dest, scale in ((m['val'], o, None), (hidden, a, 1.0 / F)):
            L0 = np.zeros((len(rows), len(m['lw'])))
            for g in range(G):
                off, _ = blocks[g]
                L0 += logG[:, off + Xg[rows, g]].T
            st = codes[rows]
            skips = {}
            for name in routed:
                s = m['gskip'][name]
                code = Xg[rows, route[name][1]]
                skips[name] = np.where(code < len(s), s[np.minimum(code, len(s) - 1)], s.mean())
            for b, temper in enumerate(TRAIT_TEMPERS):
                L = temper * L0 + m['lw'][None]
                L -= L.max(1, keepdims=True)
                P = np.exp(L)
                P /= P.sum(1, keepdims=True)
                V = np.empty((len(rows), W))
                for s in np.unique(st):
                    hit = np.flatnonzero(st == s)
                    V[hit] = P[hit] @ cat[s]
                for name in routed:
                    lo, hi = sl[name]
                    V[:, lo:hi - 1] *= (1.0 - skips[name])[:, None]
                    V[:, hi - 1] = skips[name]
                if scale is None:
                    dest[b, rows] = V
                else:
                    dest[b, rows] += scale * V
    if zero[1]:
        _fold_zero(o, zero[0], sl, zero[1])
        _fold_zero(a, zero[0], sl, zero[1])
    return _pick(o, a, work, names, sl, fold)[0]


def _trim_heap():
    import gc, sys
    gc.collect()
    if sys.platform == 'linux':
        import ctypes
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)


def _lca_fold(v, Xg, given, items, rows, sl, zero_fold):
    if zero_fold:
        first = [str(v) for v in items['wave']['values']].index('1')
        _fold_zero(v[None], Xg[rows, list(given).index('wave')] == first, sl, zero_fold)
    return v
