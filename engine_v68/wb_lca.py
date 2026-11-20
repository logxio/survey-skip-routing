"""World Bank joint latent-class regression over the GIVEN columns and all 76 targets.

Same model as the UNICEF arm: inside a latent class every GIVEN column and
every answer is an independent categorical. EM fits the classes on the fold's
training rows with every answered target, so each class pools evidence from
the scales, the cognitive tasks, both assessment surveys and the tracer. A
hidden row sees only its GIVEN codes: P(c | GIVEN) is the class posterior from
the GIVEN indicators alone and P(y_i | GIVEN) = sum_c P(c | GIVEN) P(y_i | c).

Kinds: 'p' class counts share one emission table. 's' class counts let each
randomisation group keep its own answer distribution inside a class, shrunk
to the pooled class with TAU pseudo-counts; GIVEN emissions stay pooled, so
the class posterior is the same function of GIVEN in every group. 't' fits a
nominal-response model with a standard normal trait of that dimension on a
quadrature grid, the smooth limit of many ordered classes. Each item mixes the
completed kinds with simplex weights learned on the other folds' out-of-fold
likelihoods; the held fold never sees its own weights.

A target gated on a GIVEN parent leaves its closed rows out of the likelihood
and puts all mass on the never-asked level there, when every training row in
the closed cells is never-asked.
"""
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from . import v68_config

# Value per second on the A/B rulers; 36-class and 24-class stratified fits add nothing
# on top. A closed window keeps the completed kinds.
KINDS = (('t', 1), ('p', 12), ('p', 24), ('s', 12))
# Exact mode: trait kinds run full softmax Newton steps to convergence, and children
# open only through a parent's answer take P(parent open | class) inside every class.
EXACT = v68_config.V91_W1 or v68_config.V91_W2
if v68_config.V91_W2:
    KINDS = (('t', 1), ('u', 1), ('p', 12), ('s', 12))
if v68_config.V101_W2:
    KINDS = KINDS + (('ux', 1),)
LAST_KINDS = {}
LAST_V110 = {}
LAST_KEEP = None
EXACT_ITERS = 300
EXACT_MIN_ITERS = 12
#  W1 wave split of the wb_lca window (fractions of it) and per-kind EM caps.
V115_WAVE1 = 0.43
V115_WAVE2 = 0.76
V115_T_ITERS = int(os.environ.get('V115_T_ITERS', EXACT_ITERS))
V115_U_ITERS = int(os.environ.get('V115_U_ITERS', EXACT_ITERS))
#  W3 trait fits end where the  12.5-second window ended them (12.5 / 24 of
# the W3 window) and otherwise run to TRAIT_TOL, as in : a 40-step cap cost rulers
# A and B 5e-5 to 1.6e-4 ( round 2), where t1 converges in 99 to 119 steps.
V115_W3_TRAIT = 12.5 / 24.0
V115_W3_ITERS = int(os.environ.get('V115_W3_ITERS', EXACT_ITERS))
_ITERS = [0]
# 'bound': per-column Newton under the diag(n P) curvature bound; 'exact': full block Hessian.
TRAIT_STEP = 'exact'
# Posterior mass below this is zero: float32 denormals run x86 matmuls about 100x slower.
FLUSH = 1e-30
# Per-row log-likelihood gain below which a trait fit stops.
TRAIT_TOL = 1e-5
OFFSET_RIDGE = 2.0
# A block of same-gated targets answered with one shared label all at once (tracer
# non-response) on at least this share of its open rows becomes one latent indicator.
JOINT_SHARE = 0.05
JOINT_MIN_ITEMS = 5
STRATUM = 'group'
TAU = 10.0
ITERS = 60
ALPHA = 2.0
TOL = 2e-6
MIX = 1e-4
WEIGHT_ITERS = 40
TRAIT_NODES = {1: 41}
TRAIT_ITERS = 80
RIDGE = 0.25


class _Expired(TimeoutError):
    pass


# Forked fit workers inherit this closure; only the task tuple crosses the pipe.
_TASK = None


def _fork_run(task):
    began = time.monotonic()
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            found = _TASK(task)
    except Exception:
        found = None
    return task, found, round(time.monotonic() - began, 3)


def _fork_call(key, payload):
    began = time.monotonic()
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            found = _TASK(key, payload)
    except Exception:
        found = None
    return key, found, round(time.monotonic() - began, 3)


#  W3 dispatch order: trait fits first in 's order (t1 before u1, so a slow
# host still completes t1), class kinds after them.
_CHAIN = {'t': 0, 'u': 1, 'ux': 1, 's': 2, 'p': 3}


def _pipeline(run, run_extra, tasks, workers, deadline, spent, with_extra):
    """Stage 1 and Stage 2 on one forked pool ( W3). A fold's direct-effect fit
    starts as soon as its trait fit returns and goes ahead of every queued Stage-1
    fit, so no core waits for the slowest Stage-1 fit before Stage 2 begins."""
    global _TASK
    import queue

    def call(key, payload):
        return run(key[1:]) if key[0] == 'base' else run_extra(key[1:], payload)

    _TASK = call
    base_done = {task: None for task in tasks}
    extra_done = {}
    pending = sorted(tasks, key=lambda x: _CHAIN.get(x[1], 4))
    ready = []
    finished = queue.Queue()
    width = max(1, min(workers, len(tasks)))
    with multiprocessing.get_context('fork').Pool(processes=width) as pool:
        busy = 0

        def fill():
            nonlocal busy
            while busy < width and (pending or ready):
                key, payload = (('base',) + pending.pop(0), None) if pending else ready.pop(0)
                pool.apply_async(_fork_call, (key, payload), callback=finished.put,
                                 error_callback=lambda exc, key=key: finished.put((key, None, None)))
                busy += 1

        fill()
        while busy:
            try:
                key, found, seconds = finished.get(timeout=max(0.5, deadline - time.monotonic() + 1.0))
            except queue.Empty:
                break
            busy -= 1
            task = key[1:]
            if key[0] == 'base':
                base_done[task] = found
                if found is not None:
                    spent.setdefault(f'{task[1]}{task[2]}', []).append(seconds)
                    if with_extra and task[1] in ('t', 'u'):
                        ready.append((('extra',) + task, (found[2], found[3])))
            else:
                extra_done[task] = found
            fill()
    _TASK = None
    return base_done, extra_done


def _options(items, name):
    values = list(items[name]['values'])
    return values + ['__gated__'] if items[name].get('gate') else values


def _layout_exact(Xg, levels, given, items, work):
    """_layout plus nesting: (child begin, K, parent columns, fill code, renormalize) in
    composition order. A target gated on another target is observed only where that
    parent is open; a same-gated block answered with one shared label all at once gets
    a latent two-level indicator column, and its members are observed only where it is 0."""
    O, spans, n_given_cols, targets = _layout(Xg, levels, given, items, work)
    n = len(Xg)
    raw = list(given[:Xg.shape[1]])
    begin = {name: targets[name][0] for name in targets}
    train = next(iter(work.values()))['train']
    joint, nest = [], []
    signature = {}
    for name in work:
        gate = items[name].get('gate')
        if isinstance(gate, dict) and gate.get('parent') in raw and targets[name][2].any():
            signature.setdefault((gate['parent'], tuple(gate.get('observed_if', ()))), []).append(name)
    extra = []
    for key, members in signature.items():
        if len(members) < JOINT_MIN_ITEMS:
            continue
        open_rows = train & ~targets[members[0]][2]
        if open_rows.sum() < 20:
            continue
        shared = set(items[members[0]]['values'])
        for name in members[1:]:
            shared &= set(items[name]['values'])
        best, share = None, 0.0
        for label in sorted(shared):
            every = np.ones(n, bool)
            for name in members:
                every &= work[name]['y'] == _options(items, name).index(label)
            s = float(every[open_rows].mean())
            if s > share:
                best, share = (label, every & open_rows), s
        if best is None or share < JOINT_SHARE:
            continue
        label, every = best
        z = np.zeros((n, 2), np.float32)
        z[open_rows & ~every, 0] = 1.0
        z[every, 1] = 1.0
        at = O.shape[1] + sum(b.shape[1] for b in extra)
        extra.append(z)
        spans.append((at, at + 2))
        for name in members:
            b, K = begin[name], targets[name][1]
            O[every, b:b + K] = 0.0
            nest.append((b, K, [at], _options(items, name).index(label), False))
        joint.append({'members': len(members), 'label': label, 'share': round(share, 4)})
    if extra:
        O = np.hstack([O] + extra)
    for name in work:
        gate = items[name].get('gate')
        parent = gate.get('parent') if isinstance(gate, dict) else None
        if parent not in work or parent not in begin:
            continue
        y, K = work[name]['y'], work[name]['K']
        options = _options(items, parent)
        open_codes = [options.index(v) for v in gate.get('observed_if', ()) if v in options]
        shut = train & ~np.isin(work[parent]['y'], open_codes)
        if not shut.any() or not np.all(y[shut] == K - 1):
            continue
        b = begin[name]
        O[shut, b:b + K] = 0.0
        nest.append((b, K, [begin[parent] + c for c in open_codes], K - 1, True))
        b0, K0, closed = targets[name]
        parent_closed = targets[parent][2]
        if parent_closed.any() and np.all(y[train & parent_closed] == K - 1):
            targets[name] = (b0, K0, closed | parent_closed)
    return O, spans, n_given_cols, targets, nest, joint


def _compose(T, nest):
    """P(child | class) = P(parent open | class) P(child | open, class), rest on the fill code."""
    for b, K, parent_cols, fill, renorm in nest:
        po = T[..., parent_cols].sum(-1, keepdims=True)
        block = T[..., b:b + K]
        if renorm:
            inner = block.copy()
            inner[..., K - 1] = 0.0
            inner /= np.maximum(inner.sum(-1, keepdims=True), 1e-30)
        else:
            inner = block
        out = po * inner
        out[..., fill] += 1.0 - po[..., 0]
        T[..., b:b + K] = out
    return T


def _layout(Xg, levels, given, items, work):
    """Dense one-hot indicators: GIVEN blocks first, then one block per target."""
    n = len(Xg)
    raw = list(given[:Xg.shape[1]])
    cols, widths = [], []
    for j in range(Xg.shape[1]):
        L = int(max(levels[j], Xg[:, j].max() + 1))
        block = np.zeros((n, L), np.float32)
        block[np.arange(n), Xg[:, j]] = 1.0
        cols.append(block)
        widths.append(L)
    n_given_cols = sum(widths)
    targets = {}
    for name, j in work.items():
        y, K, train = j['y'], j['K'], j['train']
        observed = train.copy()
        closed = np.zeros(n, bool)
        gate = items[name].get('gate')
        if isinstance(gate, dict) and gate.get('parent') in raw:
            p = raw.index(gate['parent'])
            values = list(items[gate['parent']]['values'])
            open_codes = [values.index(v) for v in gate.get('observed_if', ()) if v in values]
            shut = ~np.isin(Xg[:, p], open_codes)
            seen = shut & train
            if seen.any() and np.all(y[seen] == K - 1):
                closed = shut
                observed &= ~closed
        block = np.zeros((n, K), np.float32)
        rows = np.flatnonzero(observed)
        block[rows, y[rows]] = 1.0
        targets[name] = (sum(widths), K, closed)
        cols.append(block)
        widths.append(K)
    edges = np.cumsum([0] + widths)
    return np.hstack(cols), [(int(edges[i]), int(edges[i + 1])) for i in range(len(widths))], n_given_cols, targets


def _normalize(c, blocks):
    for a, b in blocks:
        c[..., a:b] /= c[..., a:b].sum(-1, keepdims=True)
    return c


def _fit(O, blocks, rows, C, seed, deadline, strata=None, G=1, local=None):
    """EM on rows; returns (log emissions G x C x D, log class weights)."""
    Ot = O[rows]
    m = len(rows)
    rng = np.random.default_rng(seed)
    marg = _normalize(Ot.sum(0) + 1e-3, blocks)
    # Seed every class on one respondent's answer pattern, half-shrunk to the margin.
    picks = rng.choice(m, size=C, replace=False)
    theta = _normalize(0.5 * Ot[picks] + 0.5 * marg[None, :], blocks)
    log_g = np.log(np.maximum(theta, 1e-8)).astype(np.float32)[None]
    groups = None
    if G > 1:
        g = strata[rows]
        groups = [np.flatnonzero(g == s) for s in range(G)]
        log_g = np.repeat(log_g, G, 0)
    log_pi = np.full(C, -np.log(C), np.float32)
    L = np.empty((m, C), np.float32)
    prev = -np.inf
    for it in range(ITERS):
        if time.monotonic() >= deadline:
            #  W3: a class fit past EXACT_MIN_ITERS EM steps keeps its current M step.
            if v68_config.V115_W3 and it >= EXACT_MIN_ITERS:
                break
            raise _Expired()
        if groups is None:
            np.matmul(Ot, log_g[0].T, out=L)
        else:
            for s, idx in enumerate(groups):
                if len(idx):
                    L[idx] = Ot[idx] @ log_g[s].T
        L += log_pi
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        if EXACT:
            E[E < FLUSH] = 0.0
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        if groups is None:
            counts = R.T @ Ot
            theta = _normalize(counts + ALPHA * marg[None, :], blocks)
            log_g = np.log(np.maximum(theta, 1e-12)).astype(np.float32)[None]
        else:
            counts = np.zeros((G, C, Ot.shape[1]), np.float32)
            for s, idx in enumerate(groups):
                if len(idx):
                    counts[s] = R[idx].T @ Ot[idx]
            theta = _normalize(counts.sum(0) + ALPHA * marg[None, :], blocks)
            own = _normalize(counts + TAU * theta[None], blocks)
            own[:, :, ~local] = theta[None][:, :, ~local]
            log_g = np.log(np.maximum(own, 1e-12)).astype(np.float32)
        log_pi = np.log((R.sum(0) + 1.0) / (m + C)).astype(np.float32)
        if ll - prev < TOL * m:
            break
        prev = ll
    return log_g, log_pi


def _grid(dim):
    axis = np.linspace(-4.0, 4.0, TRAIT_NODES[dim])
    nodes = np.stack(np.meshgrid(*([axis] * dim), indexing='ij'), -1).reshape(-1, dim)
    lw = -0.5 * (nodes ** 2).sum(1)
    top = lw.max()
    return nodes, lw - top - np.log(np.exp(lw - top).sum())


def _block_lse(Z, starts, widths):
    top = np.maximum.reduceat(Z, starts, axis=1)
    s = np.add.reduceat(np.exp(Z - np.repeat(top, widths, 1)), starts, axis=1)
    return np.repeat(top + np.log(s), widths, 1)


def _fit_trait(O, blocks, rows, dim, seed, deadline):
    """Nominal-response trait model on a quadrature grid: every column block's
    answer logits are base + trait . loading, with a standard normal trait.
    Returns the grid's log emissions and log prior in _fit's layout."""
    Ot = O[rows]
    m, D = Ot.shape
    starts = np.array([a for a, _ in blocks])
    widths = np.array([b - a for a, b in blocks])
    nodes, lw = _grid(dim)
    obs = (np.add.reduceat(Ot, starts, axis=1) > 0).astype(np.float32)
    B = np.log(_normalize(Ot.sum(0).astype(np.float64) + 1e-3, blocks))
    A = np.zeros((dim, D))
    # Start the posterior around the leading principal scores of the indicators.
    rng = np.random.default_rng(seed)
    Xc = Ot - Ot.mean(0)
    V = rng.normal(size=(D, dim))
    for _ in range(6):
        V, _ = np.linalg.qr(Xc.T @ (Xc @ V))
    S = Xc @ V
    S = (S - S.mean(0)) / np.maximum(S.std(0), 1e-9)
    R = -0.5 * ((S[:, None, :] - nodes[None]) ** 2).sum(-1) / 0.36 + lw[None]
    R = np.exp(R - R.max(1, keepdims=True))
    R = (R / R.sum(1, keepdims=True)).astype(np.float32)
    X1 = np.column_stack([np.ones(len(nodes)), nodes])
    ridge = np.diag([1e-3] + [RIDGE] * dim)
    prev = -np.inf
    for _ in range(TRAIT_ITERS):
        if time.monotonic() >= deadline:
            raise _Expired()
        N = (R.T @ Ot).astype(np.float64)
        n = np.repeat((R.T @ obs).astype(np.float64), widths, 1)
        for _ in range(2):
            Z = B[None] + nodes @ A
            P = np.exp(Z - _block_lse(Z, starts, widths))
            G = N - n * P
            W = n * P * (1.0 - P) + 1e-6
            # One (dim + 1)-parameter Newton step per answer column.
            g = X1.T @ G - np.vstack([np.zeros((1, D)), RIDGE * A])
            H = np.einsum('qa,qb,qd->dab', X1, X1, W) + ridge
            # The diagonal-category Hessian overshoots on rare answers; cap every step.
            step = np.clip(np.linalg.solve(H, g.T[:, :, None])[:, :, 0], -1.0, 1.0)
            B = np.clip(B + step[:, 0], -25.0, 25.0)
            A = np.clip(A + step[:, 1:].T, -8.0, 8.0)
        Z = B[None] + nodes @ A
        log_t = (Z - _block_lse(Z, starts, widths)).astype(np.float32)
        L = Ot @ log_t.T + lw.astype(np.float32)
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        if ll - prev < TOL * m:
            break
        prev = ll
    return log_t[None], lw.astype(np.float32)


def _block_index(blocks):
    starts = np.array([a for a, _ in blocks])
    widths = np.array([b - a for a, b in blocks])
    return starts, widths, [(w, np.flatnonzero(widths == w)) for w in np.unique(widths)]


def _softmax_newton(theta, grad, curve, lam, by_width, starts):
    """One exact multinomial-logit Newton step per answer block, batched by block width.

    theta: list of parameter rows (each length D) sharing design x_r; grad: same shapes;
    curve: (x_r products, weights n_rb, probabilities P_rd) -> Hessian sum_r n (diag P - P P^T) x x^T.
    """
    X, nrb, P = curve
    p = len(theta)
    out = [t.copy() for t in theta]
    for w, idx in by_width:
        cols = (starts[idx][:, None] + np.arange(w)[None]).ravel()
        Pb = P[:, cols].reshape(len(P), len(idx), w)
        H = np.empty((len(idx), p, w, p, w))
        eye = np.eye(w)[None]
        for i in range(p):
            for j in range(i, p):
                Wt = (X[:, i] * X[:, j])[:, None] * nrb[:, idx]
                Wp = Pb * Wt[:, :, None]
                diag = Wp.sum(0)
                h = diag[:, :, None] * eye - np.einsum('rbk,rbl->bkl', Wp, Pb)
                H[:, i, :, j, :] = h
                if j != i:
                    H[:, j, :, i, :] = h.transpose(0, 2, 1)
        H = H.reshape(len(idx), p * w, p * w)
        H[:, np.arange(p * w), np.arange(p * w)] += np.repeat(lam, w)[None]
        g = np.concatenate([gr[cols].reshape(len(idx), w) for gr in grad], 1)
        step = np.clip(np.linalg.solve(H, g[:, :, None])[:, :, 0], -1.0, 1.0)
        for i in range(p):
            out[i][cols] += step[:, i * w:(i + 1) * w].ravel()
    return out


def _bound_step(theta, grad, X, W, lam):
    """Per-column Newton step under the curvature bound diag(n P) >= n (diag P - P P^T):
    never overshoots a softmax block, and costs one small solve per column."""
    p = len(theta)
    H = np.einsum('ra,rb,rd->dab', X, X, W)
    H[:, np.arange(p), np.arange(p)] += lam[None]
    g = np.stack(grad, 1)
    if p == 1:
        step = g[:, 0] / H[:, 0, 0]
        return [theta[0] + np.clip(step, -1.0, 1.0)]
    step = np.clip(np.linalg.solve(H, g[:, :, None])[:, :, 0], -1.0, 1.0)
    return [theta[i] + step[:, i] for i in range(p)]


def _fit_trait_exact(O, blocks, rows, seed, deadline, strata=None, G=1, local=None, interaction=False, max_iters=EXACT_ITERS):
    """One-trait nominal-response model, every M-step an exact softmax Newton step on
    (base, loading) and, with strata, on per-stratum target intercepts shrunk to zero.
    Runs until the log likelihood stops rising; a window that closes after
    EXACT_MIN_ITERS keeps the current fit."""
    Ot = O[rows]
    m, D = Ot.shape
    starts, widths, by_width = _block_index(blocks)
    nodes, lw = _grid(1)
    t = nodes[:, 0]
    q = len(t)
    obs = (np.add.reduceat(Ot, starts, axis=1) > 0).astype(np.float32)
    B = np.log(_normalize(Ot.sum(0).astype(np.float64) + 1e-3, blocks))
    A = np.zeros(D)
    S = 1 if G <= 1 else G
    C = np.zeros((S, D))
    U = np.zeros((S, D))
    groups = [np.arange(m)] if S == 1 else [np.flatnonzero(strata[rows] == s) for s in range(S)]
    # Row blocks per stratum are fixed for the whole fit.
    O_g = [Ot] if S == 1 else [Ot[g] for g in groups]
    obs_g = [obs] if S == 1 else [obs[g] for g in groups]
    rng = np.random.default_rng(seed)
    Xc = Ot - Ot.mean(0)
    V = rng.normal(size=(D, 1))
    for _ in range(6):
        V, _ = np.linalg.qr(Xc.T @ (Xc @ V))
    sc = Xc @ V
    sc = (sc - sc.mean(0)) / np.maximum(sc.std(0), 1e-9)
    R = -0.5 * (sc - t[None]) ** 2 / 0.36 + lw[None]
    R = np.exp(R - R.max(1, keepdims=True))
    R = (R / R.sum(1, keepdims=True)).astype(np.float32)
    R[R < FLUSH] = 0.0
    lam_ba = np.array([1e-3, RIDGE])
    lam_c = np.array([OFFSET_RIDGE])
    X1 = np.column_stack([np.ones(S * q), np.tile(t, S)])
    prev = -np.inf
    log_t = None
    for it in range(max_iters):
        if time.monotonic() >= deadline:
            if it < EXACT_MIN_ITERS or log_t is None:
                raise _Expired()
            break
        _ITERS[0] = it + 1
        R_g = [R] if S == 1 else [R[g] for g in groups]
        N = np.concatenate([(r.T @ o).astype(np.float64) for r, o in zip(R_g, O_g)])
        nb = np.concatenate([(r.T @ o).astype(np.float64) for r, o in zip(R_g, obs_g)])
        n = np.repeat(nb, widths, 1)
        Z = (B[None, None] + t[None, :, None] * A[None, None] + C[:, None] + t[None, :, None] * U[:, None]).reshape(S * q, D)
        P = np.exp(Z - _block_lse(Z, starts, widths))
        Gd = N - n * P
        gB = Gd.sum(0) - lam_ba[0] * B
        gA = (X1[:, 1:2] * Gd).sum(0) - lam_ba[1] * A
        if TRAIT_STEP == 'exact':
            B, A = _softmax_newton([B, A], [gB, gA], (X1, nb, P), lam_ba, by_width, starts)
        else:
            B, A = _bound_step([B, A], [gB, gA], X1, n * P, lam_ba)
        B = np.clip(B, -25.0, 25.0)
        A = np.clip(A, -8.0, 8.0)
        if S > 1:
            Z = (B[None, None] + t[None, :, None] * A[None, None] + C[:, None] + t[None, :, None] * U[:, None]).reshape(S * q, D)
            P = np.exp(Z - _block_lse(Z, starts, widths))
            Gd = (N - n * P).reshape(S, q, D)
            Pg = P.reshape(S, q, D)
            for s in range(S):
                gC = Gd[s].sum(0) - lam_c[0] * C[s]
                # Intercepts take the diagonal-bound step: one division per column.
                if interaction:
                    gU = (t[:, None] * Gd[s]).sum(0) - OFFSET_RIDGE * U[s]
                    C[s], U[s] = _bound_step([C[s],U[s]], [gC,gU], np.column_stack((np.ones(q),t)), n.reshape(S,q,D)[s]*Pg[s], np.array([OFFSET_RIDGE,OFFSET_RIDGE]))
                else:
                    (C[s],) = _bound_step([C[s]], [gC], np.ones((q, 1)), (n.reshape(S, q, D)[s] * Pg[s]), lam_c)
            C[:, ~local] = 0.0
            U[:, ~local] = 0.0
            U = np.clip(U, -6.0, 6.0)
            C = np.clip(C, -6.0, 6.0)
        Z = (B[None, None] + t[None, :, None] * A[None, None] + C[:, None] + t[None, :, None] * U[:, None]).reshape(S * q, D)
        log_t = (Z - _block_lse(Z, starts, widths)).astype(np.float32).reshape(S, q, D)
        L = np.empty((m, q), np.float32)
        for s, g in enumerate(groups):
            if len(g):
                L[g] = O_g[s] @ log_t[s].T
        L += lw.astype(np.float32)
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        E[E < FLUSH] = 0.0
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        if 0.0 <= ll - prev < TRAIT_TOL * m:
            break
        prev = ll
    return log_t, lw.astype(np.float32)


def _posterior(O, n_given_cols, log_g, log_pi):
    # GIVEN emissions are pooled in every kind, so stratum 0 carries them.
    Q = O[:, :n_given_cols] @ log_g[0, :, :n_given_cols].T + log_pi
    Q -= Q.max(1, keepdims=True)
    np.exp(Q, out=Q)
    if EXACT:
        Q[Q < FLUSH] = 0.0
    Q /= Q.sum(1, keepdims=True)
    return Q


def _emit(Q, T, idx, start, strata):
    """Every target block's answer distribution of one fitted model on rows idx."""
    if len(T) == 1:
        return Q[idx] @ T[0, :, start:]
    P = np.empty((len(idx), T.shape[2] - start), np.float32)
    st = strata[idx]
    for s in range(len(T)):
        sel = np.flatnonzero(st == s)
        if len(sel):
            P[sel] = Q[idx[sel]] @ T[s, :, start:]
    return P


def _mixture_weights_newton(likes, groups, em=10, steps=8):
    """Same objective as _mixture_weights: a few EM steps from uniform, then per subset
    active-set Newton on the simplex (KKT solve, feasible step, halving on a drop)."""
    k = likes.shape[1]
    w = np.full((len(groups), k), 1.0 / k)
    counts = np.maximum(groups.sum(1, keepdims=True), 1.0)
    for _ in range(em):
        ratio = groups.T / np.maximum(likes @ w.T, 1e-300)
        w = w * (ratio.T @ likes) / counts
        w /= w.sum(1, keepdims=True)
    for s in range(len(groups)):
        sel = groups[s] > 0
        Ls = likes[sel]
        ws = w[s].copy()
        f = np.log(np.maximum(Ls @ ws, 1e-300)).mean()
        for _ in range(steps):
            act = np.flatnonzero(ws > 1e-9)
            if len(act) < 2:
                break
            La = Ls[:, act]
            r = 1.0 / np.maximum(Ls @ ws, 1e-300)
            g = La.T @ r / len(Ls)
            Lr = La * r[:, None]
            H = -(Lr.T @ Lr) / len(Ls)
            a = len(act)
            K = np.zeros((a + 1, a + 1))
            K[:a, :a] = H
            K[:a, a] = K[a, :a] = 1.0
            try:
                d = np.linalg.solve(K, np.r_[-g, 0.0])[:a]
            except np.linalg.LinAlgError:
                break
            neg = d < 0
            alpha = min(1.0, float(np.min(-ws[act][neg] / d[neg]))) if neg.any() else 1.0
            for _ in range(20):
                trial = ws.copy()
                trial[act] = np.maximum(ws[act] + alpha * d, 0.0)
                trial /= trial.sum()
                ft = np.log(np.maximum(Ls @ trial, 1e-300)).mean()
                if ft >= f:
                    break
                alpha *= 0.5
            else:
                break
            done = ft - f < 1e-12
            ws, f = trial, ft
            if done:
                break
        w[s] = ws
    return w


def _mixture_weights(likes, groups):
    """Simplex weights maximizing the mean log mixed likelihood on each row
    subset (EM). likes: rows x kinds; groups: subsets x rows 0/1."""
    w = np.full((len(groups), likes.shape[1]), 1.0 / likes.shape[1])
    counts = np.maximum(groups.sum(1, keepdims=True), 1.0)
    for _ in range(WEIGHT_ITERS):
        ratio = groups.T / np.maximum(likes @ w.T, 1e-300)
        w = w * (ratio.T @ likes) / counts
        w /= w.sum(1, keepdims=True)
    return w


def attach(Xg, levels, given, items, work, fold, deadline, workers=8):
    """Returns ({name: (oof in train order, full n x K)}, info).

    Fits that miss the deadline drop out; the completed kinds still mix.
    """
    global LAST_KINDS, LAST_V110, LAST_KEEP
    LAST_KINDS = {}
    LAST_V110 = {}
    LAST_KEEP = None
    start = time.monotonic()
    raw = list(given[:Xg.shape[1]])
    kinds = [(k, C) for k, C in KINDS if k not in ('s', 'u', 'ux') or STRATUM in raw]
    info = {'kinds': [f'{k}{C}' for k, C in kinds], 'fits': 0, 'seconds': 0.0, 'status': 'complete', 'tau': TAU}
    train = next(iter(work.values()))['train'] if work else None
    if not work or not kinds or time.monotonic() >= deadline or (fold[train] < 0).any():
        info['status'] = 'no_budget'
        return {}, info
    nest = []
    if EXACT:
        O, blocks, n_given_cols, targets, nest, info['joint'] = _layout_exact(Xg, levels, given, items, work)
        info['nested'] = len(nest)
    else:
        O, blocks, n_given_cols, targets = _layout(Xg, levels, given, items, work)
    strata, G, local = None, 1, None
    if any(k in ('s', 'u', 'ux') for k, _ in kinds):
        si = raw.index(STRATUM)
        strata = Xg[:, si].astype(np.int64)
        G = int(max(levels[si], strata.max() + 1))
        local = np.zeros(O.shape[1], bool)
        local[n_given_cols:] = True
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    tasks = [(f, k, C) for k, C in kinds for f in folds]
    task_deadline, caps = {}, {}
    if v68_config.V115_W1:
        # Class kinds converge in a fraction of a second, so they go first; t1 then runs
        # to the first wave line, u1 to the second, and the direct effects keep the rest.
        span = max(0.0, deadline - start)
        rank = {'p': 0, 's': 0, 't': 1}
        tasks = sorted(tasks, key=lambda x: rank.get(x[1], 2))
        task_deadline = {x: start + (V115_WAVE2 if rank.get(x[1], 2) == 2 else V115_WAVE1) * span for x in tasks}
        caps = {'t': V115_T_ITERS, 'u': V115_U_ITERS, 'ux': V115_U_ITERS}
        info['waves'] = [round(V115_WAVE1 * span, 3), round(V115_WAVE2 * span, 3), round(span, 3)]
    elif v68_config.V115_W3:
        caps = {'t': V115_W3_ITERS, 'u': V115_W3_ITERS, 'ux': V115_W3_ITERS}
        trait_stop = start + V115_W3_TRAIT * max(0.0, deadline - start)
        task_deadline = {x: trait_stop for x in tasks if x[1] in caps}
    spent = {}

    def run(task):
        f, kind, C = task
        rows = np.flatnonzero((fold >= 0) & (fold != f))
        if len(rows) < 2 * C:
            return None
        began = time.monotonic()
        stop = task_deadline.get(task, deadline)
        cap = caps.get(kind, EXACT_ITERS)
        _ITERS[0] = 0
        try:
            if kind == 't' and EXACT:
                log_g, log_pi = _fit_trait_exact(O, blocks, rows, 20261108 + 97 * f + C, stop, max_iters=cap)
            elif kind in ('u','ux'):
                log_g, log_pi = _fit_trait_exact(O, blocks, rows, 20261109 + 97 * f + C, stop, strata, G, local, interaction=(kind=='ux'), max_iters=cap)
            elif kind == 't':
                log_g, log_pi = _fit_trait(O, blocks, rows, C, 20261108 + 97 * f + C, stop)
            elif kind == 'p':
                log_g, log_pi = _fit(O, blocks, rows, C, 20261106 + 97 * f + C, stop)
            else:
                log_g, log_pi = _fit(O, blocks, rows, C, 20261107 + 97 * f + C, stop, strata, G, local)
        except _Expired:
            return None
        spent.setdefault(f'{kind}{C}', []).append(round(time.monotonic() - began, 3))
        T = np.exp(log_g)
        if EXACT:
            T[T < FLUSH] = 0.0
        if nest:
            T = _compose(T, nest)
        Q = _posterior(O, n_given_cols, log_g, log_pi)
        return Q, T, log_g, log_pi, _ITERS[0]

    def _map_tasks(fn, items):
        if not items:
            return {}
        if EXACT and sys.platform.startswith('linux'):
            # Forked workers share O read-only and keep the Python half of every EM step
            # off one GIL; a fit still running at the deadline is dropped with its pool.
            global _TASK
            _TASK = fn
            out = {task: None for task in items}
            # The parent's BLAS pools are never resized around the fork (a resize there
            # leaves scipy's later L-BFGS-B calls segfaulting); each child limits itself.
            with multiprocessing.get_context('fork').Pool(processes=max(1, min(workers, len(items)))) as pool:
                results = pool.imap_unordered(_fork_run, items, chunksize=1)
                for _ in items:
                    try:
                        task, found, seconds = results.next(timeout=max(0.5, deadline - time.monotonic() + 1.0))
                    except (StopIteration, multiprocessing.TimeoutError):
                        break
                    out[task] = found
                    if found is not None and fn is run:
                        spent.setdefault(f'{task[1]}{task[2]}', []).append(seconds)
            _TASK = None
            return out
        with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
            return dict(zip(items, pool.map(fn, items)))

    def run_extra(task, fitted=None):
        f, kind, C = task
        if time.monotonic() >= deadline:
            return None, None
        rows = np.flatnonzero((fold >= 0) & (fold != f))
        log_g, log_pi = fitted if fitted is not None else base_done[task][2:4]
        extra_started = time.monotonic()
        from . import wb_trait_effects as fx
        try:
            pairs = tuple((raw.index(STRATUM), a) for a in range(len(raw)) if raw[a] != STRATUM) if STRATUM in raw else ()
            off, inv, stratum = fx._direct_effects(O, blocks, n_given_cols, Xg, rows, log_g, log_pi,
                strata if kind == 'u' else None, deadline, pairs=pairs, rank=3, refresh=True)
            extra = fx.predict_effects(O, n_given_cols, log_g, log_pi, off, inv, stratum, blocks, targets, nest)
        except fx._Expired:
            extra = None
        return extra, time.monotonic() - extra_started

    if v68_config.V115_W3 and EXACT and sys.platform.startswith('linux'):
        base_done, extra_done = _pipeline(run, run_extra, tasks, workers, deadline, spent, v68_config.V110_W1)
        mixed = [(k, C) for k, C in kinds if all(base_done[(f, k, C)] is not None for f in folds)]
        if not mixed:
            info.update(status='budget_expired', seconds=time.monotonic() - start)
            return {}, info
    else:
        base_done = _map_tasks(run, tasks)
        mixed = [(k, C) for k, C in kinds if all(base_done[(f, k, C)] is not None for f in folds)]
        if not mixed:
            info.update(status='budget_expired', seconds=time.monotonic() - start)
            return {}, info
        extra_tasks = [(f, k, C) for k, C in mixed if k in ('t', 'u') for f in folds] if v68_config.V110_W1 else []
        extra_done = _map_tasks(run_extra, extra_tasks) if extra_tasks else {}
    done = {}
    for task, val in base_done.items():
        if val is None:
            done[task] = None
        else:
            Q, T = val[:2]
            ex = extra_done.get(task)
            done[task] = (Q, T, ex[0] if ex else None, ex[1] if ex else None)
    info['fit_task_iters'] = {f'{k}{C}/fold{f}': v[4] for (f, k, C), v in base_done.items() if v is not None and k in ('t', 'u', 'ux')}
    del base_done, extra_done
    info['direct_effects'] = {f'{k}{C}/fold{f}': {'complete': value is not None and value[2] is not None,
        'seconds': value[3] if value is not None else None} for (f,k,C),value in done.items() if k in ('t','u')}
    info['direct_effects_completed'] = sum(x['complete'] for x in info['direct_effects'].values())
    info['fits'] = sum(v is not None for v in done.values())
    info['fit_task_seconds'] = spent
    info['mixed'] = [f'{k}{C}' for k, C in mixed]
    info['fit_seconds'] = time.monotonic() - start
    n = len(Xg)
    # Every item shares the training rows. One product per kind and fold gives
    # every target's out-of-fold rows and that fold's share of the hidden rows.
    rows = np.flatnonzero(train)
    rest = np.flatnonzero(~train)
    rf = fold[rows]
    oof_all = np.empty((len(mixed), len(rows), O.shape[1] - n_given_cols), np.float32)
    hid_all = np.zeros((len(mixed), len(rest), O.shape[1] - n_given_cols), np.float32)
    for ki, (k, C) in enumerate(mixed):
        for f in folds:
            Q, T = done[(f, k, C)][:2]
            hit = np.flatnonzero(rf == f)
            oof_all[ki, hit] = _emit(Q, T, rows[hit], n_given_cols, strata)
            hid_all[ki] += _emit(Q, T, rest, n_given_cols, strata) / len(folds)
    extra_oof = oof_all.copy() if v68_config.V110_W1 else None
    extra_hid = hid_all.copy() if v68_config.V110_W1 else None
    complete_extra = []
    for ki, (k,C) in enumerate(mixed):
        if extra_oof is not None and all(done[(f,k,C)][2] is not None for f in folds):
            complete_extra.append(k)
            extra_hid[ki] = 0
            for f in folds:
                hit = np.flatnonzero(rf == f)
                P = done[(f,k,C)][2]
                extra_oof[ki,hit] = P[rows[hit]]
                extra_hid[ki] += P[rest] / len(folds)
    expected_extra = [k for k, _ in mixed if k in ('t', 'u')]
    if len(complete_extra) < len(expected_extra):
        complete_extra = []
    kept = {f:done[(f,'u',1)][:2] for f in folds} if v68_config.V110_W2 and ('u',1) in mixed else None
    kept_v127 = (
        {f'{k}{C}': {int(f): done[(f, k, C)][:2] for f in folds} for k, C in mixed if all(done.get((f, k, C)) is not None for f in folds)},
        [f'{k}{C}' for k, C in mixed if all(done.get((f, k, C)) is not None for f in folds)],
        targets,
        n_given_cols,
        strata,
    ) if v68_config.V127_W2 else None
    del done
    direct, reopt, union = {}, {}, {}
    w_by_item = {}
    info['v110_completed_effect_kinds'] = complete_extra
    #  W1: the Stage-1 arms that a completed direct-effect arm replaces join the union.
    swapped = [ki for ki, (k, _) in enumerate(mixed) if k in complete_extra]
    # Each fold's rows mix with weights learned on the other folds only; the
    # last subset is every row and weighs the hidden rows.
    groups = np.vstack([(rf != f).astype(np.float64) for f in folds] + [np.ones(len(rows))])
    found, weights = {}, {}
    for name, j in work.items():
        if not np.array_equal(j['train'], train):
            continue
        begin, K, closed = targets[name]
        a = begin - n_given_cols
        oof_c, hid_c = oof_all[:, :, a:a + K], hid_all[:, :, a:a + K]
        if closed.any():
            oof_c, hid_c = oof_c.copy(), hid_c.copy()
            for c, part in ((oof_c, closed[rows]), (hid_c, closed[rest])):
                c[:, part] = 0.0
                c[:, part, -1] = 1.0
        y = np.maximum(j['y'][rows], 0)
        likes = oof_c[:, np.arange(len(rows)), y].T.astype(np.float64)
        w = _mixture_weights_newton(likes, groups) if EXACT else _mixture_weights(likes, groups)
        oof = np.empty((len(rows), K), np.float32)
        for i, f in enumerate(folds):
            hit = rf == f
            oof[hit] = np.tensordot(w[i], oof_c[:, hit], 1)
        full = np.zeros((n, K), np.float32)
        full[rows] = oof
        full[rest] = np.tensordot(w[-1], hid_c, 1)
        found[name] = ((1.0 - MIX) * oof + MIX / K, (1.0 - MIX) * full + MIX / K)
        if v68_config.V101_W2:
            LAST_KINDS[name] = {}
            for ki,(kind,count) in enumerate(mixed):
                fp=np.zeros((n,K),np.float32);fp[rows]=oof_c[ki];fp[rest]=hid_c[ki]
                LAST_KINDS[name][f'{kind}{count}']=((1-MIX)*oof_c[ki]+MIX/K,(1-MIX)*fp+MIX/K)
        if complete_extra:
            eo, eh = extra_oof[:,:,a:a+K].copy(), extra_hid[:,:,a:a+K].copy()
            if closed.any():
                for c, part in ((eo, closed[rows]),(eh,closed[rest])):
                    c[:,part]=0; c[:,part,-1]=1
            def mix(wk, so, sh):
                co=np.empty_like(oof)
                for i,f in enumerate(folds):
                    hit=rf==f; co[hit]=np.tensordot(wk[i],so[:,hit],1)
                cp=np.zeros_like(full);cp[rows]=co;cp[rest]=np.tensordot(wk[-1],sh,1)
                return ((1-MIX)*co+MIX/K,(1-MIX)*cp+MIX/K)
            direct[name]=mix(w, eo, eh)
            if v68_config.V118_W1:
                # Stage-1 weights were fit before the direct effects existed; refit them on
                # the completed arms, and on the union where Stage-1 t1/u1 stay available.
                ue, uh = np.concatenate([oof_c[swapped], eo]), np.concatenate([hid_c[swapped], eh])
                ul = ue[:, np.arange(len(rows)), y].T.astype(np.float64)
                k4, mw = len(swapped), (_mixture_weights_newton if EXACT else _mixture_weights)
                reopt[name]=mix(mw(ul[:, k4:], groups), eo, eh)
                union[name]=mix(mw(ul, groups), ue, uh)
        weights[name] = [round(float(v), 4) for v in w[-1]]
        w_by_item[name] = w.copy()
    if direct:
        from .wb_trait_effects import guarded
        LAST_V110['base'] = found
        LAST_V110['raw_w1'] = direct
        cands = {'w1': direct}
        if reopt:
            cands.update(w1_reopt=reopt, w1_6arm=union)
            LAST_V110['raw_w1_reopt'], LAST_V110['raw_w1_6arm'] = reopt, union
        found, info['v110_w1_gate'] = guarded(found, cands, work, fold)
    if kept_v127 is not None:
        LAST_KEEP = (*kept_v127, found, w_by_item)
    elif kept is not None:
        LAST_KEEP = (kept, targets, n_given_cols, strata, found)
    info['weights'] = weights
    info['mix_seconds'] = time.monotonic() - start - info['fit_seconds']
    info['seconds'] = time.monotonic() - start
    return found, info
