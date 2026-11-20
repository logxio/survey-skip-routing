"""Joint latent-class regression over every GIVEN column and every target.

Each respondent belongs to one of C latent classes; inside a class all GIVEN
columns and all twelve answers are independent categoricals. EM fits the
classes on the fold's training rows using every answered target, so each
class pools evidence across the questionnaire. A hidden row sees only its
GIVEN codes: P(c | GIVEN) is the class posterior from the GIVEN indicators
alone and P(y_i | GIVEN) = sum_c P(c | GIVEN) P(y_i | c), a rank-C mixture
of the full GIVEN-by-answer table that no per-item learner can express.

Targets gated on a GIVEN parent drop their closed rows from the likelihood
and put all mass on the never-asked level there. Out-of-fold rows come from
the model that never saw them; other rows average the five fold models.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

import ortho_config as _ortho

CLASSES = (12, 24)
ITERS = 70
ALPHA = 2.0
TOL = 2e-6
MIX = 1e-4


class _Expired(TimeoutError):
    pass


def _layout(Xg, levels, given, items, work):
    """Dense one-hot indicator matrix: GIVEN blocks first, then target blocks."""
    n = len(Xg)
    blocks, cols = [], []
    for j in range(Xg.shape[1]):
        L = int(max(levels[j], Xg[:, j].max() + 1))
        block = np.zeros((n, L), np.float32)
        block[np.arange(n), Xg[:, j]] = 1.0
        cols.append(block)
        blocks.append(L)
    n_given_cols = sum(blocks)
    targets = {}
    for name, j in work.items():
        y, K = j['y'], j['K']
        observed = y >= 0
        closed = np.zeros(n, bool)
        gate = items[name].get('gate')
        if isinstance(gate, dict) and gate.get('parent') in given[:Xg.shape[1]]:
            p = given.index(gate['parent'])
            values = list(items[gate['parent']]['values'])
            open_codes = [values.index(v) for v in gate.get('observed_if', ()) if v in values]
            closed = ~np.isin(Xg[:, p], open_codes)
            observed &= ~closed
        block = np.zeros((n, K), np.float32)
        rows = np.flatnonzero(observed)
        block[rows, y[rows]] = 1.0
        targets[name] = (sum(blocks), K, closed)
        cols.append(block)
        blocks.append(K)
    edges = np.cumsum([0] + blocks)
    return np.hstack(cols), [(int(edges[i]), int(edges[i + 1])) for i in range(len(blocks))], n_given_cols, targets


def _fit(O, blocks, rows, C, seed, deadline):
    Ot = O[rows]
    m = len(rows)
    rng = np.random.default_rng(seed)
    marg = Ot.sum(0) + 1e-3
    for a, b in blocks:
        marg[a:b] /= marg[a:b].sum()
    # Seed every class on one respondent's answer pattern, half-shrunk to the margin.
    picks = rng.choice(m, size=C, replace=False)
    theta = 0.5 * Ot[picks] + 0.5 * marg[None, :]
    for a, b in blocks:
        theta[:, a:b] /= theta[:, a:b].sum(1, keepdims=True)
    log_t = np.log(np.maximum(theta, 1e-8)).astype(np.float32)
    log_pi = np.full(C, -np.log(C), np.float32)
    prev = -np.inf
    for _ in range(ITERS):
        if time.monotonic() >= deadline:
            raise _Expired()
        L = Ot @ log_t.T + log_pi
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        s = E.sum(1, keepdims=True)
        R = E / s
        ll = float(np.log(s).sum() + top.sum())
        counts = R.T @ Ot
        for a, b in blocks:
            c = counts[:, a:b] + ALPHA * marg[None, a:b]
            log_t[:, a:b] = np.log(c / c.sum(1, keepdims=True))
        size = R.sum(0)
        log_pi = np.log((size + 1.0) / (m + C)).astype(np.float32)
        if ll - prev < TOL * m:
            break
        prev = ll
    return log_t, log_pi


def _predict(O, n_given_cols, log_t, log_pi, targets, beta=None):
    L = O[:, :n_given_cols] @ log_t[:, :n_given_cols].T
    if beta is not None:
        L *= np.float32(beta)
    L += log_pi
    L -= L.max(1, keepdims=True)
    Q = np.exp(L)
    Q /= Q.sum(1, keepdims=True)
    T = np.exp(log_t)
    out = {}
    for name, (start, K, closed) in targets.items():
        P = Q @ T[:, start:start + K]
        if closed.any():
            P[:, -1] = 0.0
            P[closed] = 0.0
            P[closed, -1] = 1.0
        P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
        out[name] = P
    return out


def attach(Xg, levels, given, items, work, fold, deadline, workers=8, betas=None, keep=None):
    """Returns ({name: (oof in train order, full n x K)}, info). Empty on timeout.

    betas ( U2): predict each fit at these GIVEN-likelihood temperatures and
    mix the (classes, temperature) candidates per item out of fold. keep: dict
    that receives every fit's parameters for later posteriors."""
    start = time.monotonic()
    info = {'classes': list(CLASSES), 'fits': 0, 'seconds': 0.0, 'status': 'complete'}
    if not work or time.monotonic() >= deadline:
        info['status'] = 'no_budget'
        return {}, info
    O, blocks, n_given_cols, targets = _layout(Xg, levels, given, items, work)
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    tasks = [(f, C) for f in folds for C in CLASSES]
    if keep is not None:
        keep['layout'] = (O, blocks, n_given_cols, targets)

    def run(task):
        f, C = task
        rows = np.flatnonzero((fold >= 0) & (fold != f))
        log_t, log_pi = _fit(O, blocks, rows, C, 20261106 + 97 * f + C, deadline)
        if keep is not None:
            keep[('g', f, C)] = (log_t, log_pi)
        if betas:
            return {b: _predict(O, n_given_cols, log_t, log_pi, targets, beta=b) for b in betas}
        return _predict(O, n_given_cols, log_t, log_pi, targets)

    if betas:
        try:
            with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
                fitted = dict(zip(tasks, pool.map(run, tasks)))
        except _Expired:
            info.update(status='budget_expired', seconds=time.monotonic() - start)
            return {}, info
        info['fits'] = len(fitted)
        info['betas'] = list(betas)
        cands = [(C, b) for C in CLASSES for b in betas]
        found, info['weights'] = _mix_folds(work, fold, folds, cands, lambda f, c, name: fitted[(f, c[0])][c[1]][name], len(Xg), workers=workers)
        info['seconds'] = time.monotonic() - start
        return found, info

    try:
        # BLAS limits are process-wide: one single-threaded BLAS for all fits.
        with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
            fitted = dict(zip(tasks, pool.map(run, tasks)))
    except _Expired:
        info.update(status='budget_expired', seconds=time.monotonic() - start)
        return {}, info
    info['fits'] = len(fitted)
    n = len(Xg)
    found = {}
    for name, j in work.items():
        K = j['K']
        by_fold = {f: np.mean([fitted[(f, C)][name] for C in CLASSES], axis=0) for f in folds}
        full = np.mean([by_fold[f] for f in folds], axis=0)
        rows = np.flatnonzero(j['train'])
        oof = np.empty((len(rows), K), np.float64)
        for f in folds:
            hit = fold[rows] == f
            oof[hit] = by_fold[f][rows[hit]]
        stray = fold[rows] < 0
        if stray.any():
            oof[stray] = full[rows[stray]]
        oof = (1.0 - MIX) * oof + MIX / K
        full = (1.0 - MIX) * full + MIX / K
        found[name] = (oof.astype(np.float32), full.astype(np.float32).reshape(n, K))
    info['seconds'] = time.monotonic() - start
    return found, info


#  U1: the answer distribution inside a class also depends on the country.
# Each country's class emissions shrink to the class-wide ones with TAU
# pseudo-counts, so a country-class cell with few respondents stays close to
# the pooled profile. Country fits at several class counts are mixed per item
# with weights learned on the other folds' out-of-fold predictions. Pooled
# fits (GLOBAL_CLASSES) can join the mix; on the B rulers they only lose.
GLOBAL_CLASSES = ()
COUNTRY_CLASSES = (8, 16, 28, 40)
TAU = 10.0
STRATIFY_GIVEN = False
WEIGHT_ITERS = 60


def _normalize_blocks(c, blocks):
    for a, b in blocks:
        c[..., a:b] /= c[..., a:b].sum(-1, keepdims=True)
    return c


def _fit_country(O, blocks, rows, C, seed, deadline, strata, G, strat_cols):
    Ot = O[rows]
    m = len(rows)
    g = strata[rows]
    groups = [np.flatnonzero(g == s) for s in range(G)]
    rng = np.random.default_rng(seed)
    marg = Ot.sum(0) + 1e-3
    _normalize_blocks(marg, blocks)
    picks = rng.choice(m, size=C, replace=False)
    theta = _normalize_blocks(0.5 * Ot[picks] + 0.5 * marg[None, :], blocks)
    log_g = np.repeat(np.log(np.maximum(theta, 1e-8)).astype(np.float32)[None], G, 0)
    log_pi = np.full(C, -np.log(C), np.float32)
    L = np.empty((m, C), np.float32)
    prev = -np.inf
    for _ in range(ITERS):
        if time.monotonic() >= deadline:
            raise _Expired()
        for s, idx in enumerate(groups):
            if len(idx):
                L[idx] = Ot[idx] @ log_g[s].T
        L += log_pi
        top = L.max(1, keepdims=True)
        E = np.exp(L - top)
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        counts_g = np.zeros((G, C, Ot.shape[1]), np.float32)
        for s, idx in enumerate(groups):
            if len(idx):
                counts_g[s] = R[idx].T @ Ot[idx]
        theta = _normalize_blocks(counts_g.sum(0) + ALPHA * marg[None, :], blocks)
        local = _normalize_blocks(counts_g + TAU * theta[None], blocks)
        local[:, :, ~strat_cols] = theta[None][:, :, ~strat_cols]
        log_g = np.log(np.maximum(local, 1e-12)).astype(np.float32)
        log_pi = np.log((R.sum(0) + 1.0) / (m + C)).astype(np.float32)
        if ll - prev < TOL * m:
            break
        prev = ll
    return log_g, log_pi


def _predict_country(O, n_given_cols, log_g, log_pi, targets, strata, beta=None, extra=None):
    n = len(O)
    Q = np.empty((n, len(log_pi)), np.float32)
    for s in range(len(log_g)):
        idx = np.flatnonzero(strata == s)
        if len(idx):
            Q[idx] = O[idx, :n_given_cols] @ log_g[s, :, :n_given_cols].T
    if beta is not None:
        Q *= np.float32(beta)
    if extra is not None:
        Q += extra
    Q += log_pi
    Q -= Q.max(1, keepdims=True)
    np.exp(Q, out=Q)
    Q /= Q.sum(1, keepdims=True)
    T = np.exp(log_g)
    out = {}
    for name, (start, K, closed) in targets.items():
        P = np.empty((n, K), np.float32)
        for s in range(len(log_g)):
            idx = np.flatnonzero(strata == s)
            if len(idx):
                P[idx] = Q[idx] @ T[s, :, start:start + K]
        if closed.any():
            P[:, -1] = 0.0
            P[closed] = 0.0
            P[closed, -1] = 1.0
        P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
        out[name] = P
    return out, Q


def _mixture_weights(likes):
    """Simplex weights maximizing the mean log of the mixed likelihood (EM)."""
    w = np.full(likes.shape[1], 1.0 / likes.shape[1])
    for _ in range(WEIGHT_ITERS):
        mixed = likes @ w
        w = w * (likes / np.maximum(mixed, 1e-300)[:, None]).mean(0)
        w /= w.sum()
    return w


def _mix_folds(work, fold, folds, cands, pred, n, names=None, workers=8):
    """Per item, mix candidate predictions with simplex weights; each fold's rows
    use weights learned on the other folds' out-of-fold likelihoods only.
    pred(f, cand, name) is the n x K prediction of the model fit without fold f."""
    def one(name):
        j = work[name]
        K = j['K']
        rows = np.flatnonzero(j['train'])
        rf = fold[rows]
        y = np.maximum(j['y'][rows], 0)
        oof_c = np.empty((len(cands), len(rows), K), np.float64)
        full_c = np.zeros((len(cands), n, K), np.float64)
        for ki, c in enumerate(cands):
            for f in folds:
                P = pred(f, c, name)
                hit = rf == f
                oof_c[ki, hit] = P[rows[hit]]
                full_c[ki] += P / len(folds)
            stray = rf < 0
            if stray.any():
                oof_c[ki, stray] = full_c[ki, rows[stray]]
        likes = oof_c[:, np.arange(len(rows)), y].T
        oof = np.empty((len(rows), K), np.float64)
        for f in folds:
            hit = rf == f
            oof[hit] = np.tensordot(_mixture_weights(likes[~hit]), oof_c[:, hit], 1)
        w = _mixture_weights(likes)
        stray = rf < 0
        if stray.any():
            oof[stray] = np.tensordot(w, oof_c[:, stray], 1)
        full = np.tensordot(w, full_c, 1)
        oof = (1.0 - MIX) * oof + MIX / K
        full = (1.0 - MIX) * full + MIX / K
        return (oof.astype(np.float32), full.astype(np.float32).reshape(n, K)), [round(float(v), 4) for v in w]

    order = list(work if names is None else names)
    with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(order)))) as pool:
        mixed = dict(zip(order, pool.map(one, order)))
    return {k: v[0] for k, v in mixed.items()}, {k: v[1] for k, v in mixed.items()}


def attach_country(Xg, levels, given, items, work, fold, deadline, workers=8, betas=None, keep=None):
    """Country-stratified, multi-scale variant of attach(); same return contract."""
    start = time.monotonic()
    kinds = [('g', C) for C in GLOBAL_CLASSES] + [('c', C) for C in COUNTRY_CLASSES]
    info = {'classes': [f'{k}{C}' for k, C in kinds], 'fits': 0, 'seconds': 0.0, 'status': 'complete', 'tau': TAU}
    if not work or time.monotonic() >= deadline or 'Country' not in given[:Xg.shape[1]]:
        info['status'] = 'no_budget'
        return {}, info
    O, blocks, n_given_cols, targets = _layout(Xg, levels, given, items, work)
    ci = given.index('Country')
    strata = Xg[:, ci].astype(np.int64)
    G = int(max(levels[ci], strata.max() + 1))
    strat_cols = np.zeros(O.shape[1], bool)
    strat_cols[(0 if STRATIFY_GIVEN else n_given_cols):] = True
    strat_cols[blocks[ci][0]:blocks[ci][1]] = False
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    # Pooled fits first: if the window closes, the completed class counts still mix.
    tasks = [(f, k, C) for k, C in kinds for f in folds]
    if keep is not None:
        keep['country_layout'] = (strata, G)

    def run(task):
        f, kind, C = task
        rows = np.flatnonzero((fold >= 0) & (fold != f))
        try:
            if kind == 'g':
                log_t, log_pi = _fit(O, blocks, rows, C, 20261106 + 97 * f + C, deadline)
                if betas:
                    return {b: _predict(O, n_given_cols, log_t, log_pi, targets, beta=b) for b in betas}
                return _predict(O, n_given_cols, log_t, log_pi, targets)
            log_g, log_pi = _fit_country(O, blocks, rows, C, 20261106 + 97 * f + C, deadline, strata, G, strat_cols)
            if keep is not None:
                keep[('c', f, C)] = (log_g, log_pi)
            if betas:
                return {b: _predict_country(O, n_given_cols, log_g, log_pi, targets, strata, beta=b)[0] for b in betas}
            return _predict_country(O, n_given_cols, log_g, log_pi, targets, strata)[0]
        except _Expired:
            return None

    with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        done = dict(zip(tasks, pool.map(run, tasks)))
    classes = [(k, C) for k, C in kinds if all(done[(f, k, C)] is not None for f in folds)]
    if not classes:
        info.update(status='budget_expired', seconds=time.monotonic() - start)
        return {}, info
    if keep is not None:
        keep['country_classes'] = [C for k, C in classes if k == 'c']
    fitted = {(f, (k, C)): done[(f, k, C)] for f in folds for k, C in classes}
    info['fits'] = len(fitted)
    info['mixed'] = [f'{k}{C}' for k, C in classes]
    n = len(Xg)
    if betas:
        info['betas'] = list(betas)
        cands = [(kc, b) for kc in classes for b in betas]
        found, info['weights'] = _mix_folds(work, fold, folds, cands, lambda f, c, name: fitted[(f, c[0])][c[1]][name], n, workers=workers)
        info['seconds'] = time.monotonic() - start
        return found, info
    found = {}
    weights = {}
    for name, j in work.items():
        K = j['K']
        rows = np.flatnonzero(j['train'])
        rf = fold[rows]
        y = np.maximum(j['y'][rows], 0)
        oof_c = np.empty((len(classes), len(rows), K), np.float64)
        full_c = np.zeros((len(classes), n, K), np.float64)
        for ki, C in enumerate(classes):
            for f in folds:
                P = fitted[(f, C)][name]
                hit = rf == f
                oof_c[ki, hit] = P[rows[hit]]
                full_c[ki] += P / len(folds)
            stray = rf < 0
            if stray.any():
                oof_c[ki, stray] = full_c[ki, rows[stray]]
        likes = oof_c[:, np.arange(len(rows)), y].T
        oof = np.empty((len(rows), K), np.float64)
        # Each fold's rows mix with weights learned on the other folds only.
        for f in folds:
            hit = rf == f
            oof[hit] = np.tensordot(_mixture_weights(likes[~hit]), oof_c[:, hit], 1)
        w = _mixture_weights(likes)
        stray = rf < 0
        if stray.any():
            oof[stray] = np.tensordot(w, oof_c[:, stray], 1)
        full = np.tensordot(w, full_c, 1)
        weights[name] = [round(float(v), 4) for v in w]
        oof = (1.0 - MIX) * oof + MIX / K
        full = (1.0 - MIX) * full + MIX / K
        found[name] = (oof.astype(np.float32), full.astype(np.float32).reshape(n, K))
    info['weights'] = weights
    info['seconds'] = time.monotonic() - start
    return found, info


#  U3: the child items' class posterior also reads the four anchor items
# through their stacked predictions S_a (out-of-fold on visible rows), as soft
# evidence: L_ic = O_given log theta_c + gamma sum_a S_ia log theta_{c,a} + log pi_c.
# The fits are the ones attach()/attach_country() already made, so a fold's rows
# only meet parameters fitted without them; gamma and class count mix per item.
STAGE2_GAMMAS = (0.3, 0.6, 1.0)
#  U2: the GIVEN-only class posterior of a fit with smoothed emissions is
# too flat (B28 dev mix: weight falls monotonically from beta 1 to 0.4), so the
# candidates sharpen it with beta >= 1.
BETAS = (1.0, 1.3, 1.7)


def stage2(keep, anchor_probs, work, names, fold, deadline, gammas=STAGE2_GAMMAS, workers=8):
    start = time.monotonic()
    info = {'gammas': list(gammas), 'anchors': sorted(anchor_probs), 'status': 'complete'}
    if 'layout' not in keep or not anchor_probs or not names or time.monotonic() >= deadline:
        info['status'] = 'no_budget'
        return {}, info
    O, blocks, n_given_cols, targets = keep['layout']
    n = len(O)
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    evidence = []
    for a, S in anchor_probs.items():
        a_start, a_K, closed = targets[a]
        S = np.asarray(S, np.float32).copy()
        S[closed] = 0.0
        evidence.append((S, a_start, a_K))
    kinds = [('g', C) for C in CLASSES if all(('g', f, C) in keep for f in folds)]
    if 'country_layout' in keep:
        kinds += [('c', C) for C in keep.get('country_classes', ()) if all(('c', f, C) in keep for f in folds)]
    if not kinds:
        info['status'] = 'no_fits'
        return {}, info

    def run(task):
        kind, C, f = task
        if time.monotonic() >= deadline:
            raise _Expired()
        out = {}
        if kind == 'g':
            log_t, log_pi = keep[('g', f, C)]
            base = O[:, :n_given_cols] @ log_t[:, :n_given_cols].T
            soft = sum(S @ log_t[:, s:s + K].T for S, s, K in evidence)
            T = np.exp(log_t)
            for g in gammas:
                L = base + np.float32(g) * soft + log_pi
                L -= L.max(1, keepdims=True)
                Q = np.exp(L)
                Q /= Q.sum(1, keepdims=True)
                out[g] = {name: Q @ T[:, targets[name][0]:targets[name][0] + targets[name][1]] for name in names}
        else:
            strata, G = keep['country_layout']
            log_g, log_pi = keep[('c', f, C)]
            soft = np.empty((n, C), np.float32)
            for s in range(G):
                idx = np.flatnonzero(strata == s)
                if len(idx):
                    soft[idx] = sum(S[idx] @ log_g[s, :, a:a + K].T for S, a, K in evidence)
            for g in gammas:
                out[g] = _predict_country(O, n_given_cols, log_g, log_pi, {name: targets[name] for name in names}, strata, extra=np.float32(g) * soft)[0]
            return out
        for name in names:
            closed = targets[name][2]
            for g in gammas:
                P = out[g][name]
                if closed.any():
                    P[:, -1] = 0.0
                    P[closed] = 0.0
                    P[closed, -1] = 1.0
                P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
        return out

    tasks = [(k, C, f) for k, C in kinds for f in folds]
    try:
        with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
            done = dict(zip(tasks, pool.map(run, tasks)))
    except _Expired:
        info.update(status='budget_expired', seconds=time.monotonic() - start)
        return {}, info
    cands = [((k, C), g) for k, C in kinds for g in gammas]
    found, info['weights'] = _mix_folds(work, fold, folds, cands, lambda f, c, name: done[(c[0][0], c[0][1], f)][c[1]][name], n, names=names, workers=workers)
    info['kinds'] = [f'{k}{C}' for k, C in kinds]
    info['seconds'] = time.monotonic() - start
    return found, info


#  U2: class posteriors as features. One fit per training partition on that
# partition's rows only; every row's posterior uses its GIVEN codes alone, so
# fit rows and query rows see the same function of GIVEN.
FEATURE_CLASSES = 16


def layout(Xg, levels, given, items, work):
    return _layout(Xg, levels, given, items, work)


def posterior(fixed, rows, deadline):
    O, blocks, n_given_cols, _ = fixed
    log_t, log_pi = _fit(O, blocks, np.asarray(rows, dtype=np.int64), FEATURE_CLASSES, 20261107, deadline)
    L = O[:, :n_given_cols] @ log_t[:, :n_given_cols].T + log_pi
    L -= L.max(1, keepdims=True)
    Q = np.exp(L)
    Q /= Q.sum(1, keepdims=True)
    return Q.astype(np.float32)


#  U1: a nominal-response trait arm. Every column block's answer logits are
# base + theta * loading with a standard normal theta on a quadrature grid, the
# smooth limit of many ordered classes. GIVEN columns that a GIVEN gate closed
# carry no evidence on those rows (the country already says it), and a target
# gated on another target is fitted on its open rows only and predicted as
# P(open | theta) * P(answer | theta), with the never-asked level taking the rest.
TRAIT_NODES = 41
TRAIT_ITERS = 80
RIDGE = 0.25
SHIFT_RIDGE = 1.0


def _layout_gated(Xg, levels, given, items, work):
    """_layout with gate-aware evidence; chains: {child: (parent start, parent K, open codes)}."""
    O, blocks, n_given_cols, targets = _layout(Xg, levels, given, items, work)
    raw = list(given[:Xg.shape[1]])
    for j, name in enumerate(raw):
        gate = items.get(name, {}).get('gate')
        if isinstance(gate, dict) and gate.get('parent') in raw:
            p = raw.index(gate['parent'])
            values = list(items[gate['parent']]['values'])
            open_codes = [values.index(v) for v in gate.get('observed_if', ()) if v in values]
            shut = ~np.isin(Xg[:, p], open_codes)
            a, b = blocks[j]
            O[np.ix_(np.flatnonzero(shut), np.arange(a, b))] = 0.0
    chains = {}
    for name, j in work.items():
        gate = items[name].get('gate')
        if not (isinstance(gate, dict) and gate.get('parent') in work):
            continue
        parent = gate['parent']
        values = list(items[parent]['values'])
        open_codes = [values.index(v) for v in gate.get('observed_if', ()) if v in values]
        py = work[parent]['y']
        shut = (py >= 0) & ~np.isin(py, open_codes)
        start, K, closed = targets[name]
        O[np.ix_(np.flatnonzero(shut), np.arange(start, start + K))] = 0.0
        chains[name] = (targets[parent][0], targets[parent][1], open_codes)
    return O, blocks, n_given_cols, targets, chains


def _grid():
    nodes = np.linspace(-4.0, 4.0, TRAIT_NODES)
    lw = -0.5 * nodes ** 2
    lw -= lw.max() + np.log(np.exp(lw - lw.max()).sum())
    return nodes, lw


def _block_lse(Z, starts, widths):
    top = np.maximum.reduceat(Z, starts, axis=1)
    s = np.add.reduceat(np.exp(Z - np.repeat(top, widths, 1)), starts, axis=1)
    return np.repeat(top + np.log(s), widths, 1)




def _fit_trait(O, blocks, rows, seed, deadline, strata=None, G=1, n_given_cols=0):
    """Returns (log emissions strata x nodes x columns, log prior over nodes).

    With strata, every target column adds a per-stratum offset to its base
    logit, shrunk to zero with SHIFT_RIDGE; loadings and GIVEN columns stay
    pooled, so the trait posterior is the same function of GIVEN everywhere."""
    if strata is not None:
        # Rows sorted by stratum: every stratum is a contiguous slice, not a copy.
        rows = rows[np.argsort(strata[rows], kind='stable')]
    Ot = O[rows]
    m, D = Ot.shape
    starts = np.array([a for a, _ in blocks])
    widths = np.array([b - a for a, b in blocks])
    nodes, lw = _grid()
    obs = (np.add.reduceat(Ot, starts, axis=1) > 0).astype(np.float32)
    if strata is None:
        groups = [slice(0, m)]
    else:
        edges = np.searchsorted(strata[rows], np.arange(G + 1))
        groups = [slice(int(edges[s]), int(edges[s + 1])) for s in range(G)]
    S = np.zeros((len(groups), D))
    local = np.zeros(D, bool)
    if strata is not None:
        local[n_given_cols:] = True
    B = np.log(_normalize_blocks(Ot.sum(0).astype(np.float64) + 1e-3, blocks))
    A = np.zeros(D)
    # Start the posterior around the leading principal score of the indicators.
    rng = np.random.default_rng(seed)
    Xc = Ot - Ot.mean(0)
    v = rng.normal(size=(D, 1))
    for _ in range(6):
        v, _ = np.linalg.qr(Xc.T @ (Xc @ v))
    s = (Xc @ v)[:, 0]
    s = (s - s.mean()) / max(s.std(), 1e-9)
    R = -0.5 * (s[:, None] - nodes[None]) ** 2 / 0.36 + lw[None]
    R = np.exp(np.maximum(R - R.max(1, keepdims=True), -80.0))
    R = (R / R.sum(1, keepdims=True)).astype(np.float32)
    q = nodes[None, :, None]
    Qn = len(nodes)

    def emissions():
        Z = (B[None, None] + q * A[None, None] + S[:, None, :]).reshape(-1, D)
        return (Z - _block_lse(Z, starts, widths)).reshape(len(groups), Qn, D)

    prev = -np.inf
    L = np.empty((m, Qn), np.float32)
    for _ in range(TRAIT_ITERS):
        if time.monotonic() >= deadline:
            raise _Expired()
        N = np.stack([R[idx].T @ Ot[idx] for idx in groups]).astype(np.float64)
        n = np.stack([np.repeat(R[idx].T @ obs[idx], widths, 1) for idx in groups]).astype(np.float64)
        for _ in range(2):
            # n P bounds the softmax Hessian n (diag P - P P^T) from above, so
            # every step is majorize-minimize and the likelihood never falls.
            P = np.exp(emissions())
            Gr = N - n * P
            W = n * P + 1e-6
            g0 = Gr.sum((0, 1))
            g1 = (q * Gr).sum((0, 1)) - RIDGE * A
            h00 = W.sum((0, 1)) + 1e-3
            h01 = (q * W).sum((0, 1))
            h11 = (q * q * W).sum((0, 1)) + RIDGE
            det = h00 * h11 - h01 ** 2
            B = np.clip(B + np.clip((h11 * g0 - h01 * g1) / det, -1.0, 1.0), -25.0, 25.0)
            A = np.clip(A + np.clip((h00 * g1 - h01 * g0) / det, -1.0, 1.0), -8.0, 8.0)
            if local.any():
                P = np.exp(emissions())
                Gr = N - n * P
                W = n * P + 1e-6
                step = (Gr.sum(1) - SHIFT_RIDGE * S) / (W.sum(1) + SHIFT_RIDGE)
                S[:, local] = np.clip(S[:, local] + np.clip(step[:, local], -1.0, 1.0), -12.0, 12.0)
        # Floors at e^-80 keep every float32 normal: x86 runs denormals ~100x slower.
        log_g = np.maximum(emissions(), -80.0).astype(np.float32)
        for k, idx in enumerate(groups):
            L[idx] = Ot[idx] @ log_g[k].T
        L += lw.astype(np.float32)
        top = L.max(1, keepdims=True)
        E = np.exp(np.maximum(L - top, -80.0))
        tot = E.sum(1, keepdims=True)
        R = E / tot
        ll = float(np.log(tot).sum() + top.sum())
        if ll - prev < TOL * m:
            break
        prev = ll
    return log_g, lw.astype(np.float32)


def _predict_chained(O, n_given_cols, log_g, log_pi, targets, chains, strata=None):
    """Answer distributions from the GIVEN-only posterior of one fit (pooled or
    per-stratum target emissions), composing target-gated children."""
    L = O[:, :n_given_cols] @ log_g[0, :, :n_given_cols].T + log_pi
    L -= L.max(1, keepdims=True)
    Q = np.exp(np.maximum(L, -80.0))
    Q /= Q.sum(1, keepdims=True)
    T = np.exp(log_g)
    parts = [np.arange(len(O))] if len(T) == 1 else [np.flatnonzero(strata == s) for s in range(len(T))]
    out = {}
    for name, (start, K, closed) in targets.items():
        P = np.empty((len(O), K), np.float32)
        for s, idx in enumerate(parts):
            if not len(idx):
                continue
            Ts = T[s]
            if name in chains:
                p_start, p_K, open_codes = chains[name]
                Tp = Ts[:, p_start:p_start + p_K]
                go = Tp[:, open_codes].sum(1) / Tp.sum(1)
                Tc = Ts[:, start:start + K - 1]
                Tc = Tc / Tc.sum(1, keepdims=True)
                P[idx, :K - 1] = Q[idx] @ (Tc * go[:, None])
                P[idx, K - 1] = Q[idx] @ (1.0 - go)
            else:
                P[idx] = Q[idx] @ Ts[:, start:start + K]
        if closed.any():
            P[:, -1] = 0.0
            P[closed] = 0.0
            P[closed, -1] = 1.0
        P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
        out[name] = P
    return out


TRAIT_KINDS = ('tc', 'tcx', 't')


def attach_trait(Xg, levels, given, items, work, fold, deadline, workers=8, kinds=TRAIT_KINDS, pairs=(), v106=None, state_keep=None):
    """Trait arms, one per kind: 't' pooled, 'tc' with country target offsets,
    and their 'x' versions with direct GIVEN effects on the targets (same fit).
    pairs: GIVEN name pairs for the 'xp' kinds, whose joint levels add a shrunk rank-one effect.
    Returns ({kind: {name: (oof in train order, full n x K)}}, info)."""
    start = time.monotonic()
    raw = list(given[:Xg.shape[1]])
    kinds = [k for k in kinds if not k.startswith('tc') or 'Country' in raw]
    info = {'kinds': list(kinds), 'fits': 0, 'seconds': 0.0, 'status': 'complete', 'nodes': TRAIT_NODES}
    if not work or not kinds or time.monotonic() >= deadline:
        info['status'] = 'no_budget'
        return {}, info
    O, blocks, n_given_cols, targets, chains = _layout_gated(Xg, levels, given, items, work)
    index_pairs = tuple((raw.index(a), raw.index(b)) for a, b in pairs if a in raw and b in raw)
    #  U2 'xv' kinds: their own pairs, pair-table rank gate and posterior refresh.
    v_pairs, v_rank, v_refresh = (), 1, False
    if v106 and _ortho.V115_U2 and 'Country' in raw:
        v106 = dict(v106, pairs=tuple(('Country', b) for b in raw if b != 'Country'))
    if v106:
        v_pairs = tuple((raw.index(a), raw.index(b)) for a, b in v106['pairs'] if a in raw and b in raw)
        v_rank, v_refresh = v106['rank'], v106['refresh']
    if index_pairs:
        info['pairs'] = [list(p) for p in pairs]
    strata, G = None, 1
    if any(k.startswith('tc') for k in kinds):
        ci = raw.index('Country')
        strata = Xg[:, ci].astype(np.int64)
        G = int(max(levels[ci], strata.max() + 1))
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    bases = list(dict.fromkeys(k.rstrip('xpv') for k in kinds))
    tasks = [(b, f) for b in bases for f in folds]
    spent = {}
    n = len(Xg)

    def run(task):
        kind, f = task
        rows = np.flatnonzero((fold >= 0) & (fold != f))
        began = time.monotonic()
        res = {}
        try:
            if kind == 'tc':
                log_g, lw = _fit_trait(O, blocks, rows, 20261108 + 97 * f, deadline, strata, G, n_given_cols)
            else:
                log_g, lw = _fit_trait(O, blocks, rows, 20261108 + 97 * f, deadline)
            if kind in kinds:
                res[kind] = _predict_chained(O, n_given_cols, log_g, lw, targets, chains, strata)
            if kind + 'x' in kinds:
                off, inv, stratum = _direct_effects(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata if kind == "tc" else None, deadline)
                res[kind + "x"] = _predict_effects(O, n_given_cols, log_g, lw, off, inv, stratum, blocks, targets, chains, n)
            if kind + 'xp' in kinds and index_pairs:
                off, inv, stratum = _direct_effects(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata if kind == "tc" else None, deadline, index_pairs, *((1, True) if v106 and v_refresh else ()))
                res[kind + "xp"] = _predict_effects(O, n_given_cols, log_g, lw, off, inv, stratum, blocks, targets, chains, n)
                if state_keep is not None and kind == "tc":
                    res["_params_xp"] = (log_g, lw, off, inv, stratum)
            if kind + 'xv' in kinds and v_pairs:
                off, inv, stratum = _direct_effects(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata if kind == "tc" else None, deadline, v_pairs, v_rank, v_refresh)
                res[kind + "xv"] = _predict_effects(O, n_given_cols, log_g, lw, off, inv, stratum, blocks, targets, chains, n)
                if state_keep is not None and kind == "tc":
                    res["_params_xv"] = (log_g, lw, off, inv, stratum)
        except _Expired:
            pass
        spent.setdefault(kind, []).append(round(time.monotonic() - began, 3))
        return res

    with threadpool_limits(limits=1, user_api='blas'), ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        fitted = dict(zip(tasks, pool.map(run, tasks)))
    done = {(k, f): fitted[(k.rstrip('xpv'), f)].get(k) for k in kinds for f in folds}
    finished = [k for k in kinds if all(done[(k, f)] is not None for f in folds)]
    info['fits'] = sum(len(v) for v in fitted.values())
    info['fit_task_seconds'] = spent
    info['mixed'] = finished
    if not finished:
        info.update(status='budget_expired', seconds=time.monotonic() - start)
        return {}, info
    n = len(Xg)
    out = {}
    for kind in finished:
        found = {}
        for name, j in work.items():
            K = j['K']
            full = np.mean([done[(kind, f)][name] for f in folds], axis=0)
            rows = np.flatnonzero(j['train'])
            oof = np.empty((len(rows), K), np.float64)
            for f in folds:
                hit = fold[rows] == f
                oof[hit] = done[(kind, f)][name][rows[hit]]
            stray = fold[rows] < 0
            if stray.any():
                oof[stray] = full[rows[stray]]
            oof = (1.0 - MIX) * oof + MIX / K
            full = (1.0 - MIX) * full + MIX / K
            found[name] = (oof.astype(np.float32), full.astype(np.float32).reshape(n, K))
        out[kind] = found
    for base_kind in [k[:-1] for k in out if k.endswith('xv') and k[:-1] + 'p' in out]:
        #  U2: the 'xv' fit replaces an item's 'xp' answers only when it is
        # the configured option nats better out of fold in both fold-parity halves.
        cand, keep = out.pop(base_kind + 'v'), out[base_kind + 'p']
        info['xv_items'] = []
        for name, j in work.items():
            rows = np.flatnonzero(j['train'])
            y, half = j['y'][rows], fold[rows] % 2 == 0
            gain = [_oof_nll(keep[name][0], y, m) - _oof_nll(cand[name][0], y, m) for m in (half, ~half)]
            if min(gain) >= V106_MARGIN:
                keep[name] = cand[name]
                info['xv_items'].append(name)
    if state_keep is not None:
        xv_set = set(info.get('xv_items', ()))
        best_folds = {}
        for name in work:
            chosen_k = 'tcxv' if (name in xv_set and all(done.get(('tcxv', f)) is not None for f in folds)) else (
                'tcxp' if all(done.get(('tcxp', f)) is not None for f in folds) else (
                    'tcx' if all(done.get(('tcx', f)) is not None for f in folds) else finished[0]
                )
            )
            best_folds[name] = {f: done[(chosen_k, f)][name] for f in folds}
        state_keep['trait_best_folds'] = best_folds
        if 'tc' in bases and all(fitted.get(('tc', f), {}).get('_params_xp') is not None for f in folds):
            state_keep['trait_model'] = {
                'layout': (O, blocks, n_given_cols, targets, chains, strata),
                'folds': folds,
                'params_xp': {f: fitted[('tc', f)]['_params_xp'] for f in folds},
                'params_xv': {f: fitted[('tc', f)].get('_params_xv', fitted[('tc', f)]['_params_xp']) for f in folds},
                'xv_items': xv_set,
            }
    info['seconds'] = time.monotonic() - start
    return out, info


V106_MARGIN = 0.001


def _oof_nll(P, y, mask):
    if not mask.any():
        return 0.0
    return float(-np.log(np.maximum(P[mask][np.arange(int(mask.sum())), y[mask]], 1e-12)).mean())


#  U1 'x' kinds: with a trait fit's full-information posterior of every
# training row held fixed, every target logit also takes additive offsets for
# each GIVEN column's level (direct demographic effects that the trait does not
# carry), shrunk to zero with EFFECT_RIDGE. Everything runs on the distinct
# GIVEN cells, so the cost does not grow with the row count.
EFFECT_RIDGE = 20.0
EFFECT_ITERS = 15
PAIR_RIDGE = 20.0
PAIR_RANK = True
PAIR_EDGE = 1.1


def _cell_sums(R, idx, size):
    """sum of R rows into flat bins idx: (size, nodes)."""
    return np.stack([np.bincount(idx, weights=R[:, q], minlength=size) for q in range(R.shape[1])], 1)


def _cell_tables(log_g, stratum, t0, tb):
    """Per-cell, per-target-block emission tables exp(log_g[stratum]): (cells, nodes, K)."""
    T = np.exp(log_g[:, :, t0:])
    return [np.ascontiguousarray(T[stratum][:, :, a:b]) for a, b in tb]


def _pair_levels(cells, pairs):
    """Joint level codes of GIVEN column pairs on the distinct cells: cells x pairs."""
    cols = []
    for a, b in pairs:
        _, code = np.unique(cells[:, a] * (int(cells[:, b].max()) + 1) + cells[:, b], return_inverse=True)
        cols.append(code.ravel())
    return np.column_stack(cols)


def _direct_effects(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata, deadline, pairs=(), rank=1, refresh=False):
    """Returns (target offsets per cell: cells x target columns, row -> cell, cell stratum).

    P(answer | node, cell) = T * w / sum(T * w) per block with w = exp(offset), so
    an iteration rescales the fixed tables instead of recomputing a softmax.
    pairs: GIVEN column index pairs that also take a joint-level offset ( U1).
    rank > 1 keeps every noise-cleared component of a pair table ( U2);
    refresh re-reads the training posterior under the fitted offsets and refits once."""
    Ot = O[rows]
    m, D = Ot.shape
    G, Qn = log_g.shape[0], log_g.shape[1]
    parts = [np.arange(m)] if G == 1 else [np.flatnonzero(strata[rows] == s) for s in range(G)]
    L = np.empty((m, Qn), np.float32)
    for s, idx in enumerate(parts):
        if len(idx):
            L[idx] = Ot[idx] @ log_g[s].T
    L += lw
    L -= L.max(1, keepdims=True)
    R = np.exp(np.maximum(L, -80.0)).astype(np.float64)
    R /= R.sum(1, keepdims=True)
    cells, inv = np.unique(Xg, axis=0, return_inverse=True)
    inv = inv.ravel()
    C = len(cells)
    t0 = n_given_cols
    tb = [(a - t0, b - t0) for a, b in blocks if a >= t0]
    Dt = D - t0
    starts = np.array([a for a, _ in tb])
    # A cell's stratum is the stratum of any of its rows (strata are GIVEN codes).
    stratum = np.zeros(C, np.int64)
    if G > 1:
        stratum[inv] = strata
    c_rows = inv[rows]
    seen = np.flatnonzero(np.bincount(c_rows, minlength=C) > 0)
    where = np.full(C, -1, np.int64)
    where[seen] = np.arange(len(seen))
    cr = where[c_rows]
    Cs = len(seen)
    Yt = Ot[:, t0:]
    ii, dd = np.nonzero(Yt)
    Ns = np.bincount(cr[ii] * Dt + dd, minlength=Cs * Dt).reshape(Cs, Dt)
    bo = np.add.reduceat(Yt, starts, axis=1) > 0
    ib, bb = np.nonzero(bo)
    n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
    Ts = _cell_tables(log_g, stratum[seen], t0, tb)
    ridge = [EFFECT_RIDGE] * cells.shape[1]
    if pairs:
        cells = np.column_stack([cells, _pair_levels(cells, pairs)])
        ridge += [PAIR_RIDGE] * len(pairs)
    cs = cells[seen]
    J = cs.shape[1]
    # One-hot level maps: effect gradients are matrix products, not scatter-adds.
    E = [np.eye(int(cells[:, j].max()) + 1)[cs[:, j]].T for j in range(J)]
    beta = [np.zeros((len(E[j]), Dt)) for j in range(J)]
    h = np.empty((Cs, Dt))

    def sweep(update):
        update = list(update)
        main_only = max(update) < J - len(pairs)
        #  U3: exact curvature on main-effect sweeps; a pair sweep keeps the J step,
        # whose 15 short steps are the pair tables' shrinkage (U1 lost ruler C 4/4 without it).
        K = len(update) if _ortho.V115_U1 or _ortho.V115_U2 or (_ortho.V115_U3 and main_only) else J
        for _ in range(EFFECT_ITERS):
            if time.monotonic() >= deadline:
                raise _Expired()
            w = np.exp(sum(beta[j][cs[:, j]] for j in range(J))).astype(np.float32)
            for b, (a, e) in enumerate(tb):
                wb = w[:, a:e]
                den = (Ts[b] @ wb[:, :, None])[:, :, 0]
                h[:, a:e] = wb * (((n[:, b, :] / den).astype(np.float32))[:, None, :] @ Ts[b])[:, 0, :]
            g = Ns - h
            for j in update:
                # Jensen over the columns this sweep moves: curvature K h keeps the joint step a majorizer.
                beta[j] += np.clip((E[j] @ g - ridge[j] * beta[j]) / (K * (E[j] @ h) + ridge[j]), -1.0, 1.0)

    def fit():
        if pairs and PAIR_RANK and rank > 1:
            #  U2: main effects first, the free pair tables on what they leave,
            # so a pair with no signal gates to zero without moving the main effects.
            main = range(J - len(pairs))
            sweep(main)
            if _ortho.V115_U2:
                # Block coordinate ascent: each pair table takes its full step on what the others leave.
                for j in range(J - len(pairs), J):
                    sweep([j])
            else:
                sweep(range(J - len(pairs), J))
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(main)
            return
        sweep(range(J))
        if pairs and PAIR_RANK:
            # A pair's free table is mostly noise; keep its shrunk rank-one part (one
            # score per joint level times one direction over every target option),
            # then refit the main effects around it.
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_one(beta[j], E[j] @ rows_per_cell) if rank <= 1 else _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(range(J - len(pairs)))

    fit()
    if refresh:
        # EM's next E step: n came from the trait-only posterior; re-read it with
        # the offsets inside the target emissions, then refit from where we are.
        w = np.exp(sum(beta[j][cs[:, j]] for j in range(J)))
        L = np.empty((m, Qn), np.float64)
        for s, idx in enumerate(parts):
            if len(idx):
                L[idx] = Ot[idx, :t0] @ log_g[s, :, :t0].T
        L += lw
        for b, (a, e) in enumerate(tb):
            hit = (dd >= a) & (dd < e)
            lt = np.log(np.maximum(Ts[b] * w[:, None, a:e], 1e-300))
            lt -= np.log(np.exp(lt).sum(2, keepdims=True))
            np.add.at(L, ii[hit], lt[cr[ii[hit]], :, dd[hit] - a] * Yt[ii[hit], dd[hit]][:, None])
        L -= L.max(1, keepdims=True)
        R = np.exp(np.maximum(L, -80.0))
        R /= R.sum(1, keepdims=True)
        n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
        fit()
    off = sum(beta[j][cells[:, j]] for j in range(J))
    return off, inv, stratum


def _rank_gated(B, count, rank):
    """_rank_one generalized ( U2): a singular component is kept only above
    PAIR_EDGE times the Marchenko-Pastur edge of the noise, where pure noise's top
    component sits, then takes _rank_one's James-Stein share. The noise is
    re-read from the components left out; a table with no signal returns zero."""
    keep = count > 0
    r = np.sqrt(count[keep])[:, None]
    M = r * B[keep]
    L, D = M.shape
    out = np.zeros_like(B)
    if L < 3 or D < 3:
        return out
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = 0
    for _ in range(4):
        noise = float((s[k:] ** 2).sum()) / max((L - k) * (D - k), 1)
        k_new = int(min(rank, len(s) - 1, np.count_nonzero(s ** 2 > PAIR_EDGE * noise * (np.sqrt(L) + np.sqrt(D)) ** 2)))
        if k_new == k:
            break
        k = k_new
    if not k:
        return out
    shrink = np.maximum(0.0, 1.0 - noise * (L + D) / np.maximum(s[:k] ** 2, 1e-30))
    out[keep] = ((U[:, :k] * (s[:k] * shrink)) @ Vt[:k]) / r
    return out


def _rank_one(B, count):
    """James-Stein shrunk rank-one part of a levels x columns offset table, rows
    scaled by sqrt(row count) so every entry carries about the same noise."""
    keep = count > 0
    r = np.sqrt(count[keep])[:, None]
    M = r * B[keep]
    L, D = M.shape
    out = np.zeros_like(B)
    if L < 3 or D < 3:
        return out
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    noise = float((s[1:] ** 2).sum()) / ((L - 1) * (D - 1))
    shrink = max(0.0, 1.0 - noise * (L + D) / max(float(s[0]) ** 2, 1e-30))
    out[keep] = (s[0] * shrink) * np.outer(U[:, 0], Vt[0]) / r
    return out


def _predict_effects(O, n_given_cols, log_g, lw, off, inv, stratum, blocks, targets, chains, n):
    """Per-cell answer distributions from the GIVEN posterior, the trait tables and
    the cell offsets, spread to rows."""
    t0 = n_given_cols
    C = len(off)
    tb = [(a - t0, b - t0) for a, b in blocks if a >= t0]
    rep = np.zeros(C, np.int64)
    rep[inv[::-1]] = np.arange(n)[::-1]
    L = O[rep, :t0] @ log_g[0, :, :t0].T + lw
    L -= L.max(1, keepdims=True)
    Qc = np.exp(np.maximum(L, -80.0))
    Qc /= Qc.sum(1, keepdims=True)
    Ts = _cell_tables(log_g, stratum, t0, tb)
    first = {a: b for b, (a, _) in enumerate(tb)}
    w = np.exp(off)

    def scaled(start, K):
        b = first[start - t0]
        a = start - t0
        return Ts[b][:, :, :K] * w[:, None, a:a + K]

    out = {}
    for name, (start, K, closed) in targets.items():
        if name in chains:
            p_start, p_K, open_codes = chains[name]
            Tp = scaled(p_start, p_K)
            go = Tp[:, :, open_codes].sum(2) / Tp.sum(2)
            Tc = scaled(start, K - 1)
            Tc = Tc / Tc.sum(2, keepdims=True)
            Pc = np.empty((C, K))
            Pc[:, :K - 1] = ((Qc * go)[:, None, :] @ Tc)[:, 0, :]
            Pc[:, K - 1] = (Qc * (1.0 - go)).sum(1)
        else:
            Tk = scaled(start, K)
            Pc = ((Qc / Tk.sum(2))[:, None, :] @ Tk)[:, 0, :]
        P = Pc[inv].astype(np.float32)
        if closed.any():
            P[:, -1] = 0.0
            P[closed] = 0.0
            P[closed, -1] = 1.0
        P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
        out[name] = P
    return out
