"""Directional copy law with endpoint-excluded nested calibration.

The infinite diagonal prior exactly recovers the original copy branch.
Finite priors count visible donor/target transitions, splitting a wave-one
zero by its observed parent state. No hidden answer enters these tables.
"""
import time
import numpy as np
from .twin_base import _loss

ALPHAS = (1.5, 4.0, 12.0, np.inf)
LAMBDAS = (0.0, 0.25, 0.5, 0.7, 0.85, 1.0)


def refine(grid, d, tv, sv, th, sh, mv, mh, qv, qh, work, final,
           final_oofs, stacked, fold, pos, report, deadline):
    ft, fs = fold[tv], fold[sv]
    groups = sorted(set(tuple(sorted((int(a), int(b)))) for a, b in zip(ft, fs)))
    changed, picks = {}, {}
    for i, k in enumerate(grid.names):
        if time.monotonic() >= deadline:
            report['deadline'] = True
            break
        if k not in final:
            continue
        K, y = int(grid.K[i]), work[k]['y'][tv]
        src = grid.Y[sv, i]
        hsrc = grid.Y[sh, i]
        augment = grid.zero[i] >= 0
        code = 2 * src + grid.opened[sv, i].astype(int) if augment else src
        hcode = 2 * hsrc + grid.opened[sh, i].astype(int) if augment else hsrc
        S = 2 * K if augment else K
        # Identity includes the original wave-specific skip mapping.
        identity = np.eye(K)[np.repeat(np.arange(K), 2) if augment else np.arange(K)]
        z = grid.zero[i]
        if augment:
            if d == 1:
                identity[0::2] = np.eye(K)[K - 1]
            else:
                identity[2 * (K - 1):] = np.eye(K)[z]
        onev = grid._copy(i, sv, d == 1)
        oneh = grid._copy(i, sh, d == 1)
        wv, wh = qv[:, grid.item_root[i]:grid.item_root[i] + 1], qh[:, grid.item_root[i]:grid.item_root[i] + 1]
        redv, redh = mv[i] - wv * onev, mh[i] - wh * oneh
        cur = np.asarray(final_oofs.get(k, stacked[k][0]), float)
        cv = cur[pos[tv]].copy()
        cv /= np.maximum(cv.sum(1, keepdims=True), 1e-300)
        bh = np.asarray(final[k][th], float)
        bh = bh / np.maximum(bh.sum(1, keepdims=True), 1e-300)
        memo = {}

        def tables(excluded):
            excluded = tuple(sorted(set(excluded)))
            if excluded not in memo:
                fit = ~np.isin(ft, excluded) & ~np.isin(fs, excluded)
                C = np.bincount(code[fit] * K + y[fit], minlength=S * K).reshape(S, K)
                prior = np.bincount(y[fit], minlength=K) + 1. / K
                if d == 0 and z >= 0:
                    C[:, z] += C[:, -1]
                    C[:, -1] = 0
                    prior[z] += prior[-1]
                    prior[-1] = 0
                prior /= prior.sum()
                ts = []
                for alpha in ALPHAS:
                    if np.isinf(alpha):
                        ts.append(identity)
                    else:
                        T = C + alpha * identity + .5 * prior[None]
                        ts.append(T / T.sum(1, keepdims=True))
                memo[excluded] = ts
            return memo[excluded]

        def choose(excluded):
            eligible = ~np.isin(ft, excluded) & ~np.isin(fs, excluded)
            if eligible.sum() < 30:
                return len(ALPHAS) - 1, 0.0
            # Every inner validation edge excludes both endpoints as well as
            # the outer pair's folds from its own sufficient statistics.
            predictions = np.empty((len(ALPHAS), len(tv)))
            for pairfold in groups:
                ev = eligible & (np.minimum(ft, fs) == pairfold[0]) & (np.maximum(ft, fs) == pairfold[1])
                if not ev.any():
                    continue
                rows = np.flatnonzero(ev)
                for a, T in enumerate(tables((*excluded, *pairfold))):
                    predictions[a, ev] = redv[rows, y[ev]] + wv[ev, 0] * T[code[ev], y[ev]]
            rows = np.flatnonzero(eligible)
            base_prob = cv[rows, y[eligible]]
            lam = np.asarray(LAMBDAS)[:, None, None]
            # All components are normalised; only the answered columns enter
            # the proper log score during the inner choice.
            mix = lam * predictions[None, :, eligible] + (1 - lam) * base_prob[None, None]
            losses = -np.log(1e-3 + (1 - K * 1e-3) * np.maximum(mix, 0.)).sum(2)
            li, ai = np.unravel_index(int(losses.argmin()), losses.shape)
            return ai, LAMBDAS[li]

        newv = cv.copy()
        for excluded in groups:
            ev = (np.minimum(ft, fs) == excluded[0]) & (np.maximum(ft, fs) == excluded[1])
            a, lam = choose(excluded)
            T = tables(excluded)[a]
            newv[ev] = lam * (redv[ev] + wv[ev] * T[code[ev]]) + (1 - lam) * cv[ev]
        oldloss, newloss = _loss(cv, y), _loss(newv, y)
        half = ft % 2 == 0
        if not all(newloss[m].mean() < oldloss[m].mean() - 1e-6
                   for m in (np.ones(len(y), bool), half, ~half) if m.any()):
            continue
        a, lam = choose(())
        if lam <= 0:
            continue
        T = tables(())[a]
        candidate = lam * (redh + wh * T[hcode]) + (1 - lam) * bh
        candidate = np.maximum(candidate, 0.)
        candidate /= np.maximum(candidate.sum(1, keepdims=True), 1e-300)
        final[k][th] = candidate
        upd = cur.copy(); upd[pos[tv]] = newv
        final_oofs[k] = upd
        changed[k] = float(oldloss.mean() - newloss.mean())
        picks[k] = {'alpha': float(ALPHAS[a]) if np.isfinite(ALPHAS[a]) else 'identity', 'lambda': lam}
    report[str(d)] = {'changed': len(changed), 'gain': changed, 'picks': picks}
