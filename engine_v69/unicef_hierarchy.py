import ortho_config as _ortho
"""Nested demographic simplex estimates and residual conditional coupling."""
from itertools import combinations
import time
import numpy as np
from .full_cell_eb import Cells, _GIVEN
from .base import _nll
from .cross_round import dominates

HIERARCHY_ENABLED = _ortho.V66_HIERARCHY
GRAPH_ENABLED = _ortho.V66_GRAPH


def normalize(p):
    p = np.maximum(p, 1e-9)
    return p / p.sum(axis=-1, keepdims=True)


class PrecisionCells(Cells):
    """All 21 fifth-order and seven sixth-order views share one count bank."""
    def __init__(self, X, given, items):
        # Integer category codes allow collision-free mixed-radix keys.
        import json
        from pathlib import Path
        self.n = len(X)
        self.X, self.given, self.items = X, given, items or {}
        self.config, self.count_totals = {}, {}
        self.country = given.index('Country')
        self.external = {}
        for row in json.loads(Path(__file__).with_name('country_marginals.json').read_text()):
            self.external.setdefault((row['country'], row['item']), {})[row['option']] = float(row['reported_percent']) / 100.
        self.paths = [tuple(p) for d in (5, 6, 7) for p in combinations(_GIVEN, d)]
        self.ids, self.orders = [], []
        for names in self.paths:
            key = np.zeros(len(X), np.int64)
            for name in names:
                col = X[:, given.index(name)].astype(np.int64)
                key = key * (int(col.max()) + 1) + col
            _, ids = np.unique(key, return_inverse=True)
            self.ids.append(ids)
            self.orders.append(len(names))
        self.sizes = [int(ids.max()) + 1 for ids in self.ids]
        _, self.representatives = np.unique(self.ids[-1], return_index=True)
        self.parents = [[j for j, p in enumerate(self.paths) if len(p) == len(names)-1 and set(p).issubset(names)] for names in self.paths]
        self.grid = [(12., 4.)]

    @staticmethod
    def pool(ps, ns):
        ps, ns = np.asarray(ps), np.asarray(ns)
        # Inverse total Dirichlet posterior variance, tempered by observed support.
        variance = (1. - np.square(ps).sum(-1, keepdims=True)) / (ns + 1.)
        precision = (ns / (ns + 4.)) / np.maximum(variance, 1. / ((ns + 1.) * ps.shape[-1] * 16.))
        precision += 1e-12
        return (ps * precision).sum(0) / precision.sum(0)

    def predict(self, counts, alphas=(12., 4.)):
        prior, values = counts
        weighted, precisions = [], []
        for i, c in enumerate(values):
            parent_ids = self.parents[i]
            # Each parent's precision is reused by both of its supersets.
            parent = (sum(weighted[j] for j in parent_ids) /
                      sum(precisions[j] for j in parent_ids)) if parent_ids else prior
            n = c.sum(1, keepdims=True)
            alpha = alphas[0 if self.orders[i] == 5 else 1]
            p = (c + alpha * parent) / (n + alpha)
            if i + 1 == len(values):
                return p
            support = n + alpha
            variance = (1. - np.einsum('ij,ij->i', p, p)[:, None]) / (support + 1.)
            precision = (support / (support + 4.)) / np.maximum(
                variance, 1. / ((support + 1.) * p.shape[1] * 16.)) + 1e-12
            weighted.append(p * precision); precisions.append(precision)

    def predict_grid(self, counts):
        return {a: self.predict(counts, a) for a in self.grid}


def _basis(X):
    # Full raw categorical design, so no category ordering enters a residual fit.
    out = [np.ones((len(X), 1), dtype=np.float64)]
    for col in X.T:
        _, code = np.unique(col, return_inverse=True)
        out.append(np.eye(int(code.max()) + 1)[code][:, 1:])
    return np.column_stack(out)


def _ridge(Z, target, alpha, weight=None):
    from scipy.linalg import solve
    weighted = Z if weight is None else Z * weight[:, None]
    gram = Z.T @ weighted
    return solve(gram + alpha * np.eye(gram.shape[0]), weighted.T @ target, assume_a='pos')


def _offset_fit(Z, y, p0, alpha=36., weight=None):
    """Convex multinomial offset fit, diagonal-class Newton preconditioning."""
    k = p0.shape[1]
    target = np.eye(k)[y] if y.ndim == 1 else y
    weighted = Z if weight is None else Z * weight[:, None]
    w = np.zeros((Z.shape[1], k))
    logp = np.log(np.maximum(p0, 1e-9))
    gram = .5 * Z.T @ weighted + alpha * np.eye(Z.shape[1])
    from scipy.linalg import cho_factor, cho_solve
    factor = cho_factor(gram)
    for _ in range(12):
        logits = logp + Z @ w
        logits -= logits.max(1, keepdims=True)
        p = normalize(np.exp(logits))
        gradient = weighted.T @ (p-target) + alpha*w
        step = cho_solve(factor, gradient)
        w -= step
        w -= w.mean(1, keepdims=True)
        if np.max(np.abs(step)) < 1e-5:
            break
    return w


def _offset(p0, z, coef):
    logits = np.log(np.maximum(p0, 1e-9)) + z @ coef
    logits -= logits.max(1, keepdims=True)
    return normalize(np.exp(logits))


def _features(name, names, strong, eb):
    peers = [n for n in names if n != name]
    blocks = [strong[n][:, :-1] for n in peers]
    blocks += [(strong[n]-eb[n])[:, :-1] for n in peers]
    for a, b in (('Q16','Q17'), ('Q06','Q07')):
        if a != name and b != name and a in peers and b in peers:
            blocks.append((strong[a][:, :, None] * strong[b][:, None, :]).reshape(len(strong[a]), -1))
    return np.column_stack(blocks)


def attach(X, given, items, work, fold, deadline):
    """All fitted context excludes an entire outer fold, across every target."""
    start = time.monotonic()
    info = {'targets': {}, 'completed_folds': 0, 'paths_5': 21, 'paths_6': 7}
    if len(work) != 12 or not set(_GIVEN).issubset(given):
        return info
    names = list(work)
    common = np.logical_and.reduce([j['train'] for j in work.values()])
    if any(not np.array_equal(j['train'], common) for j in work.values()):
        return info
    rows = np.flatnonzero(common); rest = np.flatnonzero(~common)
    if len(rows) < 60:
        return info
    cells = PrecisionCells(X, given, items)
    totals = {n: cells.sufficient_counts(work[n]['y'], work[n]['K'], rows) for n in names}
    ans_rows = {n: rows[work[n]['y'][rows] < work[n]['K'] - 1]
                for n in names if items[n].get('gate')}
    ans_totals = {n: cells.sufficient_counts(work[n]['y'], work[n]['K'] - 1, r)
                  for n, r in ans_rows.items()}
    folds = tuple(int(f) for f in np.unique(fold[rows]))
    masks = {f: fold[rows] == f for f in folds}
    fold_rows = {f: rows[masks[f]] for f in folds}
    fold_totals = {f: {n: cells.sufficient_counts(work[n]['y'], work[n]['K'], fold_rows[f]) for n in names} for f in folds}
    fold_ans_totals = {f: {n: cells.sufficient_counts(work[n]['y'], work[n]['K']-1,
        fold_rows[f][work[n]['y'][fold_rows[f]] < work[n]['K']-1]) for n in ans_rows} for f in folds}
    representatives = cells.representatives
    ids = cells.ids[-1]
    if GRAPH_ENABLED:
        # Equal (demographic cell, fold) rows have exactly equal model inputs.
        # Sum their response counts, preserving the original likelihood weights.
        group_key = (np.where(common, fold, max(folds)+1) * len(representatives) + ids)
        _, group_reps, group_ids = np.unique(group_key, return_index=True, return_inverse=True)
        group_fold = np.where(common[group_reps], fold[group_reps], -1)
        group_weights = np.bincount(group_ids).astype(float)
        Z = _basis(X[group_reps])
        group_targets = {}
        for name in names:
            k = work[name]['K']
            counts = np.bincount(group_ids[rows]*k+work[name]['y'][rows], minlength=len(group_reps)*k).reshape(-1,k)
            group_targets[name] = counts / group_weights[:, None]
    cache = {}

    def subtract(total, removed):
        return (total[0]-removed[0], total[1]-removed[1],
                [a-b for a,b in zip(total[2], removed[2])])

    def predictions(excluded):
        key = tuple(sorted(excluded))
        if key in cache:
            return cache[key]
        selected = ~np.isin(fold[rows], key)
        fit = rows[selected]
        needed = np.unique(ids[rest if not key else rows[~selected]])
        queries = representatives[needed]
        result = {}
        for name in names:
            j = work[name]; y,k = j['y'],j['K']
            stats = totals[name]
            for f in key:
                stats = subtract(stats, fold_totals[f][name])
            raw = cells.predict(cells.counts_from_totals(y,k,fit,queries,name,stats))
            conditioned = raw.copy()
            if name in ans_rows:
                qa = fit[y[fit] < k-1]
                if len(qa) >= 20:
                    astats = ans_totals[name]
                    for f in key:
                        astats = subtract(astats, fold_ans_totals[f][name])
                    cond = cells.predict(cells.counts_from_totals(y,k-1,qa,queries,name,astats,answered_only=True))
                    conditioned[:,:-1] = cond * raw[:,:-1].sum(1,keepdims=True)
            full_raw = np.zeros((len(representatives), k)); full_raw[needed] = raw
            full_cond = np.zeros_like(full_raw); full_cond[needed] = conditioned
            result[name] = full_raw, full_cond
        cache[key] = result
        return result

    oo = {n: np.zeros((len(rows), work[n]['K'])) for n in names}
    pp = {n: np.zeros_like(work[n]['all'][0]) for n in names}
    full_oo = {n: np.zeros_like(oo[n]) for n in names}
    full_pp = {n: np.zeros_like(pp[n]) for n in names}
    go = {n: np.zeros_like(oo[n]) for n in names}
    gp = {n: np.zeros_like(pp[n]) for n in names}
    for f in [*folds, None]:
        if time.monotonic() >= deadline:
            info['seconds'] = time.monotonic()-start
            return info
        fit = rows if f is None else rows[~masks[f]]
        query = rest if f is None else fold_rows[f]
        excluded = () if f is None else (f,)
        pred = predictions(excluded)
        eb, strong = {}, {}
        for name in names:
            raw,cond = pred[name]
            if f is None:
                pp[name][query] = cond[ids[query]]; full_pp[name][query] = raw[ids[query]]
            else:
                oo[name][masks[f]] = cond[ids[query]]; full_oo[name][masks[f]] = raw[ids[query]]
        if GRAPH_ENABLED:
            gf = np.flatnonzero((group_fold >= 0) & ((group_fold != f) if f is not None else True))
            gq = np.flatnonzero(group_fold == (-1 if f is None else f))
            gw = group_weights[gf]
            pf_by_name = {n: np.zeros((len(gf), work[n]['K'])) for n in names}
            local = group_fold[gf]
            for h in folds:
                va = local == h
                if not va.any():
                    continue
                inner = predictions((*excluded,h))
                for name in names:
                    pf_by_name[name][va] = inner[name][1][ids[group_reps[gf[va]]]]
            residual = np.column_stack([group_targets[n][gf]-pf_by_name[n] for n in names])
            sf = np.zeros_like(residual)
            # One weighted multi-response factorization fits all twelve targets.
            for h in folds:
                va = local == h
                if va.any():
                    coef = _ridge(Z[gf[~va]],residual[~va],36.,gw[~va])
                    sf[va] = Z[gf[va]] @ coef
            coef = _ridge(Z[gf], residual, 36.,gw)
            sq = Z[gq] @ coef
            offset = 0
            for name in names:
                k = work[name]['K']; pq = pred[name][1][ids[group_reps[gq]]]
                arr = np.zeros((len(group_reps),k)); arr[gf] = pf_by_name[name]; arr[gq] = pq
                eb[name] = arr
                st = arr.copy()
                st[gf] = normalize(np.maximum(arr[gf]+sf[:,offset:offset+k],1e-5))
                st[gq] = normalize(np.maximum(pq+sq[:,offset:offset+k],1e-5))
                strong[name] = st
                offset += k
            for name in names:
                j = work[name]
                z = _features(name,names,strong,eb)
                mu = np.average(z[gf],axis=0,weights=gw)
                sd = np.sqrt(np.average((z[gf]-mu)**2,axis=0,weights=gw)); sd[sd<.01]=1.
                z=np.column_stack((np.ones(len(group_reps)),(z-mu)/sd))
                coef=_offset_fit(z[gf],group_targets[name][gf],eb[name][gf],weight=gw)
                out = np.zeros_like(eb[name])
                out[gq]=_offset(eb[name][gq],z[gq],coef)
                if f is None: gp[name][query]=out[group_ids[query]]
                else: go[name][masks[f]]=out[group_ids[query]]
        info['completed_folds'] += 1
    half = fold[rows] % 2 == 0
    for name in names:
        j=work[name]; k=j['K']; y=j['y'][rows]
        selected_arm='full_state'
        if dominates(oo[name],full_oo[name],y,k,half):
            selected_arm='answered_conditioned'
        else:
            oo[name],pp[name]=full_oo[name],full_pp[name]
        old=j['oof'][j['arms'].index('full_cell_eb')] if 'full_cell_eb' in j['arms'] else j['oof'][0]
        rec={'selected_arm':selected_arm,'hierarchy_nll':float(_nll(oo[name],y,k)), 'reference_nll':float(_nll(old,y,k))}
        admitted = HIERARCHY_ENABLED and dominates(oo[name],old,y,k,half)
        if admitted:
            j['arms'].append('hierarchy_eb'); j['oof'].append(oo[name]); j['all'].append(pp[name])
            rec['hierarchy_admitted']=True
        if GRAPH_ENABLED:
            rec['graph_nll']=float(_nll(go[name],y,k))
            # The residual must beat the incumbent cell estimate, not only a rejected hierarchy.
            if dominates(go[name],oo[name],y,k,half) and (admitted or not _ortho.V72_K1 or dominates(go[name],old,y,k,half)):
                j['arms'].append('graph_residual'); j['oof'].append(go[name]); j['all'].append(gp[name])
                rec['graph_admitted']=True
        info['targets'][name]=rec
    info['unique_cells']=len(representatives)
    info['count_partitions']=len(folds)
    info['seconds']=time.monotonic()-start
    return info
