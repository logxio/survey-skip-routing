"""Fold-local observed-parent transition tensors and shared response statistics."""
import time
import numpy as np
from .base import _nll
from .cross_round import dominates

ANCHORS = ('Q06', 'Q07', 'Q08')
CHILDREN = ('Q09', 'Q10', 'Q11', 'Q12', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18')


def normalize(p):
    p = np.maximum(p, 1e-10)
    return p / p.sum(-1, keepdims=True)


def codes(X):
    # Mixed radix uses only GIVEN categories, with no target-dependent grouping.
    out = np.zeros(len(X), np.int64)
    for col in X.T:
        out = out * (int(col.max()) + 1) + col
    _, inverse = np.unique(out, return_inverse=True)
    return inverse


def count(ids, y, k, fit, groups):
    return np.bincount(ids[fit] * k + y[fit], minlength=groups * k).reshape(groups, k).astype(float)


def transition_tensor(y, z, nz, k, fit, query, country, coarse, posterior):
    nc, ng = int(country.max()) + 1, int(coarse.max()) + 1
    marginal = (np.bincount(y[fit], minlength=k) + 1. / k) / (len(fit) + 1.)
    cz = np.bincount(z[fit] * k + y[fit], minlength=nz * k).reshape(nz, k)
    global_t = (cz + 8. * marginal) / (cz.sum(1, keepdims=True) + 8.)
    ids = country * nz + np.maximum(z, 0)
    cc = count(ids, y, k, fit, nc * nz).reshape(nc, nz, k)
    ct = (cc + 16. * global_t) / (cc.sum(2, keepdims=True) + 16.)
    ids = coarse * nz + np.maximum(z, 0)
    cg = count(ids, y, k, fit, ng * nz).reshape(ng, nz, k)
    gc = np.zeros(ng, np.int64)
    gc[coarse] = country
    gt = (cg + 12. * ct[gc]) / (cg.sum(2, keepdims=True) + 12.)
    return normalize(np.einsum('nz,nzk->nk', posterior, gt[coarse[query]], optimize=False))


def joint_posterior(work, fit, query, country, anchor_p):
    sizes = [work[a]['K'] for a in ANCHORS]
    z = np.zeros(len(country), np.int64)
    for a, size in zip(ANCHORS, sizes):
        z = z * size + np.maximum(work[a]['y'], 0)
    nz = int(np.prod(sizes))
    nc = int(country.max()) + 1
    c = count(country, z, nz, fit, nc)
    global_p = (c.sum(0) + .5 / nz) / (len(fit) + .5)
    empirical = (c + 16. * global_p) / (c.sum(1, keepdims=True) + 16.)
    cube = empirical.reshape(nc, *sizes)
    independent = np.ones_like(cube)
    product = np.ones((len(query), 1))
    for axis, a in enumerate(ANCHORS):
        dims = tuple(i for i in range(1, 4) if i != axis + 1)
        margin = cube.sum(axis=dims)
        shape = [nc, 1, 1, 1]; shape[axis + 1] = sizes[axis]
        independent *= margin.reshape(shape)
        product = (product[:, :, None] * anchor_p[a][:, None, :]).reshape(len(query), -1)
    dependence = empirical / np.maximum(independent.reshape(nc, nz), 1e-9)
    # Empirical joint dependence preserves observed anchor associations.
    joint = normalize(product * dependence[country[query]])
    return z, nz, joint


def shared_support(work, items, fit, query, cell, country, name, base):
    """Other-question sufficient statistics with exact row exclusion on fit."""
    nc, nu = int(country.max()) + 1, int(cell.max()) + 1
    residual = np.zeros(len(cell), float); weight = np.zeros(len(cell), float)
    for other, j in work.items():
        if other == name:
            continue
        vals = items[other]['values']
        positive = [i for i, v in enumerate(vals) if str(v).lower() in ('yes', 'agree', 'strongly agree')]
        if not positive:
            positive = [0]
        valid = fit[(j['y'][fit] >= 0) & (j['y'][fit] < len(vals))]
        support = np.isin(j['y'][valid], positive).astype(float)
        n = np.bincount(country[valid], minlength=nc)
        c = np.bincount(country[valid], weights=support, minlength=nc)
        rate = (c + 1.) / (n + 2.)
        residual[valid] += support - rate[country[valid]]
        weight[valid] += 1.
    value = residual / np.maximum(weight, 1.)
    sums = np.bincount(cell[fit], weights=value[fit], minlength=nu)
    n = np.bincount(cell[fit], minlength=nu)
    feature_fit = (sums[cell[fit]] - value[fit]) / (n[cell[fit]] - 1 + 4.)
    feature_query = sums[cell[query]] / (n[cell[query]] + 4.)
    y = work[name]['y']; k = work[name]['K']
    # Learn a class-specific conditional correction, retaining the gate mass.
    cc = count(country, y, k, fit, nc)
    prior = (cc + 1. / k) / (cc.sum(1, keepdims=True) + 1.)
    target = np.eye(k)[y[fit]] - prior[country[fit]]
    slope = feature_fit @ target / (feature_fit @ feature_fit + 8.)
    shift = feature_query[:, None] * slope
    p = base.copy()
    width = k - 1 if items[name].get('gate') else k
    logits = np.log(np.maximum(p[:, :width], 1e-8)) + shift[:, :width] / np.maximum(p[:, :width], .08)
    logits -= logits.max(1, keepdims=True)
    p[:, :width] = normalize(np.exp(logits)) * p[:, :width].sum(1, keepdims=True)
    return normalize(p)


def attach(X, given, work, fold, chain, items, deadline=float('inf')):
    start = time.monotonic()
    info = {'seconds': 0., 'offered': 0, 'targets': {}}
    if not set(ANCHORS).issubset(work) or 'Country' not in given:
        return info
    country = X[:, given.index('Country')]
    coarse = codes(X[:, [given.index(a) for a in ('Country', 'Q05', 'Q13')]])
    cell = codes(X)
    rows = np.flatnonzero(np.logical_and.reduce([j['train'] for j in work.values()]))
    rest = np.flatnonzero(~np.isin(np.arange(len(X)), rows))
    if len(rows) < 25 or len(np.unique(fold[rows])) != 5:
        return info
    targets = [name for name in CHILDREN if name in work]
    outputs = {name: [np.zeros((4, len(rows), work[name]['K'])), np.zeros((4, len(X), work[name]['K']))] for name in targets}
    bases = {}
    for name in targets:
        j = work[name]
        idx = j['arms'].index('conditional_cell_eb') if 'conditional_cell_eb' in j['arms'] else j['arms'].index('full_cell_eb') if 'full_cell_eb' in j['arms'] else 0
        bases[name] = (j['oof'][idx], j['all'][idx])
    for f in [*np.unique(fold[rows]), None]:
        if time.monotonic() >= deadline:
            info['seconds'] = time.monotonic() - start
            return info
        fit = rows if f is None else rows[fold[rows] != f]
        mask = None if f is None else fold[rows] == f
        query = rest if f is None else rows[mask]
        anchor_p = {a: normalize(chain.probabilities(a, fit)[query]) for a in ANCHORS}
        z, nz, joint = joint_posterior(work, fit, query, country, anchor_p)
        for name in targets:
            j = work[name]; k = j['K']; y = j['y']
            single = [transition_tensor(y, work[a]['y'], work[a]['K'], k, fit, query, country, coarse, anchor_p[a]) for a in ANCHORS]
            # Parent-specific law, pooled anchor law, and true joint-state law.
            parent = (items[name].get('gate') or {}).get('parent')
            p_parent = single[ANCHORS.index(parent)] if parent in ANCHORS else sum(single) / len(single)
            p_joint = transition_tensor(y, z, nz, k, fit, query, country, coarse, joint)
            base = bases[name][1][query] if f is None else bases[name][0][mask]
            p_shared = shared_support(work, items, fit, query, cell, country, name, base) if name in ('Q14', 'Q15', 'Q16', 'Q17', 'Q18') else base
            preds = np.stack((p_parent, sum(single) / len(single), p_joint, p_shared))
            # GIVEN-gated targets preserve the empirical baseline's asked mass.
            if parent in given:
                mass = base[:, :-1].sum(1)
                preds[:, :, :-1] = normalize(preds[:, :, :-1]) * mass[None, :, None]
                preds[:, :, -1] = 1. - mass
            if f is None:
                outputs[name][1][:, query] = preds
            else:
                outputs[name][0][:, mask] = preds
    for name in targets:
        j = work[name]; y = j['y'][rows]; k = j['K']
        bo, bp = bases[name]; half = fold[rows] % 2 == 0
        oo, pp = outputs[name]
        best = _nll(bo, y, k); selected = None
        metrics = []
        for arm in range(4):
            metrics.append(float(_nll(oo[arm], y, k)))
            for alpha in (.125, .25, .5, 1.):
                o = (1. - alpha) * bo + alpha * oo[arm]
                score = _nll(o, y, k)
                if score < best and dominates(o, bo, y, k, half):
                    selected = (arm, alpha, o); best = score
        info['targets'][name] = {'base_nll': float(_nll(bo, y, k)), 'raw_nll': metrics, 'selected': None, 'gain': float(_nll(bo, y, k) - best)}
        if selected is not None:
            arm, alpha, o = selected
            p = bp.copy(); p[rest] = (1. - alpha) * bp[rest] + alpha * pp[arm, rest]
            j['arms'].append('rb_cell_eb'); j['oof'].append(o); j['all'].append(p)
            info['targets'][name]['selected'] = {'law': ('parent', 'pooled', 'joint', 'shared_support')[arm], 'alpha': alpha}
            info['offered'] += 1
    info['seconds'] = time.monotonic() - start
    return info
