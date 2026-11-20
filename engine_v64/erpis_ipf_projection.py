"""Fold-excluded ERPIS conditional KL tilts of the completed ensemble.

Published constraints modify conditional answer shape only. The empirical
reference and parent association table exclude the query fold; queried parent
labels are never read. One fixed shrinkage rule, with no outcome-based search,
precedes the separate three-standard-difference acceptance gate.
"""
import time
import numpy as np
from .erpis_source_data import ALIGNED_PRIORS
PRIORS = {"unhcr": ALIGNED_PRIORS}
from .survey_priors import _report_vector
from .propagate_structural_gates import ZERO_ON_SKIP, _rake


def _unit(p):
    p = np.maximum(np.asarray(p, float), 1e-12)
    return p / p.sum(axis=-1, keepdims=True)


def tilt(base, log_ratio, weight, asked):
    """KL exponential tilt; preserve all non-answer coordinates exactly."""
    out = np.array(base, dtype=float, copy=True)
    mass = out[:, :asked].sum(1, keepdims=True)
    z = np.log(np.maximum(out[:, :asked], 1e-300)) + weight[:, None] * log_ratio
    z -= z.max(1, keepdims=True)
    out[:, :asked] = _unit(np.exp(z)) * mass
    return out


def apply(schema, X, given, work, fold, final, final_oofs, stacked, deadline):
    start = time.monotonic()
    info = {'rules_available': len(PRIORS.get('unhcr', ())), 'rules_applied': 0,
            'items_proposed': 0, 'folds': 5, 'trace': {}, 'seconds': 0.0}
    if 'wave' not in given:
        return info
    # Freeze all input predictions: rule order must not update a later parent.
    old = {k: np.asarray(v) for k, v in final.items()}
    oo = {k: np.asarray(final_oofs.get(k, stacked[k][0])) for k in stacked}
    rules_by = {}
    for name, cond, constraints in PRIORS.get('unhcr', ()):
        if name in work and name in stacked:
            rules_by.setdefault(name, []).append((cond, constraints))
    for name, rules in rules_by.items():
        if time.monotonic() >= deadline:
            break
        j = work[name]
        rows, rest = np.flatnonzero(j['train']), np.flatnonzero(~j['train'])
        fr = fold[rows]
        if len(rows) < 40 or len(np.unique(fr)) != 5:
            continue
        values = schema['items'][name]['values']
        A = len(values)
        if A > j['K'] or A < 2:
            continue
        excluded_wave = None
        if name in ZERO_ON_SKIP:
            excluded_wave = [str(v) for v in schema['items']['wave']['values']].index('1')
        valid_rows = np.ones(len(X), bool)
        if excluded_wave is not None:
            valid_rows &= X[:, given.index('wave')] != excluded_wave
        delta_o = np.zeros((len(rows), A))
        delta_r = np.zeros((len(rest), A))
        used = 0
        for conditions, constraints in rules:
            if time.monotonic() >= deadline:
                break
            exact = valid_rows.copy()
            event = np.ones(len(rows), bool)
            parents, choices = [], []
            supported = True
            for key, opts in conditions:
                if key not in schema['items']:
                    supported = False
                    break
                ids = [i for i, v in enumerate(schema['items'][key]['values']) if v in opts]
                if key in given:
                    exact &= np.isin(X[:, given.index(key)], ids)
                elif key in work and key != name and key in oo and np.array_equal(work[key]['train'], j['train']):
                    parents.append(key)
                    choices.append(ids)
                    event &= np.isin(work[key]['y'][rows], ids)
                else:
                    supported = False
                    break
            if not supported or len(parents) > 2 or not exact.any():
                continue
            eligible = exact[rows] & event & (j['y'][rows] < A)
            if not eligible.any():
                continue
            # Counts are labels on fit rows only, never other-fold OOF models.
            counts = np.array([np.bincount(j['y'][rows[(fr == f) & eligible]], minlength=A)[:A]
                               for f in range(5)], float)
            total = counts.sum(0)
            def ratio(c):
                emp = _unit(c + 0.5)
                target = _report_vector(emp, constraints, values)
                # Empirical Bayes effective sample size: published margins
                # contribute 32 observations in each country/wave condition.
                posterior = _unit(c + 0.5 + 32.0 * target)
                return np.log(posterior / emp)
            def event_weight(query, sel, pred):
                if not parents:
                    return exact[query].astype(float)
                if len(parents) == 1:
                    prob = pred[parents[0]][:, choices[0]].sum(1)
                else:
                    a, b = parents
                    Ka, Kb = work[a]['K'], work[b]['K']
                    state = work[a]['y'][rows[sel]] * Kb + work[b]['y'][rows[sel]]
                    M = (np.bincount(state, minlength=Ka*Kb) + 0.5).reshape(Ka, Kb)
                    joint = _rake(M / M.sum(), pred[a], pred[b]).reshape(len(query), Ka, Kb)
                    prob = joint[:, choices[0]][:, :, choices[1]].sum((1, 2))
                return exact[query] * prob
            for f in range(5):
                m = fr == f
                pred = {p: oo[p][m] for p in parents}
                weight = event_weight(rows[m], ~m, pred)
                delta_o[m] += weight[:, None] * ratio(total - counts[f])
            pred = {p: old[p][rest] for p in parents}
            delta_r += event_weight(rest, np.ones(len(rows), bool), pred)[:, None] * ratio(total)
            used += 1
        if used:
            # Rules for disjoint countries/waves have disjoint support;
            # overlapping reported events compose in log-probability space.
            o = tilt(oo[name], delta_o, np.ones(len(rows)), A)
            p = np.array(old[name], copy=True)
            p[rest] = tilt(p[rest], delta_r, np.ones(len(rest)), A)
            final_oofs[name], final[name] = o, p
            info['items_proposed'] += 1
            info['rules_applied'] += used
            info['trace'][name] = {'rules': used, 'max_skip_change': float(np.max(np.abs(o[:, A:] - oo[name][:, A:]), initial=0.0))}
    info['seconds'] = time.monotonic() - start
    return info
