import ortho_config as _ortho
"""Finite-population complement projections with rounded report constraints."""
import json
import time
from pathlib import Path
import numpy as np
from .base import _nll
from .cross_round import dominates
from .finite_pop_raking import normalize, kl_project

ENABLED = _ortho.V66_POPULATION


def compile_rules(X, given, items, work):
    grouped = {}
    for r in json.loads(Path(__file__).with_name('country_marginals.json').read_text()):
        if r['item'] not in work or any(n not in given for n in r.get('conditions', {})):
            continue
        key = (r['item'], r['country'], json.dumps(r.get('conditions', {}), sort_keys=True))
        grouped.setdefault(key, []).append(r)
    out = {}
    for (name, country, conditions), records in grouped.items():
        countries = items['Country']['values']
        if country not in countries:
            continue
        mask = X[:, given.index('Country')] == countries.index(country)
        for col, val in json.loads(conditions).items():
            values = val if isinstance(val, list) else [val]
            ids = [items[col]['values'].index(v) for v in values if v in items[col]['values']]
            mask &= np.isin(X[:, given.index(col)], ids)
        rates = {items[name]['values'].index(r['option']): r['reported_percent']/100.
                 for r in records if r['option'] in items[name]['values']}
        out.setdefault(name, []).append((mask, rates))
    return out


def project(job, fit, query, pred, rules, population_rows):
    """Subtract observed counts; project only outside feasible rounded intervals.

    Missing phase-one rows are exchangeable members of the complement. Religion
    strata retain their row odds under the common country multiplier.
    """
    out = pred.copy(); y = job['y']; K = job['K']; used = 0
    for mask, rates in rules:
        ff = fit[mask[fit]]; qq = mask[query]
        if len(ff) < 20 or not qq.any():
            continue
        observed = np.bincount(y[ff], minlength=K).astype(float)
        present = int(mask.sum())
        # The unexposed TEST cohort is included in phase-one remaining mass.
        n_total = population_rows * present / len(mask)
        remaining = n_total-len(ff)
        if remaining <= 0:
            continue
        current = out[qq].mean(0); target = current.copy(); fixed = []
        for col, rate in rates.items():
            sample = observed[col]/len(ff)
            fpc = max(0., 1-len(ff)/max(n_total, 1.))
            variance = max(rate*(1-rate), .0025)/len(ff)*fpc + .01**2/12
            if abs(sample-rate) > 3*np.sqrt(variance):
                continue
            # Percent rounding and one omitted respondent bound report counts.
            lo = max(0., (n_total*max(0.,rate-.005)-1-observed[col])/remaining)
            hi = min(1., (n_total*min(1.,rate+.005)+1-observed[col])/remaining)
            if lo > hi:
                continue
            target[col] = np.clip(current[col], lo, hi)
            fixed.append(col)
        if not fixed:
            continue
        free = np.ones(K, bool); free[fixed] = False
        total = target[fixed].sum()
        if total >= 1. or not free.any():
            target = normalize(target)
        else:
            target[free] = current[free]*(1-total)/max(current[free].sum(),1e-12)
        if np.max(np.abs(target-current)) > 1e-10:
            out[qq] = kl_project(out[qq], target)
            used += 1
    return out, used


def apply(schema, X, given, work, fold, final, oofs, deadline):
    info = {}
    if not ENABLED or len(work) != 12:
        return info
    rules = compile_rules(X, given, schema['items'], work)
    N = int(schema.get('dataset', {}).get('n_rows', len(X)))
    for name, j in work.items():
        if time.monotonic() >= deadline:
            break
        rows=np.flatnonzero(j['train']); rest=np.flatnonzero(~j['train'])
        if len(rows)<60 or len(np.unique(fold[rows]))!=5:
            continue
        old=oofs[name]; trial=old.copy(); used=0
        for f in range(5):
            va=fold[rows]==f
            trial[va], n=project(j,rows[~va],rows[va],old[va],rules.get(name,()),N)
            used+=n
        y=j['y'][rows]; half=fold[rows]%2==0
        gain=float(_nll(old,y,j['K'])-_nll(trial,y,j['K']))
        accepted=dominates(trial,old,y,j['K'],half)
        if accepted:
            final[name][rest],_=project(j,rows,rest,final[name][rest],rules.get(name,()),N)
            oofs[name]=trial
        info[name]={'groups':used,'gain':gain,'accepted':bool(accepted),
                    'population_scope':'rounded_report_exchangeable_complement',
                    'exact_census_counts':False}
    return info
