"""Cross-round demographic conservation and mutual neighbourhood tiers.

Candidate geometry reads GIVEN only. The cohort check for a visible pair
excludes both endpoint folds, so its answers never select its own match.
"""
import numpy as np
from . import cross_round as CR, twin_identity as TI, twin_collision as TC


def bracket_pairs(X, nraw, wcol, given, items, taken):
    wave = X[:, wcol]
    age = given.index('s2_1b_age')
    av = items['s2_1b_age']['values']
    ar = np.full(len(av) + 1, -100)
    ix = [i for i, v in enumerate(av) if str(v)[:1].isdigit()]
    ar[ix] = np.arange(len(ix))
    cand = {}
    for sex in CR.COUNTS:
        for young, old in zip(sex[:-1], sex[1:]):
            a, b = given.index(young), given.index(old)
            vals = []
            for name, col in ((young, a), (old, b)):
                lo, hi = CR._bounds(items[name]['values'])
                vals.append(np.where(np.isfinite(lo) & (lo == hi), lo, np.nan)[X[:, col]])
            va, vb = vals
            cols = [c for c in range(nraw) if c not in (wcol, age, a, b)]
            groups = {}
            for i in np.flatnonzero(~taken & (wave < 2) & np.isfinite(va + vb)):
                key = X[i, cols].tobytes()
                groups.setdefault(key, ([], []))[int(wave[i])].append(i)
            for left, right in groups.values():
                for i in left:
                    for j in right:
                        if va[j] == va[i] - 1 and vb[j] == vb[i] + 1 and (
                                X[i, age] == X[j, age] or ar[X[j, age]] == ar[X[i, age]] + 1):
                            cand.setdefault(i, set()).add(j)
                            cand.setdefault(j, set()).add(i)
    return [(i, next(iter(s))) for i, s in cand.items()
            if wave[i] == 0 and len(s) == 1 and cand.get(next(iter(s))) == {i}]


def kernel_pairs(X, given, items, taken):
    kernel = CR.Kernel({'items': items}, X, given)
    n = len(X)
    near = np.full(n, -1, int)
    second = np.full(n, -1, int)
    best = np.full(n, np.inf)
    unique = np.zeros(n, bool)
    origin = given.index(CR.BLOCK[2])
    for query, donor, distance in kernel.edges:
        ok = ~taken[donor]
        if not ok.any():
            continue
        ds = distance[:, ok].astype(float)
        dd = donor[ok]
        ds[X[query, origin, None] != X[dd, origin][None]] = np.inf
        k = ds.argmin(1)
        r = np.arange(len(query))
        v = ds[r, k]
        eq = (np.abs(ds - v[:, None]) < 1e-7).sum(1)
        near[query] = dd[k]
        best[query] = v
        unique[query] = eq == 1
        # Runner-up donor: same block, same origin, nearly as similar, but not
        # the mutual match. It carries the demographic agreement without identity.
        ds[r, k] = np.inf
        k2 = ds.argmin(1)
        second[query] = np.where(np.isfinite(ds[r, k2]), dd[k2], -1)
    pairs = [(int(a), int(near[a])) for a in np.flatnonzero(
        ~taken & (kernel.wave == 0) & unique & (best <= 0.65))
        if near[a] >= 0 and unique[near[a]] and near[near[a]] == a and best[near[a]] <= 0.65]
    return pairs, [(int(second[a]), int(second[b])) for a, b in pairs]


def cohort_guard(pairs, exact, work, fold, wave, null=None):
    """Tier-level identity test on visible pairs outside both endpoint folds.
    With runner-up donors (null) the pair must agree beyond its own runner-ups
    by three standard differences; demographic look-alikes agree as much as theirs."""
    if not pairs:
        return [], {}
    Y, train = TC._answers(work)
    ab = np.asarray(pairs, int).reshape(-1, 2)
    vis = train[ab[:, 0]] & train[ab[:, 1]]
    agreement = (Y[ab[:, 0]] == Y[ab[:, 1]]).mean(1)
    if null is not None:
        nl = np.asarray(null, int).reshape(-1, 2)
        sides = []
        for s in (0, 1):
            ok = (nl[:, s] >= 0) & train[ab[:, s]] & train[np.maximum(nl[:, s], 0)]
            sides.append(np.where(ok, (Y[ab[:, s]] == Y[np.maximum(nl[:, s], 0)]).mean(1), np.nan))
        base = np.nanmean(np.vstack(sides), 0) if len(ab) else np.zeros(0)
        lift = agreement - base
    masks = {}
    keep = []
    for i, (a, b) in enumerate(ab):
        excluded = tuple(sorted({int(fold[a]), int(fold[b])} - {-1})) if vis[i] else ()
        if excluded not in masks:
            tr = train & ~np.isin(fold, excluded)
            vv = vis & tr[ab[:, 0]] & tr[ab[:, 1]]
            if null is None:
                tau = TC.agreement_threshold(Y, tr, exact, wave, np.random.default_rng(20261114))
                masks[excluded] = bool(vv.sum() >= 20 and agreement[vv].mean() >= tau)
            else:
                d = lift[vv & np.isfinite(lift)]
                masks[excluded] = bool(len(d) >= 20 and d.mean() > 3.0 * d.std(ddof=1) / np.sqrt(len(d)))
        if masks[excluded]:
            keep.append((int(a), int(b)))
    report = {'geometry_pairs': len(pairs), 'accepted_pairs': len(keep),
              'visible_geometry_pairs': int(vis.sum()),
              'visible_agreement': float(agreement[vis].mean()) if vis.any() else None}
    if null is not None:
        d = lift[vis & np.isfinite(lift)]
        report.update(runner_up_lift=float(d.mean()) if len(d) else None,
                      runner_up_z=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std() > 0 else None)
    return keep, report


def tiers(X, nraw, wcol, given, items, taken, exact, work, fold):
    used = taken.copy()
    found, report = {}, {}
    for label, fn in (
            ('bracket', lambda: bracket_pairs(X, nraw, wcol, given, items, used)),
            ('kernel', lambda: kernel_pairs(X, given, items, used)),
            ('household', lambda: TC.household_pairs(X, nraw, wcol, given, used))):
        raw = fn()
        # A rejected household tier stays with the late   pass (main.py).
        if label == 'kernel':
            pairs, report[label] = cohort_guard(raw[0], exact, work, fold, X[:, wcol], null=raw[1])
        else:
            pairs, report[label] = cohort_guard(raw, exact, work, fold, X[:, wcol])
        found[label] = pairs
        if pairs:
            used[np.asarray(pairs).ravel()] = True
    return found, report
