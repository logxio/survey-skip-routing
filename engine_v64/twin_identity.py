"""Exact cross-round household identity, calibrated on visible pairs.

Two rows of opposite rounds whose GIVEN columns other than the round agree
exactly, and whose signature is unique inside each round, are one household
seen twice.  Per item and per direction, a transition T[v_twin -> y] (pair
counts shrunk toward the identity) and a copy weight rho are fitted by
maximum likelihood on visible pairs, mixed on top of the final prediction:

    p(y) = rho * T[y_twin, y] + (1 - rho) * p_final(y).

Visible-row OOF uses T and rho fitted without any pair touching the row's
fold; an item is changed only when that cross-fitted OOF improves on both
fold halves.  Without real twins the visible pairs agree at chance and rho
falls to zero, so the final prediction is returned unchanged.

DRIFT adds a second tier for rows left unmatched: one count/age column one
band apart, every other column equal, unique and mutual.

the configured option conditions the twin row on the target row's own routing: classes the
final prediction holds below tau (a closed gate in this round) get no twin
mass, T[y_twin, .] is renormalised over the open ones, and tau is fitted with
rho (tau = 0 is the unmasked mixture).
"""
import numpy as np

RHOS = np.linspace(0.0, 0.95, 20)
IDENTITY_PRIOR = 2.0
# Coincidental signature matches number in the tens; real households in the thousands.
MIN_EDGES = 200
V85_K2 = True
TAUS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2)
NUMERIC = ('s2_1b_age', 's2_3_boys_0_5', 's2_3_girls_0_5', 's2_3_boys_6_17', 's2_3_girls_6_17',
           's2_3_men_18_59', 's2_3_women_18_59', 's2_3_men_60plus', 's2_3_women_60plus')


def _unique_pairs(keys, wave, taken):
    """Rows of round 0 and 1 sharing a key that occurs once in each round."""
    groups = {}
    for i in np.flatnonzero(~taken):
        groups.setdefault((keys[i], int(wave[i])), []).append(i)
    pairs = []
    for (key, w), rows in groups.items():
        if w != 0 or len(rows) != 1:
            continue
        other = groups.get((key, 1))
        if other is not None and len(other) == 1:
            pairs.append((rows[0], other[0]))
    return pairs


def exact_pairs(X, n_raw, wave_col):
    cols = [c for c in range(n_raw) if c != wave_col]
    keys = [r.tobytes() for r in np.ascontiguousarray(X[:, cols])]
    wave = X[:, wave_col]
    return _unique_pairs(keys, wave, wave > 1)


def drift_pairs(X, n_raw, wave_col, given, schema, taken):
    """Unique mutual one-band drift on exactly one numeric column."""
    wave = X[:, wave_col]
    cand = {}
    for name in NUMERIC:
        if name not in given:
            continue
        c = given.index(name)
        vals = schema['items'][name]['values']
        rank = np.full(len(vals) + 1, -100)
        num = [i for i, v in enumerate(vals) if str(v)[:1].isdigit()]
        rank[num] = np.arange(len(num))
        r = rank[X[:, c]]
        cols = [k for k in range(n_raw) if k not in (wave_col, c)]
        rest = [row.tobytes() for row in np.ascontiguousarray(X[:, cols])]
        groups = {}
        for i in np.flatnonzero(~taken & (wave <= 1) & (r >= 0)):
            groups.setdefault(rest[i], []).append(i)
        for rows in groups.values():
            if len(rows) < 2:
                continue
            for a in rows:
                for b in rows:
                    if wave[a] == 0 and wave[b] == 1 and abs(r[a] - r[b]) == 1:
                        cand.setdefault(a, set()).add(b)
                        cand.setdefault(b, set()).add(a)
    return [(a, next(iter(s))) for a, s in cand.items()
            if wave[a] == 0 and len(s) == 1 and cand.get(next(iter(s))) == {a}]


def _open_rows(T, src, base, tau):
    """Twin rows T[src] restricted to the classes base leaves at or above tau."""
    R = T[src]
    if tau <= 0.0:
        return R
    M = R * (base >= tau)
    s = M.sum(1, keepdims=True)
    return np.where(s > 0, M / np.maximum(s, 1e-300), base)


def _fit(src, dst, base, K, mask):
    """T (K x K), rho and tau by ML on the masked edges; src/dst are answer codes."""
    C = np.bincount(src[mask] * K + dst[mask], minlength=K * K).reshape(K, K)
    T = C + IDENTITY_PRIOR * np.eye(K)
    T /= T.sum(1, keepdims=True)
    if not mask.any():
        return T, 0.0, 0.0
    B, d = base[mask], dst[mask]
    rows = np.arange(len(d))
    b = B[rows, d]
    R = T[src[mask]]
    td = R[rows, d]
    best = (np.inf, 0.0, 0.0)
    for tau in (TAUS if V85_K2 else (0.0,)):
        # The answered column of _open_rows(T, src, base, tau).
        if tau <= 0.0:
            t = td
        else:
            mass = (R * (B >= tau)).sum(1)
            t = np.where(mass > 0, td * (b >= tau) / np.maximum(mass, 1e-300), b)
        mix = RHOS[:, None] * t[None, :] + (1 - RHOS)[:, None] * b[None, :]
        loss = -np.log(1e-3 + (1 - K * 1e-3) * mix).sum(1)
        k = int(np.argmin(loss))
        if loss[k] < best[0]:
            best = (float(loss[k]), float(RHOS[k]), tau)
    return T, best[1], best[2]


def _nll(p, y, K):
    return float(-np.log(1e-3 + (1 - K * 1e-3) * p[np.arange(len(y)), y]).mean())


def apply(work, final, final_oofs, stacked, pairs, fold, wave, report, context=None, deadline=float('inf')):
    """Mix twin evidence into final (hidden rows) and final_oofs (visible rows).

    Directed edges target <- twin; T and rho are fitted per target round."""
    if context is not None:
        from .twin_posterior import apply_open
        return apply_open(work, final, final_oofs, stacked, pairs, fold, wave, report, *context)
    n = len(fold)
    report.update(pairs=len(pairs), changed=0)
    if not pairs:
        return
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    tgt = np.concatenate([a, b]); twin = np.concatenate([b, a])
    train = next(iter(work.values()))['train']
    pos = np.full(n, -1); pos[np.flatnonzero(train)] = np.arange(train.sum())
    ve = np.flatnonzero(train[tgt] & train[twin])
    he = np.flatnonzero(~train[tgt] & train[twin])
    report.update(visible_edges=len(ve), hidden_edges=len(he))
    if len(ve) < MIN_EDGES or not len(he):
        return
    vt, vw, ht, hw = tgt[ve], twin[ve], tgt[he], twin[he]
    ft, fw = fold[vt], fold[vw]
    half = ft % 2 == 0
    rhos = {}
    masked, gains = 0, []
    import time
    for name, j in work.items():
        if time.monotonic() >= deadline:
            break
        if name not in final or not np.array_equal(j['train'], train):
            continue
        y, K = j['y'], j['K']
        oof = np.asarray(final_oofs.get(name, stacked[name][0]), float)
        oof = oof / np.maximum(oof.sum(1, keepdims=True), 1e-300)
        src, dst = y[vw], y[vt]
        base_v = oof[pos[vt]]
        new_v = base_v.copy()
        hid_p = np.asarray(final[name][ht], float)
        hid_p = hid_p / np.maximum(hid_p.sum(1, keepdims=True), 1e-300)
        new_h = hid_p.copy()
        used, taus = [], []
        for d in (0, 1):
            dv = wave[vt] == d
            for f in np.unique(ft[dv]):
                ev = dv & (ft == f)
                T, r, tau = _fit(src, dst, base_v, K, dv & (ft != f) & (fw != f))
                new_v[ev] = r * _open_rows(T, src[ev], base_v[ev], tau) + (1 - r) * base_v[ev]
            T, r, tau = _fit(src, dst, base_v, K, dv)
            dh = wave[ht] == d
            new_h[dh] = r * _open_rows(T, y[hw[dh]], hid_p[dh], tau) + (1 - r) * hid_p[dh]
            used.append(r)
            taus.append(tau)
        if max(used) <= 0.0:
            continue
        if not all(_nll(new_v[m], dst[m], K) < _nll(base_v[m], dst[m], K) - 1e-6 for m in (np.ones(len(dst), bool), half, ~half) if m.any()):
            continue
        final[name][ht] = new_h
        upd = oof.copy()
        upd[pos[vt]] = new_v
        final_oofs[name] = upd
        rhos[name] = float(np.mean(used))
        masked += sum(t > 0 for t in taus)
        gains.append(_nll(base_v, dst, K) - _nll(new_v, dst, K))
    report.update(changed=len(rhos), rho_mean=float(np.mean(list(rhos.values()))) if rhos else 0.0,
                  masked_directions=int(masked), oof_edge_gain=float(np.mean(gains)) if gains else 0.0)


#  the twin passes move a gate parent's prediction but leave its
# children's skip where the gates put it.  A child takes the parent's change
# through the visible transition C[v, y] = P(child = y | parent = v):
#     p_child += lam * (p_parent_after - p_parent_before) @ C,
# lam on the visible OOF (C cross-fitted by fold), kept only when it wins on
# both fold halves.  Children run in topological order, so a moved child
# carries its own change to the next generation.
LAMS = (0.25, 0.5, 0.75, 1.0)


def snapshot(final, final_oofs, stacked, parents, train):
    hid = np.flatnonzero(~train)
    return {p: (np.array(final_oofs.get(p, stacked[p][0]), float), np.array(final[p][hid], float))
            for p in parents if p in final and p in stacked}


def _fix(P):
    P = np.maximum(P, 1e-9)
    return P / P.sum(1, keepdims=True)


def regate(work, final, final_oofs, stacked, before, names, items, fold, report):
    train = next(iter(work.values()))['train']
    rows = np.flatnonzero(train)
    hid = np.flatnonzero(~train)
    rf = fold[rows]
    half = rf % 2 == 0
    lams, gains = {}, []
    for name in names:
        gate = items[name].get('gate')
        p = gate.get('parent') if isinstance(gate, dict) else None
        if p not in before or name not in final or name not in stacked or not np.array_equal(work[name]['train'], train):
            continue
        Pb_o, Pb_h = before[p]
        Pa_o = np.asarray(final_oofs.get(p, stacked[p][0]), float)
        do = Pa_o / np.maximum(Pa_o.sum(1, keepdims=True), 1e-300) - Pb_o / np.maximum(Pb_o.sum(1, keepdims=True), 1e-300)
        dh = np.asarray(final[p][hid], float)
        dh = dh / np.maximum(dh.sum(1, keepdims=True), 1e-300) - Pb_h / np.maximum(Pb_h.sum(1, keepdims=True), 1e-300)
        if np.abs(do).max() < 1e-9:
            continue
        K, Kp = work[name]['K'], work[p]['K']
        yc, yp = work[name]['y'][rows], work[p]['y'][rows]
        ok = (yc >= 0) & (yp >= 0)
        marg = np.bincount(yc[ok], minlength=K) + 1.0
        marg /= marg.sum()

        def table(mask):
            C = np.bincount(yp[mask] * K + yc[mask], minlength=Kp * K).reshape(Kp, K) + 0.5 * marg
            return C / C.sum(1, keepdims=True)
        base = np.asarray(final_oofs.get(name, stacked[name][0]), float)
        base = base / np.maximum(base.sum(1, keepdims=True), 1e-300)
        shift = np.zeros_like(base)
        for f in np.unique(rf):
            hit = rf == f
            shift[hit] = do[hit] @ table(ok & ~hit)
        moved = np.abs(shift).max(1) > 1e-9
        if not moved.any():
            continue
        y = np.maximum(yc, 0)
        b0 = _nll(base, y, K)
        cands = [_fix(base + lam * shift) for lam in LAMS]
        k = int(np.argmin([_nll(c, y, K) for c in cands]))
        new = cands[k]
        if not all(_nll(new[m], y[m], K) < _nll(base[m], y[m], K) - 1e-7 for m in (half, ~half) if m.any()):
            continue
        final_oofs[name] = new
        cur = np.asarray(final[name][hid], float)
        final[name][hid] = _fix(cur / np.maximum(cur.sum(1, keepdims=True), 1e-300) + LAMS[k] * (dh @ table(ok)))
        lams[name] = LAMS[k]
        gains.append(b0 - _nll(new, y, K))
    report.update(changed=len(lams), lam_mean=float(np.mean(list(lams.values()))) if lams else 0.0,
                  oof_gain_sum=float(np.sum(gains)) if gains else 0.0)
