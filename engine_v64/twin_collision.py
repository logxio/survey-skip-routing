"""Cross-round identity beyond unique signatures ().

COLLISION: rows left unmatched whose GIVEN signature (all raw columns but the
round) occurs 1..MAX_K times in each round, at least twice in one.  A target
row's candidate twins are the visible rows of the other round in its group,
minus every candidate already claimed by another same-round visible row: a
claim is a mutual best answer agreement at or above the midpoint between
random cross-round agreement and the agreement of exact twins.  Claims never
read the target's own answers, so a visible target sees exactly the evidence
a hidden one would.  Evidence is the mean transition of the remaining
candidates' answers,

    p(y) = rho_s * mean_b T[y_b, y] + (1 - rho_s) * p_final(y),

with T and rho fitted per item and round on visible targets, rho split by
whether the candidate set resolved to one visible row (stratum 0) or not.
Fold-f OOF uses only labels outside fold f; an item is changed only when that
OOF improves overall and on both fold halves.

HOUSEHOLD: rows still unmatched, signature without the respondent's own
columns (relationship, age, education, marital status): a second adult may
answer the second round of the same household.  Unique on both sides, then
the exact-twin calibration of twin_identity.apply.
"""
import numpy as np

from . import twin_identity as TI

MAX_K = 4
MIN_TARGETS = 30
RESPONDENT = ('s2_1_hh_relationship_lbl', 's2_1a_gender_lbl', 's2_1b_age', 's2_1c_education_level_lbl',
              's2_1d_marital_status_lbl', 's2_1c_marital_lbl', 's2_1d_education_lbl')


def _keys(X, cols):
    return [r.tobytes() for r in np.ascontiguousarray(X[:, cols])]


def household_pairs(X, n_raw, wave_col, given, taken):
    drop = {given.index(c) for c in RESPONDENT if c in given[:n_raw]}
    cols = [c for c in range(n_raw) if c != wave_col and c not in drop]
    return TI._unique_pairs(_keys(X, cols), X[:, wave_col], taken)


def collision_groups(X, n_raw, wave_col, taken):
    cols = [c for c in range(n_raw) if c != wave_col]
    keys = _keys(X, cols)
    wave = X[:, wave_col]
    groups = {}
    for i in np.flatnonzero(~taken):
        groups.setdefault(keys[i], ([], []))[int(wave[i])].append(i)
    return [(np.array(a), np.array(b)) for a, b in groups.values()
            if a and b and len(a) <= MAX_K and len(b) <= MAX_K and max(len(a), len(b)) >= 2]


def _answers(work):
    names = [n for n, j in work.items()]
    train = work[names[0]]['train']
    names = [n for n in names if np.array_equal(work[n]['train'], train)]
    return np.column_stack([work[n]['y'] for n in names]), train


def agreement_threshold(Y, train, exact, wave, rng):
    """Midpoint of random and exact-twin answer agreement; inf without twins."""
    ab = np.array([p for p in exact if train[p[0]] and train[p[1]]]).reshape(-1, 2)
    if len(ab) < 50:
        return np.inf
    twin = float((Y[ab[:, 0]] == Y[ab[:, 1]]).mean())
    v0 = np.flatnonzero(train & (wave == 0)); v1 = np.flatnonzero(train & (wave == 1))
    m = min(4000, len(v0), len(v1))
    rand = float((Y[rng.choice(v0, m)] == Y[rng.choice(v1, m)]).mean())
    return 0.5 * (twin + rand) if twin > rand else np.inf


def collision_targets(groups, Y, train, tau):
    """[(target, visible candidates, stratum)] for every row of every group."""
    out = []
    for S in groups:
        agree = {}
        vis = [S[0][train[S[0]]], S[1][train[S[1]]]]
        if np.isfinite(tau) and len(vis[0]) and len(vis[1]):
            M = (Y[vis[0]][:, None, :] == Y[vis[1]][None, :, :]).mean(2)
            agree = {(int(a), int(b)): M[i, k] for i, a in enumerate(vis[0]) for k, b in enumerate(vis[1])}
        for d in (0, 1):
            others = S[1 - d]
            cand_vis = vis[1 - d]
            hidden_others = int((~train[others]).sum())
            for t in S[d]:
                mates = [a for a in vis[d] if a != t]
                claimed = set()
                if agree and mates:
                    for b in cand_vis:
                        key = (lambda a: (int(a), int(b))) if d == 0 else (lambda a: (int(b), int(a)))
                        sc = [agree[key(a)] for a in mates]
                        a = mates[int(np.argmax(sc))]
                        if max(sc) < tau:
                            continue
                        # mutual: b is a's best among the other round's visible rows
                        row = [agree[(int(a), int(c))] if d == 0 else agree[(int(c), int(a))] for c in cand_vis]
                        if cand_vis[int(np.argmax(row))] == b:
                            claimed.add(int(b))
                cands = [int(b) for b in cand_vis if int(b) not in claimed]
                if not cands:
                    continue
                stratum = 0 if len(cands) == 1 and hidden_others == 0 else 1
                out.append((int(t), cands, stratum))
    return out


def _evidence(T, src_codes, ti, m, emask):
    E = np.zeros((m, T.shape[1]))
    np.add.at(E, ti[emask], T[src_codes[emask]])
    c = np.bincount(ti[emask], minlength=m)
    ok = c > 0
    E[ok] /= c[ok, None]
    return E, ok


def _fit_T(src, dst, w, K, augmented=False):
    S = 2*K if augmented else K
    C = np.zeros(S * K)
    np.add.at(C, src * K + dst, w)
    T = C.reshape(S, K) + TI.IDENTITY_PRIOR * (np.repeat(np.eye(K), 2, axis=0) if augmented else np.eye(K))
    return T / T.sum(1, keepdims=True)


def _fit_rho(E, base, y, K):
    if not len(y):
        return 0.0
    t = E[np.arange(len(y)), y]; b = base[np.arange(len(y)), y]
    mix = TI.RHOS[:, None] * t[None, :] + (1 - TI.RHOS)[:, None] * b[None, :]
    loss = -np.log(1e-3 + (1 - K * 1e-3) * mix).sum(1)
    return float(TI.RHOS[int(np.argmin(loss))])


def apply_collision(work, final, final_oofs, stacked, targets, fold, wave, report, anchors=(), context=None):
    """anchors: exact twin pairs; their visible edges join T's fit at weight one.

    Runs only after the exact tier has shown twins exist (finite claim
    threshold upstream), so MIN_TARGETS visible targets suffice for rho."""
    n = len(fold)
    report.update(targets=len(targets), changed=0)
    if not targets:
        return
    train = next(iter(work.values()))['train']
    tg = np.array([t for t, _, _ in targets]); st = np.array([s for _, _, s in targets])
    ti = np.concatenate([np.full(len(c), i) for i, (_, c, _) in enumerate(targets)])
    eb = np.concatenate([np.array(c) for _, c, _ in targets])
    ew = np.concatenate([np.full(len(c), 1.0 / len(c)) for _, c, _ in targets])
    m = len(tg)
    vis = train[tg]
    report.update(visible_targets=int(vis.sum()), hidden_targets=int((~vis).sum()),
                  resolved=int((st == 0).sum()), candidates=float(np.mean([len(c) for _, c, _ in targets])))
    if vis.sum() < MIN_TARGETS or vis.all():
        return
    an = np.array(list(anchors), int).reshape(-1, 2)
    at = np.concatenate([an[:, 0], an[:, 1]]); aw = np.concatenate([an[:, 1], an[:, 0]])
    keep = train[at] & train[aw]
    at, aw = at[keep], aw[keep]
    pos = np.full(n, -1); pos[np.flatnonzero(train)] = np.arange(train.sum())
    vt = np.flatnonzero(vis); ht = np.flatnonzero(~vis)
    ft = fold[tg]; fb = fold[eb]
    half = ft[vt] % 2 == 0
    rhos = {}
    for name, j in work.items():
        if name not in final or not np.array_equal(j['train'], train):
            continue
        y, K = j['y'], j['K']
        oof = np.asarray(final_oofs.get(name, stacked[name][0]), float)
        oof = oof / np.maximum(oof.sum(1, keepdims=True), 1e-300)
        base_v = np.zeros((m, K)); base_v[vt] = oof[pos[tg[vt]]]
        new_v = base_v.copy()
        hid_p = np.asarray(final[name][tg[ht]], float)
        hid_p = hid_p / np.maximum(hid_p.sum(1, keepdims=True), 1e-300)
        new_h = hid_p.copy()
        ytg = np.where(vis, y[tg], 0); src = y[eb]
        used = []
        augmented = context is not None and context[0][name].get('gate') is not None
        ay = y[aw]
        if augmented:
            from .twin_posterior import source_open
            opened = source_open(name, work, *context)
            src = 2 * src + opened[eb].astype(int)
            ay = 2 * ay + opened[aw].astype(int)

        def fit_T(emask, amask):
            return _fit_T(np.concatenate([src[emask], ay[amask]]), np.concatenate([ytg[ti[emask]], y[at[amask]]]),
                          np.concatenate([ew[emask], np.ones(int(amask.sum()))]), K, augmented)
        for d in (0, 1):
            dv = vis & (wave[tg] == d)
            ad = wave[at] == d
            eval_groups = {}
            for target in np.flatnonzero(dv):
                excluded = tuple(sorted(set([int(ft[target]), *map(int, fb[ti == target])]))) if augmented else (int(ft[target]),)
                eval_groups.setdefault(excluded, []).append(target)
            for excluded, ev_targets in eval_groups.items():
                emask = dv[ti] & ~np.isin(ft[ti], excluded) & ~np.isin(fb, excluded)
                T = fit_T(emask, ad & ~np.isin(fold[at], excluded) & ~np.isin(fold[aw], excluded))
                E, ok = _evidence(T, src, ti, m, emask)
                fe = np.zeros(m, bool); fe[ev_targets] = True
                Ef, okf = _evidence(T, src, ti, m, fe[ti])
                for s in (0, 1):
                    tr = ok & (st == s) & dv & ~np.isin(ft, excluded)
                    r = _fit_rho(E[tr], base_v[tr], ytg[tr], K)
                    sel = fe & okf & (st == s)
                    new_v[sel] = r * Ef[sel] + (1 - r) * base_v[sel]

            emask = dv[ti]
            T = fit_T(emask, ad)
            E, ok = _evidence(T, src, ti, m, emask)
            dh = ~vis & (wave[tg] == d)
            Eh, okh = _evidence(T, src, ti, m, dh[ti])
            for s in (0, 1):
                tr = ok & (st == s) & dv
                r = _fit_rho(E[tr], base_v[tr], ytg[tr], K)
                sel = dh & okh & (st == s)
                k = np.searchsorted(ht, np.flatnonzero(sel))
                new_h[k] = r * Eh[sel] + (1 - r) * hid_p[k]
                used.append(r)
        if max(used) <= 0.0:
            continue
        nv, bv, yv = new_v[vt], base_v[vt], ytg[vt]
        if not all(TI._nll(nv[q], yv[q], K) < TI._nll(bv[q], yv[q], K) - 1e-6
                   for q in (np.ones(len(vt), bool), half, ~half) if q.any()):
            continue
        final[name][tg[ht]] = new_h
        upd = oof.copy()
        upd[pos[tg[vt]]] = nv
        final_oofs[name] = upd
        rhos[name] = float(np.mean(used))
    report.update(changed=len(rhos), rho_mean=float(np.mean(list(rhos.values()))) if rhos else 0.0)
