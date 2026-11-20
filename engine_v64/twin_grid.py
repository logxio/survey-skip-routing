""" : twin copy posterior per root subtree on the trait grid.

A round-two revisit row takes its round-one donor's answers one root subtree
at a time (copy with probability rho), else keeps answers drawn from its own
trait.  With the pooled trait fold models of latent_classes.fit_trait:

Round-one target t (the donor), visible twin s (the revisit):
  J(td, tr) = prior(td) P(GIVEN_t | td) prior(tr) prod_R [rho e^C_R(td) + (1 - rho) e^C_R(tr)]
  C_R(theta) = sum over R's answered cells of the twin of log p(y_s | theta)
  P(y_t,k) = q_R onehot(map(y_s,k)) + (1 - q_R) sum_td pi0_R(td) E_k(td)
q_R the posterior copy probability of k's root, pi0_R the posterior of td
given no copy (sharpened by the twin's other, mostly copied, subtrees),
map reading round two's skip as round one's '0' on zero families.

Round-two target t (the revisit), visible twin s (the donor): the donor's
answers say nothing about the switch or about the revisit's own trait, so
P = rho onehot(donor answer, closed zero-fill -> skip) + (1 - rho) prior law.

rho by EM on visible round-one edges.  Per item and target round, the model is
blended into the current output with a weight cross-fitted by fold and kept
only when it beats the current output on all visible edges and both halves.
"""
import time
import ortho_config as _ortho
import numpy as np

from . import latent_classes as LC
from . import twin_identity as TI
from . import twin_posterior as TP
from .twin_base import _loss, _edges, _zero_code

LAMBDAS = np.array([0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0])
CHUNK = 24
RHO_ITERS = 2
SECONDS = 45.0


class _Grid:
    def __init__(self, state, work, items, X, given, wave):
        self.models, self.blocks, self.route = state['models'], state['blocks'], state['route']
        self.names, self.G = list(state['names']), state['G']
        self.X, self.wave, self.items = X, wave, items
        self.wcol = list(given[:self.G]).index('wave')
        present = set(self.names)
        roots = {}
        for k in self.names:
            r = k
            while items[r].get('gate') and items[r]['gate']['parent'] in present:
                r = items[r]['gate']['parent']
            roots[k] = r
        self.root_names = sorted(set(roots.values()))
        ridx = {r: i for i, r in enumerate(self.root_names)}
        I, R = len(self.names), len(self.root_names)
        self.item_root = np.array([ridx[roots[k]] for k in self.names])
        self.M = np.zeros((I, R))
        self.M[np.arange(I), self.item_root] = 1.0
        self.off = np.array([self.blocks[self.G + i][0] for i in range(I)])
        self.routed = np.array([self.route[k] is not None for k in self.names])
        self.K = np.array([work[k]['K'] for k in self.names])
        self.Y = np.column_stack([work[k]['y'] for k in self.names])
        self.zero = np.array([_zero_code(items, k) for k in self.names])
        self.opened = np.column_stack([TP.source_open(k, work, items, X, given) for k in self.names])

    def _given_ll(self, m, rows):
        L = np.zeros((len(rows), len(m['lw'])))
        for g in range(self.G):
            if g != self.wcol:
                L += m['log_t'][:, self.blocks[g][0] + self.X[rows, g]].T
        return L

    def _twin_ll(self, m, s):
        y = self.Y[s]
        ok = (y >= 0) & ~(self.routed[None] & (y == self.K[None] - 1))
        L = m['log_t'].T[self.off[None] + np.where(ok, y, 0)]
        L[~ok] = 0.0
        return np.einsum('eiq,ir->erq', L, self.M)

    def round_one(self, m, t, s, rho, deadline):
        lw = m['lw']
        lr, l1 = np.log(rho), np.log1p(-rho)
        R, Q = len(self.root_names), len(lw)
        q = np.empty((len(t), R))
        pi0 = np.empty((len(t), R, Q))
        for a in range(0, len(t), CHUNK):
            if time.monotonic() >= deadline:
                raise TimeoutError
            tt, ss = t[a:a + CHUNK], s[a:a + CHUNK]
            C = self._twin_ll(m, ss)
            cp = lr + C[:, :, :, None]
            rd = l1 + C[:, :, None, :]
            mix = np.logaddexp(cp, rd)
            J = mix.sum(1) + (self._given_ll(m, tt) + lw[None])[:, :, None] + lw[None, None, :]
            J -= J.max((1, 2), keepdims=True)
            P = np.exp(J)
            P /= P.sum((1, 2), keepdims=True)
            w1 = np.exp(cp - mix)
            q[a:a + CHUNK] = np.einsum('edr,eRdr->eR', P, w1)
            p0 = np.einsum('edr,eRdr->eRd', P, 1.0 - w1)
            pi0[a:a + CHUNK] = p0 / np.maximum(p0.sum(2, keepdims=True), 1e-300)
        return q, pi0

    def _redraw(self, m, i, post, rows):
        k = self.names[i]
        E = m['E'][k]
        if k in m['gskip']:
            sk = m['gskip'][k]
            code = self.X[rows, self.route[k][1]]
            sk = np.where(code < len(sk), sk[np.minimum(code, len(sk) - 1)], sk.mean())[:, None]
            nom = E[:, :-1] / np.maximum(1.0 - E[:, -1:], 1e-12)
            return np.column_stack([(1.0 - sk) * (post @ nom), sk])
        return post @ E

    def _copy(self, i, s, round_two):
        K = self.K[i]
        y = self.Y[s, i].copy()
        z = self.zero[i]
        if z >= 0:
            if round_two:
                y[(self.wave[s] == 0) & ~self.opened[s, i]] = K - 1
            else:
                y[(y == K - 1) & (self.wave[s] == 1)] = z
        out = np.zeros((len(s), K))
        out[np.arange(len(s)), y] = 1.0
        return out

    def predict(self, d, t, s, rho, groups, deadline, cached=None):
        """groups: [(edge index, model ids)]; cached: {fold: (q, pi0)} for round-one visible groups."""
        out = [np.zeros((len(t), self.K[i])) for i in range(len(self.names))]
        R = len(self.root_names)
        copy_weights = np.zeros((len(t), R))
        for gi, (ix, fs) in enumerate(groups):
            if not len(ix):
                continue
            w = 1.0 / len(fs)
            for f in fs:
                m = self.models[f]
                if d == 0:
                    q, pi0 = cached[gi] if cached is not None and gi in cached else self.round_one(m, t[ix], s[ix], rho, deadline)
                else:
                    q = np.full((len(ix), R), rho)
                    pi0 = np.broadcast_to(np.exp(m['lw'])[None, None, :], (len(ix), R, len(m['lw'])))
                copy_weights[ix] += w * q
                for i in range(len(self.names)):
                    r = self.item_root[i]
                    red = self._redraw(m, i, pi0[:, r, :], t[ix])
                    z = self.zero[i]
                    if d == 0 and z >= 0:
                        red[:, z] += red[:, -1]
                        red[:, -1] = 0.0
                    qr = q[:, r:r + 1]
                    out[i][ix] += w * (qr * self._copy(i, s[ix], d == 1) + (1.0 - qr) * red)
        self.copy_weights = copy_weights
        return out


SOLO_LAMS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def solo(work, final, final_oofs, stacked, pairs, fold, report, B, wave=None):
    """: rows with no visible twin.  The trees read a twin's answers off its
    identical GIVEN row, so the stack weights, calibrated on visible rows most
    of which have a twin in another fold, overweight the trees for rows that
    have none.  Per item, lam current + (1 - lam) clean (the stack without tree
    arms, twin_base.bases) is cross-fitted by fold on visible rows with no
    visible twin, kept only when it beats the
    current output there overall and on both halves, and applied to hidden
    rows without a visible twin."""
    report.update(changed=0)
    train = next(iter(work.values()))['train']
    n = len(train)
    partner = np.full(n, -1)
    if len(pairs):
        a = np.asarray(pairs, int).reshape(-1, 2)
        partner[a[:, 0]] = a[:, 1]
        partner[a[:, 1]] = a[:, 0]
    rows = np.flatnonzero(train)
    p = partner[rows]
    pp = np.maximum(p, 0)
    # Without twins the trees have nothing to memorise and the stack weights already fit these rows.
    if int((train[pp] & (p >= 0)).sum()) < TI.MIN_EDGES:
        return
    # Rows with no visible twin at all: the twin passes left their outputs as stacked.
    solo_v = (p < 0) | ~train[pp]
    hidden = np.flatnonzero(~train)
    hp = partner[hidden]
    solo_h = hidden[(hp < 0) | ~train[np.maximum(hp, 0)]]
    if solo_v.sum() < 200 or not len(solo_h):
        return
    ft = fold[rows][solo_v]
    half = ft % 2 == 0
    changed = {}
    for k, b in B.items():
        if k not in final or not np.array_equal(work[k]['train'], train):
            continue
        y = work[k]['y'][rows][solo_v]
        cur = np.asarray(final_oofs.get(k, stacked[k][0]), float)
        cv = cur[solo_v] / np.maximum(cur[solo_v].sum(1, keepdims=True), 1e-300)
        cl = np.asarray(b['clean'][rows][solo_v], float)
        losses = np.stack([_loss(l * cv + (1 - l) * cl, y) for l in SOLO_LAMS])
        new = cv.copy()
        for f in np.unique(ft):
            ev = ft == f
            l = SOLO_LAMS[int(np.argmin(losses[:, ~ev].sum(1)))]
            new[ev] = l * cv[ev] + (1 - l) * cl[ev]
        lc, ln = _loss(cv, y), _loss(new, y)
        if not all(ln[m].mean() < lc[m].mean() - 1e-6 for m in (np.ones(len(y), bool), half, ~half)):
            continue
        l = SOLO_LAMS[int(np.argmin(losses.sum(1)))]
        fh = np.asarray(final[k][solo_h], float)
        fh = fh / np.maximum(fh.sum(1, keepdims=True), 1e-300)
        final[k][solo_h] = l * fh + (1 - l) * np.asarray(b['clean'][solo_h], float)
        upd = cur.copy()
        upd[np.flatnonzero(solo_v)] = new
        final_oofs[k] = upd
        changed[k] = float(lc.mean() - ln.mean())
    report.update(changed=len(changed), solo_visible=int(solo_v.sum()), solo_hidden=int(len(solo_h)),
                  oof_gain=float(np.mean(list(changed.values()))) if changed else 0.0)
    if RESTACK:
        report['restack'] = {}
        _restack(work, final, final_oofs, stacked, B, rows, solo_v, solo_h, ft, half, report['restack'])
    if REVISIT_PRIOR and wave is not None:
        report['revisit_prior'] = {}
        _revisit_prior(work, final, final_oofs, stacked, partner, train, fold, wave, report['revisit_prior'])


#  a round-two row whose round-one twin is hidden (or outside the
# frame) repeats that twin, whose GIVEN row it carries, on each root subtree
# with the panel share, and keeps its own trait elsewhere; for that share its
# GIVEN row says nothing.  Per class of twin-less round-two rows (paired with a
# hidden row; unpaired), one weight lam on the round-two prior,
# (1 - lam) current + lam prior, is cross-fitted by fold over all items on the
# visible rows of the class, kept only when it beats the current output there
# overall and on both halves, and applied to the hidden rows of the class.
REVISIT_PRIOR = False
REVISIT_LAMS = np.array([0.0, 0.02, 0.04, 0.07, 0.1, 0.14, 0.2, 0.3])
REVISIT_MIN_ROWS = 100


def _revisit_prior(work, final, final_oofs, stacked, partner, train, fold, wave, report):
    n = len(train)
    pp = np.maximum(partner, 0)
    lone = (partner < 0) | ~train[pp]
    classes = {'paired': (wave == 1) & (partner >= 0) & ~train[pp], 'unpaired': (wave == 1) & (partner < 0)}
    names = [k for k in final if k in work and np.array_equal(work[k]['train'], train)]
    rows = np.flatnonzero(train)
    pos = np.full(n, -1)
    pos[rows] = np.arange(len(rows))
    two = train & (wave == 1)
    F = int(fold[rows].max()) + 1 if len(rows) else 0
    for label, mask in classes.items():
        vis = np.flatnonzero(mask & train & lone)
        hid = np.flatnonzero(mask & ~train)
        if len(vis) < REVISIT_MIN_ROWS or not len(hid):
            report[label] = {'visible': int(len(vis)), 'hidden': int(len(hid)), 'kept': False}
            continue
        fv = fold[vis]
        L = np.zeros((len(REVISIT_LAMS), len(vis)))
        cur_v, pri_v = {}, {}
        for k in names:
            j = work[k]
            K = j['K']
            y = j['y'][vis]
            cur = np.asarray(final_oofs.get(k, stacked[k][0]), float)[pos[vis]]
            cur = cur / np.maximum(cur.sum(1, keepdims=True), 1e-300)
            pri = np.zeros((len(vis), K))
            for f in range(F):
                ev = fv == f
                if not ev.any():
                    continue
                yy = j['y'][two & (fold != f)]
                c = np.bincount(yy[yy >= 0], minlength=K)[:K] + 0.5
                pri[ev] = c / c.sum()
            cur_v[k], pri_v[k] = cur, pri
            for a, lam in enumerate(REVISIT_LAMS):
                L[a] += _loss((1 - lam) * cur + lam * pri, y)
        L /= max(len(names), 1)
        pick = np.zeros(len(vis), int)
        for f in np.unique(fv):
            ev = fv == f
            pick[ev] = int(np.argmin(L[:, ~ev].sum(1)))
        lc, ln = L[0], L[pick, np.arange(len(vis))]
        half = fv % 2 == 0
        kept = all(ln[m].mean() < lc[m].mean() - 1e-6 for m in (np.ones(len(vis), bool), half, ~half))
        lam = float(REVISIT_LAMS[int(np.argmin(L.sum(1)))])
        report[label] = {'visible': int(len(vis)), 'hidden': int(len(hid)), 'kept': bool(kept), 'lam': lam,
                         'oof_gain': float(lc.mean() - ln.mean())}
        if not kept or lam == 0.0:
            continue
        lv = REVISIT_LAMS[pick][:, None]
        for k in names:
            j = work[k]
            K = j['K']
            yy = j['y'][two]
            c = np.bincount(yy[yy >= 0], minlength=K)[:K] + 0.5
            fh = np.asarray(final[k][hid], float)
            fh = fh / np.maximum(fh.sum(1, keepdims=True), 1e-300)
            final[k][hid] = (1 - lam) * fh + lam * (c / c.sum())[None]
            upd = np.asarray(final_oofs.get(k, stacked[k][0]), float).copy()
            upd[pos[vis]] = (1 - lv) * cur_v[k] + lv * pri_v[k]
            final_oofs[k] = upd


#  after 's two-stack blend, rows with no visible twin get their own
# stack over the current output, the clean stack and every arm, fitted on
# visible rows with no visible twin (cross-fitted by fold), kept per item only
# when it beats the current output there overall and on both halves.
RESTACK = False
RESTACK_SECONDS = 40.0


def _restack(work, final, final_oofs, stacked, B, rows, solo_v, solo_h, ft, half, report):
    from .base import _stack
    began = time.monotonic()
    sv = np.flatnonzero(solo_v)
    norm = lambda a: a / np.maximum(a.sum(1, keepdims=True), 1e-300)
    changed, arms_n, status = {}, [], 'complete'
    for k, b in B.items():
        if k not in final or not np.array_equal(work[k]['train'], next(iter(work.values()))['train']):
            continue
        if time.monotonic() - began > RESTACK_SECONDS:
            status = 'time_cap'
            break
        j = work[k]
        y = j['y'][rows][sv]
        cur = np.asarray(final_oofs.get(k, stacked[k][0]), float)
        ov = [norm(cur[sv]), norm(np.asarray(b['clean'][rows][sv], float))]
        oh = [norm(np.asarray(final[k][solo_h], float)), norm(np.asarray(b['clean'][solo_h], float))]
        for o, a in zip(j['oof'], j['all']):
            ov.append(norm(np.asarray(o, float)[sv]))
            oh.append(norm(np.asarray(a, float)[solo_h]))
        K = j['K']
        new = ov[0].copy()
        for f in np.unique(ft):
            ev = ft == f
            new[ev] = _stack([o[~ev] for o in ov], [o[ev] for o in ov], y[~ev], K)[0]
        lc, ln = _loss(ov[0], y), _loss(new, y)
        if not all(ln[m].mean() < lc[m].mean() - 1e-6 for m in (np.ones(len(y), bool), half, ~half)):
            continue
        final[k][solo_h] = _stack(ov, oh, y, K)[0]
        upd = cur.copy()
        upd[sv] = new
        final_oofs[k] = upd
        changed[k] = float(lc.mean() - ln.mean())
        arms_n.append(len(ov))
    report.update(changed=len(changed), solo_visible=int(len(sv)), solo_hidden=int(len(solo_h)), restack=True, status=status,
                  arms_mean=float(np.mean(arms_n)) if arms_n else 0.0, seconds=round(time.monotonic() - began, 2),
                  oof_gain=float(np.mean(list(changed.values()))) if changed else 0.0)


def _groups(train, fold, t, F):
    vis = train[t]
    out = [(np.flatnonzero(vis & (fold[t] == f)), [f]) for f in range(F)]
    out.append((np.flatnonzero(~vis), list(range(F))))
    return out


def apply(work, final, final_oofs, stacked, pairs, fold, wave, report, items, X, given, deadline, workers=1):
    state = LC.TRAIT_MODELS
    report.update(changed=0)
    if not state or not len(pairs):
        return
    began = time.monotonic()
    train = next(iter(work.values()))['train']
    vt, vs, ht, hs = _edges(train, pairs)
    if len(vt) < TI.MIN_EDGES or not len(ht):
        return
    if any(k not in work or not np.array_equal(work[k]['train'], train) for k in state['names']):
        return
    grid = _Grid(state, work, items, X, given, wave)
    F = len(grid.models)
    pos = np.full(len(train), -1)
    pos[train] = np.arange(train.sum())
    try:
        d0 = wave[vt] == 0
        t1, s1 = vt[d0], vs[d0]
        g1 = _groups(train, fold, t1, F)
        rho, cached = 0.85, {}
        active = [gi for gi, (ix, _) in enumerate(g1[:F]) if len(ix)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(len(active), workers))) as ex:
            for it in range(RHO_ITERS + 1):
                got = list(ex.map(lambda gi: grid.round_one(grid.models[g1[gi][1][0]], t1[g1[gi][0]], s1[g1[gi][0]], rho, deadline), active))
                cached = dict(zip(active, got))
                if it == RHO_ITERS:
                    break
                num = sum(q.sum() for q, _ in got)
                den = sum(q.size for q, _ in got)
                rho = float(np.clip(num / max(den, 1.0), 0.05, 0.995))
        report['rho'] = round(rho, 4)
        changed = {}
        directional_inputs = []
        for d in (0, 1):
            dv, dh = wave[vt] == d, wave[ht] == d
            if dv.sum() < 30 or not dh.any():
                continue
            tv, sv_, th, sh = vt[dv], vs[dv], ht[dh], hs[dh]
            mv = grid.predict(d, tv, sv_, rho, _groups(train, fold, tv, F), deadline, cached if d == 0 else None)
            qv = grid.copy_weights.copy()
            mh = grid.predict(d, th, sh, rho, _groups(train, fold, th, F), deadline)
            qh = grid.copy_weights.copy()
            pv = pos[tv]
            ft = fold[tv]
            half = ft % 2 == 0
            for i, k in enumerate(grid.names):
                if k not in final:
                    continue
                y = work[k]['y'][tv]
                cur = np.asarray(final_oofs.get(k, stacked[k][0]), float)
                cv = cur[pv] / np.maximum(cur[pv].sum(1, keepdims=True), 1e-300)
                losses = np.stack([_loss(l * mv[i] + (1 - l) * cv, y) for l in LAMBDAS])
                newv = cv.copy()
                for f in np.unique(ft):
                    ev = ft == f
                    lam = LAMBDAS[int(np.argmin(losses[:, ~ev].sum(1)))]
                    newv[ev] = lam * mv[i][ev] + (1 - lam) * cv[ev]
                lc, ln = _loss(cv, y), _loss(newv, y)
                if not all(ln[m].mean() < lc[m].mean() - 1e-6 for m in (np.ones(len(y), bool), half, ~half) if m.any()):
                    continue
                lam = LAMBDAS[int(np.argmin(losses.sum(1)))]
                bh = np.asarray(final[k][th], float)
                bh = bh / np.maximum(bh.sum(1, keepdims=True), 1e-300)
                final[k][th] = lam * mh[i] + (1 - lam) * bh
                upd = cur.copy()
                upd[pv] = newv
                final_oofs[k] = upd
                changed[(k, d)] = float(lc.mean() - ln.mean())
            if _ortho.V114_K2:
                directional_inputs.append((d, tv, sv_, th, sh, mv, mh, qv, qh))
        if _ortho.V114_K2:
            from . import panel_transition
            for args in directional_inputs:
                panel_transition.refine(grid, *args, work, final, final_oofs, stacked, fold, pos,
                                        report.setdefault('directional', {}), deadline)
    except TimeoutError:
        report.update(status='deadline', seconds=round(time.monotonic() - began, 2))
        return
    report.update(changed=len(changed), round_one=sum(1 for _, d in changed if d == 0), round_two=sum(1 for _, d in changed if d == 1),
                  oof_edge_gain=float(np.mean(list(changed.values()))) if changed else 0.0, seconds=round(time.monotonic() - began, 2))
