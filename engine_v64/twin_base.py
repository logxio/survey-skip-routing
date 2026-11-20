""": which base the twin mix rests on, and per-row copy evidence.

. The twin mix p = rho T[y_twin] + (1 - rho) base puts the final stack in
the non-copy branch, and on the B ruler the trees in that stack have already
read the twin's answers off its identical GIVEN row.  Per item and target
round, the same twin pass is rerun over three other bases: the stack of the
non-tree arms ("clean"), its half mix with the final stack, and the
fold-honest marginal.  An alternative replaces the current output only when
its cross-fitted visible-edge loss beats it overall and on both fold halves.

. A revisit row copies each root subtree from its donor with probability
.85, else redraws it from its own trait.  For a round-one target, the twin's
answers on the rest of the subtree carry that switch: under copy they follow
the target's own prediction, under a redraw the marginal.  With

    s = sum over the other subtree items of log p_clean,target(y_twin) - log p_marg(y_twin)

(round two's skip read as round one's '0' on zero families), rho becomes
RMAX sigmoid(a + b s); a, b by maximum likelihood on visible edges, the item
switched only when the cross-fitted loss beats the current output overall and
on both halves.  Subtrees with fewer than MIN_SUBTREE items are left alone.
"""
import gc
import os
import sys
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
import ortho_config as _ortho

from . import twin_identity as TI
from . import twin_posterior as TP
from .base import _stack

V107_K1 = os.environ.get('V107_K1', '1') == '1'
V107_K2 = os.environ.get('V107_K2', '1') == '1'
V125_K1 = _ortho.V125_K1
NON_BASE = frozenset(('shallow', 'shallow_asked', 'two_stage', 'deep', 'hgb', 'catboost', 'strong'))
MIN_SUBTREE = 5
RMAX = 0.99
GA = np.linspace(-2.0, 6.0, 17)
GB = np.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])
ZERO_FAMILIES = ('s6_4_', 's6_6_', 's6_11_', 's7a_7_', 's7b_1_')
_FORKED_BASES = {}


def _use_v125_k1():
    if V125_K1 != _ortho.V125_K1:
        return bool(V125_K1 and _ortho.V125_K1)
    return _ortho._flag('V125_K1', bool(V125_K1))


def _trim_heap():
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            trim = getattr(libc, 'malloc_trim', None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
        except Exception:
            pass


def _loss(p, y):
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
    K = p.shape[1]
    return -np.log(1e-3 + (1 - K * 1e-3) * p[np.arange(len(y)), y])


def _aligned(oof, hidden, train):
    """One row per respondent: OOF on visible rows, the prediction on hidden rows."""
    A = np.zeros((len(train), oof.shape[1]), np.float32)
    A[train] = oof
    A[~train] = hidden
    return A / np.maximum(A.sum(1, keepdims=True), 1e-30)


def current(name, work, final, final_oofs, stacked):
    train = work[name]['train']
    return _aligned(np.asarray(final_oofs.get(name, stacked[name][0]), float), np.asarray(final[name][~train], float), train)


def _compute_base_item(j, fold):
    keep = [i for i, a in enumerate(j['arms']) if a not in NON_BASE]
    if not keep:
        return None
    train = j['train']
    rest = np.flatnonzero(~train)
    y, K = j['y'][train], j['K']
    p, o, _, _, _ = _stack([j['oof'][i] for i in keep], [np.asarray(j['all'][i])[rest] for i in keep], y, K)
    f = fold[train]
    mo = np.zeros((len(y), K))
    for g in np.unique(f):
        c = np.bincount(y[f != g], minlength=K) + 0.5
        mo[f == g] = c / c.sum()
    c = np.bincount(y, minlength=K) + 0.5
    return {'clean': _aligned(o, p, train), 'marg': _aligned(mo, np.tile(c / c.sum(), (len(rest), 1)), train)}


def _forked_base_worker(name):
    with threadpool_limits(1):
        return name, _compute_base_item(_FORKED_BASES['work'][name], _FORKED_BASES['fold'])


def bases(work, stacked, fold, workers=8):
    """Row-aligned clean stack (no tree arms) and fold-honest marginal per item."""
    tasks = [
        name for name, j in work.items()
        if name in stacked and any(a not in NON_BASE for a in j['arms'])
    ]
    if not tasks:
        return {}
    if _use_v125_k1() and sys.platform.startswith('linux') and workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        if 'fork' in mp.get_all_start_methods():
            gc.collect()
            _trim_heap()
            _FORKED_BASES.update(work=work, fold=fold)
            got = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                pool = mp.get_context('fork').Pool(min(int(workers), len(tasks)))
            try:
                for name, res in pool.imap_unordered(_forked_base_worker, tasks, chunksize=1):
                    if res is not None:
                        got[name] = res
            finally:
                pool.terminate()
                pool.join()
                _FORKED_BASES.clear()
            return {name: got[name] for name in tasks if name in got}
    out = {}
    for name in tasks:
        res = _compute_base_item(work[name], fold)
        if res is not None:
            out[name] = res
    return out


def _edges(train, pairs):
    a = np.asarray(pairs, int).reshape(-1, 2)
    t = np.r_[a[:, 0], a[:, 1]]
    s = np.r_[a[:, 1], a[:, 0]]
    vv = train[t] & train[s]
    hh = ~train[t] & train[s]
    return t[vv], s[vv], t[hh], s[hh]


def snapshot(work, final, final_oofs, stacked, names):
    return {k: current(k, work, final, final_oofs, stacked) for k in names}


def select(work, final, final_oofs, stacked, pairs, fold, wave, report, context, B, pre, used):
    """ after one twin pass.  pre: row-aligned bases before the pass; used
    collects, per item, the base each target row's twin output rests on."""
    train = next(iter(work.values()))['train']
    report.update(switched=0)
    if not len(pairs):
        return
    vt, vs, ht, hs = _edges(train, pairs)
    if len(vt) < TI.MIN_EDGES or not len(ht):
        return
    pos = np.full(len(train), -1)
    pos[train] = np.arange(train.sum())
    names = [k for k in B if k in final and k in pre and np.array_equal(work[k]['train'], train)]
    outs = {}
    for tag in ('clean', 'mix', 'marg'):
        fa, oa, base = {}, {}, {}
        for k in names:
            A = 0.5 * pre[k] + 0.5 * B[k]['clean'] if tag == 'mix' else B[k][tag]
            base[k] = A
            fa[k] = np.zeros((len(train), A.shape[1]))
            fa[k][~train] = A[~train]
            oa[k] = np.asarray(A[train], float)
        TI.apply(work, fa, oa, {k: (oa[k], None, None) for k in names}, pairs, fold, wave, {}, context=context)
        outs[tag] = {k: (oa[k][pos[vt]], fa[k][ht], base[k]) for k in names}
    half = fold[vt] % 2 == 0
    picks = {}
    for k in names:
        y = work[k]['y']
        ref_v = np.asarray(final_oofs.get(k, stacked[k][0]), float)[pos[vt]]
        for d in (0, 1):
            dv = wave[vt] == d
            dh = wave[ht] == d
            if not dv.any():
                continue
            lref = _loss(ref_v[dv], y[vt[dv]])
            hv = half[dv]
            best, tag_best = lref.mean(), None
            for tag, got in outs.items():
                l = _loss(got[k][0][dv], y[vt[dv]])
                if l.mean() < best and all(l[m].mean() < lref[m].mean() - 1e-6 for m in (np.ones(len(l), bool), hv, ~hv) if m.any()):
                    best, tag_best = l.mean(), tag
            if tag_best is None:
                continue
            ov, oh, A = outs[tag_best][k]
            upd = np.array(final_oofs.get(k, stacked[k][0]), float)
            upd[pos[vt[dv]]] = ov[dv]
            final_oofs[k] = upd
            final[k][ht[dh]] = oh[dh]
            rows = np.r_[vt[dv], ht[dh]]
            used[k][rows] = A[rows]
            picks[tag_best] = picks.get(tag_best, 0) + 1
    report.update(switched=int(sum(picks.values())), picks=picks)


def _zero_code(items, k):
    v = items[k]['values']
    if k.startswith(ZERO_FAMILIES) and not any(x in k for x in ('other', 'pna', 'PNA', 'dont_know')) and '0' in v:
        return v.index('0')
    return -1


def _mapped(y, K, z, wave, s, t):
    out = y[s].copy()
    if z >= 0:
        out[(out == K - 1) & (wave[t] == 0) & (wave[s] == 1)] = z
    return out


def _transfer(src, dst, opened, K, mask, gated):
    if gated:
        code = 2 * src + opened.astype(int)
        C = np.bincount(code[mask] * K + dst[mask], minlength=2 * K * K).reshape(2 * K, K).astype(float)
        T = C + TI.IDENTITY_PRIOR * np.repeat(np.eye(K), 2, axis=0)
    else:
        code = src
        C = np.bincount(src[mask] * K + dst[mask], minlength=K * K).reshape(K, K).astype(float)
        T = C + TI.IDENTITY_PRIOR * np.eye(K)
    return T / T.sum(1, keepdims=True), code


def _fit_ab(R, b, s, K, mask):
    if mask.sum() < 20:
        return None
    z = GA[:, None, None] + GB[None, :, None] * s[mask][None, None, :]
    rho = RMAX / (1 + np.exp(-z))
    loss = -np.log(1e-3 + (1 - K * 1e-3) * (rho * R[mask] + (1 - rho) * b[mask])).sum(2)
    i, j = np.unravel_index(int(np.argmin(loss)), loss.shape)
    return GA[i], GB[j]


def subtree(work, final, final_oofs, stacked, pairs, fold, wave, report, items, X, given, B, used):
    """ over round-one target edges of every multi-item root subtree."""
    train = next(iter(work.values()))['train']
    report.update(changed=0)
    if not len(pairs):
        return
    vt, vs, ht, hs = _edges(train, pairs)
    if len(vt) < TI.MIN_EDGES or not len(ht):
        return
    pos = np.full(len(train), -1)
    pos[train] = np.arange(train.sum())
    names = [k for k in work if k in final and k in B and np.array_equal(work[k]['train'], train)]
    present = set(names)
    roots = {}
    for k in names:
        r = k
        while items[r].get('gate') and items[r]['gate']['parent'] in present:
            r = items[r]['gate']['parent']
        roots.setdefault(r, []).append(k)
    t = np.r_[vt, ht]
    s = np.r_[vs, hs]
    nv = len(vt)
    r_all = np.arange(len(t))
    dv = wave[vt] == 0
    dh = wave[ht] == 0
    ft, fs = fold[vt], fold[vs]
    half = ft % 2 == 0
    rv = np.arange(nv)
    changed = {}
    for root, members in roots.items():
        if len(members) < MIN_SUBTREE:
            continue
        L = np.zeros((len(t), len(members)))
        for i, k in enumerate(members):
            ys = _mapped(work[k]['y'], work[k]['K'], _zero_code(items, k), wave, s, t)
            L[:, i] = np.log(np.maximum(B[k]['clean'][t][r_all, ys], 1e-4)) - np.log(np.maximum(B[k]['marg'][t][r_all, ys], 1e-4))
        total = L.sum(1)
        for i, k in enumerate(members):
            y, K = work[k]['y'], work[k]['K']
            gated = bool(items[k].get('gate'))
            opened = TP.source_open(k, work, items, X, given)
            sl = total - L[:, i]
            sv, sh = sl[:nv], sl[nv:]
            bo = np.asarray(used[k][vt], float)
            bh = np.asarray(used[k][ht], float)
            src, dst = y[vs], y[vt]
            cur_v = np.asarray(final_oofs.get(k, stacked[k][0]), float)[pos[vt]]
            cur_v = cur_v / np.maximum(cur_v.sum(1, keepdims=True), 1e-300)
            newv = cur_v.copy()
            for f, g in sorted(set(zip(ft[dv], fs[dv]))):
                ev = dv & (ft == f) & (fs == g)
                fit = dv & ~np.isin(ft, [f, g]) & ~np.isin(fs, [f, g])
                T, code = _transfer(src, dst, opened[vs], K, fit, gated)
                ab = _fit_ab(T[code][rv, dst], bo[rv, dst], sv, K, fit)
                if ab is None:
                    continue
                rho = RMAX / (1 + np.exp(-(ab[0] + ab[1] * sv[ev])))
                newv[ev] = rho[:, None] * T[code[ev]] + (1 - rho[:, None]) * bo[ev]
            cur = _loss(cur_v[dv], dst[dv])
            new = _loss(newv[dv], dst[dv])
            hv = half[dv]
            if not all(new[m].mean() < cur[m].mean() - 1e-6 for m in (np.ones(len(new), bool), hv, ~hv) if m.any()):
                continue
            T, code = _transfer(src, dst, opened[vs], K, dv, gated)
            ab = _fit_ab(T[code][rv, dst], bo[rv, dst], sv, K, dv)
            if ab is None:
                continue
            ysh = y[hs[dh]]
            hcode = 2 * ysh + opened[hs[dh]].astype(int) if gated else ysh
            rho = RMAX / (1 + np.exp(-(ab[0] + ab[1] * sh[dh])))
            final[k][ht[dh]] = rho[:, None] * T[hcode] + (1 - rho[:, None]) * bh[dh]
            upd = np.array(final_oofs.get(k, stacked[k][0]), float)
            upd[pos[vt[dv]]] = newv[dv]
            final_oofs[k] = upd
            changed[k] = float(cur.mean() - new.mean())
    report.update(changed=len(changed), oof_edge_gain=float(np.mean(list(changed.values()))) if changed else 0.0)
