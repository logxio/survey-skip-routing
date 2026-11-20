"""Visible-row gate for every module that edits the final vectors after the
structural gates, and the closing gate-consistent re-projection.

``snapshot`` freezes the structurally gated output (hidden rows and visible-row
OOF). ``gate`` keeps a later module's edit to an item only when the per-row
floored log-loss gain on the visible rows is positive on both fold-parity halves
and at least ``MIN_SE`` standard differences overall; every other edit is reverted to
the snapshot. ``reproject`` then closes the gate graph: a gated child whose
declared parent moved after the snapshot shifts its asked mass by
sum_a (P_new(a) - P_old(a)) r(a), with r(a) the fold-excluded visible-row rate
P(child asked | parent = a), keeping the child's answer shape.
"""
import numpy as np

MIN_SE = 3.0


def _rows_loss(p, y, K):
    p = np.asarray(p, float)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
    return -np.log(1e-3 + (1.0 - K * 1e-3) * p[np.arange(len(y)), y])


def snapshot(final, final_oofs, stacked):
    f0 = {k: np.array(v, copy=True) for k, v in final.items()}
    o0 = {k: np.array(final_oofs.get(k, stacked[k][0]), copy=True) for k in stacked}
    return f0, o0


def gate(work, final, final_oofs, stacked, snap, fold, min_se=MIN_SE):
    f0, o0 = snap
    info = {'changed': 0, 'kept': 0, 'reverted': 0, 'gain_nats_item_sum': 0.0, 'items': {}}
    for name in stacked:
        if name not in final or name not in f0:
            continue
        j = work[name]
        new_o = final_oofs.get(name, stacked[name][0])
        old_o = o0[name]
        same_o = np.shape(new_o) == np.shape(old_o) and np.array_equal(new_o, old_o)
        if same_o and np.array_equal(final[name], f0[name]):
            continue
        info['changed'] += 1
        y = j['y'][j['train']]
        keep = False
        stats = {'kept': False, 'n': int(len(y)), 'gain': None, 'se': None, 'half_even': None, 'half_odd': None}
        if not same_o and np.shape(new_o) == np.shape(old_o):
            d = _rows_loss(old_o, y, j['K']) - _rows_loss(new_o, y, j['K'])
            half = fold[j['train']] % 2 == 0
            if len(d) > 2 and half.any() and not half.all():
                se = float(d.std(ddof=1) / np.sqrt(len(d)))
                stats.update(gain=float(d.mean()), se=se, half_even=float(d[half].mean()), half_odd=float(d[~half].mean()))
                keep = d.mean() > 0 and d.mean() >= min_se * se and d[half].mean() > 0 and d[~half].mean() > 0
        stats['kept'] = bool(keep)
        info['items'][name] = stats
        if keep:
            info['kept'] += 1
            info['gain_nats_item_sum'] += float(d.mean())
        else:
            info['reverted'] += 1
            final[name] = f0[name].copy()
            final_oofs[name] = old_o.copy()
    return info


def _asked_rates(parent_true, Kp, asked, fold_rows, folds):
    """Fold-excluded and full P(asked | parent level), shrunk to the global rate."""
    def table(sel):
        cnt = np.bincount(parent_true[sel], minlength=Kp).astype(float)
        hit = np.bincount(parent_true[sel], weights=asked[sel], minlength=Kp)
        g = (hit.sum() + 0.5) / (cnt.sum() + 1.0)
        return (hit + 2.0 * g) / (cnt + 2.0)
    return {f: table(fold_rows != f) for f in folds}, table(np.ones(len(asked), bool))


def _shift(dist, delta):
    K = dist.shape[1]
    asked = np.clip(1.0 - dist[:, K - 1], 0.0, 1.0)
    shape = np.clip(dist[:, :K - 1], 0.0, None)
    total = shape.sum(1, keepdims=True)
    shape = np.divide(shape, total, out=np.full_like(shape, 1.0 / (K - 1)), where=total > 0)
    new = np.clip(asked + delta, 0.0, 1.0)
    out = np.empty_like(dist)
    out[:, :K - 1] = shape * new[:, None]
    out[:, K - 1] = 1.0 - new
    return out


def reproject(items, work, final, final_oofs, stacked, snap, fold, order, declared, skip=(), X=None, given=()):
    """Close single and joint parent updates, preserving conditional shapes.

    Parent joint odds come from fold-excluded labels and IPF to the predicted
    marginals. Wave-one zero-coded skips retain their original mass exactly.
    No child-by-child score veto can leave a moved parent's graph half closed.
    """
    from .propagate_structural_gates import _rake, ZERO_ON_SKIP
    f0, o0 = snap
    info = {'checked': 0, 'moved_parents': 0, 'projected': 0,
            'joint_projected': 0, 'zero_wave_preserved': 0}
    state_cache = {}
    for name in order:
        if name not in work or name not in final or not items[name].get('gate'):
            continue
        info['checked'] += 1
        parents = tuple(declared(name, items))
        if name in skip or not 1 <= len(parents) <= 2 or any(p not in work or p not in o0 or p not in final for p in parents):
            continue
        j = work[name]
        if j['K'] < 2 or any(not np.array_equal(work[p]['train'], j['train']) for p in parents):
            continue
        if all(np.array_equal(final_oofs.get(p, stacked[p][0]), o0[p]) and np.array_equal(final[p], f0[p]) for p in parents):
            continue
        rows, rest = np.flatnonzero(j['train']), np.flatnonzero(~j['train'])
        fr = fold[rows]
        key = (parents, rows.tobytes())
        if key not in state_cache:
            if len(parents) == 1:
                p = parents[0]
                true, P = work[p]['y'][rows], work[p]['K']
                do = np.asarray(final_oofs.get(p, stacked[p][0]), float) - o0[p]
                dr = np.asarray(final[p][rest], float) - f0[p][rest]
            else:
                a, b = parents
                Ka, Kb = work[a]['K'], work[b]['K']
                P = Ka * Kb
                true = work[a]['y'][rows] * Kb + work[b]['y'][rows]
                ao, bo = final_oofs.get(a, stacked[a][0]), final_oofs.get(b, stacked[b][0])
                do = np.zeros((len(rows), P))
                def table(sel):
                    c = np.bincount(true[sel], minlength=P) + 0.5
                    return (c / c.sum()).reshape(Ka, Kb)
                for f in np.unique(fr):
                    m = fr == f
                    M = table(~m)
                    do[m] = _rake(M, ao[m], bo[m]) - _rake(M, o0[a][m], o0[b][m])
                M = table(np.ones(len(rows), bool))
                dr = _rake(M, final[a][rest], final[b][rest]) - _rake(M, f0[a][rest], f0[b][rest])
            state_cache[key] = true, P, do, dr
        true, P, do, dr = state_cache[key]
        active = np.ones(len(rows), bool)
        hidden_active = np.ones(len(rest), bool)
        if name in ZERO_ON_SKIP and X is not None and 'wave' in given:
            first = [str(v) for v in items['wave']['values']].index('1')
            active = X[rows, given.index('wave')] != first
            hidden_active = X[rest, given.index('wave')] != first
            info['zero_wave_preserved'] += 1
        asked = (j['y'][rows] != j['K'] - 1).astype(float)
        def rate(sel):
            sel = sel & active
            c = np.bincount(true[sel], minlength=P).astype(float)
            h = np.bincount(true[sel], weights=asked[sel], minlength=P)
            g = (h.sum() + .5) / (c.sum() + 1.)
            return (h + 2. * g) / (c + 2.)
        delta_o = np.zeros(len(rows))
        for f in np.unique(fr):
            m = fr == f
            delta_o[m] = do[m] @ rate(~m)
        delta_r = dr @ rate(np.ones(len(rows), bool))
        delta_o[~active] = 0.
        delta_r[~hidden_active] = 0.
        cur_o = np.asarray(final_oofs.get(name, stacked[name][0]), float)
        new_o = _shift(cur_o, delta_o)
        new_r = _shift(final[name][rest], delta_r)
        # Preserve byte-for-byte values wherever no probability mass moved.
        new_o[delta_o == 0] = cur_o[delta_o == 0]
        new_r[delta_r == 0] = final[name][rest][delta_r == 0]
        final_oofs[name] = new_o
        final[name] = np.array(final[name], float, copy=True)
        final[name][rest] = new_r
        info['moved_parents'] += 1
        info['projected'] += 1
        info['joint_projected'] += int(len(parents) == 2)
    return info
