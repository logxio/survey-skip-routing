"""KKT inverse of the scoring floor, with held-out strength selection."""
import numpy as np
STANDARD = (0., .5, .75, 1.)
EXTENDED = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)

def _waterfill(P, g):
    P = np.asarray(P, dtype=float)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    K = P.shape[1]
    if g <= 0 or K*g >= .5:
        return P.copy()
    c = g/(1-K*g)
    s = -np.sort(-P, axis=1)
    lam = np.cumsum(s, axis=1)/(1+c*np.arange(1,K+1))
    count = (s > c*lam).sum(1)
    level = lam[np.arange(len(P)), np.maximum(1,count)-1]
    Q = np.maximum(P/np.maximum(level[:,None],1e-300)-c,0.)
    return Q/np.maximum(Q.sum(1,keepdims=True),1e-300)

def _floor_fill(work, stacked, final, final_oofs, fold, items, policy='extended'):
    taus = STANDARD if policy == 'standard' else EXTENDED
    chosen = {}; total_before = []; total_after = []
    for name, fin in final.items():
        j = work.get(name)
        if j is None or name not in stacked: continue
        tr = j['train']; y = j['y'][tr]; K = j['K']
        o = final_oofs.get(name, stacked[name][0])
        if o is None or np.shape(o) != (len(y),K) or len(y)<10: continue
        o = np.asarray(o, float); o = o/np.maximum(o.sum(1,keepdims=True),1e-300)
        ps = [_waterfill(o,t*.001) for t in taus]
        loss = np.array([-np.log(.001+(1-K*.001)*v[np.arange(len(y)),y]) for v in ps])
        rf = np.asarray(fold)[tr]
        # Groups depend only on input predictions. Labels never define row regimes.
        group = np.zeros(len(y),int); hg = np.zeros(np.sum(~tr),int)
        values = items[name].get('values',[])
        skip = K-1 if items[name].get('gate') and K == len(values)+1 else None
        if policy == 'conditional' and items[name].get('gate') and skip is not None and skip<K:
            group = (o[:,skip]>.95).astype(int)
            hg = (fin[~tr,skip]>.95).astype(int)
        def select(mask):
            if np.sum(mask)<20: return 0
            best = int(np.argmin(loss[:,mask].mean(1)))
            # Two independent fold groups must agree on the sign of this choice.
            fs = np.unique(rf[mask]); half = np.isin(rf,fs[::2])
            for part in (mask & half, mask & ~half):
                if not np.any(part) or (loss[0,part]-loss[best,part]).mean() <= 0: return 0
            return best
        adjusted = o.copy(); hidden = np.asarray(fin[~tr],float).copy(); picks = {}
        for g in np.unique(group):
            mask = group == g
            for f in np.unique(rf):
                hit = mask & (rf == f)
                if np.any(hit): adjusted[hit] = ps[select(mask & (rf != f))][hit]
            best = select(mask)
            hit = hg == g
            hidden[hit] = _waterfill(hidden[hit],taus[best]*.001)
            picks[str(int(g))] = taus[best]
        before = float(loss[0].mean())
        after = float(-np.log(.001+(1-K*.001)*adjusted[np.arange(len(y)),y]).mean())
        fin[~tr] = hidden
        fin[tr] = adjusted
        final_oofs[name] = adjusted
        chosen[name] = dict(taus=picks, oof_gain=before-after, rows=len(y), gated=bool(items[name].get('gate')))
        total_before.append(before); total_after.append(after)
    return dict(policy=policy, items=sum(any(t>0 for t in c['taus'].values()) for c in chosen.values()),
                oof_gain=float(np.mean(total_before)-np.mean(total_after)) if chosen else 0., chosen=chosen)
