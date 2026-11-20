"""Cross-round answer evidence with fold-isolated household posteriors."""
import time
import numpy as np

V104_K1 = False
V104_K2 = True


def source_open(name, work, items, X, given):
    gate = items[name].get('gate')
    if not gate:
        return np.ones(len(X), bool)
    p = gate['parent']
    vals = items[p]['values']
    allowed = [vals.index(v) for v in gate['observed_if'] if v in vals]
    if p in work:
        return np.isin(work[p]['y'], allowed)
    if p in given:
        return np.isin(X[:, given.index(p)], allowed)
    raise KeyError(p)


def augmented_fit(src, dst, base, opened, K, mask):
    from . import twin_identity as TI
    code = src * 2 + opened.astype(int)
    C = np.bincount(code[mask] * K + dst[mask], minlength=2*K*K).reshape(2*K, K)
    T = C + TI.IDENTITY_PRIOR * np.repeat(np.eye(K), 2, axis=0)
    T /= T.sum(1, keepdims=True)
    rr, tt = np.zeros(2), np.zeros(2)
    for g in (0, 1):
        hit = mask & (opened == g)
        if hit.sum() < 12:
            continue
        B, d = base[hit], dst[hit]
        R = T[code[hit]]; row = np.arange(len(d)); b = B[row,d]
        best = (np.inf, 0., 0.)
        for tau in TI.TAUS:
            M = R * (B >= tau); sm = M.sum(1)
            t = np.where(sm > 0, M[row,d] / np.maximum(sm,1e-300), b)
            mix = TI.RHOS[:,None]*t + (1-TI.RHOS[:,None])*b
            loss = -np.log(1e-3+(1-K*1e-3)*mix).sum(1)
            k = int(loss.argmin())
            if loss[k] < best[0]: best = (loss[k],TI.RHOS[k],tau)
        rr[g], tt[g] = best[1:]
    return T, rr, tt


def augmented_predict(T, rr, tt, src, opened, base):
    from . import twin_identity as TI
    out = base.copy()
    for g in (0,1):
        h = opened == g
        R = TI._open_rows(T, src[h]*2+g, base[h], tt[g])
        out[h] = rr[g]*R + (1-rr[g])*base[h]
    return out


def apply_open(work, final, final_oofs, stacked, pairs, fold, wave, report, items, X, given):
    from . import twin_identity as TI
    train = next(iter(work.values()))['train']
    a = np.asarray(pairs,int).reshape(-1,2)
    if not len(a): return
    t = np.r_[a[:,0],a[:,1]]; s = np.r_[a[:,1],a[:,0]]
    vv = train[t]&train[s]; hh = ~train[t]&train[s]
    vt,vs,ht,hs=t[vv],s[vv],t[hh],s[hh]
    report.update(pairs=len(pairs),visible_edges=len(vt),hidden_edges=len(ht),changed=0,augmented=True)
    if len(vt)<TI.MIN_EDGES or not len(ht): return
    pos=np.full(len(train),-1);pos[train]=np.arange(train.sum())
    ft,fs=fold[vt],fold[vs]; half=ft%2==0
    changes={}; groups={}
    for name,j in work.items():
        if name not in final or not np.array_equal(train,j['train']):continue
        if not items[name].get('gate'):
            # Preserve the proven ungated transfer exactly.
            TI.apply({name:j},final,final_oofs,stacked,pairs,fold,wave,{},context=None)
            continue
        y,K=j['y'],j['K']; opened=source_open(name,work,items,X,given)
        o=np.asarray(final_oofs.get(name,stacked[name][0]),float);o=o/o.sum(1,keepdims=True)
        bv=o[pos[vt]]; bh=np.asarray(final[name][ht],float);bh=bh/bh.sum(1,keepdims=True)
        nv,nh=bv.copy(),bh.copy(); rr=[]
        for d in (0,1):
            dv=wave[vt]==d
            for f,g in sorted(set(zip(ft[dv],fs[dv]))):
                ev=dv&(ft==f)&(fs==g)
                fit=dv&~np.isin(ft,[f,g])&~np.isin(fs,[f,g])
                T,r,tau=augmented_fit(y[vs],y[vt],bv,opened[vs],K,fit)
                nv[ev]=augmented_predict(T,r,tau,y[vs[ev]],opened[vs[ev]],bv[ev])
            T,r,tau=augmented_fit(y[vs],y[vt],bv,opened[vs],K,dv)
            hit=wave[ht]==d
            nh[hit]=augmented_predict(T,r,tau,y[hs[hit]],opened[hs[hit]],bh[hit]);rr.append(r.tolist())
        if not all(TI._nll(nv[h],y[vt[h]],K)<TI._nll(bv[h],y[vt[h]],K)-1e-6 for h in (np.ones(len(vt),bool),half,~half) if h.any()):continue
        final[name][ht]=nh; upd=o.copy();upd[pos[vt]]=nv;final_oofs[name]=upd
        changes[name]=TI._nll(bv,y[vt],K)-TI._nll(nv,y[vt],K);groups[name]=rr
    report.update(changed=len(changes),oof_edge_gain=float(np.mean(list(changes.values()))) if changes else 0.,by_item_gain=changes,rhos=groups)


def _normal(L):
    L=L-L.max(1,keepdims=True); P=np.exp(L);return P/P.sum(1,keepdims=True)


def apply_latent(state, work, final, final_oofs, stacked, pairs, fold, wave, items, X, given, report, collision=()):
    """Same-fold visible edges and source-held-out hidden edges share read-back.

    Target-subtree answers are removed from the source likelihood, so this arm
    conveys household trait evidence independently of the single-item copy arm.
    """
    from . import twin_identity as TI
    began=time.monotonic();report.update(changed=0)
    if not state or not pairs:return
    train=next(iter(work.values()))['train'];a=np.asarray(pairs,int).reshape(-1,2)
    t=np.r_[a[:,0],a[:,1]];s=np.r_[a[:,1],a[:,0]]
    if collision:
        t=np.r_[t, np.concatenate([np.full(len(c),row,int) for row,c,_ in collision])]
        s=np.r_[s, np.concatenate([np.asarray(c,int) for _,c,_ in collision])]
    valid=train[s]&((~train[t])|(fold[t]==fold[s]))
    t,s=t[valid],s[valid]
    vv=train[t]; hh=~vv;vt,ht=t[vv],t[hh]
    report.update(visible_edges=int(vv.sum()),hidden_edges=int(hh.sum()),excluded_crossfold=True)
    if vv.sum()<80 or not hh.any():return
    names=state['names'];blocks=state['blocks'];G=len(given);models=state['models']
    pos=np.full(len(train),-1);pos[train]=np.arange(train.sum())
    roots={}
    for name in names:
        p=name
        while items[p].get('gate') and items[p]['gate']['parent'] in work:p=items[p]['gate']['parent']
        roots[name]=p
    slices={};pv={name:np.empty((len(t),work[name]['K'])) for name in names}
    opened={name:source_open(name,work,items,X,given) for name in names}
    for f,m in enumerate(models):
        hit=np.flatnonzero(fold[s]==f)
        if not len(hit):continue
        src=s[hit];dst=t[hit]
        assert np.all(~train[dst]|(fold[dst]==f)) and np.all(fold[src]==f)
        Lg=np.zeros((len(src),len(m['lw'])))
        for g in range(G):Lg+=m['log_t'][:,blocks[g][0]+X[src,g]].T
        contributions={r:np.zeros_like(Lg) for r in set(roots.values())}
        for i,name in enumerate(names):
            off,w=blocks[G+i];y=work[name]['y'][src]
            ok=(y>=0)&(y<w)&opened[name][src]
            contributions[roots[name]][ok]+=m['log_t'][:,off+y[ok]].T
        total=sum(contributions.values())
        post={r:_normal(Lg+0.5*(total-c)+m['lw'][None]) for r,c in contributions.items()}
        for name in names:
            # Target routing remains the calibrated base routing. This arm
            # changes option probabilities inside the open state only.
            if 'strat' in m and state.get('strata_codes') is not None:
                codes=state['strata_codes'][dst]
                for code in np.unique(codes):
                    ix=np.flatnonzero(codes==code)
                    pv[name][hit[ix]]=post[roots[name]][ix]@m['strat'][1][code][name]
            else:
                pv[name][hit]=post[roots[name]]@m['E'][name]
    # Collision donors are averaged after each donor's own held-out read-back.
    targets, inverse=np.unique(t,return_inverse=True)
    count=np.bincount(inverse)
    source_group={}
    for name in names:
        combined=np.zeros((len(targets),work[name]['K']))
        np.add.at(combined,inverse,pv[name]);pv[name]=combined/count[:,None]
        source_group[name]=np.bincount(inverse,weights=opened[name][s].astype(float))/count>=0.5
    t=targets;vv=train[t];hh=~vv;vt,ht=t[vv],t[hh]
    changes={}; weights={};ft=fold[vt];half=ft%2==0
    for name in names:
        if name not in final:continue
        y,K=work[name]['y'],work[name]['K']
        o=np.asarray(final_oofs.get(name,stacked[name][0]),float);o=o/o.sum(1,keepdims=True)
        B=np.empty_like(pv[name]);B[vv]=o[pos[vt]]
        bh=np.asarray(final[name][ht],float); B[hh]=bh/bh.sum(1,keepdims=True)
        P=pv[name]; gate=items[name].get('gate')
        if gate and K>len(items[name]['values']):
            # All closed-state mass, including wave-one zero-fill routing,
            # stays with the existing gate; reweight only nominal answers.
            P[:,:-1]/=np.maximum(P[:,:-1].sum(1,keepdims=True),1e-300)
            P[:,:-1]*=(1-B[:,-1:]);P[:,-1]=B[:,-1]
        P/=P.sum(1,keepdims=True)
        nv=B[vv].copy();nh=B[hh].copy();rec={}
        for d in (0,1):
            for op in (0,1):
                hv=(wave[vt]==d)&(source_group[name][vv]==op)
                hd=(wave[ht]==d)&(source_group[name][hh]==op)
                if hv.sum()<30:continue
                b=B[vv][np.arange(vv.sum()),y[vt]];p=P[vv][np.arange(vv.sum()),y[vt]]
                def weight(mask):
                    mix=TI.RHOS[:,None]*p[mask]+(1-TI.RHOS[:,None])*b[mask]
                    return TI.RHOS[int(np.argmin(-np.log(1e-3+(1-K*1e-3)*mix).sum(1)))] if mask.sum()>=20 else 0.
                for f in np.unique(ft[hv]):
                    ev=hv&(ft==f);r=weight(hv&(ft!=f));nv[ev]=r*P[vv][ev]+(1-r)*B[vv][ev]
                r=weight(hv);rec[f'{d}:{op}']=float(r)
                # Gate each direction/open stratum independently on both halves.
                if all(TI._nll(nv[h],y[vt[h]],K)<TI._nll(B[vv][h],y[vt[h]],K)-1e-6 for h in (hv,hv&half,hv&~half) if h.any()):
                    nh[hd]=r*P[hh][hd]+(1-r)*B[hh][hd]
                else:nv[hv]=B[vv][hv]
        gain=TI._nll(B[vv],y[vt],K)-TI._nll(nv,y[vt],K)
        if gain<=1e-7:continue
        final[name][ht]=nh;upd=o.copy();upd[pos[vt]]=nv;final_oofs[name]=upd
        changes[name]=gain;weights[name]=rec
    report.update(changed=len(changes),by_item_gain=changes,weights=weights,seconds=time.monotonic()-began)
