"""Event-conditioned reduced-rank response models with item-specific masks."""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .chain_features import check_clock, BudgetExpired
from .cross_round import dominates
from .base import _nll
from . import v68_config

RIASEC=('sc_nstd_realistic','sc_nstd_invest','sc_nstd_artistic','sc_nstd_social','sc_nstd_enterp','sc_nstd_convent')


def event_mask(work, names, items):
    present = np.zeros(len(work[names[0]]['y']), bool)
    for name in names:
        j = work[name]; y = j['y']; na = items[name]['values'].index('Not answered')
        present |= (y >= 0) & (y != na) & (y != j['K'] - 1)
    return present


def compose(cond, event, randomized, na, k):
    """P(Y)=P(event)P(Y|event)+P(no event) delta_NA."""
    cond = np.maximum(np.asarray(cond, float), 0.)
    cond[:, k-1] = 0.
    s = cond.sum(1, keepdims=True)
    reserve = np.zeros_like(cond); reserve[:, :k-1] = 1./(k-1)
    cond = np.divide(cond, s, out=reserve, where=s > 1e-12)
    e = np.clip(event, 0., 1.)
    out = cond * e[:, None]; out[:, na] += 1.-e
    out[~randomized] = 0.; out[~randomized, k-1] = 1.
    return out


def design(X, given, chain, train, anchors=RIASEC):
    raw = np.column_stack([np.eye(int(X[:, c].max())+1)[X[:, c]] for c in range(X.shape[1])])
    if chain is None:
        return raw
    cols = [raw]
    probs = [chain.probabilities(a, train) for a in anchors if a in chain.anchors]
    if probs:
        cols.extend(probs)
        ranks = np.column_stack([p @ np.linspace(-1., 1., p.shape[1]) for p in probs])
        centered = ranks-ranks.mean(1, keepdims=True)
        cols.append(centered)
        if anchors == RIASEC:
            cols.append(np.column_stack([centered[:, i]-centered[:, j] for i in range(centered.shape[1]) for j in range(i)]) if centered.shape[1]>1 else centered)
            for g in ('user.educationLevel', 'user.hadJob', 'group'):
                if g in given:
                    one = np.eye(int(X[:, given.index(g)].max())+1)[X[:, given.index(g)]]
                    cols.append((one[:, :, None]*centered[:, None, :]).reshape(len(X), -1))
    cols.append(chain._latent(train, train.tobytes())[:, :6])
    return np.column_stack(cols)


def standardize(F, fit):
    mu=F[fit].mean(0); sd=F[fit].std(0); sd[sd<.05]=1.
    return np.column_stack((np.ones(len(F)), (F-mu)/sd))


def ridge(A, Y, strength=.04):
    gram=A.T@A
    penalty=np.full(A.shape[1], max(10., strength*len(A)));penalty[0]=1e-5
    gram.flat[::len(gram)+1] += penalty
    return cho_solve(cho_factor(gram, check_finite=False), A.T@Y, check_finite=False)


def probabilities(scores):
    # Project a regularized one-hot regression onto the probability simplex.
    p=np.maximum(scores, 1e-5)
    return p/p.sum(1, keepdims=True)


def admit(j, arm, o, full, visible, fold, deadline):
    """Five-fold mixture comparison, followed by independent half-fold checks."""
    from .main import crossfit_calibrate
    keep=[i for i,a in enumerate(j['arms']) if j.get('partition_k',{}).get(a,5)==5]
    if not keep:keep=[0]
    y=j['y'][visible]; split=fold[visible]; k=j['K']
    base_o=[j['oof'][i] for i in keep]
    empty=[np.empty((0,k)) for _ in keep]
    _, old, _, _, old_loss=crossfit_calibrate(base_o, empty, y, k, split, deadline)
    _, new, w, _, new_loss=crossfit_calibrate(base_o+[o], empty+[np.empty((0,k))], y, k, split, deadline)
    added=bool(w[-1]>1e-4 and dominates(new,old,y,k,split%2==0))
    trace=dict(added=added, weight=float(w[-1]), gain=float(old_loss-new_loss),
               raw_gain=float(_nll(j['oof'][0],y,k)-_nll(o,y,k)))
    if added:
        j['arms'].append(arm); j['oof'].append(o); j['all'].append(full)
        j['check_arms']=tuple(dict.fromkeys((*j.get('check_arms',()),arm)))
    return trace


def attach(schema,X,given,work,fold,chain,names,visible,rest,event_oof,event_full,randomized,deadline):
    if not v68_config.K1:
        from .wb_multitask_legacy import attach as legacy
        return legacy(schema,X,given,work,fold,chain,names,visible,rest,event_oof,event_full,randomized,deadline)
    started=time.monotonic(); items=schema['items']; present=event_mask(work,names,items)&randomized
    offsets=np.r_[0,np.cumsum([work[n]['K']-1 for n in names])]
    arms=('present_multitask','answered_multitask')
    oofs={a:{n:np.zeros((len(visible),work[n]['K']),float) for n in names} for a in arms}
    fulls={a:{n:np.zeros((len(X),work[n]['K']),float) for n in names} for a in arms}
    counts={n:[] for n in names}; fit_counts=[]; partitions=0
    try:
        for f in [*np.unique(fold[visible]),None]:
            check_clock(deadline)
            train=visible if f is None else visible[fold[visible]!=f]
            query=rest if f is None else visible[fold[visible]==f]
            fit=train[present[train]]; fit_counts.append(len(fit))
            if not len(fit):
                return dict(partitions=partitions,completed=0,present_rows_by_partition=fit_counts)
            F=standardize(design(X,given,chain,train),fit)
            Y=np.column_stack([np.eye(work[n]['K']-1)[work[n]['y'][fit]] for n in names])
            B=ridge(F[fit],Y); fitted=F[fit]@B
            _,vectors=np.linalg.eigh(fitted.T@fitted)
            basis=vectors[:,-min(8,Y.shape[1]):]
            Z=np.column_stack((F,F@B@basis))
            predicted=Z[query]@ridge(Z[fit],Y)
            e=(event_full if f is None else event_oof)[query]
            for i,n in enumerate(names):
                check_clock(deadline)
                j=work[n]; k=j['K']; na=items[n]['values'].index('Not answered')
                cond=np.zeros((len(query),k));cond[:,:k-1]=probabilities(predicted[:,offsets[i]:offsets[i+1]])
                pn=compose(cond,e,randomized[query],na,k)
                answered=fit[j['y'][fit]!=na];counts[n].append(len(answered))
                pa=pn.copy()
                if len(answered)>=25:
                    idx=np.array([c for c in range(k-1) if c!=na])
                    yy=(j['y'][answered,None]==idx).astype(float)
                    q=probabilities(Z[query]@ridge(Z[answered],yy))
                    rate=np.clip(Z[query]@ridge(Z[fit],(j['y'][fit]!=na).astype(float)[:,None]),0.,1.)[:,0]
                    cond[:]=0.;cond[:,idx]=rate[:,None]*q;cond[:,na]=1.-rate
                    pa=compose(cond,e,randomized[query],na,k)
                for a,p in zip(arms,(pn,pa)):
                    if f is None:fulls[a][n][query]=p
                    else:oofs[a][n][fold[visible]==f]=p
            partitions+=1
    except BudgetExpired:
        return dict(partitions=partitions,completed=0,present_rows_by_partition=fit_counts,seconds=time.monotonic()-started)
    trace={}
    for n in names:
        trace[n]={'answered_rows_by_partition':counts[n]}
        for a in arms:
            fulls[a][n][visible]=oofs[a][n]
            trace[n][a]=admit(work[n],a,oofs[a][n],fulls[a][n],visible,fold,deadline)
    return dict(partitions=partitions,completed=len(trace),present_rows_by_partition=fit_counts,
                admitted={a:sum(t[a]['added'] for t in trace.values()) for a in arms},
                seconds=time.monotonic()-started,items=trace)
