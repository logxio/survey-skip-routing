"""Hierarchical arm x primary x secondary response tensors."""
import numpy as np
from .chain_features import check_clock, BudgetExpired, PRE_POST_MIRRORS
from .cross_round import dominates
from .rao_blackwell import adjusted_mi
from .base import _nll


def project(a, b, y, group, fit, query, pa, pb, ka, kb, ky, alpha=12.):
    fit=fit[(a[fit]>=0)&(b[fit]>=0)&(y[fit]>=0)]
    ng=int(group.max())+1
    prior=(np.bincount(y[fit],minlength=ky)+.5)/(len(fit)+.5*ky)
    def count(code,size):
        return np.bincount(code[fit]*ky+y[fit],minlength=size*ky).reshape(size,ky).astype(float)
    ca=count(a,ka);ta=(ca+alpha*prior)/(ca.sum(1,keepdims=True)+alpha)
    cb=count(b,kb);tb=(cb+alpha*prior)/(cb.sum(1,keepdims=True)+alpha)
    cg=count(group,ng);tg=(cg+alpha*prior)/(cg.sum(1,keepdims=True)+alpha)
    cab=count(a*kb+b,ka*kb).reshape(ka,kb,ky)
    tab=(cab+alpha*.5*(ta[:,None]+tb[None]))/(cab.sum(2,keepdims=True)+alpha)
    cga=count(group*ka+a,ng*ka).reshape(ng,ka,ky)
    tga=(cga+alpha*.5*(ta[None]+tg[:,None]))/(cga.sum(2,keepdims=True)+alpha)
    cgb=count(group*kb+b,ng*kb).reshape(ng,kb,ky)
    tgb=(cgb+alpha*.5*(tb[None]+tg[:,None]))/(cgb.sum(2,keepdims=True)+alpha)
    prior3=(tab[None]+tga[:,:,None]+tgb[:,None])/3
    cgab=count((group*ka+a)*kb+b,ng*ka*kb).reshape(ng,ka,kb,ky)
    tensor=(cgab+alpha*prior3)/(cgab.sum(3,keepdims=True)+alpha)
    # Arm is a GIVEN state. Integrate both uncertain anchors exactly under
    # the supplied joint approximation P(a,b)=P(a)P(b).
    p=np.empty((len(query),ky))
    for g in np.unique(group[query]):
        m=group[query]==g
        p[m]=np.einsum('na,nb,aby->ny',pa[m],pb[m],tensor[g],optimize=True)
    return p/np.maximum(p.sum(1,keepdims=True),1e-12)


def _group_logodds(L, y, g, ng, K, ridge=1.):
    """Per-group temperature and class offsets: softmax(t_g * log p + b_g)."""
    from scipy.optimize import minimize
    Y = np.zeros_like(L); Y[np.arange(len(y)), y] = 1.
    def objective(theta):
        t, b = theta[:ng], theta[ng:].reshape(ng, K)
        z = t[g][:, None] * L + b[g]
        z -= z.max(1, keepdims=True)
        P = np.exp(z); s = P.sum(1); P /= s[:, None]
        R = P - Y
        loss = -(z[np.arange(len(y)), y] - np.log(s)).sum() + ridge * (((t - 1.) ** 2).sum() + (b ** 2).sum())
        gt = np.bincount(g, weights=(R * L).sum(1), minlength=ng) + 2. * ridge * (t - 1.)
        gb = np.zeros((ng, K)); np.add.at(gb, g, R); gb += 2. * ridge * b
        return loss, np.concatenate((gt, gb.ravel()))
    theta = np.concatenate((np.ones(ng), np.zeros(ng * K)))
    result = minimize(objective, theta, jac=True, method='L-BFGS-B', options={'maxiter': 60})
    return result.x if np.isfinite(result.x).all() else theta

def _apply_logodds(L, g, theta, ng, K):
    z = theta[:ng][g][:, None] * L + theta[ng:].reshape(ng, K)[g]
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(1, keepdims=True)

def attach_logodds(work, X, given, fold, deadline, names):
    """Fold-out treatment-group recalibration of each item's best five-fold arm."""
    groups = X[:, given.index('group')]; ng = int(groups.max()) + 1
    trace = {}
    for child in names:
        if child not in work: continue
        j = work[child]; rows = np.flatnonzero(j['train']); rest = np.flatnonzero(~j['train'])
        if len(rows) < 40 or len(np.unique(fold[rows])) != 5: continue
        yy, K, pk = j['y'][rows], j['K'], j.get('partition_k', {})
        five = [i for i, a in enumerate(j['arms']) if pk.get(a, 5) == 5]
        if not five: continue
        base = min(five, key=lambda i: _nll(j['oof'][i], yy, K))
        L_oof = np.log(np.asarray(j['oof'][base], float) + 1e-6)
        L_rest = np.log(np.asarray(j['all'][base][rest], float) + 1e-6)
        g_rows = groups[rows]
        out = np.zeros_like(L_oof); full = np.zeros((len(X), K))
        try:
            for f in [*np.unique(fold[rows]), None]:
                check_clock(deadline)
                fit = np.ones(len(rows), bool) if f is None else fold[rows] != f
                theta = _group_logodds(L_oof[fit], yy[fit], g_rows[fit], ng, K)
                if f is None: full[rest] = _apply_logodds(L_rest, groups[rest], theta, ng, K)
                else: out[~fit] = _apply_logodds(L_oof[~fit], g_rows[~fit], theta, ng, K)
        except BudgetExpired:
            break
        base_oof = j['oof'][base]
        ok = dominates(out, base_oof, yy, K, fold[rows] % 2 == 0)
        trace[child] = dict(added=bool(ok), base=j['arms'][base], gain=float(_nll(base_oof, yy, K) - _nll(out, yy, K)))
        if ok:
            j['arms'].append('rct_logodds'); j['oof'].append(out); j['all'].append(full)
    return trace

def attach(work,X,given,fold,bank,stacked,deadline,children,items):
    from .paired_transition import _parent_probs
    groups=X[:,given.index('group')];anchors=[a for a in stacked if not a.startswith('tr_')]
    cache={a:_parent_probs(a,work,stacked,None) for a in anchors}
    trace={}
    for child in children:
        if child not in work or child.startswith('tr_'):continue
        j=work[child];rows=np.flatnonzero(j['train']);rest=np.flatnonzero(~j['train'])
        if len(rows)<40:continue
        out=np.zeros_like(j['oof'][0]);full=np.zeros_like(j['all'][0]);chosen=[]
        primary=PRE_POST_MIRRORS.get(child)
        # Wage and reservation-pay baselines are observed GIVEN columns.
        observed=child[:-5] if child.endswith('After') and child[:-5] in given else None
        try:
            for f in [*np.unique(fold[rows]),None]:
                check_clock(deadline)
                fit=rows if f is None else rows[fold[rows]!=f];q=rest if f is None else rows[fold[rows]==f]
                ranked=sorted(((adjusted_mi(work[a]['y'][fit],work[a]['K'],j['y'][fit],j['K']),a) for a in anchors if a!=child),reverse=True)
                if observed:
                    a=observed;ay=X[:,given.index(a)];ka=int(ay.max())+1;pa=np.eye(ka)[ay[q]]
                else:
                    a=primary if primary in cache else ranked[0][1];ay=work[a]['y'];ka=work[a]['K'];pa=cache[a][q]
                b=next(a2 for _,a2 in ranked if a2!=a);by=work[b]['y'];kb=work[b]['K'];pb=cache[b][q]
                p=project(ay,by,j['y'],groups,fit,q,pa,pb,ka,kb,j['K'])
                if f is None:full[q]=p
                else:out[fold[rows]==f]=p
                chosen.append([a,b])
            if len(chosen)!=6:continue
            ok=dominates(out,j['oof'][0],j['y'][rows],j['K'],fold[rows]%2==0)
            trace[child]=dict(added=bool(ok),anchors=chosen,gain=float(_nll(j['oof'][0],j['y'][rows],j['K'])-_nll(out,j['y'][rows],j['K'])))
            if ok:j['arms'].append('rct_tensor');j['oof'].append(out);j['all'].append(full)
        except BudgetExpired:break
    return trace
