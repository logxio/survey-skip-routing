"""Fold-local joint-parent, wave and latent-grid observation model.

Answers retain the original nominal-response fit. A separate arm estimates
who answers from soft grid counts, conditional on the joint parent state.
Schema corrections supply features, never force a delivered answer to skip.
"""
import numpy as np
from .propagate_structural_gates import GATE_CORRECTIONS, _joint_mi

TAU = 10.0


def _parents(name, items, work, given, rows, earlier):
    gate = items[name].get('gate')
    if not isinstance(gate, dict):
        return (), None
    correction = GATE_CORRECTIONS.get(name)
    parents = list(correction[0] if correction else (gate['parent'],))
    if any(p not in given and p not in earlier for p in parents):
        return (), None
    if correction:
        return tuple(parents), correction if len(parents) > 2 else None
    p = parents[0]
    if p not in work:
        return tuple(parents), None
    y = work[name]['y'][rows]
    target = (y == work[name]['K'] - 1).astype(float)
    yp = work[p]['y'][rows]
    ok = (y >= 0) & (yp >= 0)
    if ok.sum() < 100 or np.unique(target[ok]).size < 2:
        return tuple(parents), None
    kp = int(work[p]['K']);single = _joint_mi(yp[ok], kp, target[ok])
    best, gain = None, .015
    # Root candidates precede their children and are independent of held-out labels.
    for a in earlier:
        if a == p or items[a].get('gate'):
            continue
        ya = work[a]['y'][rows];hit = ok & (ya >= 0)
        if hit.sum() < 100:
            continue
        ka = int(work[a]['K'])
        delta = _joint_mi(yp[hit]*ka+ya[hit], kp*ka, target[hit])-single
        if delta > gain:
            best, gain = a, delta
    if best is not None:
        parents.append(best)
    return tuple(parents), None


def fit(X, given, work, names, items, rows, R, E, route):
    """All target/parent counts and parent selection use this fold's rows only."""
    wave = given.index('wave') if 'wave' in given else None
    nw = int(X[:, wave].max())+1 if wave is not None else 1
    waves = X[rows, wave] if wave is not None else np.zeros(len(rows), int)
    found, earlier = {}, []
    for name in names:
        if route[name] is None:
            earlier.append(name);continue
        parents, count_spec = _parents(name, items, work, given, rows, earlier)
        earlier.append(name)
        if not parents:
            continue
        dims=[];pc=[];valid=np.ones(len(rows),bool);active=[]
        for j,p in enumerate(parents):
            vals=X[rows,given.index(p)] if p in given else work[p]['y'][rows]
            valid &= vals>=0
            dims.append(int(X[:,given.index(p)].max())+1 if p in given else int(work[p]['K']))
            pc.append(np.maximum(vals,0))
            if count_spec:
                active.append([items[p]['values'].index(v) for v in count_spec[1][j] if v in items[p]['values']])
        if count_spec:
            code=np.minimum(sum(np.isin(v,a).astype(int) for v,a in zip(pc,active)),2);size=3
        else:
            code=np.zeros(len(rows),int);size=1
            for v,d in zip(pc,dims):code=code*d+v;size*=d
        if size>512:
            continue
        y=work[name]['y'][rows];valid &= y>=0;skip=y==work[name]['K']-1
        c=waves*size+code
        cnt=np.zeros((nw*size,R.shape[1]));hit=np.zeros_like(cnt)
        np.add.at(cnt,c[valid],R[valid]);np.add.at(hit,c[valid],R[valid]*skip[valid,None])
        cnt=cnt.reshape(nw,size,-1);hit=hit.reshape(nw,size,-1)
        # Hierarchy: pooled parent state -> wave-parent state -> grid effect.
        pooled_cnt=cnt.sum((0,2));pooled_hit=hit.sum((0,2))
        overall=float(skip[valid].mean()) if valid.any() else 0.
        pooled=(pooled_hit+.5*overall)/(pooled_cnt+.5)
        local=(hit.sum(2)+TAU*pooled[None])/(cnt.sum(2)+TAU)
        rates=(hit+TAU*local[:,:,None])/(cnt+TAU)
        found[name]={'parents':parents,'dims':dims,'count':bool(count_spec),'active':active,
                     'rates':rates.transpose(0,2,1),'wave':wave,'given':tuple(given),
                     'joint_counts':cnt.transpose(0,2,1)}
    return found


def _state_prob(spec, E, X, row):
    """Integrate PREDICT parents on the grid; read GIVEN parents exactly."""
    Q=next(iter(E.values())).shape[0]
    if spec['count']:
        p=np.zeros((Q,3));p[:,0]=1
        for name,active in zip(spec['parents'],spec['active']):
            if name in spec['given']:
                yes=np.full(Q,float(X[row,spec['given'].index(name)] in active))
            else:yes=E[name][:,active].sum(1)
            p=np.column_stack([p[:,0]*(1-yes),p[:,1]*(1-yes)+p[:,0]*yes,p[:,2]+p[:,1]*yes])
        return p
    p=np.ones((Q,1))
    for name,d in zip(spec['parents'],spec['dims']):
        if name in spec['given']:
            a=np.zeros((Q,d));a[:,X[row,spec['given'].index(name)]]=1
        else:a=E[name]
        p=(p[:,:,None]*a[:,None,:]).reshape(Q,-1)
    return p


def predict(spec, E, name, X, rows, P):
    """One grid hazard per observed wave/GIVEN-parent group, bounded memory."""
    wave=spec['wave'];cols=([wave] if wave is not None else [])+[spec['given'].index(p) for p in spec['parents'] if p in spec['given']]
    key=X[np.ix_(rows,cols)] if cols else np.zeros((len(rows),1),int)
    groups, inv=np.unique(key,axis=0,return_inverse=True)
    e=E[name];nom=e[:,:-1]/np.maximum(1-e[:,-1:],1e-12)
    out=np.empty((len(rows),e.shape[1]))
    for g in range(len(groups)):
        ix=np.flatnonzero(inv==g);row=rows[ix[0]];w=int(X[row,wave]) if wave is not None else 0
        state=_state_prob(spec,E,X,row)
        # Empirical joint-state mass preserves correlated PREDICT parents;
        # the product law is a prior, and GIVEN states stay clamped.
        counts=spec['joint_counts'][w]
        support=(state>0).astype(float)
        joint=counts*support+TAU*state
        joint/=np.maximum(joint.sum(1,keepdims=True),1e-12)
        sk=(joint*spec['rates'][w]).sum(1)
        out[ix,:-1]=P[ix]@((1-sk[:,None])*nom)
        out[ix,-1]=P[ix]@sk
    return out


def read(models,X,blocks,route,names,sl,W,hidden,G,work,fold,codes=None,zero=(None,{})):
    from . import latent_classes as L
    n=len(X);T=len(L.TRAIT_TEMPERS);F=len(models)
    o=np.zeros((T,n,W),np.float32);a=np.zeros_like(o)
    for m in models:
        logG=m['strat'][0] if codes is not None else m['log_t']
        for rows,dest,factor in ((m['val'],o,1.),(hidden,a,1./F)):
            ll=np.zeros((len(rows),len(m['lw'])))
            for g in range(G):ll+=logG[:,blocks[g][0]+X[rows,g]].T
            st=codes[rows] if codes is not None else np.zeros(len(rows),int)
            for b,t in enumerate(L.TRAIT_TEMPERS):
                lp=t*ll+m['lw'][None];lp-=lp.max(1,keepdims=True);P=np.exp(lp);P/=P.sum(1,keepdims=True)
                for s in np.unique(st):
                    ix=np.flatnonzero(st==s);rr=rows[ix];pp=P[ix]
                    E=m['strat'][1][s] if codes is not None else m['E']
                    for name in names:
                        lo,hi=sl[name]
                        if name in m['joint_routes']:v=predict(m['joint_routes'][name],E,name,X,rr,pp)
                        elif name in m['gskip']:
                            rates=m['gskip'][name];sk=rates[np.minimum(X[rr,route[name][1]],len(rates)-1),None]
                            e=E[name];nom=e[:,:-1]/np.maximum(1-e[:,-1:],1e-12)
                            v=np.column_stack([(1-sk)*(pp@nom),sk])
                        else:v=pp@E[name]
                        dest[b,rr,lo:hi]+=factor*v
    if zero[1]:L._fold_zero(o,zero[0],sl,zero[1]);L._fold_zero(a,zero[0],sl,zero[1])
    return L._pick(o,a,work,names,sl,fold)[0]
