"""Answered-only reduced-rank log-odds regression with shared RIASEC features."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from .chain_features import check_clock, BudgetExpired
from .cross_round import dominates
from .base import _nll

RIASEC=('sc_nstd_realistic','sc_nstd_invest','sc_nstd_artistic','sc_nstd_social','sc_nstd_enterp','sc_nstd_convent')


def attach(schema,X,given,work,fold,chain,names,visible,rest,event_oof,event_full,randomized,deadline):
    if chain is None:return {}
    offsets=np.r_[0,np.cumsum([work[n]['K']-2 for n in names])]
    if offsets[-1]<1:return {}
    raw=[]
    for col in range(X.shape[1]):raw.append(np.eye(int(X[:,col].max())+1)[X[:,col]])
    raw=np.column_stack(raw)
    oofs={n:np.zeros_like(work[n]['oof'][0]) for n in names}
    fulls={n:np.zeros_like(work[n]['all'][0]) for n in names}
    partitions=0
    try:
        for f in [*np.unique(fold[visible]),None]:
            check_clock(deadline)
            train=visible if f is None else visible[fold[visible]!=f]
            query=rest if f is None else visible[fold[visible]==f]
            # Joint answered subset uses fitting rows only, never held-out answers.
            answered=randomized[train].copy()
            for n in names:
                j=work[n];na=schema['items'][n]['values'].index('Not answered')
                answered &= (j['y'][train]>=0)&(j['y'][train]!=na)&(j['y'][train]!=j['K']-1)
            fit=train[answered]
            if len(fit)<25:return {}
            rs=[]
            for a in RIASEC:
                if a in work:
                    pp=chain.probabilities(a,train)
                    # Mid-bin ordinal rank, with missing bins carrying no rank.
                    z=np.zeros(work[a]['K']);z[:min(6,len(z))]=np.arange(min(6,len(z)))
                    rs.append(pp@z)
            R=np.column_stack(rs) if rs else np.zeros((len(X),0))
            R=R-R.mean(1,keepdims=True) if R.shape[1] else R
            dif=np.column_stack([R[:,i]-R[:,j] for i in range(R.shape[1]) for j in range(i)]) if R.shape[1]>1 else R
            latent=chain._latent(train,train.tobytes())[:,:6]
            cols=[raw,R,dif,latent]
            for g in ('user.educationLevel','user.hadJob','group'):
                if g in given and R.shape[1]:
                    one=np.eye(int(X[:,given.index(g)].max())+1)[X[:,given.index(g)]]
                    cols.append((one[:,:,None]*R[:,None,:]).reshape(len(X),-1))
            F=np.column_stack(cols);mu=F[fit].mean(0);sd=F[fit].std(0);sd[sd<.05]=1.
            F=np.column_stack((np.ones(len(X)),(F-mu)/sd))
            blocks=[];priors=[]
            for n in names:
                j=work[n];na=schema['items'][n]['values'].index('Not answered')
                idx=np.array([i for i in range(j['K']) if i not in (na,j['K']-1)])
                counts=np.bincount(j['y'][fit],minlength=j['K'])[idx]+.5;prior=counts/counts.sum();priors.append((idx,prior))
                Y=(j['y'][fit,None]==idx).astype(float)
                log=np.log(.05*prior[None]+.95*Y)-np.log(prior)[None];log-=log.mean(1,keepdims=True);blocks.append(log)
            Y=np.column_stack(blocks);A=F[fit]
            gram=A.T@A;penalty=np.full(A.shape[1],max(10.,.04*len(fit)));penalty[0]=1e-5
            gram.flat[::len(gram)+1]+=penalty
            coef=cho_solve(cho_factor(gram,check_finite=False),A.T@Y,check_finite=False)
            # Shared task subspace couples the 19 responses, fitted on this fold.
            fitted=A@coef;_,vectors=np.linalg.eigh(fitted.T@fitted)
            basis=vectors[:,-min(8,Y.shape[1]):];pred=(F[query]@coef@basis)@basis.T
            for i,n in enumerate(names):
                j=work[n];idx,prior=priors[i];na=schema['items'][n]['values'].index('Not answered')
                cond=softmax(np.log(prior)[None]+pred[:,offsets[i]:offsets[i+1]],axis=1)
                p=np.zeros((len(query),j['K']));e=np.clip((event_full if f is None else event_oof)[query],0,1)
                p[:,idx]=e[:,None]*cond;p[:,na]=1-e;p[~randomized[query]]=0;p[~randomized[query],j['K']-1]=1
                if f is None:fulls[n][query]=p
                else:oofs[n][fold[visible]==f]=p
            partitions+=1
    except BudgetExpired:return {'partitions':partitions,'completed':0}
    trace={}
    for n in names:
        j=work[n];o=oofs[n];ok=dominates(o,j['oof'][0],j['y'][visible],j['K'],fold[visible]%2==0)
        trace[n]=dict(added=bool(ok),gain=float(_nll(j['oof'][0],j['y'][visible],j['K'])-_nll(o,j['y'][visible],j['K'])))
        if ok:j['arms'].append('answered_multitask');j['oof'].append(o);j['all'].append(fulls[n])
    return dict(partitions=partitions,completed=len(trace),items=trace)
