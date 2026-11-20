"""Active-face Newton likelihood maximization with inactive-arm KKT reentry."""
import time
import numpy as np


def weights(L, deadline=float('inf')):
    n,m=L.shape
    if m==1:return np.ones(1)
    if time.monotonic()>=deadline:
        w=np.zeros(m);w[int(np.argmin(-np.log(np.maximum(L,1e-300)).mean(0)))]=1.;return w
    w=np.full(m,1/m)
    for _ in range(8):
        if time.monotonic()>=deadline:break
        w*=np.mean(L/np.maximum(L@w,1e-300)[:,None],axis=0);w/=w.sum()
    for _ in range(64):
        if time.monotonic()>=deadline:break
        q=np.maximum(L@w,1e-300);R=L/q[:,None];g=R.mean(0)
        if g.max()-w@g<1e-8:break
        active=np.flatnonzero(w>1e-12);enter=int(np.argmax(g))
        if enter not in active:active=np.r_[active,enter]
        if len(active)==1:break
        # Eliminate the equality constraint against the largest-weight member.
        anchor=int(active[np.argmax(w[active])]);other=active[active!=anchor]
        D=R[:,other]-R[:,anchor,None];H=D.T@D/n
        grad=g[other]-g[anchor]
        try:d0=np.linalg.solve(H+np.eye(len(other))*1e-10,grad)
        except np.linalg.LinAlgError:d0=np.linalg.lstsq(H,grad,rcond=1e-10)[0]
        d=np.zeros(m);d[other]=d0;d[anchor]=-d0.sum()
        neg=d < -1e-14
        alpha=min(1.,float(np.min(-w[neg]/d[neg]))) if neg.any() else 1.
        # A blocked face must release a zero-weight coordinate, or move along
        # a feasible pair direction to admit an arm whose KKT gain is positive.
        if alpha<1e-10 or g@d<=0:
            donor=int(active[np.argmin(g[active])]);d=np.zeros(m)
            if donor==enter:break
            d[enter]=1.;d[donor]=-1.;alpha=w[donor]
        f=np.log(q).mean();accepted=False
        for _ in range(32):
            trial=np.maximum(w+alpha*d,0.);trial/=trial.sum()
            ft=np.log(np.maximum(L@trial,1e-300)).mean()
            if ft>=f+1e-4*alpha*(g@d)-1e-14:
                accepted=True;break
            alpha*=.5
        if not accepted:break
        w=trial
    return w


def stack(oofs,rests,y,K,deadline=float('inf')):
    from .base import _FLOOR,_nll,_softmax
    L=_FLOOR+(1-K*_FLOOR)*np.stack([a[np.arange(len(y)),y] for a in oofs],1).astype(float)
    w=weights(L,deadline)
    o=sum(x*a for x,a in zip(w,oofs));p=sum(x*a for x,a in zip(w,rests))
    t,loss=1.,_nll(o,y,K);log=np.log(np.clip(o,1e-12,None))
    for candidate in (.8,.9,1.1,1.2,1.35,1.5,1.75,2.,2.5,3.,4.,6.):
        if time.monotonic()>=deadline:break
        value=_nll(_softmax(log*candidate),y,K)
        if value<loss:t,loss=candidate,value
    if t!=1.:
        o=_softmax(log*t);p=_softmax(np.log(np.clip(p,1e-12,None))*t)
    return p,o,w,t,loss
