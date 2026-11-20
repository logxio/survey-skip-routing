"""Nested multiresolution Dirichlet empirical Bayes on categorical grids.

Discount zero is the Dirichlet member of the Pitman-Yor family. Concentration
and simplex weights are learned on a rotating inner fold, separately inside
each outer fold. Count banks contain only labelled TRAIN rows. Parent states
are rebuilt inside each exclusion set, with leave-fold-out training states.
"""
from itertools import combinations
import time
import numpy as np
from scipy.sparse import csr_matrix

V137_U1 = True
GIVEN=('Country','Q05','Q04','Q01','Q02','Q03','Q13')


def audit(candidate, baseline, y, fold):
    k=candidate.shape[1]
    d=np.log(.001+(1-k*.001)*candidate[np.arange(len(y)),y])-np.log(.001+(1-k*.001)*baseline[np.arange(len(y)),y])
    half=fold%2==0
    gains=[float(d[m].mean()) if m.any() else float('-inf') for m in (half,~half)]
    se=float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else float('inf')
    return dict(gain=float(d.mean()),se=se,half_gains=gains,added=bool(min(gains)>1e-5 and d.mean()>=max(1e-5,2*se)))


def dense(X):
    # Collision-free mixed radix, with a distinct category for missing values.
    key=np.zeros(len(X),np.int64)
    for c in X.T:
        _,v=np.unique(c,return_inverse=True)
        width=int(v.max())+1
        if int(key.max())>(np.iinfo(np.int64).max-width)//width:
            return np.unique(X,axis=0,return_inverse=True)[1]
        key=key*width+v
    return np.unique(key,return_inverse=True)[1]


class Grid:
    def __init__(self,X,given,work,fold,items):
        active_given = [n for n in GIVEN if n in given] + [n for n in given if n not in GIVEN]
        X=np.asarray(X)[:,[given.index(n) for n in active_given[:7]]]
        self.ids=dense(X); self.g=int(self.ids.max())+1
        _,rep=np.unique(self.ids,return_index=True); small=X[rep]
        self.names=list(work); self.work=work; self.fold=fold; self.items=items
        self.ks=np.array([work[n]['K'] for n in self.names]); self.starts=np.r_[0,np.cumsum(self.ks)[:-1]]
        self.width=int(self.ks.sum()); self.owner=np.repeat(np.arange(len(work)),self.ks)
        self.slices={n:slice(int(s),int(s+k)) for n,s,k in zip(self.names,self.starts,self.ks)}
        self.counts=np.zeros((5,self.g,self.width))
        for name in self.names:
            j=work[name]; mask=j['train'] & (fold>=0)&(fold<5); k=j['K']
            keys=(fold[mask]*self.g+self.ids[mask])*k+j['y'][mask]
            self.counts[:,:,self.slices[name]]=np.bincount(keys,minlength=5*self.g*k).reshape(5,self.g,k)
        self.views=[]
        for cols in [(0,1,2)]+[c for d in (5,6,7) for c in combinations(range(7),d)]:
            ids=dense(small[:,cols]); size=int(ids.max())+1
            project=csr_matrix((np.ones(self.g),(ids,np.arange(self.g))),shape=(size,self.g))
            self.views.append((len(cols),ids,project))
        self.core=self.views[0]

    def norm(self,c):
        denom=np.add.reduceat(c,self.starts,axis=-1)[...,self.owner]
        return c/np.maximum(denom,1e-30)

    def core_prob(self,counts):
        _,ids,P=self.core
        prior=self.norm(counts.sum(0,keepdims=True)+.5)
        return self.norm(P@counts+8*prior)[ids]

    def candidates(self,excluded,deadline=float('inf')):
        active=[f for f in range(5) if f not in excluded]
        counts=self.counts[active].sum(0); prior=self.norm(counts.sum(0,keepdims=True)+.5)
        core=self.core_prob(counts)
        layers=[]
        for alpha in (4.,24.):
            pools={d:np.zeros_like(counts) for d in (5,6,7)}
            for d,ids,P in self.views[1:]:
                if time.monotonic()>=deadline: raise TimeoutError('candidate budget')
                # Hierarchical prior is the country/religion/sex posterior.
                parent=P@core; parent=self.norm(parent)
                pools[d]+=self.norm(P@counts+alpha*parent)[ids]
            layers.extend((core,pools[5]/21,pools[6]/7,pools[7]))
        layers=np.asarray(layers)
        # Parent-state cell estimates are a ninth, separately weighted view.
        coupled=layers[3].copy()
        root_oof={f:self.core_prob(counts-self.counts[f]) for f in active}
        for name in self.names:
            gate=self.items[name].get('gate')
            if not isinstance(gate,dict) or gate['parent'] not in self.slices: continue
            parent=gate['parent']; ps=self.slices[parent]; cs=self.slices[name]; pk=self.work[parent]['K']; k=self.work[name]['K']
            table=np.zeros((self.g*pk,k))
            for f in active:
                states=root_oof[f][:,ps].argmax(1)
                table[np.arange(self.g)*pk+states]+=self.counts[f,:,cs]
            states=core[:,ps].argmax(1)
            c=table[np.arange(self.g)*pk+states]
            coupled[:,cs]=(c+8*coupled[:,cs])/(c.sum(1,keepdims=True)+8)
        return np.concatenate((layers,coupled[None,:,:]),axis=0)

    def predict_fold(self,held,deadline=float('inf')):
        inner=(held+1)%5
        validation=self.candidates({held,inner},deadline)
        final=self.candidates({held},deadline)
        weights={}; out={}
        for name in self.names:
            j=self.work[name]; s=self.slices[name]
            mask=j['train']&(self.fold==inner); yy=j['y'][mask]; ids=self.ids[mask]
            w=np.full(len(final),1/len(final))
            if len(yy):
                p=validation[:,ids,s][:,np.arange(len(yy)),yy]
                for _ in range(16):
                    w*=np.mean(p/np.maximum(w@p,1e-12),axis=1)
                    w/=w.sum()
            out[name]=np.einsum('a,agk->gk',w,final[:,:,s])[self.ids]
            weights[name]=w.tolist()
        return out,weights


def build(X,given,work,fold,items,deadline=float('inf')):
    grid=Grid(X,given,work,fold,items)
    out={n:np.zeros((len(X),j['K'])) for n,j in work.items()}; weights={}
    try:
        for f in range(5):
            if time.monotonic()>=deadline: raise TimeoutError('candidate budget')
            pred,weights[f]=grid.predict_fold(f,deadline)
            for n,p in pred.items():
                visible=work[n]['train']; mask=visible&(fold==f)
                out[n][mask]=p[mask]; out[n][~visible]+=p[~visible]/5
    except TimeoutError:
        return {},dict(complete=False,cells=grid.g)
    return out,dict(complete=True,cells=grid.g,weights=weights,folds=5,inner_folds=1,discount=0.)


def attach(X,given,items,work,fold,deadline):
    started=time.monotonic(); found,trace=build(X,given,work,fold,items,deadline)
    checks={}; pending=[]
    for name,p in found.items():
        j=work[name]; rows=j['train']; b=j['arms'].index('table') if 'table' in j['arms'] else 0
        o=.5*p[rows]+.5*j['oof'][b]
        info=audit(o,j['oof'][b],j['y'][rows],fold[rows]); checks[name]=info
        if info['added']: pending.append((j,o,.5*p+.5*j['all'][b]))
    if time.monotonic()<deadline and trace['complete']:
        for j,o,p in pending:
            j['oof'].append(o); j['all'].append(p); j['arms'].append('pyp_eb')
            j.setdefault('partition_k',{})['pyp_eb']=5
    else:
        for info in checks.values(): info['added']=False
    trace.update(items=checks,seconds=time.monotonic()-started)
    return trace
