"""Shared-gate Cartesian targets and exact probability marginalization."""
import json
import numpy as np
import gpu

def pairs(schema,work,subset=None):
    parents={v['gate']['parent'] for v in schema['items'].values() if isinstance(v.get('gate'),dict) and 'parent' in v['gate']}
    groups={};allowed=set(work if subset is None else subset)
    for name,item in schema['items'].items():
        if name not in allowed or name not in work or name in parents:continue
        if item.get('class')!='PREDICT' or not item.get('gate') or work[name]['K']>3:continue
        key=json.dumps(item['gate'],sort_keys=True)
        groups.setdefault(key,[]).append(name)
    result=[]
    for group in groups.values():
        for i in range(0,len(group)-1,2):
            a,b=group[i:i+2]
            if np.array_equal(work[a]['train'],work[b]['train']):result.append((a,b))
    return result

def marginalize(p,ka,kb):
    grid=np.asarray(p).reshape((-1,ka,kb))
    return grid.sum(axis=2),grid.sum(axis=1)

def candidate(a,b,schema,X,given,work,fold,bank,deadline,timing,arm):
    ja,jb=work[a],work[b];ka,kb=ja['K'],jb['K']
    visible=ja['train'] & jb['train'];rows=np.flatnonzero(visible);rest=np.flatnonzero(~visible)
    if len(rows)<20 or not np.array_equal(ja['train'],jb['train']):return None
    y=np.full(len(X),-1,dtype=np.int64);y[rows]=ja['y'][rows]*kb+jb['y'][rows]
    # Every parent feature is rebuilt inside this exact five-fold fit set.
    # Both children share a gate, so the first child's bank contains the parent.
    def features(train,part):
        return bank.matrix(a,train)
    strata=tuple(i for i,k in enumerate(given) if k=='wave' or 'country' in k.lower())
    result=gpu.candidate(X,y,rows,rest,ka*kb,X.shape[1],fold,deadline,timing,
                         strata=strata,feature_factory=features,arm=arm)
    if result is None:return None
    o,p=result;oa,ob=marginalize(o,ka,kb);pa,pb=marginalize(p,ka,kb)
    return [(a,oa,pa),(b,ob,pb)]

