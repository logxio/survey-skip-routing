"""Four-factor nominal IRT with training-residual GH81 integration."""
import itertools, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from chain_features import check_clock, BudgetExpired

class LogisticFM(torch.nn.Module):
    def __init__(self, subjects, categories, factors, device='cpu'):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(subjects, factors, device=device))
        self.V = torch.nn.Parameter(torch.randn(categories, factors, device=device))
        self.Z = torch.nn.Parameter(torch.randn(categories, device=device))

def options_for(schema,name):
    """Official option order with the final structural skip category."""
    item=schema['items'][name];values=list(item['values'])
    if item.get('gate') and schema.get('gated_value','NA_GATED') not in values:
        values.append(schema.get('gated_value','NA_GATED'))
    return values

def gate_order(schema,names):
    remaining=list(names);order=[]
    while remaining:
        ready=[name for name in remaining if (schema['items'][name].get('gate') or {}).get('parent') not in remaining]
        if not ready:raise ValueError('Cyclic questionnaire gates')
        order.extend(ready);remaining=[name for name in remaining if name not in ready]
    return order

def predict(train_frame,query_frame,schema,*,seed=20261114,factors=4,epochs=100,apply_schema_gates=False, deadline=float("inf")):
    torch.manual_seed(seed)
    if factors!=4:raise ValueError('This arm uses fixed four-dimensional quadrature')
    given=[k for k,v in schema['items'].items() if v['class']=='GIVEN']
    targets=[k for k,v in schema['items'].items() if v['class']=='PREDICT']
    if not given or not targets or not len(train_frame):raise ValueError('IRT requires train rows, GIVEN and PREDICT items')
    query_given=query_frame[given].copy()
    sizes=[len(options_for(schema,k)) for k in targets]
    offsets=np.r_[0,np.cumsum(sizes)]
    model=LogisticFM(len(train_frame),int(offsets[-1]),factors,device='cpu')
    labels=[]
    for j,name in enumerate(targets):
        levels=options_for(schema,name)
        y=pd.Categorical(train_frame[name],categories=levels).codes.astype(np.int64)

        labels.append(torch.tensor(y,dtype=torch.long))
    with torch.no_grad():
        model.U.mul_(0.1);model.V.mul_(0.1)
        for j,y in enumerate(labels):
            counts=torch.bincount(y[y>=0],minlength=sizes[j]).float()+0.5
            model.Z[offsets[j]:offsets[j+1]]=torch.log(counts/counts.sum())
    optimizer=torch.optim.Adam(model.parameters(),lr=0.045)
    last=0.
    for epoch in range(epochs):
        check_clock(deadline)
        optimizer.zero_grad()
        logits=model.U @ model.V.T + model.Z
        terms=[]
        for j,y in enumerate(labels):
            ok=y>=0
            if ok.any():
                terms.append(torch.nn.functional.cross_entropy(logits[ok,offsets[j]:offsets[j+1]],y[ok]))
        loss=torch.stack(terms).mean()+0.025*model.U.square().mean()+0.01*model.V.square().mean()
        loss.backward();optimizer.step();last=float(loss.detach())
    # Query rows supply demographics only. All preprocessing is fitted on train.
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    xt=enc.fit_transform(train_frame[given].fillna('NA').astype(str))
    xq=enc.transform(query_given.fillna('NA').astype(str))
    traits=model.U.detach().numpy()
    reg=Ridge(alpha=20.,solver='lsqr').fit(xt,traits)
    mu=np.asarray(reg.predict(xq),dtype=np.float64)
    # Integrate the nonlinear response distribution over uncertainty left by
    # GIVEN. Both the conditional mean and its residual covariance use this
    # training fold only. The held-out rows contribute demographics only.
    residual=traits.astype(np.float64)-np.asarray(reg.predict(xt),dtype=np.float64)
    covariance=np.cov(residual,rowvar=False,ddof=1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    # The empirical covariance is PSD. Clamp only numerical roundoff at zero.
    sqrtcov=(eigenvectors*np.sqrt(np.maximum(eigenvalues,0.)))@eigenvectors.T
    nodes,weights=np.polynomial.hermite.hermgauss(3)
    nodes=np.sqrt(2.)*nodes
    weights=weights/np.sqrt(np.pi)
    out={name:np.zeros((len(query_given),sizes[j]),dtype=np.float64) for j,name in enumerate(targets)}
    with torch.no_grad():
        # Stream 3^4 nodes instead of materializing query by node tensors.
        for coordinates in itertools.product(range(3),repeat=4):
            check_clock(deadline)
            node=nodes[np.asarray(coordinates)]
            weight=float(np.prod(weights[np.asarray(coordinates)]))
            uq=torch.tensor(mu+node@sqrtcov,dtype=torch.float32)
            logits=uq @ model.V.T + model.Z
            for j,name in enumerate(targets):
                out[name]+=weight*torch.softmax(logits[:,offsets[j]:offsets[j+1]],dim=1).numpy()
    # The clean IRT arm retains its native nominal distribution. Projection
    # through predicted parents is an explicit diagnostic option only.
    for name in gate_order(schema,targets):
        item=schema['items'][name];gate=item.get('gate')
        if not gate:continue
        parent=gate['parent'];allowed=gate['observed_if']
        if parent in given:
            values=query_given[parent]
            unknown=values.isna() | values.astype(str).isin(['No response','NA','nan','None'])
            # Source nonresponse codes do not establish questionnaire eligibility.
            unknown=unknown | values.astype(str).str.startswith('-')
            closed=(~unknown & ~values.isin(allowed)).to_numpy()
            if not apply_schema_gates:
                col=options_for(schema,name).index(schema.get('gated_value','NA_GATED'))
                out[name][closed]=0.;out[name][closed,col]=1.
                continue
            asked=values.isin(allowed).to_numpy(dtype=float)
            preserve=unknown.to_numpy()
        elif parent in out:
            if not apply_schema_gates:continue
            parent_options=options_for(schema,parent)
            asked=out[parent][:,[parent_options.index(v) for v in allowed]].sum(1)
            preserve=np.zeros(len(query_given),dtype=bool)
        else:raise ValueError('Unknown gate parent for '+name)
        asked=np.clip(asked,0.,1.)
        col=options_for(schema,name).index(schema.get('gated_value','NA_GATED'))
        p=out[name].astype(float);p[:,col]=0.
        y=labels[targets.index(name)].numpy()
        prior=np.bincount(y,minlength=p.shape[1]).astype(float)+.5;prior[col]=0.;prior/=prior.sum()
        denom=p.sum(1,keepdims=True)
        conditional=np.divide(p,denom,out=np.broadcast_to(prior,p.shape).copy(),where=denom>0)
        p=conditional*asked[:,None];p[:,col]=1.-asked
        p[preserve]=out[name][preserve];out[name]=p
    return out,dict(train_objective=last,epochs=epochs,n_factors=factors,category_items=int(offsets[-1]),query_answer_access=False,torch_measure_model='LogisticFM',response_link='expected per-item nominal softmax',integration={'method':'tensor Gauss-Hermite','order':3,'dimensions':4,'nodes':81,'normal_node_scale':'sqrt(2)','normal_weights':'weights/sqrt(pi)','accumulation':'streamed nodes','variance_source':'training-fold latent traits minus training-fold ridge predictions','covariance_estimator':'full empirical covariance, ddof=1','covariance':covariance.tolist(),'eigenvalues':eigenvalues.tolist(),'variance_scale':1.0,'heldout_label_selection':False},apply_schema_gates=apply_schema_gates,gates='native nominal output; known closed GIVEN hard lock; unknown GIVEN unchanged' if not apply_schema_gates else 'diagnostic recursive schema projection; unknown GIVEN unchanged')

def crossfit(frame, schema, work, allowed, fold, deadline):
    targets = list(work)
    local_schema = dict(schema, items={k: v for k, v in schema['items'].items()
        if v['class'] == 'GIVEN' or k in work})
    visible = frame.copy()
    # Only visible target labels enter any latent fit, including partial rows.
    for name in targets:
        visible.loc[~work[name]['train'], name] = np.nan
    allowed = np.asarray(allowed)
    out = {name: (np.zeros((int(j['train'].sum()), j['K']), np.float32),
                  np.zeros((len(frame), j['K']), np.float32)) for name, j in work.items()}
    try:
        for f in range(5):
            check_clock(deadline)
            tr = allowed[fold[allowed] != f]
            va = allowed[fold[allowed] == f]
            query = visible.iloc[va].copy()
            query[targets] = np.nan
            probs, _ = predict(visible.iloc[tr], query, local_schema, seed=20261114+f, deadline=deadline)
            for name, j in work.items():
                rows = np.flatnonzero(j['train'])
                take = fold[rows] == f
                out[name][0][take] = probs[name][np.searchsorted(va, rows[take])]
        query = visible.copy()
        query[targets] = np.nan
        probs, _ = predict(visible.iloc[allowed], query, local_schema, deadline=deadline)
        for name in targets:
            out[name][1][:] = probs[name]
        return out
    except BudgetExpired:
        return None
