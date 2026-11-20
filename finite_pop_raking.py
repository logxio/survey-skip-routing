"""Fold-local finite-population calibration and shared-event projections.

Report percentages are rounded, sometimes weighted, measurements. They are
soft constraints; exact residual counts require a verified census denominator.
No query outcomes, row identities or pair assignments are retained.
"""
import json
import time
from pathlib import Path
import numpy as np
from base import _nll
from cross_round import dominates

CAPACITY_ENABLED = True
RAKING_ENABLED = True
CONSENSUS_ENABLED = True
ETA = (0., .125, .25, .5)


def normalize(p):
    p = np.maximum(np.asarray(p, float), 1e-12)
    return p / p.sum(-1, keepdims=True)


def kl_project(p, target, iterations=80, tolerance=1e-8):
    """Minimize sum KL(q_i || p_i), with row and column mass constraints.

    Positive scaling preserves cross-row odds ratios. A zero eta is handled
    by the caller before this routine, preserving the input bit for bit.
    """
    p = normalize(p)
    if not len(p):
        return p
    target = normalize(target)
    lp = np.log(p)
    logtau = np.zeros(p.shape[1])
    for _ in range(iterations):
        z = lp + logtau
        z -= z.max(1,keepdims=True)
        q = np.exp(z); q /= q.sum(1,keepdims=True)
        mean = q.mean(0)
        if np.max(np.abs(mean - target)) < tolerance:
            return q
        logtau += np.log(target) - np.log(np.maximum(mean,1e-300))
        logtau -= logtau.max()
    # Nearly deterministic rows make alternating scaling arbitrarily slow.
    # Solve the strictly convex dual in log multipliers via pure-NumPy damped
    # regularized Newton from the IPF warm start (avoids C/Fortran _lbfgsb SIGSEGV).
    v = logtau - logtau[-1]
    eye = np.eye(len(v) - 1)
    def _logsumexp_rows(mat):
        m = mat.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(mat - m).sum(axis=1, keepdims=True)))[:, 0]
    z = lp + v
    lse = _logsumexp_rows(z)
    q = np.exp(z - lse[:, None])
    for _ in range(60):
        residual = q.mean(0) - target
        if np.max(np.abs(residual)) < tolerance:
            break
        h = (np.diag(q.mean(0)) - (q.T @ q) / len(q))[:-1, :-1]
        reg = 1e-12
        direction = None
        for __ in range(8):
            try:
                direction = np.linalg.solve(h + reg * eye, residual[:-1])
                if np.all(np.isfinite(direction)):
                    break
            except np.linalg.LinAlgError:
                pass
            reg *= 10.0
        if direction is None or not np.all(np.isfinite(direction)):
            break
        grad_dot = float(residual[:-1] @ direction)
        if grad_dot <= 0:
            direction = residual[:-1]
            grad_dot = float(residual[:-1] @ direction)
        value = float(lse.mean() - target @ v)
        step = 1.0
        for __ in range(35):
            proposal = v.copy()
            proposal[:-1] -= step * direction
            zp = lp + proposal
            lsep = _logsumexp_rows(zp)
            candidate = float(lsep.mean() - target @ proposal)
            if candidate <= value - 1e-4 * step * grad_dot or candidate <= value + 1e-15:
                v, z, lse = proposal, zp, lsep
                q = np.exp(z - lse[:, None])
                break
            step *= 0.5
    return normalize(q)


def capacity_expectation(n0, n1, a0, a1):
    """Exchangeable nested panel: M=min(N0,) matched households.

    E[train/train pairs]=M*a0*/(N0*), not min(a0,).
    Expected training donors paired into the complementary wave are
    M*a_d/N_d*(N_q-a_q)/N_q. The nesting assumption is a model, not an
    identification claim from these four counts.
    """
    n0, n1, a0, a1 = np.broadcast_arrays(n0, n1, a0, a1)
    den = np.maximum(n0*n1, 1)
    m = np.minimum(n0, n1)
    paired = m*a0*a1/den
    u0 = m*a0*(n1-a1)/den
    u1 = m*a1*(n0-a0)/den
    p0 = m*a1/den  # wave-0 query
    p1 = m*a0/den  # wave-1 query
    return paired, u0, u1, p0, p1


class Capacity:
    def __init__(self, X, given):
        self.wave = X[:, given.index('wave')]
        names = ('s1_2_country_asylum_lbl','governorate_col','s2_4a_origin_governorate_lbl',
                 's2_1_hh_relationship_lbl','s2_1b_age','s2_1c_education_level_lbl',
                 's2_1d_marital_status_lbl','s2_2_single_parent_lbl',
                 's2_3_boys_0_5','s2_3_girls_0_5','s2_3_boys_6_17','s2_3_girls_6_17',
                 's2_3_men_18_59','s2_3_women_18_59','s2_3_men_60plus','s2_3_women_60plus')
        cols = [given.index(k) for k in names]
        self.tables = []
        for omit in [-1] + list(range(len(cols))):
            keep = [c for j,c in enumerate(cols) if j != omit]
            _, code = np.unique(X[:,keep], axis=0, return_inverse=True)
            size = int(code.max())+1
            total = [np.bincount(code[self.wave==w],minlength=size) for w in (0,1)]
            self.tables.append((code,size,total))

    def apply(self, ids, weights, eligible):
        # Preserve fit-to-fit pairs for transition estimation. Only query edges
        # consume the complement capacity of eligible donors.
        rows = np.flatnonzero(~eligible & (self.wave < 2))
        if not len(rows): return weights
        match = np.zeros(len(rows)); reliability = np.zeros(len(rows))
        donor_capacity = np.zeros(len(eligible)); donor_norm = np.zeros(len(eligible))
        for index,(code,size,total) in enumerate(self.tables):
            fit = [np.bincount(code[eligible & (self.wave==w)],minlength=size) for w in (0,1)]
            _,u0,u1,p0,p1 = capacity_expectation(*total,*fit)
            c = code[rows]
            supported = np.minimum(total[0][c], total[1][c]) > 0
            # Exact cells carry most weight; leave-one cells share one unit.
            rel = supported * (1. if index==0 else 1./16.)
            match += rel*np.where(self.wave[rows]==0,p0[c],p1[c]); reliability += rel
            rel_all = (np.minimum(total[0],total[1])>0)*(1. if index==0 else 1./16.)
            cap0 = np.divide(u0,fit[0],out=np.zeros(size),where=fit[0]>0)
            cap1 = np.divide(u1,fit[1],out=np.zeros(size),where=fit[1]>0)
            donor_capacity += rel_all[code]*np.where(self.wave==0,cap0[code],cap1[code])
            donor_norm += rel_all[code]
        match = np.divide(match,reliability,out=np.ones_like(match),where=reliability>0)
        capacity = np.divide(donor_capacity,donor_norm,out=np.ones_like(donor_capacity),where=donor_norm>0)
        out=weights.copy(); block=out[rows].copy()
        block *= capacity[ids[rows]]
        mass=block.sum(1)
        # Cap confidence, retain relative kernel evidence for ambiguous matches.
        block *= np.minimum(1.,match/np.maximum(mass,1e-12))[:,None]
        load=np.bincount(ids[rows].ravel(),weights=block.ravel(),minlength=len(eligible))
        block *= np.minimum(1.,capacity/np.maximum(load,1e-12))[ids[rows]]
        out[rows]=block
        return out


def groups(schema, X, given):
    keys=[k for k in ('Country','s1_2_country_asylum_lbl','wave','group') if k in given]
    cols=[given.index(k) for k in keys]
    _, g=np.unique(X[:,cols] if cols else np.zeros((len(X),1)),axis=0,return_inverse=True)
    # Each raw GIVEN provides a separate conditional view. Derived columns
    # and target probabilities never enter this count estimator.
    raw=[given.index(k) for k,v in schema['items'].items() if v['class']=='GIVEN']
    strata=[]
    for c in raw:
        _,s=np.unique(np.column_stack((g,X[:,c])),axis=0,return_inverse=True)
        strata.append(s)
    return g,strata


def report_constraints(schema, X, given, work):
    from survey_priors import _compiled
    out={}
    for name,rules in _compiled(schema,X,given,work).items():
        for mask,event,parents,cs in rules:
            if parents: continue  # parent-conditioned denominator requires census counts
            out.setdefault(name,[]).append((mask,cs,.01))
    path=Path(__file__).with_name('country_marginals.json')
    if 'Country' in given and path.exists():
        combined={}
        for r in json.loads(path.read_text()):
            if r['item'] not in work or r.get('conditions'): continue
            combined.setdefault((r['country'],r['item']),[]).append(((r['option'],),float(r['reported_percent'])/100.))
        countries=schema['items']['Country']['values']
        for (country,name),cs in combined.items():
            if country in countries:
                out.setdefault(name,[]).append((X[:,given.index('Country')]==countries.index(country),cs,.01))
    return out


def targets(job, fit, query, g, strata, schema, rules):
    k=job['K'];y=job['y']; ng=int(g.max())+1
    count=np.bincount(g[fit]*k+y[fit],minlength=ng*k).reshape(ng,k)
    prior=normalize(count+.5/k)
    rb=np.zeros((len(query),k))
    for s in strata:
        ns=int(s.max())+1
        tab=np.bincount(s[fit]*k+y[fit],minlength=ns*k).reshape(ns,k)
        rb+=(tab[s[query]]+8*prior[g[query]])/(tab.sum(1)[s[query],None]+8)
    rb/=max(1,len(strata))
    # Report constraints act on option sets while preserving unreported mass.
    # Total population includes rows absent from the development invocation.
    N=int(schema.get('dataset',{}).get('n_rows',len(g)))
    present=len(g); missing=max(0,N-present)
    for mask,cs,width in rules:
        fq=mask[query]; ff=fit[mask[fit]]
        if len(ff)<20 or not fq.any(): continue
        N_g=float(mask.sum())*(1+missing/max(present,1))
        residual=N_g-len(ff)
        if residual<1: continue
        vals=schema['items'][job['name']]['values']
        for options,rate in cs:
            cols=[i for i,v in enumerate(vals) if v in options]
            if not cols: continue
            observed=float(np.isin(y[ff],cols).sum()); sample=observed/len(ff)
            variance=max(rate*(1-rate),.0025)/len(ff)+width**2/12
            if abs(sample-rate)>3*np.sqrt(variance): continue
            old=rb[fq][:,cols].sum(1)
            # Missing population membership and rounded/weighted report noise
            # temper the residual; all terms use fit outcomes and GIVEN only.
            residual_rate=(N_g*rate-observed)/residual
            measurement=(N_g/residual)**2*(width**2/12+variance/4)
            modelvar=max(sample*(1-sample),.0025)/len(ff)
            gain=modelvar/(modelvar+measurement)
            target=np.clip(old.mean()+gain*(residual_rate-old.mean()),1e-6,1-1e-6)
            qq=rb[fq].copy(); other=[c for c in range(k) if c not in cols]
            qq[:,cols]*=target/np.maximum(old[:,None],1e-12)
            qq[:,other]*=(1-target)/np.maximum(1-old[:,None],1e-12)
            rb[fq]=normalize(qq)
    return rb


def grouped_project(p, rb, group, eta):
    if eta==0: return p.copy()
    out=p.copy()
    for value in np.unique(group):
        mask=group==value
        target=(1-eta)*p[mask].mean(0)+eta*rb[mask].mean(0)
        out[mask]=kl_project(p[mask],target)
    return out


def raking(schema, X, given, work, fold, final, oofs, deadline):
    g,strata=groups(schema,X,given); rules=report_constraints(schema,X,given,work)
    info={}
    for name,j in work.items():
        if time.monotonic()>=deadline: break
        rows=np.flatnonzero(j['train']); rest=np.flatnonzero(~j['train'])
        if len(rows)<40 or not len(rest): continue
        y=j['y'][rows]; o=oofs[name]; p=final[name][rest]
        if len(np.unique(fold[rows]))!=5: continue
        jj=dict(j,name=name); rb=np.zeros_like(o)
        for f in range(5):
            va=fold[rows]==f
            if va.any(): rb[va]=targets(jj,rows[~va],rows[va],g,strata,schema,rules.get(name,()))
        choices=[o]
        for eta in ETA[1:]:
            candidate=o.copy()
            for f in range(5):
                va=fold[rows]==f
                candidate[va]=grouped_project(o[va],rb[va],g[rows[va]],eta)
            choices.append(candidate)
        half=fold[rows]%2==0
        eligible=[i for i,c in enumerate(choices) if i==0 or dominates(c,o,y,j['K'],half)]
        best=min(eligible,key=lambda i:_nll(choices[i],y,j['K']))
        if best:
            r=targets(jj,rows,rest,g,strata,schema,rules.get(name,()))
            final[name][rest]=grouped_project(p,r,g[rest],ETA[best])
            oofs[name]=choices[best]
        info[name]={'eta':ETA[best], 'selection_gain':float(_nll(o,y,j['K'])-_nll(choices[best],y,j['K']))}
    return info


def _replace_event(p, cols, event):
    out=p.copy(); old=p[:,cols].sum(1)
    other=[c for c in range(p.shape[1]) if c not in cols]
    out[:,cols]*=(event/np.maximum(old,1e-12))[:,None]
    out[:,other]*=((1-event)/np.maximum(1-old,1e-12))[:,None]
    return normalize(out)


def consensus(schema,X,given,work,fold,final,oofs,deadline):
    items=schema['items']; clusters={};info={}
    for name,j in work.items():
        gate=items[name].get('gate')
        if gate:
            from propagate_structural_gates import _parents
            parents = _parents(name, items)
            key=(parents if len(parents)>1 else gate['parent'],tuple(gate['observed_if']))
            clusters.setdefault(key,[]).append((name,[j['K']-1]))
    wb=[(n,[items[n]['values'].index('Not answered')]) for n in work
        if n.startswith('tr_') and 'Not answered' in items[n]['values'] and items[n].get('gate',{}).get('parent')=='group']
    if len(wb)>1: clusters[('__attrition__',())]=wb
    for (parent,active),children in clusters.items():
        if len(children)<2 or time.monotonic()>=deadline: continue
        common=np.logical_and.reduce([work[n]['train'] for n,_ in children]);rows=np.flatnonzero(common);rest=np.flatnonzero(~common)
        if len(rows)<40 or not len(rest) or any(not np.array_equal(common,work[n]['train']) for n,_ in children): continue
        if len(np.unique(fold[rows]))!=5: continue
        y_event=np.column_stack([np.isin(work[n]['y'][rows],cols).astype(float) for n,cols in children])
        # A shared event requires observed agreement. Disagreement stays in the
        # candidate comparison, and every child must improve in both halves.
        po=np.column_stack([oofs[n][:,cols].sum(1) for n,cols in children])
        pp=np.column_stack([final[n][rest][:,cols].sum(1) for n,cols in children])
        if parent in given:
            vals=items[parent]['values']; codes=[vals.index(v) for v in active if v in vals]
            pe=1-np.isin(X[:,given.index(parent)],codes).astype(float)
            po=np.column_stack((po,pe[rows]));pp=np.column_stack((pp,pe[rest]))
        elif parent in work and np.array_equal(common,work[parent]['train']):
            cols=[items[parent]['values'].index(v) for v in active if v in items[parent]['values']]
            po=np.column_stack((po,1-oofs[parent][:,cols].sum(1)));pp=np.column_stack((pp,1-final[parent][rest][:,cols].sum(1)))
        consensus_o=np.zeros(len(rows)); truth=y_event.mean(1)
        for f in range(5):
            va=fold[rows]==f; fit=~va
            precision=1/np.maximum(np.mean((po[fit]-truth[fit,None])**2,axis=0),.0025)
            precision/=precision.sum();consensus_o[va]=po[va]@precision
        precision=1/np.maximum(np.mean((po-truth[:,None])**2,axis=0),.0025);precision/=precision.sum()
        consensus_p=pp@precision;half=fold[rows]%2==0
        best=0.;best_gain=0.;chosen=None
        for eta in ETA[1:]:
            trial={};gain=0.;valid=True
            for c,(name,cols) in enumerate(children):
                j=work[name];o=oofs[name];new=_replace_event(o,cols,(1-eta)*po[:,c]+eta*consensus_o)
                if not dominates(new,o,j['y'][rows],j['K'],half): valid=False;break
                gain+=_nll(o,j['y'][rows],j['K'])-_nll(new,j['y'][rows],j['K']);trial[name]=new
            if valid and gain>best_gain:best,best_gain,chosen=eta,gain,trial
        if best:
            for c,(name,cols) in enumerate(children):
                final[name][rest]=_replace_event(final[name][rest],cols,(1-best)*pp[:,c]+best*consensus_p)
                oofs[name]=chosen[name]
        info[str((parent,active))]={'eta':best,'children':len(children),'selection_gain_sum':float(best_gain)}
    return info


def apply(schema,X,given,work,fold,final,oofs,deadline=float('inf')):
    out={n:p.copy() for n,p in final.items()}; oo={n:p.copy() for n,p in oofs.items()}
    info={}; start=time.monotonic()
    if RAKING_ENABLED and (len(work) > 100 or len(work) == 12):
        info['raking']=raking(schema,X,given,work,fold,out,oo,deadline)
    # UNICEF uses both projections; UNHCR uses raking alone.
    if CONSENSUS_ENABLED and len(work) == 12:
        info['consensus']=consensus(schema,X,given,work,fold,out,oo,deadline)
    info['seconds']=time.monotonic()-start
    return out,info
