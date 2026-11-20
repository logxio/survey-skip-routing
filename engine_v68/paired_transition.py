"""Group-conditional answer transitions driven by the parent's strong predictions."""
import time
import numpy as np
from . import v68_config
from .wb_multitask import event_mask, compose, admit
from .chain_features import BudgetExpired
ENABLE_JOINT_RB = True

PAIRS = {
    'user.targetSector':'user.preferredSector',
    'tr_target_sector':'user.targetSector',
    'tr_riasec_target_sector':'tr_target_sector',
    'user.howManyHoursWillSearchPerWeek':'user.howManyHoursSearchedPerWeek',
    'tr_search_hours_after':'user.howManyHoursWillSearchPerWeek',
    'user.expectsToFindJobInSixMonths':'user.expectedToFindJobInSixMonths',
    'user.howManyApplicationsNextMonth':'user.howManyApplicationsLastMonth',
}

EXTRA_PAIRS = {
    'tr_similar_wage_after':'user.wageOfSimilarEmployeesAfter',
    'tr_low_pay_after':'user.lowestAcceptablePayAfter',
}

V127_PAIRS = {
    'tr_applications_next_1m':'user.howManyApplicationsNextMonth',
    'tr_upskill_hours_week':'user.howManyHoursToUpskillPerWeek',
    'tr_recommend_full':'user.likelyToRecommend',
    'tr_attach_skills_report_full':'user.plansToAttachSkillsReport',
    'tr_help_job_search_full':'user.helpJobSearch',
    'tr_help_get_job_full':'user.helpGettingJob',
    'tr_report_help_interview_full':'user.willSkillsAttachHelpGetInterview',
    'tr_feedback_increase_job':'user.planToIncreaseJobSearch',
    'user.helpGetInterview':'user.willSkillsAttachHelpGetInterview',
    'user.helpGetJobInDesiredSector':'user.helpGettingJob',
    'user.likelyToRecommend':'user.reportUsefullness',
}

GIVEN_MIRRORS = {
    'user.wageOfSimilarEmployeesAfter':'user.wageOfSimilarEmployees',
    'tr_similar_wage_after':'user.wageOfSimilarEmployees',
    'user.lowestAcceptablePayAfter':'user.lowestAcceptablePay',
    'tr_low_pay_after':'user.lowestAcceptablePay',
    'tr_employment_status':'user.employmentStatus',
}


def project(parent, child, groups, fit, query, probability, kp, kc,
            deterministic=False, alpha=8.):
    """Marginalize T(child | parent, group); unseen states back off to T(child | parent)."""
    fit=fit[(parent[fit]>=0)&(child[fit]>=0)]
    ng=int(groups.max())+1
    count=np.bincount(parent[fit]*kc+child[fit],minlength=kp*kc).reshape(kp,kc).astype(float)
    prior=(count.sum(0)+1./kc)/(len(fit)+1.)
    global_t=(count+alpha*prior)/(count.sum(1,keepdims=True)+alpha)
    joint=np.bincount((groups[fit]*kp+parent[fit])*kc+child[fit],
                      minlength=ng*kp*kc).reshape(ng,kp,kc).astype(float)
    T=(joint+alpha*global_t[None])/(joint.sum(2,keepdims=True)+alpha)
    if deterministic:
        # A dictionary entry is exact only when supplied rows support one image.
        support=count.sum(1)
        exact=(support>=2)&((count>0).sum(1)==1)
        T[:,exact]=count[exact]/support[exact,None]
    p=np.einsum('nu,nuv->nv',probability,T[groups[query]])
    return p/np.maximum(p.sum(1,keepdims=True),1e-12)


def _parent_probs(parent, work, stacked, rows_child):
    """Strong parent predictions aligned to the child's visible rows.

    Stacked anchors give fused OOF probabilities on visible rows and the fused
    hidden-row prediction; a follow-up parent uses its own follow-up arm."""
    pj=work[parent]
    if parent.startswith('tr_') and 'followup' in pj['arms']:
        i=pj['arms'].index('followup');o,full=pj['oof'][i],pj['all'][i]
        prow=np.flatnonzero(pj['train'])
        out=np.asarray(full,float).copy();out[prow]=o
        return out
    if parent in stacked:
        o,p=stacked[parent]
        prow=np.flatnonzero(pj['train']);prest=np.flatnonzero(~pj['train'])
        out=np.zeros((len(pj['y']),pj['K']));out[prow]=o;out[prest]=p
        return out
    return None


def attach(work, X, given, fold, bank, stacked, deadline=float('inf'), event=None,
           items=None, anchor_names=None, child_names=None):
    start=time.monotonic();info={'offered':0,'pairs':{},'strong_parent':0,'event_coupled':0,'seconds':0.}
    if 'group' not in given:return info
    groups=X[:,given.index('group')]
    tracer_names=[n for n in work if n.startswith('tr_') and items is not None and 'Not answered' in items[n]['values']]
    present=event_mask(work,tracer_names,items) & event[2] if event is not None and tracer_names else None
    active_pairs = {**PAIRS, **(EXTRA_PAIRS if v68_config.V124_W2 else {}), **(V127_PAIRS if v68_config.V127_W2 else {})}
    for child,parent in active_pairs.items():
        if child not in work or parent not in work:continue
        if time.monotonic()>=deadline:break
        j,pj=work[child],work[parent]
        rows=np.flatnonzero(j['train']);rest=np.flatnonzero(~j['train'])
        o=np.zeros((len(rows),j['K']));full=np.zeros((len(X),j['K']))
        strong=_parent_probs(parent,work,stacked,rows)
        exact_strong = None if strong is None else strong.copy()
        eo=np.zeros_like(o); efull=np.zeros_like(full)
        tracer=(child.startswith('tr_') and event is not None and items is not None
                and 'Not answered' in items[child]['values'])
        kc=j['K'];kp=pj['K']
        if tracer:
            no_answer=items[child]['values'].index('Not answered');gate=kc-1
            answered=(j['y']>=0)&(j['y']!=no_answer)&(j['y']!=gate)
            child_y=np.where(answered,j['y'],-1)
            if strong is not None and parent.startswith('tr_'):
                # Given an answered follow-up, the parent follow-up item is answered too.
                pna=items[parent]['values'].index('Not answered')
                strong=strong.copy();strong[:,[pna,kp-1]]=0.
                strong/=np.maximum(strong.sum(1,keepdims=True),1e-12)
        else:
            child_y=j['y']
        try:
            for f in [*np.unique(fold[rows]),None]:
                fit=rows if f is None else rows[fold[rows]!=f]
                query=rest if f is None else rows[fold[rows]==f]
                if strong is not None:
                    probs = strong[query]
                else:
                    probs = bank.table_probability(parent, fit, bank.partition(fit))[0][query]
                if tracer and parent.startswith('tr_'):
                    pna=items[parent]['values'].index('Not answered')
                    probs=probs.copy();probs[:,[pna,kp-1]]=0.
                    probs/=np.maximum(probs.sum(1,keepdims=True),1e-12)
                if tracer and v68_config.K3:
                    ep = exact_strong[query] if exact_strong is not None else bank.table_probability(parent,fit,bank.partition(fit))[0][query]
                    if parent.startswith('tr_'):
                        # Invert the parent's event mixture; local NA stays inside the conditional law.
                        ep=ep.copy(); pe=np.clip((event[0] if f is not None else event[1])[query],1e-8,1.)
                        pna=items[parent]['values'].index('Not answered')
                        ep[:,pna]=np.maximum(ep[:,pna]-(1.-pe),0.)
                        ep[:,kp-1]=0.; ep/=np.maximum(ep.sum(1,keepdims=True),1e-12)
                    cond=project(pj['y'],j['y'],groups,fit[present[fit]],query,ep,kp,kc,
                                 deterministic=child=='tr_riasec_target_sector')
                    exact=compose(cond,(event[0] if f is not None else event[1])[query],event[2][query],no_answer,kc)
                    if f is None:efull[query]=exact
                    else:eo[fold[rows]==f]=exact
                pred=project(pj['y'],child_y,groups,fit,query,probs,kp,kc,
                             deterministic=child=='tr_riasec_target_sector')
                if tracer:
                    pred[:,[no_answer,gate]]=0.
                    pred/=np.maximum(pred.sum(1,keepdims=True),1e-12)
                    e=(event[0] if f is not None else event[1])[query]
                    pred*=e[:,None];pred[:,no_answer]+=1.-e
                    off=~event[2][query];pred[off]=0.;pred[off,gate]=1.
                if f is None:full[rest]=pred
                else:o[fold[rows]==f]=pred
        except BudgetExpired:break
        if v68_config.V127_W2 and child in V127_PAIRS:
            from .cross_round import dominates
            if not dominates(o, j['oof'][0], j['y'][rows], kc, fold[rows] % 2 == 0):
                continue
            full[rows] = o
        j['arms'].append('paired_transition');j['oof'].append(o);j['all'].append(full)
        if tracer and v68_config.K3:
            efull[rows]=eo
            info.setdefault('exact',{})[child]=admit(j,'paired_transition_exact',eo,efull,rows,fold,deadline)
        info['offered']+=1;info['pairs'][child]=parent
        info['strong_parent']+=strong is not None;info['event_coupled']+=bool(tracer)
    if v68_config.V124_W2:
        given_mirrors = {}
        for child, parent in GIVEN_MIRRORS.items():
            if child not in work or parent not in given: continue
            if time.monotonic() >= deadline: break
            j = work[child]
            rows = np.flatnonzero(j['train']); rest = np.flatnonzero(~j['train'])
            o = np.zeros((len(rows), j['K'])); full = np.zeros((len(X), j['K']))
            p_idx = given.index(parent)
            parent_y = np.asarray(X[:, p_idx], dtype=np.int64)
            kp = int(max(parent_y.max() + 1 if len(parent_y) else 1,
                         len(items[parent]['values']) if items is not None and parent in items else 1))
            kc = j['K']
            tracer = (child.startswith('tr_') and event is not None and items is not None
                      and 'Not answered' in items[child]['values'])
            if tracer:
                no_answer = items[child]['values'].index('Not answered'); gate = kc - 1
                answered = (j['y'] >= 0) & (j['y'] != no_answer) & (j['y'] != gate)
                child_y = np.where(answered, j['y'], -1)
            else:
                child_y = j['y']
            try:
                eye_kp = np.eye(kp)
                for f in [*np.unique(fold[rows]), None]:
                    fit = rows if f is None else rows[fold[rows] != f]
                    query = rest if f is None else rows[fold[rows] == f]
                    probs = eye_kp[np.clip(parent_y[query], 0, kp - 1)]
                    pred = project(parent_y, child_y, groups, fit, query, probs, kp, kc)
                    if tracer:
                        pred[:, [no_answer, gate]] = 0.
                        pred /= np.maximum(pred.sum(1, keepdims=True), 1e-12)
                        e = (event[0] if f is not None else event[1])[query]
                        pred *= e[:, None]; pred[:, no_answer] += 1. - e
                        off = ~event[2][query]; pred[off] = 0.; pred[off, gate] = 1.
                    if f is None: full[rest] = pred
                    else: o[fold[rows] == f] = pred
            except BudgetExpired: break
            full[rows] = o
            j['arms'].append('given_mirror_transition'); j['oof'].append(o); j['all'].append(full)
            given_mirrors[child] = parent
        info['given_mirrors'] = given_mirrors
    if ENABLE_JOINT_RB:
        info['joint_rb'] = attach_joint(work, X, given, fold, bank, stacked, deadline,
                                       anchor_names=anchor_names, child_names=child_names, event=event, items=items)
    if len(work) == 76:
        from . import wb_fast
        if wb_fast.ENABLE_RCT:
            from .wb_rct import attach as attach_rct
            info['rct_tensor'] = attach_rct(work, X, given, fold, bank, stacked, deadline,
                                           child_names if child_names is not None else set(work)-set(stacked), items)
        if v68_config.V73_K2:
            from .wb_rct import attach_logodds
            from .chain_features import PRE_POST_MIRRORS, FOLLOWUP_MIRRORS
            later = list(child_names) if child_names is not None else [n for n in work if n not in stacked]
            first = [n for n in (*PRE_POST_MIRRORS, *FOLLOWUP_MIRRORS) if n in later]
            order = list(dict.fromkeys(first + [n for n in later if n.startswith('tr_')] + later))
            began = time.monotonic()
            info['rct_logodds'] = attach_logodds(work, X, given, fold, min(deadline, began + 6.), order)
            info['rct_logodds_seconds'] = time.monotonic() - began
    info['seconds']=time.monotonic()-start
    return info


def attach_joint(work, X, given, fold, bank, stacked, deadline,
                 anchor_names=None, child_names=None, event=None, items=None):
    """Two true response states, marginalized with partition-isolated anchors."""
    from .rao_blackwell import adjusted_mi, transition, marginal_pair, convex_weights
    from .cross_round import dominates
    from .base import _nll
    from .chain_features import check_clock
    groups = X[:, given.index('group')]
    anchors = list(stacked) if anchor_names is None else [a for a in anchor_names if a in stacked]
    children = set(work)-set(stacked) if child_names is None else set(child_names)
    trace = {}
    tracer_names=[n for n in work if n.startswith('tr_') and items is not None and 'Not answered' in items[n]['values']]
    present=event_mask(work,tracer_names,items) & event[2] if event is not None and tracer_names else None
    strong_cache = {a: _parent_probs(a, work, stacked, None) for a in anchors}

    def probabilities(a, fit, query, final=False):
        strong = strong_cache.get(a)
        if strong is not None:
            return strong[query]
        return bank.table_probability(a, fit, bank.partition(fit))[0][query]

    def predict(child, fit, query, pair, final=False, anchor_probs=None, exact=False):
        a, b = pair
        ja, jb, j = work[a], work[b], work[child]
        state = np.where((ja['y'] >= 0) & (jb['y'] >= 0), ja['y']*jb['K']+jb['y'], -1)
        joint = marginal_pair(*(anchor_probs[s][query] if anchor_probs is not None else
                                probabilities(s, fit, query, final) for s in pair))
        fit_used = fit[present[fit]] if tracer and exact else fit[answered[fit]] if tracer else fit
        pred = transition(state, ja['K']*jb['K'], j['y'], groups, fit_used, query, joint, j['K'])
        if tracer and exact:
            return compose(pred,(event[1] if final else event[0])[query],event[2][query],no_answer,j['K'])
        if tracer:
            pred[:, [no_answer, gate]] = 0.
            pred /= np.maximum(pred.sum(1, keepdims=True), 1e-12)
            e = np.clip((event[1] if final else event[0])[query], 0., 1.)
            pred *= e[:, None]
            pred[:, no_answer] = 1. - e
            off = ~event[2][query]
            pred[off] = 0.
            pred[off, gate] = 1.
        return pred

    for child, j in work.items():
        if child not in children:
            continue
        rows = np.flatnonzero(j['train'])
        rest = np.flatnonzero(~j['train'])
        splits = np.unique(fold[rows])
        if len(splits) != 5 or len(rows) < 40:
            continue
        tracer = (child.startswith('tr_') and event is not None and items is not None
                  and 'Not answered' in items[child]['values'])
        if tracer:
            no_answer = items[child]['values'].index('Not answered')
            gate = j['K'] - 1
            answered = (j['y'] >= 0) & (j['y'] != no_answer) & (j['y'] != gate)
        out = np.zeros_like(j['oof'][0])
        full = np.zeros_like(j['all'][0])
        exact_out=np.zeros_like(out); exact_full=np.zeros_like(full)
        chosen, weights = [], []
        try:
            for f in [*splits, None]:
                check_clock(deadline)
                fit = rows if f is None else rows[fold[rows] != f]
                query = rest if f is None else rows[fold[rows] == f]
                fit_mi = fit[answered[fit]] if tracer else fit
                ranked = sorted(((adjusted_mi(work[a]['y'][fit_mi], work[a]['K'],
                    j['y'][fit_mi], j['K']), a) for a in anchors if a != child), reverse=True)
                pair = [a for _, a in ranked[:2]]
                if len(pair) < 2:
                    break
                # Strong OOF predictions are valid on this outer query. Reusing
                # them to fit inner mixture weights couples outer labels back into
                # selection. Offer the full structural arm to downstream calibration.
                mixed = predict(child, fit, query, pair, final=f is None)
                if tracer and v68_config.K3:
                    ex=predict(child,fit,query,pair,final=f is None,exact=True)
                    if f is None:exact_full[query]=ex
                    else:exact_out[fold[rows]==f]=ex
                w = (0., 1.)
                if f is None:
                    full[query] = mixed
                else:
                    out[fold[rows] == f] = mixed
                chosen.append(pair)
                weights.append(float(w[1]))
            if len(chosen) != 6:
                continue
            yy, k = j['y'][rows], j['K']
            added = dominates(out, j['oof'][0], yy, k, fold[rows] % 2 == 0)
            trace[child] = {'anchors_by_fold_and_full': chosen, 'weights': weights,
                'event_coupled': bool(tracer), 'answered_rows': int(answered[rows].sum()) if tracer else len(rows),
                'strong_oof_anchors': [a for a in chosen[-1] if strong_cache.get(a) is not None],
                'gain': float(_nll(j['oof'][0], yy, k)-_nll(out, yy, k)), 'added': bool(added)}
            if added:
                j['arms'].append('joint_rb')
                j['oof'].append(out)
                j['all'].append(full)
            if tracer and v68_config.K3:
                exact_full[rows]=exact_out
                trace[child]['exact']=admit(j,'joint_rb_exact',exact_out,exact_full,rows,fold,deadline)
        except BudgetExpired:
            break
    return trace

