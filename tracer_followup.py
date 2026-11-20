from threadpoolctl import threadpool_limits
"""Shared six-month follow-up response propensity for World Bank tracer items.

Visible rows reveal a common nonresponse event only if nearly all directly
gated tracer answers are 'Not answered'.  When this structure is absent, the
module returns no candidate.  The event model, conditional answer counts and
their fold-out predictions use visible rows only.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from chain_features import BudgetExpired, check_clock

ENABLE_WB_RCT = True
ENABLE_MULTI_ANCHOR_RB = True

# Pass-1 items most tied to staying in the six-month follow-up.
EVENT_ANCHORS = (
    'user.lowestAcceptablePayAfter',
    'user.willSkillsAttachHelpGetInterview',
    'user.planToIncreaseJobSearch',
    'user.helpJobSearch',
    'user.reportUsefullness',
    'user.targetSector',
    'user.howManyHoursWillSearchPerWeek',
    'user.likelyToRecommend',
    'user.plansToAttachSkillsReport',
)


def _binary_loss(p, y):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return float(-np.mean(np.where(y, np.log(p), np.log(1.0 - p))))


def _extract_pass2_consensus(tracer_names, schema, work, visible, train_randomized, rest, fold):
    """Extract fold-isolated cross-19-item median and inverse-loss weighted mean non-attrition probabilities."""
    n_total = len(work[tracer_names[0]]['y'])
    vis_to_pos = np.full(n_total, -1, dtype=np.int64)
    vis_to_pos[visible] = np.arange(len(visible))
    tr_r_vis_idx = vis_to_pos[train_randomized]

    # Collect per-item, per-arm answered probability (1 - p_NA / (1 - p_gate)) on train_randomized and on all rows
    item_oof_arms = []
    item_full_arms = []
    for name in tracer_names:
        j = work[name]
        na_idx = schema['items'][name]['values'].index('Not answered')
        gate_idx = j['K'] - 1
        o_list = []
        p_list = []
        for arm_name, o_arm, p_arm in zip(j['arms'], j['oof'], j['all']):
            if j.get('partition_k', {}).get(arm_name, 5) != 5:
                continue
            denom_o = np.maximum(1.0 - o_arm[tr_r_vis_idx, gate_idx], 1e-6)
            p_ans_o = np.clip(1.0 - o_arm[tr_r_vis_idx, na_idx] / denom_o, 1e-4, 1.0 - 1e-4)
            denom_p = np.maximum(1.0 - p_arm[:, gate_idx], 1e-6)
            p_ans_p = np.clip(1.0 - p_arm[:, na_idx] / denom_p, 1e-4, 1.0 - 1e-4)
            p_ans_p[visible] = np.clip(
                1.0 - o_arm[:, na_idx] / np.maximum(1.0 - o_arm[:, gate_idx], 1e-6), 1e-4, 1.0 - 1e-4
            )
            o_list.append(p_ans_o)
            p_list.append(p_ans_p)
        item_oof_arms.append(np.vstack(o_list))
        item_full_arms.append(np.vstack(p_list))
    return item_oof_arms, item_full_arms


def _consensus_for_partition(item_oof_arms, item_full_arms, event_r, fit_mask_r, query_rows, is_oof_query, query_mask_r=None):
    """Compute cross-19-item median and inverse-loss weighted mean on query given fit_mask_r on train_randomized."""
    y_fit = event_r[fit_mask_r]
    q_preds = []
    q_losses = []
    for o_mat, p_mat in zip(item_oof_arms, item_full_arms):
        # Score arms of this item on fit_mask_r
        fit_mat = np.clip(o_mat[:, fit_mask_r], 1e-4, 1.0 - 1e-4)
        losses = -np.mean(np.where(y_fit[None, :], np.log(fit_mat), np.log(1.0 - fit_mat)), axis=1)
        w = np.exp(-8.0 * (losses - losses.min()))
        w /= w.sum()
        if is_oof_query:
            pred_q = w @ o_mat[:, query_mask_r]
        else:
            pred_q = w @ p_mat[:, query_rows]
        pred_fit = w @ fit_mat
        item_loss = _binary_loss(pred_fit, y_fit)
        q_preds.append(pred_q)
        q_losses.append(item_loss)
    stack_q = np.vstack(q_preds)  # (19, len(query))
    med_q = np.median(stack_q, axis=0)
    losses_arr = np.asarray(q_losses)
    iw = np.exp(-10.0 * (losses_arr - losses_arr.min()))
    iw /= iw.sum()
    wmean_q = iw @ stack_q
    return med_q, wmean_q


def _event_arms(design, dense, eb_ids, rows, event, fit, query, deadline, consensus_pair=None):
    """LR, HGB, EB-table, report-prior and cross-19-item consensus probabilities of the shared answered event."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    check_clock(deadline)
    rate = (event[fit].sum() + 1.0) / (len(fit) + 2.0)
    prior_ans = 471.0 / 1229.0
    prior_eb = np.full(len(query), (event[fit].sum() + 64.0 * prior_ans) / (len(fit) + 64.0))
    if np.unique(event[fit]).size < 2:
        n_arms = 6 if consensus_pair is not None else 3
        return np.full((n_arms, len(query)), rate)
    logit = LogisticRegression(C=0.1, max_iter=45, tol=1e-3)
    logit.fit(design[fit], event[fit])
    check_clock(deadline)
    lr = logit.predict_proba(design[query])[:, 1]
    hgb = HistGradientBoostingClassifier(
        max_iter=60,
        max_leaf_nodes=7,
        learning_rate=0.06,
        l2_regularization=2.0,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=20261027,
    )
    hgb.fit(dense[fit], event[fit])
    check_clock(deadline)
    gb = hgb.predict_proba(dense[query])[:, 1]
    hit = np.bincount(eb_ids[fit], weights=event[fit].astype(float), minlength=int(eb_ids.max()) + 1)
    cnt = np.bincount(eb_ids[fit], minlength=int(eb_ids.max()) + 1)
    eb = (hit[eb_ids[query]] + 10.0 * rate) / (cnt[eb_ids[query]] + 10.0)
    if consensus_pair is None:
        return np.vstack((lr, gb, eb))
    med_q, wmean_q = consensus_pair
    return np.vstack((lr, gb, eb, prior_eb, med_q, wmean_q))


def _fuse_event(arms, event, fold_r=None):
    """Convex weights over the event arms on fold-excluded binary loss."""
    if arms.shape[0] == 3 and fold_r is None:
        best = (np.inf, np.array([1.0, 0.0, 0.0]))
        grid = np.arange(0, 11) / 10.0
        for a in grid:
            for b in grid:
                if a + b > 1.0 + 1e-9:
                    continue
                w = np.array([a, b, 1.0 - a - b])
                p = np.clip(w @ arms, 1e-4, 1 - 1e-4)
                loss = -np.mean(np.where(event, np.log(p), np.log(1 - p)))
                if loss < best[0]:
                    best = (loss, w)
        return best[1], float(best[0])

    # General simplex search over 6 arms (lr, gb, eb, prior_eb, tr_med, tr_wmean)
    from scipy.optimize import minimize

    n_arms = arms.shape[0]
    clipped = np.clip(arms, 1e-4, 1.0 - 1e-4)

    def fit_weights(sub_arms, sub_y):
        arm_losses = np.array([_binary_loss(sub_arms[i], sub_y) for i in range(n_arms)])
        w0 = np.exp(-12.0 * (arm_losses - arm_losses.min()))
        w0 /= w0.sum()

        def obj(w):
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s <= 0:
                return 1e9
            w = w / s
            p = np.clip(w @ sub_arms, 1e-4, 1.0 - 1e-4)
            return float(-np.mean(np.where(sub_y, np.log(p), np.log(1.0 - p)))) + 1e-4 * float(np.sum(w * arm_losses))

        res = minimize(
            obj,
            w0,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * n_arms,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            options={'maxiter': 35, 'ftol': 1e-6},
        )
        w = np.maximum(res.x, 0.0)
        w /= max(w.sum(), 1e-12)
        # Also check pure single best arm and 2-arm blends with prior_eb/consensus
        best_w = w
        best_l = _binary_loss(best_w @ sub_arms, sub_y)
        for i in range(n_arms):
            ei = np.zeros(n_arms)
            ei[i] = 1.0
            li = _binary_loss(sub_arms[i], sub_y)
            if li < best_l:
                best_l = li
                best_w = ei
        return best_w

    if fold_r is not None and len(np.unique(fold_r)) >= 2:
        oof_fused = np.zeros(len(event), float)
        for f in np.unique(fold_r):
            tr = fold_r != f
            va = fold_r == f
            wf = fit_weights(clipped[:, tr], event[tr])
            oof_fused[va] = wf @ clipped[:, va]
        w_full = fit_weights(clipped, event)
        return w_full, _binary_loss(oof_fused, event), oof_fused
    w_full = fit_weights(clipped, event)
    p_full = w_full @ clipped
    return w_full, _binary_loss(p_full, event), p_full


def _joint_event(
    design,
    dense,
    eb_ids,
    rows,
    event,
    fold,
    rest,
    deadline,
    workers=1,
    extra_fn=None,
    consensus_data=None,
):
    """Five-fold OOF and full-fit probability of answering the follow-up."""
    folds = np.unique(fold[rows])
    n_arms = 6 if consensus_data is not None else 3
    oof = np.zeros((n_arms, len(rows)))
    event_r = event[rows]

    def with_extra(fit):
        if extra_fn is None:
            return design, dense
        from scipy import sparse

        extra = extra_fn(fit)
        if not extra.shape[1]:
            return design, dense
        return (
            sparse.hstack((design, sparse.csr_matrix(extra))).tocsr(),
            np.column_stack((dense, extra)),
        )

    def one_fold(f):
        check_clock(deadline)
        if f is None:
            d, x = with_extra(rows)
            cpair = None
            if consensus_data is not None:
                item_oof_arms, item_full_arms = consensus_data(rows) if callable(consensus_data) else consensus_data
                cpair = _consensus_for_partition(
                    item_oof_arms,
                    item_full_arms,
                    event_r,
                    np.ones(len(rows), bool),
                    rest,
                    is_oof_query=False,
                )
            return None, _event_arms(d, x, eb_ids, rows, event, rows, rest, deadline, consensus_pair=cpair)
        fit_mask = fold[rows] != f
        mask = fold[rows] == f
        fit = rows[fit_mask]
        d, x = with_extra(fit)
        cpair = None
        if consensus_data is not None:
            item_oof_arms, item_full_arms = consensus_data(fit) if callable(consensus_data) else consensus_data
            cpair = _consensus_for_partition(
                item_oof_arms,
                item_full_arms,
                event_r,
                fit_mask,
                rows[mask],
                is_oof_query=True,
                query_mask_r=mask,
            )
        return mask, _event_arms(d, x, eb_ids, rows, event, fit, rows[mask], deadline, consensus_pair=cpair)

    with ThreadPoolExecutor(
        max_workers=workers, initializer=lambda: threadpool_limits(limits=1, user_api='openmp')
    ) as executor:
        for mask, probabilities in executor.map(one_fold, [*folds, None]):
            if mask is None:
                full = probabilities
            else:
                oof[:, mask] = probabilities
    if consensus_data is not None:
        weights, loss, oof_pred = _fuse_event(oof, event[rows], fold_r=fold[rows])
        return oof_pred, weights @ full, weights, loss
    weights, loss = _fuse_event(oof, event[rows])
    return weights @ oof, weights @ full, weights, loss


def _conditional(ids, labels, fit_rows, query_rows, k, alpha=20.0):
    counts = np.bincount(ids[fit_rows] * k + labels[fit_rows], minlength=(int(ids.max()) + 1) * k).reshape(-1, k)
    global_counts = np.bincount(labels[fit_rows], minlength=k).astype(float)
    prior = (global_counts + 1.0 / k) / (global_counts.sum() + 1.0)
    local = counts[ids[query_rows]]
    return (local + alpha * prior) / (local.sum(1, keepdims=True) + alpha)


def _anchor_transition(fit_anchor, k_anchor, fit_labels, query_anchor, k, alpha=8.0):
    """Baseline-answer -> follow-up-answer table on fit rows, applied to the
    anchor's stacked probabilities on query rows."""
    counts = np.bincount(fit_anchor * k + fit_labels, minlength=k_anchor * k).reshape(k_anchor, k).astype(float)
    global_counts = np.bincount(fit_labels, minlength=k).astype(float)
    prior = (global_counts + 1.0 / k) / (global_counts.sum() + 1.0)
    T = (counts + alpha * prior) / (counts.sum(1, keepdims=True) + alpha)
    out = np.clip(query_anchor, 0.0, None) @ T
    total = out.sum(1, keepdims=True)
    return np.divide(out, total, out=np.tile(prior, (len(out), 1)), where=total > 0)


def _adjusted_mi(a, k_a, b, k_b):
    """Plug-in MI minus its independence bias; the StackedChain.informative rule."""
    n = len(a)
    if not n:
        return 0.0
    joint = np.bincount(a * k_b + b, minlength=k_a * k_b).reshape(k_a, k_b) / n
    independent = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    valid = joint > 0
    mi = float(np.sum(joint[valid] * np.log(joint[valid] / independent[valid])))
    return mi - (k_a - 1) * (k_b - 1) / (2.0 * n)


def _answer_probability(
    coarse_ids,
    all_ids,
    design,
    X,
    labels,
    fit_rows,
    query_rows,
    k,
    deadline=float('inf'),
    slices=None,
    extra=None,
    anchor=None,
):
    """All-GIVEN EB table blended with two models fit on the same eligible rows."""
    check_clock(deadline)
    if not len(query_rows):
        return np.empty((0, k))
    if slices is None:
        slices = (
            design[fit_rows],
            design[query_rows],
            X[fit_rows],
            X[query_rows],
            coarse_ids[fit_rows],
            coarse_ids[query_rows],
            all_ids[fit_rows],
            all_ids[query_rows],
        )
    fit_design, query_design, fit_X, query_X, fit_ids, query_ids, fit_all, query_all = slices
    raw_fit_X, raw_query_X = fit_X, query_X
    if extra is not None and extra[0].shape[1]:
        from scipy import sparse

        fit_design = sparse.hstack((fit_design, sparse.csr_matrix(extra[0]))).tocsr()
        query_design = sparse.hstack((query_design, sparse.csr_matrix(extra[1]))).tocsr()
        fit_X = np.column_stack((fit_X, extra[0]))
        query_X = np.column_stack((query_X, extra[1]))
    fit_labels = labels[fit_rows]
    counts = np.bincount(fit_ids * k + fit_labels, minlength=(int(coarse_ids.max()) + 1) * k).reshape(-1, k)
    global_counts = np.bincount(fit_labels, minlength=k).astype(float)
    prior = (global_counts + 1.0 / k) / (global_counts.sum() + 1.0)
    local = counts[query_ids]
    coarse = (local + 20.0 * prior) / (local.sum(1, keepdims=True) + 20.0)
    counts = np.bincount(fit_all * k + fit_labels, minlength=(int(all_ids.max()) + 1) * k).reshape(-1, k)
    local = counts[query_all]
    eb = (local + 20.0 * coarse) / (local.sum(1, keepdims=True) + 20.0)
    if ENABLE_WB_RCT and raw_fit_X.shape[1] >= 1:
        log_p0 = np.log(np.maximum(prior, 1e-12))
        ll = np.log(np.maximum(coarse, 1e-12))
        for c in range(raw_fit_X.shape[1]):
            col_f = raw_fit_X[:, c].astype(np.int64)
            col_q = raw_query_X[:, c].astype(np.int64)
            nc = int(max(col_f.max() if len(col_f) else 0, col_q.max() if len(col_q) else 0)) + 1
            ct = np.bincount(col_f * k + fit_labels, minlength=nc * k).reshape(nc, k).astype(float)
            pc = (ct + 12.0 * prior) / (ct.sum(1, keepdims=True) + 12.0)
            ll += 0.16 * (np.log(np.maximum(pc[col_q], 1e-12)) - log_p0)
        ll -= ll.max(1, keepdims=True)
        add_eb = np.exp(ll)
        add_eb /= add_eb.sum(1, keepdims=True)
        eb = 0.5 * eb + 0.5 * add_eb
    classes = np.unique(fit_labels)
    if len(classes) < 2:
        return eb

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    check_clock(deadline)
    logit = LogisticRegression(C=0.1, max_iter=28, tol=2e-3)
    logit.fit(fit_design, fit_labels)
    check_clock(deadline)
    logit_p = np.zeros_like(eb)
    logit_p[:, logit.classes_.astype(int)] = logit.predict_proba(query_design)
    shallow = HistGradientBoostingClassifier(
        max_iter=15, max_leaf_nodes=7, learning_rate=0.1, early_stopping=False, random_state=20261021
    )
    check_clock(deadline)
    shallow.fit(fit_X, fit_labels)
    check_clock(deadline)
    shallow_p = np.zeros_like(eb)
    shallow_p[:, shallow.classes_.astype(int)] = shallow.predict_proba(query_X)
    check_clock(deadline)
    if anchor is None:
        return 0.5 * eb + 0.3 * logit_p + 0.2 * shallow_p
    anchor_eb = (anchor if isinstance(anchor, np.ndarray) else
                 _anchor_transition(anchor[0], anchor[1], fit_labels, anchor[2], k))
    return 0.35 * eb + 0.25 * anchor_eb + 0.20 * logit_p + 0.20 * shallow_p


def _rescale_randomized_na(prob, r_mask, na_idx, gate_idx, target_na):
    """Preserve conditional answer odds on randomized rows while setting P(Not answered) = target_na."""
    out = prob.copy()
    if not np.any(r_mask):
        return out
    sub = out[r_mask].copy()
    sub[:, gate_idx] = 0.0
    ans_sum = sub[:, :na_idx].sum(1, keepdims=True)
    cond = np.divide(
        sub[:, :na_idx],
        ans_sum,
        out=np.full_like(sub[:, :na_idx], 1.0 / max(1, na_idx)),
        where=ans_sum > 1e-12,
    )
    t_na = np.clip(target_na, 1e-5, 1.0 - 1e-5)
    sub[:, :na_idx] = cond * (1.0 - t_na[:, None])
    sub[:, na_idx] = t_na
    out[r_mask] = sub
    return out


def _shrink_item_attrition(job, visible, rest, randomized, fold, event_oof, event_full, na_idx, gate_idx):
    """Nested 5-fold shrinkage of each single-item arm's attrition probability toward cross-19-item consensus."""
    from base import _nll
    from cross_round import dominates

    y_vis = job['y'][visible]
    k = job['K']
    f_vis = fold[visible]
    half = (f_vis % 2) == 0
    r_vis = randomized[visible]
    r_rest = randomized[rest]
    cons_na_vis = np.clip(1.0 - event_oof[visible[r_vis]], 1e-4, 1.0 - 1e-4)
    cons_na_rest = np.clip(1.0 - event_full[rest[r_rest]], 1e-4, 1.0 - 1e-4)
    LAMBDAS = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)
    shrunk_count = 0

    for arm_idx in range(len(job['oof'])):
        o_orig = job['oof'][arm_idx]
        p_orig = job['all'][arm_idx]
        denom_vis = np.maximum(1.0 - o_orig[r_vis, gate_idx], 1e-6)
        arm_na_vis = np.clip(o_orig[r_vis, na_idx] / denom_vis, 1e-4, 1.0 - 1e-4)

        cand_oofs = [
            _rescale_randomized_na(o_orig, r_vis, na_idx, gate_idx, (1.0 - lam) * arm_na_vis + lam * cons_na_vis)
            for lam in LAMBDAS
        ]
        # Nested 5-fold selection of lambda
        o_nested = o_orig.copy()
        fold_lams = []
        for f in range(5):
            fit_m = f_vis != f
            val_m = f_vis == f
            losses = [_nll(c[fit_m], y_vis[fit_m], k) for c in cand_oofs]
            best_i = int(np.argmin(losses))
            fold_lams.append(LAMBDAS[best_i])
            o_nested[val_m] = cand_oofs[best_i][val_m]
        base_l = _nll(o_orig, y_vis, k)
        new_l = _nll(o_nested, y_vis, k)
        if new_l < base_l - 1e-5 and dominates(o_nested, o_orig, y_vis, k, half):
            lam_full = float(np.median(fold_lams))
            p_new = p_orig.copy()
            p_new[visible] = o_nested
            if len(rest) and np.any(r_rest):
                p_rest = p_orig[rest].copy()
                denom_rest = np.maximum(1.0 - p_rest[r_rest, gate_idx], 1e-6)
                arm_na_rest = np.clip(p_rest[r_rest, na_idx] / denom_rest, 1e-4, 1.0 - 1e-4)
                p_new[rest] = _rescale_randomized_na(
                    p_rest, r_rest, na_idx, gate_idx, (1.0 - lam_full) * arm_na_rest + lam_full * cons_na_rest
                )
            job['oof'][arm_idx] = o_nested.astype(o_orig.dtype)
            job['all'][arm_idx] = p_new.astype(p_orig.dtype)
            shrunk_count += 1
    return shrunk_count


def fit(
    frame,
    schema,
    given,
    X,
    design,
    tables,
    work,
    deadline=float('inf'),
    chain=None,
    workers=1,
    anchor_names=None,
):
    """Return item -> (OOF, full-frame, gain), plus shared-event diagnostics."""
    if 'group' not in given:
        return {}, {'reason': 'no_group'}
    if time.monotonic() >= deadline:
        return {}, {'reason': 'budget_exhausted'}
    group_col = given.index('group')
    group_values = list(schema['items']['group']['values'])
    try:
        not_randomized = group_values.index('Not randomised')
    except ValueError:
        return {}, {'reason': 'no_randomization_sentinel'}

    tracer_names = []
    for name, item in schema['items'].items():
        gate = item.get('gate')
        if (
            item['class'] == 'PREDICT'
            and name.startswith('tr_')
            and isinstance(gate, dict)
            and gate.get('parent') == 'group'
            and 'Not answered' in (item.get('values') or ())
            and name in work
        ):
            tracer_names.append(name)
    if len(tracer_names) < 8:
        return {}, {'reason': 'too_few_tracer_items', 'tracer_items': len(tracer_names)}

    first = work[tracer_names[0]]['train']
    if any(not np.array_equal(first, work[name]['train']) for name in tracer_names):
        return {}, {'reason': 'unequal_visible_rows'}
    visible = np.flatnonzero(first)
    randomized = X[:, group_col] != not_randomized
    train_randomized = visible[randomized[visible]]
    unanswered = np.zeros(len(X), np.int16)
    for name in tracer_names:
        option = schema['items'][name]['values'].index('Not answered')
        unanswered += (work[name]['y'] == option) & first
    cutoff = len(tracer_names)
    absent = unanswered >= cutoff
    n_absent = int(absent[train_randomized].sum())
    n_present = len(train_randomized) - n_absent
    mixed = int(((unanswered[train_randomized] > 0) & (unanswered[train_randomized] < cutoff)).sum())
    info = {
        'tracer_items': len(tracer_names),
        'cutoff': cutoff,
        'absent': n_absent,
        'present': n_present,
        'partial_missing': mixed,
        'event_definition': 'all_direct_items_not_answered',
    }
    if min(n_absent, n_present) < max(20, int(0.02 * len(train_randomized))):
        info['reason'] = 'no_shared_followup_event'
        return {}, info

    if chain is not None:
        original_chain = chain

        class PartitionChain:
            def __init__(self):
                self.anchors = original_chain.anchors
                self.local = {}

            def probabilities(self, name, train):
                return original_chain.bank.table_probability(name, train, original_chain.bank.partition(train))[0]

            def _latent(self, train, key):
                if key not in self.local:
                    selected = [a for a in EVENT_ANCHORS if a in self.anchors]
                    blocks = [self.probabilities(a, train) for a in selected]
                    S = np.column_stack(blocks) if blocks else np.zeros((len(X), 0))
                    rank = min(6, S.shape[1], len(train) - 1)
                    if rank < 1:
                        z = np.zeros((len(X), 0))
                    else:
                        mean = S[train].mean(0)
                        centered = S[train] - mean
                        _, e = np.linalg.eigh(centered.T @ centered)
                        z = (S - mean) @ e[:, -rank:]
                        sd = z[train].std(0)
                        sd[sd < 0.05] = 1
                        z = (z / sd).astype(np.float32)
                    self.local[key] = z
                return self.local[key]

            def informative(self, name, train, count=2):
                return original_chain.informative(name, train, count=count)

        chain = PartitionChain()
    fold = tables.fold
    event = ~absent
    event_oof = np.zeros(len(X), float)
    country = given.index('user.country') if 'user.country' in given else None
    features = (group_col,) if country is None else (group_col, country)
    event_anchors = [a for a in EVENT_ANCHORS if a in chain.anchors] if chain is not None else []

    def extra_fn(fit):
        fit = np.asarray(fit, dtype=np.int64)
        cols = [chain.probabilities(a, fit) for a in event_anchors]
        cols.append(chain._latent(fit, fit.tobytes())[:, :6])
        return np.column_stack(cols).astype(np.float32)

    info['event_anchors'] = event_anchors
    info['event_extra_columns'] = int(extra_fn(train_randomized).shape[1]) if chain is not None else 0
    dense = X.astype(np.float32)
    rest = np.flatnonzero(~first)
    consensus_data = None
    if ENABLE_WB_RCT and chain is not None:
        def consensus_data(fit):
            o_arms, p_arms = [], []
            for target in tracer_names:
                job = work[target]
                probability = chain.probabilities(target, fit)
                na = schema['items'][target]['values'].index('Not answered')
                answered = np.clip(1.-probability[:, na] / np.maximum(1.-probability[:, job['K']-1], 1e-6), 1e-4, 1.-1e-4)
                o_arms.append(answered[train_randomized][None, :])
                p_arms.append(answered[None, :])
            return o_arms, p_arms
    try:
        oof_event, full_event, weights, loss = _joint_event(
            design,
            dense,
            tables.ids(features),
            train_randomized,
            event,
            fold,
            np.arange(len(X)),
            deadline,
            workers=workers,
            extra_fn=extra_fn if chain is not None else None,
            consensus_data=consensus_data,
        )
        event_oof[train_randomized] = oof_event
        event_full = full_event
        info['_event'] = (event_oof.copy(), np.asarray(event_full, float).copy(), randomized.copy())
        info['event_weights'] = [float(w) for w in weights]
        info['event_loss'] = loss
    except BudgetExpired:
        info['reason'] = 'budget_exhausted'
        return {}, info
    if time.monotonic() >= deadline:
        info['reason'] = 'budget_exhausted'
        return {}, info

    if ENABLE_WB_RCT:
        total_shrunk = 0
        for name in tracer_names:
            na_idx = schema['items'][name]['values'].index('Not answered')
            gate_idx = work[name]['K'] - 1
            total_shrunk += _shrink_item_attrition(
                work[name], visible, rest, randomized, fold, event_oof, event_full, na_idx, gate_idx
            )
        info['attrition_shrunk_arms'] = int(total_shrunk)

    ids = tables.ids(features)
    all_ids = tables.ids(tuple(range(X.shape[1])))

    def sliced(fit_r, query_r):
        return (
            design[fit_r],
            design[query_r],
            X[fit_r],
            X[query_r],
            ids[fit_r],
            ids[query_r],
            all_ids[fit_r],
            all_ids[query_r],
        )

    partitions = []
    for f in np.unique(fold[visible]):
        train = train_randomized[(fold[train_randomized] != f) & event[train_randomized]]
        val = visible[fold[visible] == f]
        if len(train) and len(val):
            partitions.append((train, val, sliced(train, val)))
    fit_rows = train_randomized[event[train_randomized]]
    full_slices = sliced(fit_rows, rest)
    tracer_probabilities = {}

    def probabilities(a, train):
        return chain.probabilities(a, train)

    def linked(name, train, query):
        if chain is None:
            return None, None, []
        labels, k = work[name]['y'][train], int(work[name]['K'])
        pool = (list(chain.anchors if anchor_names is None else anchor_names) if ENABLE_MULTI_ANCHOR_RB else
                chain.informative(name, train, count=2) + list(tracer_probabilities))
        scored = []
        for a in pool:
            if a == name or np.any(work[a]['y'][train] < 0):
                continue
            mi = _adjusted_mi(work[a]['y'][train], int(work[a]['K']), labels, k)
            if mi > 0:
                scored.append((-mi, a))
        scored.sort()
        top = [a for _, a in scored[:3 if ENABLE_MULTI_ANCHOR_RB else 2]]
        latent = chain._latent(train, train.tobytes())[:, :6]
        cols = np.column_stack([probabilities(a, train) for a in top if a not in tracer_probabilities] + [latent])
        extra = (cols[train], cols[query])
        if not top:
            return extra, None, top
        if ENABLE_MULTI_ANCHOR_RB:
            from rao_blackwell import transition, convex_weights
            # Every risk estimate is made on rows excluded from both the
            # response table and the anchor predictor's fitting partition.
            inner = np.zeros((len(top), len(train), k), float)
            anchor_prob = {a: probabilities(a, train) for a in top}
            for f in np.unique(fold[train]):
                mask = fold[train] == f
                tr, va = train[~mask], train[mask]
                check_clock(deadline)
                for m, a in enumerate(top):
                    inner[m, mask] = transition(work[a]['y'], work[a]['K'],
                        work[name]['y'], X[:, group_col], tr, va,
                        anchor_prob[a][va], k)
            weights = convex_weights(inner, labels)
            components = [transition(work[a]['y'], work[a]['K'], work[name]['y'],
                X[:, group_col], train, query, probabilities(a, train)[query], k) for a in top]
            return extra, np.einsum('a,ank->nk', weights, components), top
        a = top[0]
        if a in tracer_probabilities:
            q = probabilities(a, train)[query].astype(float).copy()
            q[:, schema['items'][a]['values'].index('Not answered')] = 0.0
            q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
            return extra, (work[a]['y'][train], int(work[a]['K']), q), top
        return extra, (work[a]['y'][train], int(work[a]['K']), probabilities(a, train)[query]), top

    def answered(conditional, event, rows, no_answer, gate_index):
        p = event[rows, None] * conditional
        p[:, no_answer] += 1.0 - event[rows]
        p[~randomized[rows]] = 0.0
        p[~randomized[rows], gate_index] = 1.0
        return p

    anchor_hits = {}
    result = {}
    for name in tracer_names:
        if time.monotonic() >= deadline:
            info['reason'] = 'budget_exhausted'
            break
        job = work[name]
        labels, k = job['y'], job['K']
        no_answer = schema['items'][name]['values'].index('Not answered')
        gate_index = k - 1
        oof = np.zeros((len(visible), k), float)
        try:
            def one_partition(partition):
                train, val, cached_slices = partition
                check_clock(deadline)
                extra, anchor, _ = linked(name, train, val)
                conditional = _answer_probability(
                    ids, all_ids, design, X, labels, train, val, k, deadline, cached_slices, extra=extra, anchor=anchor
                )
                return val, conditional

            full = np.zeros((len(X), k), float)
            with ThreadPoolExecutor(
                max_workers=workers, initializer=lambda: threadpool_limits(limits=1, user_api='openmp')
            ) as executor:
                for i, (val, conditional) in enumerate(
                    executor.map(one_partition, [*partitions, (fit_rows, rest, full_slices)])
                ):
                    if i == len(partitions):
                        full[rest] = answered(conditional, event_full, rest, no_answer, gate_index)
                    else:
                        oof[np.searchsorted(visible, val)] = answered(
                            conditional, event_oof, val, no_answer, gate_index
                        )
            _, _, top = linked(name, fit_rows, rest)
            if top:
                anchor_hits[name] = top
            y = labels[visible]
            floor = 0.001
            c = 1.0 - k * floor
            own = -np.log(floor + c * oof[np.arange(len(visible)), y]).mean()
            base = -np.log(floor + c * job['oof'][0][np.arange(len(visible)), y]).mean()
            result[name] = (oof, full, float(base - own))
            if chain is not None:
                answer = full.copy()
                answer[visible] = oof
                answer[:, gate_index] = 0.0
                answer /= np.maximum(answer.sum(1, keepdims=True), 1e-12)
                tracer_probabilities[name] = answer.astype(np.float32)
        except BudgetExpired:
            info['reason'] = 'budget_exhausted'
            break
    info['anchors'] = anchor_hits
    info['tracer_anchors'] = sorted({a for top in anchor_hits.values() for a in top if a in tracer_probabilities})
    info['transitions'] = 0
    info['transition_arms'] = {}
    return result, info
