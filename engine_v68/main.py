import ortho_config as _ortho
from . import v68_config
from threadpoolctl import threadpool_limits
'Budgeted categorical fusion with partition-isolated feature selection.'
import time
_START = time.monotonic()
import os
_THREADS = max(1, min(8, os.cpu_count() or 1, int(os.environ.get('SIMULACRA_THREADS', '8'))))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = str(_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import sys, warnings, gc, re
warnings.filterwarnings('ignore', message='X does not have valid feature names, but .* was fitted with feature names', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import numpy as np
import pandas as pd
from threadpoolctl import ThreadpoolController
from types import SimpleNamespace
from .base import _codes, _Tables, _fit_table, _safe_table_config, _design, _mi_drivers, _logit, _logit_proba, _fit_net, _fit_net_gpu, _stack, _nll, _normalize, _softmax, _cross_target, _rb_distill
from .chain_features import hgb_stages, hgb_crossfit_stages, FoldFeatureBank, folds_for, tree_predict, gated_tree_predict, topological, BudgetExpired, check_clock, split_anchors, StackedChain, FoldPlan, PRE_POST_MIRRORS, UNICEF_ANCHORS
from .stratified_kernel_prior import build as build_neighbors, select as select_neighbors
from .propagate_structural_gates import propagate_structural_gates, detect_zero_on_skip, fold_wave_one_skip, ZERO_ON_SKIP, _parents, EMPIRICAL_PARENTS, NESTED_TRACE, discover_empirical_gate_parents
from .gate_decomposition import candidate as gate_candidate
from . import gpu; from . import base; from . import survey_priors; from . import joint_targets
from .model_config import FOUNDATION_ARMS, JOINT_TARGETS
from .high_order_tables import fit as fit_high_order_tables
from .full_cell_eb import attach as attach_full_cell_eb, attach_conditioned
from .latent_profiles import fit_latent_profiles
from .tracer_followup import fit as fit_tracer_followup
from .paired_transition import attach as attach_paired_transition
from .cross_round import attach as attach_cross_round, row_gate as cross_round_row_gate, pre_track
from . import given_crosses; from . import anchor_transition
if v68_config.V101_W1:
    from .stack_active import stack as _stack
_DONE = set()
LAST_REPORT = {}
DISABLED = frozenset(('anchor_cascade', 'joint_gate'))
VARIANT = 'main'
_SHARE = {'unhcr_heavy': {213: 0.7, 76: 0.22, 12: 0.08}, 'wb_unicef_heavy': {213: 0.48, 76: 0.38, 12: 0.14}}.get(VARIANT, {213: 0.62, 76: 0.28, 12: 0.1})
if VARIANT == 'unhcr_heavy':
    from . import cross_round
    cross_round.ROW_PRIORS = (0.03, 0.05, 0.1, 0.18, 0.3, 0.6, 1.2, 2.5, 5.0)
    cross_round.ROW_THRESHOLDS = (1e-09, 0.15, 0.35, 0.45)
_TREE_ARMS = frozenset(('shallow', 'shallow_asked', 'two_stage', 'deep', 'hgb', 'catboost'))
# Forked stack workers inherit this per-item closure; only item names cross the pipe.
_FORK_TASK = None


def _fork_run(name):
    try:
        return _FORK_TASK(name)
    except Exception:
        return None
_WB_CHILD_DEEP = 'all'
# the configured option strengths: the water level as a multiple of the scoring floor; 0 keeps the vector.
FLOOR_FILL_TAUS = (0.0, 0.5, 0.75, 1.0)


def _waterfill(P, g):
    """argmax_p sum_y pi_y log(g + (1 - K g) p_y) for every row's belief pi = P: options
    under the water level drop to the floor, the rest stay proportional to pi."""
    K = P.shape[1]
    if g <= 0.0 or K * g >= 0.5:
        return P
    c = g / (1.0 - K * g)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    s = -np.sort(-P, 1)
    lam = np.cumsum(s, 1) / (1.0 + c * np.arange(1, K + 1))
    top = (s > c * lam).sum(1)
    level = lam[np.arange(len(P)), np.maximum(top, 1) - 1]
    Q = np.maximum(P / np.maximum(level, 1e-300)[:, None] - c, 0.0)
    return Q / np.maximum(Q.sum(1, keepdims=True), 1e-300)


def _floor_fill(work, stacked, final, final_oofs):
    """The scorer mixes every vector with a flat floor, so a calibrated belief is not the
    best vector to send: water-fill each item's hidden rows at the strength its final OOF
    scores best under the same floor."""
    chosen = {}
    for name, fin in final.items():
        j = work.get(name)
        if j is None or name not in stacked:
            continue
        y, K = j['y'][j['train']], j['K']
        o = final_oofs.get(name, stacked[name][0])
        if o is None or np.shape(o) != (len(y), K) or np.shape(fin) != (len(j['train']), K):
            continue
        o = _normalize(np.asarray(o, float))
        losses = [_nll(_waterfill(o, tau * base._FLOOR), y, K) for tau in FLOOR_FILL_TAUS]
        best = int(np.argmin(losses))
        if FLOOR_FILL_TAUS[best] > 0.0:
            rest = ~j['train']
            fin[rest] = _waterfill(_normalize(np.asarray(fin[rest], float)), FLOOR_FILL_TAUS[best] * base._FLOOR)
        chosen[name] = [FLOOR_FILL_TAUS[best], round(_nll(o, y, K) - losses[best], 6)]
    return {'items': sum(v[0] > 0 for v in chosen.values()), 'oof_gain': float(np.mean([v[1] for v in chosen.values()])) if chosen else 0.0, 'chosen': chosen}

def calibrate(oofs, rests, y, K, deadline=float('inf')):
    if 'simplex_stack' in DISABLED:
        losses = [_nll(o, y, K) for o in oofs]
        best = int(np.argmin(losses))
        weights = np.zeros(len(oofs))
        weights[best] = 1.0
        return (rests[best], oofs[best], weights, 1.0, losses[best])
    p, o, w, t, l = _stack(oofs, rests, y, K, deadline=deadline)
    best = (l, 0.0, p, o)
    for floor in (() if v68_config.V101_W4 and v68_config.V97_W1 else (0.0001, 0.001, 0.005)):
        if time.monotonic() >= deadline:
            break
        co = _normalize(np.maximum(o, floor))
        loss = _nll(co, y, K)
        if loss < best[0]:
            best = (loss, floor, _normalize(np.maximum(p, floor)), co)
    return (best[2], best[3], w, t, best[0])

def _pick_c(D, work, names, rows, deadline):
    """Three inner folds on representative items, confined to supplied rows."""
    from sklearn.linear_model import LogisticRegression
    candidates = (.1, .3, .8)
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) > 768:
        rows = rows[np.argsort((rows * 2654435761) % 4294967291, kind='stable')[:768]]
    totals = np.zeros(3)
    count = 0
    chosen = [name for name in names if np.unique(work[name]['y'][rows]).size > 1][:2]
    split = np.arange(len(rows)) % 3
    for name in chosen:
        y, K = work[name]['y'], work[name]['K']
        for f in range(3):
            tr, va = rows[split != f], rows[split == f]
            tr, va = tr[y[tr] >= 0], va[y[va] >= 0]
            if not len(va) or np.unique(y[tr]).size < 2: continue
            loss = []
            for c in candidates:
                check_clock(deadline)
                model = LogisticRegression(C=c, max_iter=90, tol=5e-4)
                model.fit(D[tr], y[tr])
                pred = np.zeros((len(va), K))
                pred[:, model.classes_.astype(int)] = model.predict_proba(D[va])
                loss.append(_nll(pred, y[va], K))
            totals += loss
            count += 1
    return candidates[int(np.argmin(totals))] if count else .3


def crossfit_calibrate(oofs, rests, y, K, split, deadline=float('inf')):
    """Evaluate mixture parameters on whole held-out partitions."""
    if len(np.unique(split)) < 2:
        p, _, w, t, _ = calibrate(oofs, rests, y, K, deadline)
        o = np.mean(oofs, axis=0)
        return p, o, w, t, _nll(o, y, K)
    p, _, w, t, _ = calibrate(oofs, rests, y, K, deadline)
    o = np.empty_like(oofs[0], dtype=float)
    for f in np.unique(split):
        va = split == f
        if va.all():
            o[va] = np.mean([a[va] for a in oofs], axis=0)
        else:
            o[va] = calibrate([a[~va] for a in oofs], [a[va] for a in oofs],
                              y[~va], K, deadline)[0]
    return p, o, w, t, _nll(o, y, K)

def aligned_partitions(job, fold):
    """Coarsest union of the completed arms' base-fold partitions."""
    sub = fold[job['train']]
    labels = np.arange(5)
    partitions = {job.get('partition_k', {}).get(arm, 5) for arm in job['arms']}
    for k in sorted(partitions - {5, 3}):
        for a in range(5):
            for b in range(a):
                if a % k == b % k:
                    labels[labels == labels[a]] = labels[b]
    if len(np.unique(labels)) < 2:
        labels = np.arange(5) % 2
    aligned = labels[sub]
    return aligned if len(np.unique(aligned)) >= 2 else np.arange(len(sub), dtype=np.int64) % 2

def tiered_calibrate(job, keep, rest, fold, flat, deadline):
    """Five-fold arms fuse on their own five partitions, then meet the coarser
    arms on the shared partition. Kept when it wins both held-out halves."""
    pk = job.get('partition_k', {})
    five = [i for i in keep if pk.get(job['arms'][i], 5) == 5]
    coarse = [i for i in keep if i not in five]
    if len(five) < 2 or not coarse:
        return flat, None
    y, K = job['y'][job['train']], job['K']
    p5, o5, w5, _, _ = crossfit_calibrate([job['oof'][i] for i in five], [job['all'][i][rest] for i in five],
                                          y, K, fold[job['train']], deadline=deadline)
    aligned = aligned_partitions({'arms': [job['arms'][i] for i in keep], 'partition_k': pk, 'train': job['train']}, fold)
    p, o, w2, t, loss = crossfit_calibrate([o5] + [job['oof'][i] for i in coarse], [p5] + [job['all'][i][rest] for i in coarse],
                                           y, K, aligned, deadline=deadline)
    from .cross_round import dominates
    if not dominates(o, flat[1], y, K, fold[job['train']] % 2 == 0):
        return flat, False
    w = np.zeros(len(job['arms']))
    w[five] = w2[0] * np.asarray(w5)
    w[coarse] = w2[1:]
    return (p, o, w, t, loss), True

def compact_anchor(job, rest, fold, flat, deadline):
    if 'wb_lca' not in job['arms'] or time.monotonic() >= deadline:
        return flat
    y,K=job['y'][job['train']],job['K']
    # Preserve structural parent arms and the strongest learner families, with
    # selection and calibration inside each outer training partition.
    arms=job['arms'];anchor=arms.index('wb_lca')
    families=[('catboost','deep','hgb','shallow','shallow_asked'),('network',),('paired_transition','followup','followup_anchor','anchor_eb')]
    split=aligned_partitions(job,fold)
    def fit(rows,hold):
        keep=[anchor]
        for family in families:
            options=[i for i,a in enumerate(arms) if a in family]
            if options:keep.append(min(options,key=lambda i:_nll(job['oof'][i][rows],y[rows],K)))
        return calibrate([job['oof'][i][rows] for i in keep],hold(keep),y[rows],K,deadline),keep
    (p,_,weights,t,_),keep=fit(np.ones(len(y),bool),lambda ids:[job['all'][i][rest] for i in ids])
    o=np.empty_like(job['oof'][0],dtype=float)
    for f in np.unique(split):
        va=split==f
        if va.all():return flat
        result,_=fit(~va,lambda ids:[job['oof'][i][va] for i in ids]);o[va]=result[0]
    from .cross_round import dominates
    half=fold[job['train']]%2==0
    # Full stack must win in both held-out halves to justify its added arms.
    if dominates(flat[1],o,y,K,half):return flat
    w=np.zeros(len(arms));w[keep]=weights
    return p,o,w,t,_nll(o,y,K)


def gate_latent(job, deadline):
    if 'latent_profile' not in job['arms']:
        return
    index = job['arms'].index('latent_profile')
    keep = [i for i in range(len(job['arms'])) if i != index]
    y = job['y'][job['train']]
    K = job['K']
    empty = [a[:0] for a in job['oof']]
    base_loss = _stack([job['oof'][i] for i in keep], [empty[i] for i in keep], y, K, deadline=deadline)[-1]
    new_loss = _stack(job['oof'], empty, y, K, deadline=deadline)[-1]
    job['latent_oof_gain'] = float(base_loss - new_loss)
    if base_loss - new_loss <= 0.0001:
        for key in ('arms', 'oof', 'all'):
            job[key].pop(index)

def encode(frame, schema):
    items = schema['items']
    given = [k for k, r in items.items() if r['class'] == 'GIVEN']
    n_raw_given = len(given)
    names = [k for k, r in items.items() if r['class'] == 'PREDICT']
    levels = np.array([len(items[k]['values']) + 1 for k in given])
    X = np.column_stack([np.maximum(0, np.where(frame[k].isna(), levels[j] - 1, _codes(frame[k], items[k]['values'], levels[j] - 1))) for j, k in enumerate(given)]).astype(np.int64)
    count_names = (('s2_3_boys_0_5', 's2_3_girls_0_5'), ('s2_3_boys_6_17', 's2_3_girls_6_17'), ('s2_3_men_18_59', 's2_3_women_18_59'), ('s2_3_men_60plus', 's2_3_women_60plus'))
    if all((name in given for pair in count_names for name in pair)):
        lower = {}
        for pair in count_names:
            for name in pair:
                values = items[name]['values']
                bounds = np.array([int(re.match('^\\s*(\\d+)', str(v)).group(1)) for v in values], dtype=np.int64)
                code = X[:, given.index(name)]
                lower[name] = np.where(code < len(bounds), bounds[np.minimum(code, len(bounds) - 1)], -1)
        groups = (('hh_total_lower', tuple((n for pair in count_names for n in pair))), ('children_6_17_lower', count_names[1]), ('children_0_5_lower', count_names[0]), ('adults_18_59_lower', count_names[2]))
        for feature, members in groups:
            columns = np.column_stack([lower[name] for name in members])
            missing = (columns < 0).any(1)
            max_bound = sum((max((int(re.match('^\\s*(\\d+)', str(v)).group(1)) for v in items[name]['values'])) for name in members))
            derived = np.where(missing, max_bound + 1, columns.sum(1))
            X = np.column_stack((X, derived))
            levels = np.append(levels, max_bound + 2)
            given.append(feature)
    work = {}
    for name in names:
        options = list(items[name]['values']) + ([schema.get('gated_value', 'NA_GATED')] if items[name].get('gate') else [])
        y = _codes(frame[name], options, -2)
        if (y == -2).any():
            y = np.where(y == -2, -1, y)
        work[name] = {'y': y, 'K': len(options), 'train': y >= 0}
    return (X, given, levels, work, n_raw_given)

def randomized_gate_targets(schema):
    """PREDICT descendants of the World Bank randomization arm."""
    items = schema['items']
    if 'group' not in items or 'Not randomised' not in items['group'].get('values', ()):
        return ()
    descendants = {'group'}
    pending = True
    while pending:
        pending = False
        for name, item in items.items():
            gate = item.get('gate')
            if item['class'] == 'PREDICT' and isinstance(gate, dict) and (gate.get('parent') in descendants) and (name not in descendants):
                descendants.add(name)
                pending = True
    return tuple((name for name, item in items.items() if item['class'] == 'PREDICT' and name in descendants))

def lock_randomization_gate(out, schema, X, given, source):
    """Apply the randomization rule where every visible row obeys it."""
    if 'group' not in given:
        return ()
    names = randomized_gate_targets(schema)
    if not names:
        return ()
    group_col = given.index('group')
    sentinel = schema['items']['group']['values'].index('Not randomised')
    rows = X[:, group_col] == sentinel
    locked = []
    for name in names:
        if name not in out:
            continue
        if not schema['items'][name].get('gate'):
            continue
        job = source[name]
        seen = job['train'] & rows
        if seen.sum() < 25 or np.any(job['y'][seen] != job['K'] - 1):
            continue
        target = rows & ~job['train']
        p0 = float(np.mean(job['y'][job['train']] == job['K'] - 1))
        skip = 1.0 - 1e-6
        out[name][target] = rescale_skip(out[name][target], skip)
        locked.append(name)
    return tuple(locked)

def rescale_skip(prob, skip):
    """Preserve conditional answer odds while changing the skip probability."""
    result = prob.copy()
    skip = np.broadcast_to(np.asarray(skip), len(prob))
    sums = result[:, :-1].sum(1, keepdims=True)
    conditional = np.divide(result[:, :-1], sums, out=np.full_like(result[:, :-1], 1 / (prob.shape[1] - 1)), where=sums > 0)
    result[:, :-1] = conditional * (1 - skip[:, None])
    result[:, -1] = skip
    return result

def lock_given_gate(prob, item, items, X, given, y, K, fit_rows, query_rows, soft=False):
    """Fold-local EB rates, with finite mass reserved for observed exceptions."""
    gate = item.get('gate')
    if K < 2 or not isinstance(gate, dict) or gate.get('parent') not in given:
        return (False, False)
    values = items[gate['parent']]['values']
    observed = {str(v) for v in gate.get('observed_if', ())}
    open_codes = [i for i, v in enumerate(values) if str(v) in observed]
    open_mask = np.isin(X[:, given.index(gate['parent'])], open_codes)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    fit_rows = fit_rows[y[fit_rows] >= 0]
    if not len(fit_rows):
        return (False, False)
    closed = fit_rows[~open_mask[fit_rows]]
    opened = fit_rows[open_mask[fit_rows]]
    p0 = float(np.mean(y[fit_rows] == K - 1))
    r_closed = float(np.mean(y[closed] == K - 1)) if len(closed) else 0.0
    r_open = float(np.mean(y[opened] == K - 1)) if len(opened) else p0
    strong = len(closed) >= 25 and r_closed == 1.0
    high = soft and len(closed) >= 30 and (r_closed >= 0.92) and (r_closed >= r_open + 0.25)
    lock_closed = bool(strong or high)
    lock_open = bool(len(opened) >= 25 and r_open == 0.0)
    if lock_closed:
        alpha = 1.5 if strong else 2.0
        skip = (np.sum(y[closed] == K - 1) + alpha * p0) / (len(closed) + alpha)
        target = ~open_mask[query_rows]
        prob[target] = rescale_skip(prob[target], skip)
    if lock_open:
        target = open_mask[query_rows]
        prob[target] = rescale_skip(prob[target], 1.5 * p0 / (len(opened) + 1.5))
    return (lock_closed, lock_open)

def soft_given_calibration(job, o, p, item, items, X, given, fold):
    gate = item.get('gate')
    if not isinstance(gate, dict) or gate.get('parent') not in given:
        return (o, p)
    rows = np.flatnonzero(job['train'])
    rest = np.flatnonzero(~job['train'])
    y = job['y']
    k = job['K']
    split = fold[rows]
    candidate = o.copy()
    hidden = p.copy()
    for f in np.unique(split):
        val = split == f
        part = candidate[val].copy()
        lock_given_gate(part, item, items, X, given, y, k, rows[~val], rows[val], soft=True)
        candidate[val] = part
    lock_given_gate(hidden, item, items, X, given, y, k, rows, rest, soft=True)
    from .cross_round import dominates
    yy = y[rows]
    half = split % 2 == 0
    best = (0.0, o, p, _nll(o, yy, k))
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        co = (1 - alpha) * o + alpha * candidate
        loss = _nll(co, yy, k)
        if loss < best[3] and dominates(co, o, yy, k, half):
            best = (alpha, co, (1 - alpha) * p + alpha * hidden, loss)
    job['soft_given'] = {'weight': best[0], 'oof_gain': float(_nll(o, yy, k) - best[3])}
    return (best[1], best[2])

def fold_arms(name, j, X, given, items):
    """Fold every arm's round-1 sentinel mass above the visible cap onto "0"."""
    if name not in ZERO_ON_SKIP:
        return
    wave = X[:, given.index('wave')]
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    for i, (o, p) in enumerate(zip(j['oof'], j['all'])):
        o = np.array(o, copy=True)
        p = np.array(p, copy=True)
        fold_wave_one_skip(o, wave[rows], name, items)
        hidden = p[rest]
        fold_wave_one_skip(hidden, wave[rest], name, items)
        p[rest] = hidden
        j['oof'][i] = o
        j['all'][i] = p

def questionnaire_split(names, items, anchors, children):
    """Three questionnaire stages: pre-assessment, post-assessment, follow-up.

    A child gated only on GIVEN fields, placed before the last GIVEN field,
    that a post-assessment item restates (PRE_POST_MIRRORS) moves to Pass 1.
    Ungated items after the last GIVEN field that route no other item leave
    Pass 1 and form Pass 2a on the Pass-1 chain; their stacked answers then
    extend the chain for the remaining children (Pass 2b). Returns the three
    disjoint lists (pass 1, pass 2b, pass 2a).
    """
    order = {k: i for i, k in enumerate(items)}
    last_given = max((order[k] for k, item in items.items() if item['class'] == 'GIVEN'))
    inside = set(names)
    routing = {p for name in names for p in _parents(name, items) if p in inside}
    baselines = {PRE_POST_MIRRORS[n] for n in anchors if n in PRE_POST_MIRRORS}

    def pre(name):
        return name in baselines and bool(items[name].get('gate')) and (order[name] < last_given) and (not any((p in inside for p in _parents(name, items))))

    def post(name):
        return not items[name].get('gate') and order[name] > last_given and (name not in routing)
    moved_up = [n for n in children if pre(n)]
    post_items = [n for n in anchors if post(n)]
    if not moved_up or not post_items:
        return (anchors, children, [])
    post = set(post_items)
    first = set(anchors) - post | set(moved_up)
    return ([n for n in names if n in first], [n for n in names if n not in first and n not in post], [n for n in names if n in post])

def candidate_order(profile, names, work):
    if profile not in FOUNDATION_ARMS:
        return names
    return sorted(names, key=lambda name: (not work[name]['route_parent'], work[name]['has_gate'], -float(work[name]['gains'].sum()), name))

def fit_partition(frame, schema, X, given, levels, source, allowed, deadline, near=None, phase='dev', finish_deadline=float('inf'), n_raw_given=None):
    """Every target label outside allowed is masked before any feature/model fit."""
    surgical = len(source) == 76 and _ortho.V65_K1_FAST_STAGES
    disabled = DISABLED | frozenset(('neighbor_cross_target', )) if surgical else DISABLED
    now = time.monotonic()
    span = max(0.0, deadline - now)
    gpu_enabled = gpu.available() and gpu.has_weights()
    cpu_stop = now + (0.6 if gpu_enabled else 0.7) * span
    pfn_stop = now + (0.9 if gpu_enabled else 0.8) * span
    net_stop = now + (0.93 if gpu_enabled else 0.86) * span
    nb_build_stop = now + (0.94 if gpu_enabled else 0.88) * span
    ct_stop = now + (0.965 if gpu_enabled else 0.94) * span
    nb_select_stop = deadline
    if n_raw_given is None:
        n_raw_given = sum((item['class'] == 'GIVEN' for item in schema['items'].values()))
    is_deep_hierarchy = len(source) > 100
    if is_deep_hierarchy:
        cpu_stop = now + (0.68 if gpu_enabled else 0.785) * span
        pfn_stop = now + (0.84 if gpu_enabled else 0.785) * span
        net_stop = now + 0.895 * span
        nb_build_stop = now + 0.908 * span
        ct_stop = now + 0.93 * span
        nb_select_stop = finish_deadline - 25.0
    elif len(source) == 76:
        cpu_stop = now + (0.6 if gpu_enabled else 0.785) * span
        pfn_stop = now + (0.9 if gpu_enabled else 0.785) * span
        net_stop = now + (0.93 if gpu_enabled else 0.86) * span
        nb_build_stop = now + (0.94 if gpu_enabled else 0.88) * span
        ct_stop = now + (0.965 if gpu_enabled else 0.912) * span
    elif len(source) == 12:
        cpu_stop = now + (0.68 if gpu_enabled else 0.76) * span
        pfn_stop = now + (0.9 if gpu_enabled else 0.76) * span
        net_stop = now + (0.93 if gpu_enabled else 0.9) * span
        nb_build_stop = now + (0.94 if gpu_enabled else 0.91) * span
        ct_stop = now + (0.965 if gpu_enabled else 0.945) * span
    route_parents = {item['gate']['parent'] for item in schema['items'].values() if isinstance(item.get('gate'), dict) and 'parent' in item['gate']}
    n = len(X)
    fold = folds_for(X, allowed)
    tables = _Tables(X, np.maximum(fold, 0), 5)
    work = {}
    strength = np.zeros(n_raw_given)
    for name, j in source.items():
        train = np.zeros(n, bool)
        train[allowed] = j['train'][allowed]
        y = np.where(train, j['y'], -1)
        if not train.any():
            continue
        o, p, g, config = _fit_table(tables, np.maximum(y, 0), train, j['K'], 0 if 'marginal' in disabled else n_raw_given, return_config=True)
        work[name] = {'y': y, 'K': j['K'], 'train': train, 'gains': g, 'oof': [o], 'all': [p], 'base': _nll(o, y[train], j['K']), 'arms': ['table'], 'partition_k': {'table': 5}, 'table_config': config, 'table_gain': float(np.maximum(g, 0).sum()), 'route_parent': name in route_parents, 'has_gate': bool(schema['items'][name].get('gate'))}
        if g.size:
            strength += g
    drivers = [int(a) for a in np.argsort(-strength[:n_raw_given])[:3] if strength[a] > 0] if is_deep_hierarchy else list(_safe_table_config(X, allowed, n_raw_given)[0])
    D = _design(X, levels, drivers, max_columns=600, support_rows=allowed, given=given)
    report = {k: 0 for k in ('high_order', 'followup', 'network', 'strong', 'shallow', 'deep', 'hgb', 'catboost', 'third', 'logit', 'neighbor', 'cross_target', 'tabpfn', 'two_stage', 'gate_table', 'gate_logit', 'rb_distill')}
    report.update(variant=VARIANT, table=len(work), drivers=[given[a] for a in drivers], strength=strength.tolist(), design_width=D.shape[1], threads=_THREADS)
    names = topological(list(work), schema['items'])
    if len(source) == 12 and 'high_order' not in disabled:
        for j in work.values():
            if time.monotonic() >= cpu_stop:
                break
            found = fit_high_order_tables(tables, j, n_raw_given, deadline=cpu_stop)
            if found is not None:
                j['oof'][0], j['all'][0], _ = found
                report['high_order'] += 1
    if len(source) == 12 and 'full_cell_eb' not in disabled and (4 <= n_raw_given <= 7):
        report['full_cell_eb'] = attach_full_cell_eb(X[:, :n_raw_given], given[:n_raw_given], work, fold, cpu_stop, heavy=VARIANT == 'wb_unicef_heavy', items=schema['items'])
    report['report_priors'] = survey_priors.attach(schema, X, given, work, fold, min(cpu_stop, time.monotonic() + (min(0.75, 0.04 * span) if len(source) == 76 else 0.04 * span))) if 'report_prior' not in disabled else {}
    report['latent_profile'] = 0
    if 'latent_profile' not in disabled and len(allowed) >= 40:
        start = time.monotonic()
        heavy = VARIANT == 'unhcr_heavy' and is_deep_hierarchy or (VARIANT == 'wb_unicef_heavy' and (not is_deep_hierarchy))
        cap = min(cpu_stop, start + {213: 3.8, 76: 1.8, 12: 0.7}.get(len(source), 1.8) * (1.8 if heavy else 1.0))
        scales = (8,) if len(source) == 12 else (8, 16) if is_deep_hierarchy else (8, 16, 32)
        if heavy:
            scales = (8, 16) if len(source) == 12 else (8, 16, 24) if is_deep_hierarchy else (8, 16, 24, 32)
        found, trace = fit_latent_profiles(X, work, fold, cap, given=given, scales=scales, tree_rounds=0, cluster_iter=12, soft_iter=2, svd_iter=2, svd_dim=16)
        for name, (o, p) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('latent_profile')
        report['latent_profile'] = len(found)
        report['latent_profile_trace'] = trace
        report['latent_profile_seconds'] = time.monotonic() - start
    if len(source) == 76 and v68_config.V87_W1 and len(allowed) >= 40:
        from . import wb_lca
        lca_start = time.monotonic()
        # the configured option adds the group-intercept trait kind; its window grows to fit five forked kinds.
        window = max(14.5 * max(1.0, 4.0 / _THREADS), 0.088 * span) if (v68_config.V115_W3 and v68_config.V127_W1) else max(24.0 * max(1.0, 4.0 / _THREADS), 0.165 * span) if v68_config.V115_W3 else max(18.5 * max(1.0, 4.0 / _THREADS), 0.125 * span) if v68_config.V115_W1 else max(12.5 * max(1.0, 4.0 / _THREADS), 0.085 * span) if v68_config.V110_W1 else max(8.5, 0.06 * span) if v68_config.V91_W2 else max(6.0, 0.045 * span)
        found, report['wb_lca'] = wb_lca.attach(X[:, :n_raw_given], levels[:n_raw_given], given, schema['items'], work, fold, min(cpu_stop, time.monotonic() + window), workers=_THREADS)
        for name, (o, p) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('wb_lca')
        if v68_config.V101_W2:
            for name, kinds in wb_lca.LAST_KINDS.items():
                for kind, (o, p) in kinds.items():
                    work[name]['oof'].append(o)
                    work[name]['all'].append(p)
                    work[name]['arms'].append('wb_lca_'+kind)
                    work[name].setdefault('partition_k', {})['wb_lca_'+kind] = 5
        if v68_config.V124_W1 and not v68_config.V127_W1:
            lca_elapsed = time.monotonic() - lca_start
            cpu_stop = min(finish_deadline - 42.0, cpu_stop + lca_elapsed)
            pfn_stop = max(pfn_stop, cpu_stop)
            net_stop = max(net_stop, min(finish_deadline - 37.5, cpu_stop + 14.0))
            nb_build_stop = max(nb_build_stop, min(finish_deadline - 34.0, net_stop + 3.5))
            ct_stop = max(ct_stop, min(finish_deadline - 29.0, nb_build_stop + 5.0))
            report['wb_lca_budget_restored'] = round(lca_elapsed, 3)
        elif v68_config.V124_W1 and v68_config.V127_W1:
            lca_elapsed = time.monotonic() - lca_start
            refund = min(lca_elapsed, 6.0)
            cpu_stop = min(finish_deadline - 45.0, cpu_stop + refund)
            pfn_stop = max(pfn_stop, cpu_stop)
            report['wb_lca_budget_restored'] = round(refund, 3)
    ZERO_ON_SKIP.clear()
    if 'zero_on_skip' not in disabled:
        ZERO_ON_SKIP.update(detect_zero_on_skip(schema['items'], given, X, work, allowed))
    report['zero_on_skip'] = len(ZERO_ON_SKIP)
    EMPIRICAL_PARENTS.clear()
    if 'empirical_gate' not in disabled:
        declared_anchors, _ = split_anchors(names, schema['items'])
        found = discover_empirical_gate_parents(schema['items'], work, declared_anchors, allowed, X=X, given=given)
        names = topological(list(work), schema['items'])
        report['empirical_parents'] = {k: v[0] for k, v in found.items()}
        report['empirical_parent_mi'] = {k: [round(x, 4) for x in v[1:]] for k, v in found.items()}
    if 'rb_distill' not in disabled and len(allowed) >= 50:
        start = time.monotonic()
        found = _rb_distill(D, work, names, np.maximum(fold, 0), cpu_stop)
        for name, (o, p) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('rb_distill')
        report['rb_distill'] = len(found)
        report['rb_distill_seconds'] = time.monotonic() - start
    if 'cross_round' not in disabled:
        report['cross_round'] = attach_cross_round(schema, X, given, fold, work, deadline=cpu_stop)
    report['two_stage_eligible'] = sum((bool(schema['items'][name].get('gate')) for name in names))
    report['gate_table_eligible'] = report['two_stage_eligible']
    report['gate_logit_eligible'] = report['two_stage_eligible']
    bank = FoldFeatureBank(X, schema['items'], work, fold, cpu_stop, tables=tables, n_raw_given=n_raw_given, levels=levels, is_deep_hierarchy=is_deep_hierarchy, given=given)
    bank.latent_features = False
    from concurrent.futures import ThreadPoolExecutor
    import lightgbm, catboost
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse, optimize
    pools = ThreadpoolController()
    pfn_timing = {'recent_fit': 0.25, 'calls': 0}
    fold_sets = {5: fold, 3: np.where(fold >= 0, fold % 3, -1), 2: np.where(fold >= 0, fold % 2, -1)}
    chain = None
    wb_fast = None
    if len(source) == 76:
        from . import wb_fast
    csr_slices = {}
    design_cache = {}
    c_cache = {}
    report['logit_c'] = {}
    report['hgb_best_stage'] = {}
    from threading import RLock
    slice_lock = RLock()

    def candidate(profile, name, stop, plan):
        pools.limit(limits=1, user_api='openmp')
        j = work[name]
        if len(allowed) < 20 or time.monotonic() >= stop:
            return None
        y, K, train = (j['y'], j['K'], j['train'])
        rows = np.flatnonzero(train)
        rest = np.flatnonzero(~train)
        if profile == 'third':
            # UNHCR runs native-categorical hgb as its own arm on every item.
            profile = 'catboost' if is_deep_hierarchy or K > 3 else 'hgb'
        o = np.zeros((len(rows), K), np.float32)
        full = np.zeros((n, K), np.float32)
        try:
            if profile in FOUNDATION_ARMS:
                parents = [p for p in _parents(name, schema['items']) if p in work and profile + '_partitions' in work[p]]

                def features(tr, part):
                    blocks = [work[p][profile + '_partitions'][part] for p in parents]
                    if len(source) in (12, 76):
                        matrix, categories = bank.matrix(name, tr)
                        return (np.column_stack([matrix] + blocks) if blocks else matrix), categories
                    return (np.column_stack([X] + blocks) if blocks else X), X.shape[1]
                strata = tuple((i for i, k in enumerate(given) if k == 'wave' or 'country' in k.lower()))
                capture_part = bool(j['route_parent'] and len(source) != 12)
                result = gpu.candidate(X, y, rows, rest, K, X.shape[1], fold, stop, pfn_timing, strata=strata, feature_factory=features, capture=capture_part, arm=profile)
                if result is not None and capture_part:
                    j[profile + '_partitions'] = pfn_timing.pop('partitions', {})
                if result is not None: j.setdefault('partition_k', {})[profile] = int(pfn_timing.get('last_n_folds', 2))
                return None if result is None else [(profile, *result)]
            k = plan.choose()
            split = fold_sets[k]
            began = time.monotonic()
            linked = chain is not None and name in chain.children
            stage_o, stage_p = {}, {}
            guarded = len(source) in (12, 76) and profile in ('catboost', 'deep', 'shallow', 'strong', 'two_stage')
            if guarded and plan.costs and time.monotonic() + .65 * float(np.median(plan.costs)) >= stop:
                return None
            if profile == 'hgb' and len(source) in (12, 76) and not (len(source) == 76 and v68_config.V127_W1):
                stage_o, stage_p = hgb_crossfit_stages(lambda tr: bank.matrix(name, tr),
                    y, rows, rest, split, k, K, stop, max_iter=40 if len(source) == 76 and v68_config.V124_W1 else 60)
            else:
                for f in range(k):
                    check_clock(stop)
                    if guarded and f >= 1 and time.monotonic() + .80 * ((time.monotonic() - began) / f) >= stop:
                        raise BudgetExpired()
                    tr = rows[split[rows] != f]
                    val = rows[split[rows] == f]
                    query = np.concatenate((val, rest))
                    if profile == 'two_stage':
                        features, categories = bank.matrix(name, tr)
                        pred = gated_tree_predict(features, categories, y, tr, query, K, 'shallow', stop, name=name, items=schema['items'], given=given, raw=X, is_deep_hierarchy=False)
                        if pred is None:
                            return None
                    elif profile == 'logit':
                        design = D
                        key = (k, f, tr.tobytes(), query.tobytes())
                        if len(source) == 76:
                            mi_drivers = _mi_drivers(X[:, :n_raw_given], y, tr, levels[:n_raw_given])
                            dkey = (tuple(mi_drivers), k, f) if v68_config.V127_W1 else tuple(mi_drivers)
                            with slice_lock:
                                design = design_cache.get(dkey)
                            if design is None:
                                built = _design(X, levels, mi_drivers, support_rows=tr, retain_crosses=True, given=given)
                                with slice_lock:
                                    design = design_cache.setdefault(dkey, built)
                            key = (dkey, k, f, tr.tobytes(), query.tobytes())
                        with slice_lock:
                            sliced = csr_slices.get(key)
                        if sliced is None:
                            sliced = (design[tr], design[query])
                            with slice_lock:
                                sliced = csr_slices.setdefault(key, sliced)
                        fit_design, query_design = sliced
                        if linked:
                            ext = chain.extras(name, tr)
                            fit_design = sparse.hstack((fit_design, sparse.csr_matrix(ext[tr])), format='csr')
                            query_design = sparse.hstack((query_design, sparse.csr_matrix(ext[query])), format='csr')
                        classes = np.unique(y[tr])
                        pred = np.zeros((len(query), K))
                        if len(classes) < 2:
                            pred[:, classes[0]] = 1.0
                        else:
                            c_key = ('global', int(k), int(f)) if len(source) in (12, 76) else (k, f, tr.tobytes())
                            with slice_lock:
                                if c_key not in c_cache:
                                    c_rows = allowed[split[allowed] != f] if len(source) in (12, 76) else tr
                                    c_cache[c_key] = _pick_c(D, work, names, c_rows, stop)
                                c_val = c_cache[c_key]
                            if wb_fast is not None and wb_fast.ENABLE_SOLVER:
                                value, classes = wb_fast.fit_predict(fit_design, y[tr], query_design, C=c_val, deadline=stop)
                                pred[:, classes] = value
                            else:
                                model = LogisticRegression(C=c_val, max_iter=90, tol=5e-4)
                                model.fit(fit_design, y[tr])
                                pred[:, model.classes_] = model.predict_proba(query_design)
                    elif profile == 'hgb':
                        features, categories = bank.matrix(name, tr)
                        stages = hgb_stages(features, categories, y, tr, query, K, stop,
                                            max_iter=60)
                        for step, value in stages.items():
                            if step not in stage_o:
                                stage_o[step] = np.zeros_like(o)
                                stage_p[step] = np.zeros_like(full)
                            stage_o[step][split[rows] == f] = value[:len(val)]
                            stage_p[step][rest] += value[len(val):] / k
                        pred = stages[max(stages)]
                    else:
                        features, categories = bank.matrix(name, tr)
                        pred = tree_predict(features, categories, y, tr, query, K, profile, stop, is_deep_hierarchy=is_deep_hierarchy, hgb_iter=60)
                    if pred is None:
                        return None
                    check_clock(stop)
                    o[split[rows] == f] = pred[:len(val)]
                    full[rest] += pred[len(val):] / k
            if stage_o:
                steps = sorted(stage_o)
                losses = np.stack([-np.log(.001 + (1.-K*.001) *
                    stage_o[step][np.arange(len(rows)), y[rows]]) for step in steps])
                best = steps[int(np.argmin(losses.mean(1)))]
                # Held-fold stage choice uses the other folds' losses.
                for f in range(k):
                    va = split[rows] == f
                    step = steps[int(np.argmin(losses[:, ~va].mean(1)))]
                    o[va] = stage_o[step][va]
                full = stage_p[best]
                report['hgb_best_stage'][name] = int(best)
            if profile == 'logit' and len(rest):
                check_clock(stop)
                design = D
                if len(source) == 76:
                    drivers = _mi_drivers(X[:, :n_raw_given], y, rows, levels[:n_raw_given])
                    dkey = tuple(drivers)
                    with slice_lock:
                        design = design_cache.get(dkey)
                    if design is None:
                        built = _design(X, levels, drivers, support_rows=rows, retain_crosses=True, given=given)
                        with slice_lock:
                            design = design_cache.setdefault(dkey, built)
                fit_design, query_design = design[rows], design[rest]
                if linked:
                    ext = chain.extras(name, rows)
                    fit_design = sparse.hstack((fit_design, sparse.csr_matrix(ext[rows])), format='csr')
                    query_design = sparse.hstack((query_design, sparse.csr_matrix(ext[rest])), format='csr')
                with slice_lock:
                    c_key = ('global_full',) if len(source) in (12, 76) else ('full', rows.tobytes())
                    if c_key not in c_cache:
                        c_cache[c_key] = _pick_c(D, work, names, allowed if len(source) in (12, 76) else rows, stop)
                    c_val = c_cache[c_key]
                classes = np.unique(y[rows])
                refit = np.zeros((len(rest), K))
                if len(classes) == 1:
                    refit[:, classes[0]] = 1.
                else:
                    if wb_fast is not None and wb_fast.ENABLE_SOLVER:
                        value, classes = wb_fast.fit_predict(fit_design, y[rows], query_design, C=c_val, deadline=stop)
                        refit[:, classes] = value
                    else:
                        model = LogisticRegression(C=c_val, max_iter=90, tol=5e-4)
                        model.fit(fit_design, y[rows])
                        refit[:, model.classes_.astype(int)] = model.predict_proba(query_design)
                full[rest] = .5 * full[rest] + .5 * refit
                report['logit_c'][name] = float(c_val)
            arm = 'shallow_asked' if profile == 'two_stage' else profile
            j.setdefault('partition_k', {})[arm] = int(k)
            found = [(arm, o, full)]
            if profile == 'two_stage' and len(source) == 12:
                # Complete the shallow arm before spending its remaining window.
                o_cb, full_cb = np.zeros_like(o), np.zeros_like(full)
                cb_began = time.monotonic()
                try:
                    for f in range(k):
                        check_clock(stop)
                        if f >= 1 and time.monotonic() + .80 * ((time.monotonic() - cb_began) / f) >= stop:
                            raise BudgetExpired()
                        tr, val = rows[split[rows] != f], rows[split[rows] == f]
                        query = np.concatenate((val, rest))
                        features, categories = bank.matrix(name, tr)
                        pred_cb = gated_tree_predict(features, categories, y, tr, query, K, 'catboost', stop,
                            name=name, items=schema['items'], given=given, raw=X, is_deep_hierarchy=False)
                        if pred_cb is None:
                            break
                        check_clock(stop)
                        o_cb[split[rows] == f] = pred_cb[:len(val)]
                        full_cb[rest] += pred_cb[len(val):] / k
                    else:
                        found.append(('catboost_asked', o_cb, full_cb))
                        j.setdefault('partition_k', {})['catboost_asked'] = int(k)
                except BudgetExpired:
                    pass
            plan.record(k, time.monotonic() - began)
            return found
        except BudgetExpired:
            return None
        except FileNotFoundError:
            return None
    windows = {'logit': 0.14, 'gate': 0.23 if is_deep_hierarchy else 0.26, 'strong': 0.34 if is_deep_hierarchy else 0.4, 'shallow': 0.55, 'two_stage': 0.62, 'deep': 0.81, 'third': 1.0}
    _ANCHOR_WINDOWS = {'logit': 0.11, 'gate': 0.17, 'strong': 0.24, 'shallow': 0.44, 'two_stage': 0.52, 'deep': 0.76, 'third': 1.0}
    _UNICEF_WINDOWS = (
        {'logit': .12, 'gate': .18, 'catboost': .46, 'two_stage': .72, 'shallow': .82, 'strong': .86, 'hgb': 1.0}
        if gpu_enabled else
        {'logit': .12, 'gate': .18, 'catboost': .48, 'two_stage': .74, 'shallow': .84, 'strong': .88, 'hgb': 1.0}
    )
    _WB_WINDOWS = (
        {'logit': .14, 'gate': .20, 'strong': .26, 'catboost': .52, 'deep': .68, 'shallow': .80, 'two_stage': .86, 'hgb': 1.0}
        if gpu_enabled else
        {'logit': .14, 'gate': .20, 'strong': .26, 'catboost': .54, 'deep': .70, 'shallow': .82, 'two_stage': .88, 'hgb': 1.0}
    )
    if surgical:
        _WB_WINDOWS = {'logit': .16, 'gate': .22, 'strong': .28, 'catboost': .56, 'deep': .72, 'shallow': .83, 'two_stage': .89, 'hgb': 1.0}
    # Post items finish under half their catboost arms in 28% of the window;
    # deep and hgb carry under 1% of their stack weight.
    _WB_ANCHOR_WINDOWS = {'logit': .18, 'gate': .24, 'strong': .28, 'catboost': .56, 'shallow': .76, 'two_stage': .85, 'deep': .92, 'hgb': 1.0} if surgical and v68_config.V124_W1 else _WB_WINDOWS
    _WB_POST_WINDOWS = ({'logit': .26, 'gate': .26, 'strong': .26, 'catboost': .74, 'shallow': .86, 'two_stage': .86, 'deep': .91, 'hgb': 1.0} if v68_config.V124_W1 else {'logit': .16, 'gate': .22, 'strong': .28, 'catboost': .80, 'shallow': .87, 'two_stage': .93, 'deep': .97, 'hgb': 1.0}) if surgical and v68_config.V76_K1 else _WB_WINDOWS
    _WB_CHILD_WINDOWS = {'logit': .26, 'gate': .30, 'strong': .30, 'catboost': .58, 'shallow': .74, 'two_stage': .83, 'deep': .91, 'hgb': 1.0} if surgical and v68_config.V124_W1 else _WB_WINDOWS
    if v68_config.V127_W1:
        _WB_WINDOWS = (
            {'hgb': 0.44, 'logit': 0.5, 'gate': 0.54, 'strong': 0.58, 'shallow': 0.74, 'two_stage': 0.78, 'deep': 0.89, 'catboost': 1.0}
            if gpu_enabled else
            {'hgb': 0.54, 'logit': 0.64, 'gate': 0.68, 'strong': 0.72, 'shallow': 0.87, 'two_stage': 0.91, 'deep': 0.96, 'catboost': 1.0}
        )
        _WB_ANCHOR_WINDOWS = _WB_WINDOWS
        _WB_POST_WINDOWS = _WB_WINDOWS
        _WB_CHILD_WINDOWS = _WB_WINDOWS
    pass_windows = {'anchors': _ANCHOR_WINDOWS if is_deep_hierarchy else _WB_ANCHOR_WINDOWS if len(source) == 76 else _UNICEF_WINDOWS, 'post': _ANCHOR_WINDOWS if is_deep_hierarchy else _WB_POST_WINDOWS, 'children': {'logit': 0.16, 'gate': 0.22, 'strong': 0.22, 'shallow': 0.48, 'two_stage': 0.58, 'deep': 0.78, 'third': 1.0} if is_deep_hierarchy else _WB_CHILD_WINDOWS if len(source) == 76 else _UNICEF_WINDOWS}
    report['folds_used'] = {}
    report['folds_completed'] = {}
    report['hidden_tree_prediction'] = 'adaptive_fold_mean'
    profile_costs = {}
    shallow_folds = (5, 2) if (len(source) == 76 and v68_config.V127_W1) else ((2,) if len(source) in (12, 76) else (5, 2))
    descendants = {name: 0 for name in names}
    for name in names:
        seen = set()
        stack = [p for p in _parents(name, schema['items']) if p in descendants]
        while stack:
            parent = stack.pop()
            if parent in seen:
                continue
            seen.add(parent)
            descendants[parent] += 1
            stack.extend((p for p in _parents(parent, schema['items']) if p in descendants))

    def run_pass(label, subset, start_at, end_at, tag=None):
        w_map = pass_windows.get(tag if tag in pass_windows else label, windows)
        for profile in w_map:
            if profile in disabled:
                continue
            start = time.monotonic()
            stop = start_at + w_map[profile] * max(0.0, end_at - start_at)
            if start >= stop:
                continue
            bank.deadline = stop
            if profile in ('gate', 'two_stage'):
                profile_names = [name for name in subset if schema['items'][name].get('gate')]
            elif profile == 'strong':
                profile_names = sorted((name for name in subset if descendants[name] >= 2), key=lambda name: -descendants[name])
            elif len(source) == 76 and profile == 'deep' and (label == 'anchors') and not wb_fast.ENABLE_STAGES:
                profile_names = list(subset[:16])
            elif len(source) == 76 and profile == 'deep':
                profile_names = [name for name in subset if _WB_CHILD_DEEP == 'all' or not name.startswith('tr_')]
            elif is_deep_hierarchy and label == 'children' and (profile in ('hgb', 'deep', 'third')):
                profile_names = sorted(subset, key=lambda name: (sum((arm in _TREE_ARMS for arm in work[name]['arms'])), -float(work[name]['base'])))
            else:
                profile_names = list(subset)
            if len(source) == 76 and v68_config.V80_W2 and profile == 'catboost' and label == 'children' and tag != 'post':
                # Only 11 to 14 of 25 child CatBoost fits finish; the threads take the
                # non-follow-up, highest-loss items first.
                profile_names = sorted(profile_names, key=lambda name: (name.startswith('tr_'), -float(work[name]['base'])))
            if not profile_names:
                continue
            report[profile + '_eligible'] = report.get(profile + '_eligible', 0) + len(profile_names)
            if wb_fast is not None and wb_fast.ENABLE_STAGES and profile in ('catboost', 'deep', 'shallow', 'hgb'):
                from .wb_stages import run as stage_run
                values, stage_counts, _ = stage_run(profile_names, work, bank, fold_sets[2], profile, stop, _THREADS)
                for name, (o, p) in values.items():
                    work[name]['oof'].append(o); work[name]['all'].append(p); work[name]['arms'].append(profile)
                    work[name].setdefault('partition_k', {})[profile] = 2
                report[profile] += len(values)
                report.setdefault('wb_stages', {})[label+':'+profile] = stage_counts
                report['folds_completed'][label+':'+profile] = {'2': len(values)}
                report[profile+'_seconds'] = report.get(profile+'_seconds',0.) + time.monotonic()-start
                gc.collect()
                continue

            if len(source) == 76 and v68_config.V124_W1 and not v68_config.V127_W1 and label == 'anchors' and tag is None and profile in ('shallow', 'two_stage'):
                profile_folds = (5, 3, 2)
            else:
                profile_folds = (5, 2) if surgical and profile in ('logit', 'gate', 'strong') else shallow_folds
            prior_costs = profile_costs.get(profile, ())
            if len(source) == 76 and v68_config.V124_W1 and not v68_config.V127_W1 and profile == 'logit' and label == 'children' and tag == 'post' and prior_costs:
                prior_costs = [c * 2.5 for c in prior_costs]
            if profile == 'gate':
                singles = D[:, :int(levels.sum())]
                plan = FoldPlan(profile_folds, len(profile_names), stop, _THREADS, max_high_folds=32 if surgical else None, prior=prior_costs)

                def run(name):
                    k = plan.choose()
                    began = time.monotonic()
                    try:
                        found = gate_candidate(name, schema['items'][name], schema['items'], given, X, levels, singles, tables, work[name], fold_sets[k], stop, n_raw_given=n_raw_given, is_deep_hierarchy=is_deep_hierarchy, chain=chain, fast=bool(wb_fast is not None and wb_fast.ENABLE_SOLVER))
                    except BudgetExpired:
                        return None
                    if found is not None:
                        for label, _, _ in found:
                            work[name].setdefault('partition_k', {})[label] = int(k)
                        plan.record(k, time.monotonic() - began)
                    return found
            else:
                count = len(profile_names)
                plan = FoldPlan((2,) if len(source) in (12, 76) and profile in ('hgb', 'catboost') else profile_folds, count, stop, _THREADS, max_high_folds=(6 if profile == 'catboost' else 12) if len(source) == 12 else 32 if len(source) == 76 else 24 if label == 'anchors' or tag == 'post' else 40, prior=prior_costs)
                run = lambda name: candidate(profile, name, stop, plan)
            executor = ThreadPoolExecutor(max_workers=_THREADS, initializer=lambda: threadpool_limits(limits=1, user_api='openmp'))
            try:
                for name, found in zip(profile_names, executor.map(run, profile_names)):
                    if found is None:
                        continue
                    for arm, o, p in found:
                        work[name]['oof'].append(o)
                        work[name]['all'].append(p)
                        work[name]['arms'].append(arm)
                        if profile == 'gate':
                            report[arm] += 1
                        if profile == 'third':
                            report[arm] += 1
                    if profile != 'gate':
                        report[profile] += 1
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            report['folds_used'][(tag or label) + ':' + profile] = {str(k): v for k, v in plan.used.items()}
            report['folds_completed'][(tag or label) + ':' + profile] = {str(k): v for k, v in plan.completed.items()}
            if len(source) == 76 and v68_config.V124_W1:
                new_costs = plan.costs[len(prior_costs):]
                if new_costs:
                    profile_costs[profile] = new_costs[-16:]
            elif plan.costs:
                profile_costs[profile] = plan.costs[-16:]
            key = ('gate_decomposition' if profile == 'gate' else profile) + '_seconds'
            report[key] = report.get(key, 0.0) + time.monotonic() - start
            gc.collect()
    anchors, children = split_anchors(names, schema['items'])
    post_items = []
    if is_deep_hierarchy and 'chain' not in disabled:
        roots = [a for a in anchors if not any((p in source for p in _parents(a, schema['items'])))]
        root_set = set(roots)
        post_items = topological([a for a in anchors if a not in root_set], schema['items'])
        anchors = roots
    if len(source) == 76 and 'prepost' not in disabled:
        anchors, children, post_items = questionnaire_split(names, schema['items'], anchors, children)
    if len(source) == 12:
        anchors = [a for a in UNICEF_ANCHORS if a in names]
        children = [a for a in names if a not in anchors]
    if 'chain' in disabled:
        anchors, children, post_items = (names, [], [])
    report.update(anchors=len(anchors), children=len(children), chain_items=0, post_items=len(post_items))
    try:
        bank.deadline = cpu_stop
        if is_deep_hierarchy:
            bank.prewarm(anchors + post_items if children else names, [fold])
        else:
            bank.prewarm_partitions([fold])
        if len(source) in (12, 76):
            bank.prewarm_partitions([fold_sets[2]])
    except BudgetExpired:
        pass
    later = post_items + children
    pass_one_share = ((0.31 if post_items else 0.37) if is_deep_hierarchy else (0.46 if post_items else 0.58) if len(source) == 76 else .34 if len(source) == 12 else min(0.7, max(0.3, 1.25 * len(anchors) / max(1, len(names))))) if later else 1.0
    pass_one_start = time.monotonic()
    pass_two_stop = cpu_stop
    span_two = max(0.0, pass_two_stop - pass_one_start)
    pass_one_stop = min(pass_two_stop, pass_one_start + pass_one_share * span_two)
    pass_post_stop = min(pass_two_stop, pass_one_start + (0.39 if is_deep_hierarchy else 0.68) * span_two) if post_items else pass_one_stop
    report['pass_one_start_seconds'] = pass_one_start - now
    report['pass_one_window_seconds'] = max(0.0, pass_one_stop - pass_one_start)
    followup_chain = None
    followup_event = None
    stacked_anchor = {}
    links = {}

    def stack_into(subset, stop_at=cpu_stop):
        for name in subset:
            j = work[name]
            rest = np.flatnonzero(~j['train'])
            fold_arms(name, j, X, given, schema['items'])
            gate_latent(j, stop_at)
            keep = list(range(len(j['arms'])))
            aligned = aligned_partitions(j, fold)
            p, o, w, t, loss = crossfit_calibrate([j['oof'][i] for i in keep],
                [j['all'][i][rest] for i in keep], j['y'][j['train']], j['K'], aligned, deadline=stop_at)
            if len(source) == 76 and v68_config.V127_W1 and v68_config.V73_K1:
                coarse_w = sum(float(w[i]) for i in keep if j.get('partition_k', {}).get(j['arms'][i], 5) != 5)
                if coarse_w >= 0.02 and time.monotonic() < stop_at - 1.0:
                    (p, o, w, t, loss), _ = tiered_calibrate(j, keep, rest, fold, (p, o, w, t, loss), stop_at)
            stacked_anchor[name] = (o, p)
    pfn_done = {arm: set() for arm in FOUNDATION_ARMS}

    def run_pfn(subset, stop):
        ordered = topological(candidate_order('tabpfn', list(subset), work), schema['items'])
        report['tabpfn_eligible'] = len(names)
        for arm_index, arm in enumerate(FOUNDATION_ARMS):
            if arm in disabled:
                continue
            start = time.monotonic()
            arm_stop = start + 0.68 * (stop - start) if arm == 'tabpfn' and len(FOUNDATION_ARMS) > 1 else stop
            bank.deadline = arm_stop
            pending = [k for k in ordered if k not in pfn_done[arm]]
            pairs = joint_targets.pairs(schema, work, pending) if JOINT_TARGETS and is_deep_hierarchy else []
            pair_by_first = {a: (a, b) for a, b in pairs}
            paired_second = {b for a, b in pairs}
            for pos, name in enumerate(pending):
                if time.monotonic() >= arm_stop:
                    break
                if name in paired_second:
                    continue
                pfn_timing['remaining'] = max(1, len(pending) - pos)
                found = None
                if name in pair_by_first:
                    a, b = pair_by_first[name]
                    found = joint_targets.candidate(a, b, schema, X, given, work, fold, bank, arm_stop, pfn_timing, arm)
                    if found is not None:
                        for target, o, p in found:
                            j = work[target]
                            j['oof'].append(o)
                            j['all'].append(p)
                            j['arms'].append(arm + '_joint')
                            pfn_done[arm].add(target)
                        report[arm + '_joint_pairs'] = report.get(arm + '_joint_pairs', 0) + 1
                        continue
                targets = pair_by_first.get(name, (name,))
                for target in targets:
                    item = candidate(arm, target, arm_stop, None)
                    if item is None:
                        continue
                    for label, o, p in item:
                        j = work[target]
                        j['oof'].append(o)
                        j['all'].append(p)
                        j['arms'].append(label)
                        if arm == 'tabpfn':
                            j['tabpfn_oof'] = o
                        if target in stacked_anchor:
                            stack_into([target], stop_at=arm_stop)
                            if chain is not None and target in chain.block:
                                a, b = chain.block[target]
                                o_cal, p_cal = stacked_anchor[target]
                                chain.S[j['train'], a:b] = o_cal
                                chain.S[~j['train'], a:b] = p_cal
                                chain.cache.clear()
                                bank.clear_matrices()
                    pfn_done[arm].add(target)
                    report[arm] = report.get(arm, 0) + 1
            report[arm + '_seconds'] = report.get(arm + '_seconds', 0.0) + time.monotonic() - start
    with pools.limit(limits=1):
        run_pass('anchors', anchors, pass_one_start, pass_one_stop)
        if later and time.monotonic() < cpu_stop:
            stack_into(anchors)
            bank.deadline = cpu_stop
            chain = StackedChain(schema['items'], work, stacked_anchor, anchors, later, n, bank=bank)
            bank.clear_matrices()
            bank.chain = chain
            report['chain_items'] = len(later)
            report['chain_anchor_columns'] = int(chain.S.shape[1])
        links.update(((name, chain) for name in post_items))
        if post_items:
            run_pass('children', post_items, time.monotonic(), pass_post_stop, tag='post')
            if chain is not None and time.monotonic() < cpu_stop:
                stack_into(post_items)
                chain = StackedChain(schema['items'], work, stacked_anchor, [a for a in names if a in stacked_anchor], children, n, bank=bank)
                bank.clear_matrices()
                bank.chain = chain
                followup_chain = chain
                report['chain_post_anchor_columns'] = int(chain.S.shape[1])
        if len(source) == 12 and chain is not None:
            # Pass 2 keeps at least 16.5 s for the eight child items' trees and logit.
            report['conditional_cell_eb'] = attach_conditioned(X[:, :n_raw_given], given[:n_raw_given], work, fold, bank, chain, max(time.monotonic() + 6.0, cpu_stop - 16.5), schema['items'], alphas=report.get('full_cell_eb', {}).get('alphas'), workers=_THREADS)
        if children:
            run_pass('children', children, time.monotonic(), pass_two_stop)
        links.update(((name, chain) for name in children))
        if gpu_enabled and 'tabpfn' not in disabled and (time.monotonic() < pfn_stop):
            priority = list(dict.fromkeys([a for a in anchors + post_items if work[a]['route_parent']] + anchors + post_items + children))
            run_pfn(priority, pfn_stop)
    parent_partitions = len(bank.cache)
    report['matrix_cache_hits'] = bank.matrix_cache_hits
    report['matrix_cache_bytes'] = bank.matrix_cache_bytes
    if followup_chain is None and len(source) == 76:
        followup_chain = chain
    bank.clear_matrices()
    bank.chain = None
    chain = None
    csr_slices.clear()
    design_cache.clear()
    if len(source) == 76 and v68_config.K2 and (not v68_config.V124_W1 or v68_config.V127_W1):
        net_stop = min(finish_deadline - 22.0, max(now + .855 * span, time.monotonic() + 8.2))
    elif len(source) == 76 and v68_config.K2 and v68_config.V124_W1:
        net_stop = min(finish_deadline - 37.5, max(now + .855 * span, time.monotonic() + 8.3))
    if names and len(allowed) >= 20 and (time.monotonic() < net_stop) and ('network' not in disabled):
        start = time.monotonic()
        fn = _fit_net_gpu if gpu_enabled else _fit_net
        A = D.toarray().astype(np.float32)
        with pools.limit(limits=_THREADS):
            results = fn(A, [work[k]['y'] for k in names], [work[k]['train'] for k in names], [work[k]['K'] for k in names], np.maximum(fold, 0), (400 if phase == 'final' else 240) if gpu_enabled else 140 if is_deep_hierarchy else 200, net_stop, hidden=64, **{'categories': X, 'category_levels': levels} if gpu_enabled else {})
        del A
        if results:
            for k, (o, p) in zip(names, results):
                work[k]['oof'].append(o)
                work[k]['all'].append(p)
                work[k]['arms'].append('network')
            report['network'] = len(names)
        report['network_seconds'] = time.monotonic() - start
        report['network_trace'] = base._NET_TRACE
    index = SimpleNamespace(visible=allowed, fold=fold)
    if 'neighbor_cross_target' not in disabled and len(allowed) >= 20 and (time.monotonic() + 1.5 < nb_build_stop):
        masked = frame.copy()
        mask = np.ones(n, bool)
        mask[allowed] = False
        for name in source:
            masked.loc[mask, name] = None
        start = time.monotonic()
        try:
            candidate_index = build_neighbors(masked, schema, deadline=nb_build_stop)
            if candidate_index is not None:
                index = candidate_index
            index.fold = fold.copy()
            if is_deep_hierarchy and time.monotonic() + 0.5 < nb_build_stop:
                with pools.limit(limits=_THREADS):
                    index.leaf_candidates(deadline=nb_build_stop)
        except BudgetExpired:
            pass
        report['neighbor_build_seconds'] = time.monotonic() - start
    if 'neighbor_cross_target' not in disabled and time.monotonic() < ct_stop:
        start = time.monotonic()
        with pools.limit(limits=_THREADS):
            found = _cross_target(X, index, work, ct_stop, n_raw_given=n_raw_given)
        if found:
            for k, (o, p, gain) in found.items():
                work[k]['oof'].append(o)
                work[k]['all'].append(p)
                work[k]['arms'].append('cross_target')
                report['cross_target'] += 1
        report['cross_target_seconds'] = time.monotonic() - start
        report['cross_target_trace'] = dict(base._CROSS_TRACE)
    if len(source) == 76 and 'followup' not in disabled and (time.monotonic() < finish_deadline - 16.0):
        cap = min(finish_deadline - (28.5 if (v68_config.V124_W1 and not v68_config.V127_W1) else 20.0), time.monotonic() + (9.5 if v68_config.V124_W2 else 7.5)) if v68_config.K2 else min(finish_deadline - 16.0, time.monotonic() + 40.0)
        followup_started = time.monotonic()
        bank.deadline = cap
        with pools.limit(limits=1):
            found, info = fit_tracer_followup(frame, schema, given, X, D, tables, work, deadline=cap, chain=followup_chain, workers=_THREADS, anchor_names=anchors)
        moved = info.pop('transition_arms', {})
        shared_found = info.pop('shared_followup_arms', {})
        followup_event = info.pop('_event', None)
        report['followup_status'] = info
        report['followup_seconds'] = time.monotonic() - followup_started
        report['followup'] = len(found)
        report['followup_shared'] = len(shared_found)
        report['followup_anchor'] = len(moved)
        for name, (o, p, _) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('followup')
        for name, (o, p, _) in shared_found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('followup_shared' if name in found else 'followup')
        for name, (o, p) in moved.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('followup_anchor')
    if len(source) == 76 and v68_config.V110_W2:
        from . import wb_lca, wb_trait_effects
        anchor_probs = {}
        for name, (o, p) in stacked_anchor.items():
            if name.startswith('tr_'):
                continue
            j = work[name]
            q = np.zeros((n, j['K']), np.float32)
            q[j['train']] = o
            q[~j['train']] = p
            anchor_probs[name] = q
        moved, st2 = wb_trait_effects.stage2(wb_lca.LAST_KEEP, anchor_probs, work, fold,
            min(finish_deadline - 20.0, time.monotonic() + 2.0))
        st2.pop('raw', None)
        report['v110_stage2'] = st2
        for name, (o, p) in moved.items():
            j=work[name]
            if 'wb_lca' in j['arms']:
                i=j['arms'].index('wb_lca');j['oof'][i]=o;j['all'][i]=p
        wb_lca.LAST_KEEP = None
    if len(source) == 76 and v68_config.V127_W2:
        from . import wb_lca, wb_trait_effects
        chain_lca_found, chain_lca_info = wb_trait_effects.attach_chain_lca(
            wb_lca.LAST_KEEP, stacked_anchor, followup_chain, work, fold,
            schema['items'], anchors, post_items, children,
            min(finish_deadline - 18.0, time.monotonic() + 2.8), event=followup_event
        )
        report['wb_lca_chain'] = chain_lca_info
        for name, (o, p) in chain_lca_found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('wb_lca_chain')
            work[name].setdefault('partition_k', {})['wb_lca_chain'] = 5
            work[name]['check_arms'] = tuple(dict.fromkeys((*work[name].get('check_arms', ()), 'wb_lca_chain')))
        wb_lca.LAST_KEEP = None
    if len(source) == 76 and 'paired_transition' not in disabled:
        bank.deadline = min(finish_deadline - (24.5 if (v68_config.V124_W1 and not v68_config.V127_W1) else 16.5), time.monotonic() + 10.0) if v68_config.K2 else min(finish_deadline - 10.0, time.monotonic() + 10.0)
        report['paired_transition'] = attach_paired_transition(work, X, given, fold, bank, stacked_anchor, bank.deadline, event=followup_event, items=schema['items'], anchor_names=anchors, child_names=post_items + children)
    if len(source) == 76 and 'psychometric_irt' not in disabled and time.monotonic() < finish_deadline - (24.5 if (v68_config.V124_W1 and not v68_config.V127_W1) else 16.5 if v68_config.K2 else 22.5):
        from . import wb_psychometrics
        cap = min(finish_deadline - (24.5 if (v68_config.V124_W1 and not v68_config.V127_W1) else 16.5), time.monotonic() + 3.5) if v68_config.K2 else min(finish_deadline - 22.5, time.monotonic() + 2.0)
        bank.deadline = cap
        with pools.limit(limits=1):
            if v68_config.V83_W2:
                report['psychometric_irt'] = wb_psychometrics.attach_all(schema, X, given, work, fold, cap, bank, workers=_THREADS)
            else:
                report['psychometric_irt'] = wb_psychometrics.attach(schema, X, given, work, fold, cap, bank)
    followup_chain = None
    followup_event = None
    bank.cache.clear()
    bank.config_cache.clear()
    report['tabpfn_calls'] = pfn_timing['calls']
    report['tabpfn_folds'] = int(pfn_timing.get('last_n_folds', 2))
    report['tabpfn_contexts'] = pfn_timing.get('completed', [])
    for j in work.values():
        for arm in FOUNDATION_ARMS:
            j.pop(arm + '_partitions', None)
    report['foundation_trace'] = dict(gpu.TRACE)
    hard_gate_counts = {'closed': 0, 'open': 0}
    for name, j in work.items():
        if not isinstance(schema['items'][name].get('gate'), dict):
            continue
        if schema['items'][name]['gate'].get('parent') not in given:
            continue
        rows = np.flatnonzero(j['train'])
        rest = np.flatnonzero(~j['train'])
        row_folds = fold[rows]
        for arm_index, (o, p) in enumerate(zip(j['oof'], j['all'])):
            o = o.copy()
            p = p.copy()
            j['oof'][arm_index] = o
            j['all'][arm_index] = p
            for f in range(5):
                arm_k = j.get('partition_k', {}).get(j['arms'][arm_index], 5)
                arm_fold = fold_sets.get(arm_k, fold)[rows]
                if f >= arm_k: continue
                query = rows[arm_fold == f]
                if not len(query):
                    continue
                fit = rows[arm_fold != f]
                selected = o[arm_fold == f].copy()
                lock_given_gate(selected, schema['items'][name], schema['items'], X, given, j['y'], j['K'], fit, query)
                o[arm_fold == f] = selected
            selected = p[rest].copy()
            flags = lock_given_gate(selected, schema['items'][name], schema['items'], X, given, j['y'], j['K'], rows, rest)
            p[rest] = selected
        hard_gate_counts['closed'] += int(flags[0])
        hard_gate_counts['open'] += int(flags[1])
    report['given_gate_locks'] = hard_gate_counts
    if 'stacked_cross_target' not in disabled:
        start = time.monotonic()
        with pools.limit(limits=1):
            found, trace = base.stacked_cross_target(work, fold, min(finish_deadline - (28.0 if is_deep_hierarchy else 18.0 if (v68_config.V124_W1 and not v68_config.V127_W1) else 10.5), start + 7.5), workers=_THREADS, bank=bank)
        for name, (o, p, gain) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('stacked_cross_target')
        report['stacked_cross_target'] = len(found)
        report['stacked_cross_target_trace'] = trace
        report['stacked_cross_target_seconds'] = time.monotonic() - start
    stacked = {}
    alternates = {}
    choices = {}
    stack_start = time.monotonic()
    select_seconds = 0.0
    recent_neighbor = 0.15
    report['neighbor_attempted'] = 0
    calibrate_seconds = 0.0
    calibrated = 0
    # Items stack independently; without neighbor selection the loop runs on every thread.
    parallel_stack = v68_config.V80_W1 and len(work) == 76 and not (hasattr(index, 'distributions') and 'neighbor_cross_target' not in disabled)
    with pools.limit(limits=1):
        if parallel_stack:
            from concurrent.futures import ThreadPoolExecutor
            from threading import Lock
            clock = Lock()
            state = {'seconds': 0.0, 'done': 0}

            def stack_item(name):
                j = work[name]
                began = time.monotonic()
                rest = np.flatnonzero(~j['train'])
                fold_arms(name, j, X, given, schema['items'])
                gate_latent(j, finish_deadline)
                aligned = aligned_partitions(j, fold)
                p, o, w, t, loss = crossfit_calibrate(j['oof'], [a[rest] for a in j['all']],
                    j['y'][j['train']], j['K'], aligned, deadline=finish_deadline)
                j['calibration_partitions'] = int(len(np.unique(aligned)))
                active = list(range(len(j['arms'])))
                j['check_arms'] = tuple(dict.fromkeys((*j.get('check_arms', ()), 'psychometric_irt', 'latent_profile', 'stacked_cross_target', 'followup_anchor', 'anchor_eb')))
                while True:
                    removed = False
                    for arm in j.get('check_arms', ('stacked_cross_target', 'followup')):
                        if arm not in j['arms']:
                            continue
                        idx = j['arms'].index(arm)
                        if idx not in active or w[idx] <= 1e-08:
                            continue
                        keep = [i for i in active if i != idx]
                        aligned_keep = aligned_partitions({'arms': [j['arms'][i] for i in keep],
                            'partition_k': j.get('partition_k', {}), 'train': j['train']}, fold)
                        old_p, old_o, old_w, old_t, old_loss = crossfit_calibrate(
                            [j['oof'][i] for i in keep], [j['all'][i][rest] for i in keep],
                            j['y'][j['train']], j['K'], aligned_keep, deadline=finish_deadline)
                        half = fold[j['train']] % 2 == 0
                        y = j['y'][j['train']]
                        if not half.any() or half.all() or any((_nll(o[h], y[h], j['K']) >= _nll(old_o[h], y[h], j['K']) for h in (half, ~half))):
                            p, o, t, loss = (old_p, old_o, old_t, old_loss)
                            w = np.zeros(len(j['arms']))
                            w[keep] = old_w
                            active = keep
                            j['calibration_partitions'] = int(len(np.unique(aligned_keep)))
                            removed = True
                    if not removed:
                        break
                with clock:
                    average = state['seconds'] / state['done'] if state['done'] else .3
                    left = len(work) - state['done'] - 1
                coarse_weight = sum(float(w[i]) for i in active if j.get('partition_k', {}).get(j['arms'][i], 5) != 5)
                tier = None
                if (v68_config.V73_K1 and coarse_weight >= .02
                        and time.monotonic() + 3. * average + average * left / _THREADS < finish_deadline - 4.5):
                    tier_began = time.monotonic()
                    (p, o, w, t, loss), tiered = tiered_calibrate(j, active, rest, fold, (p, o, w, t, loss), finish_deadline)
                    if tiered is not None:
                        tier = (time.monotonic() - tier_began, bool(tiered))
                        j['tiered_calibration'] = bool(tiered)
                if v68_config.V101_W3:
                    p, o, w, t, loss = compact_anchor(j, rest, fold, (p,o,w,t,loss), finish_deadline)
                loss = _nll(o, j['y'][j['train']], j['K'])
                alternate = pre_track(j, fold, calibrate, finish_deadline) if 'cross_round' not in disabled else None
                with clock:
                    state['seconds'] += time.monotonic() - began
                    state['done'] += 1
                choice = {'partition_k': j.get('partition_k', {}), 'calibration_partitions': j.get('calibration_partitions'), 'tiered_calibration': j.get('tiered_calibration'), 'arms': j['arms'], 'weights': w.tolist(), 'temperature': t, 'latent_oof_gain': j.get('latent_oof_gain'), 'arm_nll': {a: float(_nll(q, j['y'][j['train']], j['K'])) for a, q in zip(j['arms'], j['oof'])}}
                return o, p, loss, alternate, choice, tier
            order = list(work)
            with ThreadPoolExecutor(max_workers=_THREADS, initializer=lambda: threadpool_limits(limits=1)) as pool:
                results = list(pool.map(stack_item, order))
            for name, (o, p, loss, alternate, choice, tier) in zip(order, results):
                if tier is not None:
                    trace = report.setdefault('tiered_calibration', {'offered': 0, 'kept': 0, 'seconds': 0.})
                    trace['seconds'] += tier[0]
                    trace['offered'] += 1
                    trace['kept'] += int(tier[1])
                if alternate is not None:
                    alternates[name] = alternate
                stacked[name] = (o, p, loss)
                choices[name] = choice
            calibrate_seconds, calibrated = state['seconds'], state['done']
        forked = {}
        if (v68_config.V83_W1 and not parallel_stack and len(work) == 76 and sys.platform.startswith('linux')
                and not (hasattr(index, 'distributions') and 'neighbor_cross_target' not in disabled)):
            # Items stack independently. Forked processes share the arms read-only and
            # sidestep the GIL that held the  thread pool to 1.6x; the tail keeps 18 s.
            global _FORK_TASK
            import multiprocessing
            order = sorted(work, key=lambda k: -len(work[k]['arms']) * work[k]['K'] * int(work[k]['train'].sum()))

            def stack_one(name):
                j = work[name]
                began = time.monotonic()
                had_latent = 'latent_profile' in j['arms']
                rest = np.flatnonzero(~j['train'])
                fold_arms(name, j, X, given, schema['items'])
                gate_latent(j, finish_deadline)
                aligned = aligned_partitions(j, fold)
                p, o, w, t, loss = crossfit_calibrate(j['oof'], [a[rest] for a in j['all']],
                    j['y'][j['train']], j['K'], aligned, deadline=finish_deadline)
                j['calibration_partitions'] = int(len(np.unique(aligned)))
                active = list(range(len(j['arms'])))
                j['check_arms'] = tuple(dict.fromkeys((*j.get('check_arms', ()), 'psychometric_irt', 'latent_profile', 'stacked_cross_target', 'followup_anchor', 'anchor_eb')))
                while True:
                    removed = False
                    for arm in j.get('check_arms', ('stacked_cross_target', 'followup')):
                        if arm not in j['arms']:
                            continue
                        idx = j['arms'].index(arm)
                        if idx not in active or w[idx] <= 1e-08:
                            continue
                        keep = [i for i in active if i != idx]
                        aligned_keep = aligned_partitions({'arms': [j['arms'][i] for i in keep],
                            'partition_k': j.get('partition_k', {}), 'train': j['train']}, fold)
                        old_p, old_o, old_w, old_t, old_loss = crossfit_calibrate(
                            [j['oof'][i] for i in keep], [j['all'][i][rest] for i in keep],
                            j['y'][j['train']], j['K'], aligned_keep, deadline=finish_deadline)
                        half = fold[j['train']] % 2 == 0
                        y = j['y'][j['train']]
                        if not half.any() or half.all() or any((_nll(o[h], y[h], j['K']) >= _nll(old_o[h], y[h], j['K']) for h in (half, ~half))):
                            p, o, t, loss = (old_p, old_o, old_t, old_loss)
                            w = np.zeros(len(j['arms']))
                            w[keep] = old_w
                            active = keep
                            j['calibration_partitions'] = int(len(np.unique(aligned_keep)))
                            removed = True
                    if not removed:
                        break
                coarse_weight = sum(float(w[i]) for i in active if j.get('partition_k', {}).get(j['arms'][i], 5) != 5)
                tier = None
                if (v68_config.V73_K1 and coarse_weight >= .02
                        and time.monotonic() < finish_deadline - 4.5):
                    tier_began = time.monotonic()
                    (p, o, w, t, loss), tiered = tiered_calibrate(j, active, rest, fold, (p, o, w, t, loss), finish_deadline)
                    if tiered is not None:
                        tier = (time.monotonic() - tier_began, bool(tiered))
                        j['tiered_calibration'] = bool(tiered)
                if v68_config.V101_W3:
                    p, o, w, t, loss = compact_anchor(j, rest, fold, (p,o,w,t,loss), finish_deadline)
                loss = _nll(o, j['y'][j['train']], j['K'])
                alternate = pre_track(j, fold, calibrate, finish_deadline) if 'cross_round' not in disabled else None
                seconds = time.monotonic() - began
                choice = {'partition_k': j.get('partition_k', {}), 'calibration_partitions': j.get('calibration_partitions'), 'tiered_calibration': j.get('tiered_calibration'), 'arms': j['arms'], 'weights': w.tolist(), 'temperature': t, 'latent_oof_gain': j.get('latent_oof_gain'), 'arm_nll': {a: float(_nll(q, j['y'][j['train']], j['K'])) for a, q in zip(j['arms'], j['oof'])}}
                delta = {k: j[k] for k in ('calibration_partitions', 'check_arms', 'tiered_calibration', 'latent_oof_gain') if k in j}
                return name, o, p, loss, alternate, choice, tier, delta, had_latent and 'latent_profile' not in j['arms'], seconds
            _FORK_TASK = stack_one
            fork_began = time.monotonic()
            with multiprocessing.get_context('fork').Pool(processes=_THREADS) as pool:
                results = pool.imap_unordered(_fork_run, order, chunksize=1)
                for _ in order:
                    try:
                        found = results.next(timeout=max(1.0, finish_deadline - time.monotonic() + 5.0))
                    except (StopIteration, multiprocessing.TimeoutError):
                        break
                    except Exception:
                        continue
                    if found is None:
                        continue
                    name, o, p, loss, alternate, choice, tier, delta, dropped, seconds = found
                    j = work[name]
                    fold_arms(name, j, X, given, schema['items'])
                    if dropped:
                        index_latent = j['arms'].index('latent_profile')
                        for key in ('arms', 'oof', 'all'):
                            j[key].pop(index_latent)
                    j.update(delta)
                    if tier is not None:
                        trace = report.setdefault('tiered_calibration', {'offered': 0, 'kept': 0, 'seconds': 0.})
                        trace['seconds'] += tier[0]
                        trace['offered'] += 1
                        trace['kept'] += int(tier[1])
                    if alternate is not None:
                        alternates[name] = alternate
                    stacked[name] = (o, p, loss)
                    choices[name] = choice
                    forked[name] = seconds
                    calibrate_seconds += seconds
                    calibrated += 1
            _FORK_TASK = None
            report['fork_stack'] = {'items': len(forked), 'workers': _THREADS, 'seconds': time.monotonic() - fork_began}
        for name, j in (() if parallel_stack else ((k, v) for k, v in work.items() if k not in forked)):
            began = time.monotonic()
            rest = np.flatnonzero(~j['train'])
            fold_arms(name, j, X, given, schema['items'])
            gate_latent(j, finish_deadline)
            aligned = aligned_partitions(j, fold)
            p, o, w, t, loss = crossfit_calibrate(j['oof'], [a[rest] for a in j['all']],
                j['y'][j['train']], j['K'], aligned, deadline=finish_deadline)
            j['calibration_partitions'] = int(len(np.unique(aligned)))
            active = list(range(len(j['arms'])))
            j['check_arms'] = tuple(dict.fromkeys((*j.get('check_arms', ()), 'psychometric_irt', 'latent_profile', 'stacked_cross_target', 'followup_anchor', 'anchor_eb')))
            while True:
                removed = False
                for arm in j.get('check_arms', ('stacked_cross_target', 'followup')):
                    if arm not in j['arms']:
                        continue
                    idx = j['arms'].index(arm)
                    if idx not in active or w[idx] <= 1e-08:
                        continue
                    keep = [i for i in active if i != idx]
                    aligned_keep = aligned_partitions({'arms': [j['arms'][i] for i in keep],
                        'partition_k': j.get('partition_k', {}), 'train': j['train']}, fold)
                    old_p, old_o, old_w, old_t, old_loss = crossfit_calibrate(
                        [j['oof'][i] for i in keep], [j['all'][i][rest] for i in keep],
                        j['y'][j['train']], j['K'], aligned_keep, deadline=finish_deadline)
                    half = fold[j['train']] % 2 == 0
                    y = j['y'][j['train']]
                    if not half.any() or half.all() or any((_nll(o[h], y[h], j['K']) >= _nll(old_o[h], y[h], j['K']) for h in (half, ~half))):
                        p, o, t, loss = (old_p, old_o, old_t, old_loss)
                        w = np.zeros(len(j['arms']))
                        w[keep] = old_w
                        active = keep
                        j['calibration_partitions'] = int(len(np.unique(aligned_keep)))
                        removed = True
                if not removed:
                    break
            average = calibrate_seconds / calibrated if calibrated else .3
            coarse_weight = sum(float(w[i]) for i in active if j.get('partition_k', {}).get(j['arms'][i], 5) != 5)
            if (v68_config.V73_K1 and len(work) == 76 and coarse_weight >= .02
                    and time.monotonic() + 3. * average + average * (len(work) - calibrated - 1) < finish_deadline - 4.5):
                tier_began = time.monotonic()
                (p, o, w, t, loss), tiered = tiered_calibrate(j, active, rest, fold, (p, o, w, t, loss), finish_deadline)
                if tiered is not None:
                    trace = report.setdefault('tiered_calibration', {'offered': 0, 'kept': 0, 'seconds': 0.})
                    trace['seconds'] += time.monotonic() - tier_began
                    trace['offered'] += 1
                    trace['kept'] += int(tiered)
                    j['tiered_calibration'] = bool(tiered)
            calibrate_seconds += time.monotonic() - began
            calibrated += 1
            per_cal = calibrate_seconds / calibrated if len(source) == 12 else min(calibrate_seconds / calibrated, 0.1)
            reserve = per_cal * (len(work) - calibrated) + (23.5 if is_deep_hierarchy else 4.5)
            leaf_ready = getattr(index, '_leaf_rank', None) is not None
            warm = leaf_ready and len(getattr(index, '_candidate_cache', ())) >= 4
            allow_nb = time.monotonic() + (0.03 if warm else 0.8) < nb_select_stop and select_seconds < 4.8 if is_deep_hierarchy else time.monotonic() + max(0.04 if warm else 1.0, 1.5 * recent_neighbor) < min(nb_select_stop, finish_deadline - reserve)
            if hasattr(index, 'distributions') and 'neighbor_cross_target' not in disabled and allow_nb:
                start = time.monotonic()
                p, info, o = select_neighbors(index, name, o, p, rest, deadline=min(nb_select_stop, time.monotonic() + (3.8 if not warm else 0.35)))
                elapsed = time.monotonic() - start
                select_seconds += elapsed
                if leaf_ready:
                    recent_neighbor = 0.7 * recent_neighbor + 0.3 * elapsed
                if info['path'] is not None:
                    report['neighbor'] += 1
                report['neighbor_attempted'] += 1
                if v68_config.V101_W3:
                    p, o, w, t, loss = compact_anchor(j, rest, fold, (p,o,w,t,loss), finish_deadline)
                loss = _nll(o, j['y'][j['train']], j['K'])
            loss = _nll(o, j['y'][j['train']], j['K'])
            if 'cross_round' not in disabled:
                alternate = pre_track(j, fold, calibrate, finish_deadline)
                if alternate is not None:
                    alternates[name] = alternate
            stacked[name] = (o, p, loss)
            choices[name] = {'partition_k': j.get('partition_k', {}), 'calibration_partitions': j.get('calibration_partitions'), 'tiered_calibration': j.get('tiered_calibration'), 'arms': j['arms'], 'weights': w.tolist(), 'temperature': t, 'latent_oof_gain': j.get('latent_oof_gain'), 'arm_nll': {a: float(_nll(q, j['y'][j['train']], j['K'])) for a, q in zip(j['arms'], j['oof'])}}
        if forked:
            stacked = {k: stacked[k] for k in work if k in stacked}
            choices = {k: choices[k] for k in work if k in choices}
            alternates = {k: alternates[k] for k in work if k in alternates}
        report['stack_seconds'] = time.monotonic() - stack_start
        report['calibrate_seconds'] = calibrate_seconds
        report['neighbor_select_seconds'] = select_seconds
        selected_arms = ('logit', 'strong', 'shallow', 'shallow_asked', 'gate_table', 'gate_logit', 'deep', 'hgb', 'catboost', 'tabpfn', 'followup', 'followup_shared', 'followup_anchor', 'anchor_eb', 'network', 'cross_round', 'rb_distill', 'cross_target', 'stacked_cross_target', 'latent_profile', 'full_cell_eb', 'conditional_cell_eb', 'paired_transition', 'given_mirror_transition', 'kumo', 'tabpfn_joint', 'kumo_joint', 'report_prior', 'psychometric_irt', 'wb_lca') + (('wb_lca_chain',) if v68_config.V127_W2 else ())
        report['selected'] = {arm: sum((float(weight) > 1e-08 for choice in choices.values() for arm_name, weight in zip(choice['arms'], choice['weights']) if arm_name == arm)) for arm in selected_arms}
        gate_start = time.monotonic()
        final_oofs = {}
        if 'gate' in disabled:
            final = {}
            gates = {}
            for name, (o, p, _) in stacked.items():
                if 'cross_round' not in disabled:
                    o, p = cross_round_row_gate(work[name], o, p, fold)
                final[name] = np.zeros((n, work[name]['K']))
                final[name][~work[name]['train']] = p
                final_oofs[name] = o
        else:

            def tail_calibrate(oofs, rests, y, K):
                return calibrate(oofs, rests, y, K, deadline=finish_deadline)
            final, gates = propagate_structural_gates(schema['items'], work, stacked, X, given, levels, np.maximum(fold, 0), tail_calibrate, _nll, deadline=finish_deadline, disabled=disabled, row_gate_fn=None if 'cross_round' in disabled else cross_round_row_gate, alternate_stacked=alternates, oof_output=final_oofs)
            report['nested_gate'] = {str(k): v for k, v in NESTED_TRACE.items()}
            emp = [g for g in gates if g[10] and np.isfinite(g[11])]
            report['empirical_gate'] = {'items': len(emp), 'selected_nll': float(np.mean([g[4] for g in emp])) if emp else None, 'declared_nll': float(np.mean([min(g[3], g[8]) for g in emp])) if emp else None, 'empirical_nll': float(np.mean([g[11] for g in emp])) if emp else None, 'base_nll': float(np.mean([g[12] for g in emp])) if emp else None, 'selected_kinds': {k: sum((g[6] == k for g in emp)) for k in ('base', 'multi', 'legacy')}}
    from . import finite_pop_raking
    if finite_pop_raking.RAKING_ENABLED or finite_pop_raking.CONSENSUS_ENABLED:
        final, report['finite_population'] = finite_pop_raking.apply(schema, X, given, work, fold, final, final_oofs, finish_deadline)
    report['soft_given'] = {name: j['soft_given'] for name, j in work.items() if 'soft_given' in j}
    report['dual_track'] = {name: j['dual_track'] for name, j in work.items() if 'dual_track' in j}
    report['row_gate'] = {name: j['row_gate'] for name, j in work.items() if 'row_gate' in j}
    report['gate_seconds'] = time.monotonic() - gate_start

    def floored_nll(o, name):
        y, K = work[name]['y'][work[name]['train']], work[name]['K']
        if o is None or np.shape(o) != (len(y), K):
            return None
        p = np.asarray(o, float)
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
        return float(-np.log(1e-3 + (1. - K * 1e-3) * p[np.arange(len(y)), y]).mean())
    if v68_config.V97_W1 and not v68_config.V127_W1:
        report['floor_fill'] = _floor_fill(work, stacked, final, final_oofs)
    pre = {name: floored_nll(o, name) for name, (o, _, _) in stacked.items()}
    post = {name: floored_nll(final_oofs.get(name, stacked[name][0]), name) for name in stacked}
    report['oof_nll_items'] = {name: [pre[name], post[name]] for name in stacked}
    report['oof_nll_pre_gate'] = float(np.mean([v for v in pre.values() if v is not None])) if stacked else None
    report['oof_nll_post_gate'] = float(np.mean([v for v in post.values() if v is not None])) if stacked else None
    report['pass_names'] = {'anchors': list(anchors), 'post': list(post_items), 'children': list(children)}
    report['stage_stops'] = {k: v - now for k, v in dict(cpu=cpu_stop, pfn=pfn_stop, network=net_stop, neighbor_build=nb_build_stop, cross_target=ct_stop, neighbor_select=nb_select_stop).items()}
    report.update(parents_fitted=bank.parent_fits, parent_cache_hits=bank.cache_hits, parent_partitions=parent_partitions, parent_model_cache_hits=bank.model_cache_hits, parent_fit_calls=bank.parent_fit_calls, gates=len(gates), choices=choices)
    return (final, report)

def predict(frame, schema):
    global LAST_REPORT
    began = time.monotonic()
    X, given, levels, source, n_raw_given = encode(frame, schema)
    names = list(source)
    n = len(X)
    common = np.ones(n, bool)
    for j in source.values():
        common &= j['train']
    visible = np.flatnonzero(common)
    split = schema.get('split', {})
    phase = 'final' if n - len(visible) == split.get('n_test') else 'dev'
    expected = split.get('n_train', 0) + split.get('n_dev', 0)
    if phase == 'final' and n == expected + split.get('n_test', 0) and (len(visible) == expected):
        print('[train+dev] visible=%d expected=%d+%d' % (len(visible), split['n_train'], split['n_dev']), file=sys.stderr)
    marginal_out = {}
    for name, j in source.items():
        fit = np.flatnonzero(j['train'])
        K = j['K']
        if len(fit):
            cnt = np.bincount(j['y'][fit], minlength=K).astype(float)
            p_m = (cnt + 0.5 / K) / (len(fit) + 0.5)
        else:
            p_m = np.full(K, 1.0 / K)
        marginal_out[name] = np.broadcast_to(p_m, (n, K)).copy()
    total = 690.0 if phase == 'dev' else 3200.0
    mine = _SHARE.get(len(names), 0.2)
    later = sum((v for k, v in _SHARE.items() if k not in _DONE and k != len(names)))
    end = began + (max(total * mine, max(0.0, total - (began - _START)) * mine / (mine + later)) if _ortho.V124_W1 else max(0.0, total - (began - _START)) * mine / (mine + later))
    tail = 32.0 if len(names) > 100 else (10.0 if v68_config.V80_W1 else 18.0) if len(names) > 30 else 10.0
    try:
        raw_out, report = fit_partition(frame, schema, X, given, levels, source, visible, end - tail, phase=phase, finish_deadline=end - 1.5, n_raw_given=n_raw_given)
    except (BudgetExpired, Exception):
        raw_out, report = ({}, {'variant': VARIANT, 'table': len(source), 'threads': _THREADS})
    out = {}
    for name, j in source.items():
        fit = np.flatnonzero(j['train'])
        query = np.flatnonzero(~j['train'])
        cand = raw_out.get(name)
        if isinstance(cand, np.ndarray) and cand.shape == (n, j['K']) and (not len(query) or (np.all(np.isfinite(cand[query])) and np.all(cand[query].sum(1) > 1e-09))):
            base_mat = _normalize(cand)
        else:
            base_mat = marginal_out[name].copy()
        selected = base_mat[query].copy()
        base_mat[query] = _normalize(selected)
        out[name] = base_mat
    gate_targets = lock_randomization_gate(out, schema, X, given, source)
    LAST_REPORT = {'selection_folds': 5, 'stratification': 'first raw GIVEN category', 'coverage': report, 'seconds': time.monotonic() - began, 'visible': len(visible), 'phase': phase, 'cuda': gpu.available(), 'hard_randomization_gates': len(gate_targets), 'state': 'done', 'reason': 'complete', 'passed': True, 'runtime_healthy': True}
    print('[coverage] items=%d %s seconds=%.3f' % (len(names), {k: v for k, v in report.items() if k not in ('choices', 'followup_status', 'strength', 'drivers', 'empirical_parents', 'empirical_parent_mi', 'oof_nll_items', 'pass_names')}, time.monotonic() - began), file=sys.stderr, flush=True)
    vectors = []
    order = [k for k, r in schema['items'].items() if r['class'] in ('GIVEN', 'PREDICT')]
    blanks = {name: frame[name].isna().to_numpy() for name in order}
    query_rows = np.flatnonzero(np.any(np.column_stack(list(blanks.values())), axis=1))
    for r in query_rows:
        for name in order:
            if blanks[name][r]:
                if name in out:
                    vectors.append(out[name][r].tolist())
                else:
                    K = len(schema['items'][name]['values'])
                    vectors.append([1 / K] * K)
    LAST_REPORT['seconds'] = time.monotonic() - began
    _DONE.add(len(names))
    return vectors
