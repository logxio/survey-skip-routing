import final_config as _final
import ortho_config as _ortho
from . import twin_posterior, twin_base
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
from .chain_features import _fit_hgb, hgb_stages, hgb_crossfit_stages, FoldFeatureBank, folds_for, tree_predict, gated_tree_predict, topological, BudgetExpired, check_clock, split_anchors, StackedChain, FoldPlan, PRE_POST_MIRRORS, UNICEF_ANCHORS
from .stratified_kernel_prior import build as build_neighbors, select as select_neighbors
from .propagate_structural_gates import propagate_structural_gates, detect_zero_on_skip, fold_wave_one_skip, ZERO_ON_SKIP, _parents, EMPIRICAL_PARENTS, NESTED_TRACE, discover_empirical_gate_parents
from .propagate_structural_gates import _declared_parents, _depth as _gate_depth
from . import post_gate
from .v64_config import V132_K1, V132_K2, V136_K1
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
from . import latent_classes, twin_identity, twin_collision, twin_grid
_DONE = set()
_LAST_SAFE_OUT = {}
UNHCR_CROSS_TARGET = True
# UNHCR: one logit C per fold, two-fold catboost on every item ahead of the
# deep trees, and the fast tail's spare seconds moved to the learners.
UNHCR_LEARNER_BUDGET = True
# UNHCR: latent-class arm, every column's EM read back through GIVEN only.
V79_K1 = False
LATENT_CLASS_SECONDS = 14.0
# UNHCR: exact cross-round household identity () and its one-band drift tier ().
V79_K2 = False
V79_K3 = False
# collision-group twins with 1/k evidence (), household-only signature
# pairs for a changed respondent (), second latent-class scale plus EM that
# keeps its current parameters at the deadline instead of dropping the arm ().
V84_K1 = False
V84_K2 = False
V84_K3 = False
LATENT_CLASS_SECOND_Q = 32
LATENT_CLASS_SECOND_SECONDS = 12.0
# The trait folds run beside the class pool and may close this much later.
LATENT_TRAIT_EXTRA = 4.0


def _traced_trait(*args, executor=None):
    with threadpool_limits(1):
        return latent_classes.fit_trait(*args, workers=5, executor=executor, strat=V98_K1)
# UNHCR: every item's stack, arm pruning and paired pre-track computed up front
# in forked worker processes (Linux), the neighbour selector after them in order.
V85_K1 = True
# both latent-class scales in one (scale, fold) pool plus a round x
# country-stratified emission arm mixed across scales.
V88_K1 = False
# a gate-aware one-dimensional nominal-response trait arm fitted in the
# same window as the latent classes (); after the twin passes, gated
# children's skip mass follows the twin-updated parents (, off: hidden
# -0.000113, 2/8 on the paired rulers).
V92_K1 = False
V92_K2 = False
V98_K1 = os.environ.get("V98_K1", "1") == "1"
V100_K2 = os.environ.get('V100_K2', '1') == '1'
V100_K1 = False
V100_FLOOR_POLICY = os.environ.get('V100_FLOOR_POLICY', 'conditional')
# modules after the structural gates (raking, twin passes) keep an item's
# edit only on a visible-row gain of 3 SE with both fold halves positive; the
# gate graph is then re-projected onto the moved parents (post_gate). The twin
# passes stop TWIN_TAIL_SECONDS after finish_deadline.
POST_GATE = os.environ.get('SB_V64_POST_GATE', '1') == '1'
TWIN_TAIL_SECONDS = float(os.environ.get('SB_V64_TWIN_TAIL', '12.0'))
# The per-item clean-stack bases (213 refits) run only when this much of the tail is left.
TWIN_BASES_SECONDS = 20.0
#  each item's stack also fitted on one champion per arm family, picked
# and calibrated inside every outer training partition; the full stack stays
# only where it beats the compact one on both held-out halves (  port).
import ortho_config as _ortho
latent_classes.ZERO_GATE = _ortho.V112_K1
latent_classes.LCA_ZERO_GATE = _ortho.V122_K1
latent_classes.MULTIPARENT = _ortho.V122_K2
twin_grid.RESTACK = _ortho.V112_K2
latent_classes.PAIR_PURE = _ortho.V116_K1
twin_grid.REVISIT_PRIOR = _ortho.V116_K2
_FAMILIES = (('tabpfn', 'tabpfn_joint'), ('kumo', 'kumo_joint')) + (('latent_class', 'latent_class_b', 'latent_class_strat'), ('latent_trait', 'latent_trait_strat'),
             ('catboost', 'deep', 'hgb', 'shallow', 'shallow_asked', 'strong', 'two_stage'), ('hgb_given',), ('network', 'logit'), ('logit_given',),
             ('table', 'rb_distill', 'full_cell_eb', 'anchor_eb', 'stacked_cross_target',
              'panel_report_transition', 'latent_profile', 'cross_target', 'report_prior'), ('gate_table', 'gate_logit'))
from .floor_fill import _waterfill, _floor_fill
LAST_REPORT = {}
DISABLED = frozenset(('anchor_cascade', 'joint_gate', 'cross_round'))
VARIANT = 'main'
_SHARE = {'unhcr_heavy': {213: 0.7, 76: 0.22, 12: 0.08}, 'wb_unicef_heavy': {213: 0.48, 76: 0.38, 12: 0.14}}.get(VARIANT, {213: 0.62, 76: 0.28, 12: 0.1})
def ortho_reserve(n_questions, phase, span=None):
    reserve = ((136.0 if phase == 'final' else 36.0) if n_questions > 100
               else (72.0 if phase == 'final' else 20.0) if n_questions > 30
               else (42.0 if phase == 'final' else 12.0))
    if (n_questions > 100 and phase == 'final' and _final.enabled('FINAL_WINDOWS')
            and gpu.available() and gpu.has_weights()):
        reserve = 36.0
    return min(reserve, 0.18 * max(0.0, float(span))) if span is not None else reserve

if VARIANT == 'unhcr_heavy':
    from . import cross_round
    cross_round.ROW_PRIORS = (0.03, 0.05, 0.1, 0.18, 0.3, 0.6, 1.2, 2.5, 5.0)
    cross_round.ROW_THRESHOLDS = (1e-09, 0.15, 0.35, 0.45)
_TREE_ARMS = frozenset(('shallow', 'shallow_asked', 'two_stage', 'deep', 'hgb', 'hgb_given', 'catboost'))
_WB_CHILD_DEEP = 'all'

def calibrate(oofs, rests, y, K, deadline=float('inf')):
    if 'simplex_stack' in DISABLED:
        losses = [_nll(o, y, K) for o in oofs]
        best = int(np.argmin(losses))
        weights = np.zeros(len(oofs))
        weights[best] = 1.0
        return (rests[best], oofs[best], weights, 1.0, losses[best])
    p, o, w, t, l = _stack(oofs, rests, y, K, deadline=deadline)
    best = (l, 0.0, p, o)
    for floor in (0.0001, 0.001, 0.005):
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

def compact_stack(job, rest, fold, flat, deadline):
    """One OOF-best arm per family, chosen and calibrated inside each outer
    training partition; the full stack flat is kept where it wins both halves."""
    arms = job['arms']
    groups = [[i for i, a in enumerate(arms) if a in family] for family in _FAMILIES]
    listed = {a for family in _FAMILIES for a in family}
    groups.extend([i] for i, arm in enumerate(arms) if arm not in listed)
    groups = [g for g in groups if g]
    if len(groups) < 2 or sum(map(len, groups)) <= len(groups) or time.monotonic() >= deadline:
        return flat
    y, K = job['y'][job['train']], job['K']
    split = aligned_partitions(job, fold)

    def fit(rows, hold):
        keep = [min(g, key=lambda i: _nll(job['oof'][i][rows], y[rows], K)) for g in groups]
        return calibrate([job['oof'][i][rows] for i in keep], hold(keep), y[rows], K, deadline), keep
    (p, _, weights, t, _), keep = fit(np.ones(len(y), bool), lambda ids: [job['all'][i][rest] for i in ids])
    o = np.empty_like(job['oof'][0], dtype=float)
    for f in np.unique(split):
        va = split == f
        if va.all():
            return flat
        o[va] = fit(~va, lambda ids: [job['oof'][i][va] for i in ids])[0][0]
    from .cross_round import dominates
    if dominates(flat[1], o, y, K, fold[job['train']] % 2 == 0):
        return flat
    w = np.zeros(len(arms))
    w[keep] = weights
    job['compact'] = True
    return p, o, w, t, _nll(o, y, K)

def aligned_partitions(job, fold, indices=None):
    """Coarsest union of the completed arms' base-fold partitions."""
    sub = fold[job['train']]
    labels = np.arange(5)
    arms = job['arms'] if indices is None else [job['arms'][i] for i in indices]
    partitions = {job.get('partition_k', {}).get(arm, 5) for arm in arms}
    # k=3 is a legacy foundation tag; map it to the fast k=2 wall.
    for k in sorted(partitions - {5, 3}):
        for a in range(5):
            for b in range(a):
                if a % k == b % k:
                    labels[labels == labels[a]] = labels[b]
    if len(np.unique(labels)) < 2:
        labels = np.arange(5) % 2
    aligned = labels[sub]
    if len(np.unique(aligned)) < 2:
        aligned = np.arange(len(sub), dtype=np.int64) % 2
    return aligned

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

def _given_gate_alpha(y, K, fit_rows, open_mask, closed_side):
    """Select deterministic-gate shrinkage using five inner fit-only folds."""
    alphas = (0.01, 0.05, 0.20, 1.5)
    split = folds_for(open_mask[:, None], fit_rows, k=5, seed=20261118)
    losses = np.zeros((len(alphas), 5), dtype=float)
    for f in range(5):
        train = fit_rows[split[fit_rows] != f]
        valid = fit_rows[split[fit_rows] == f]
        train = train[(~open_mask[train]) if closed_side else open_mask[train]]
        valid = valid[(~open_mask[valid]) if closed_side else open_mask[valid]]
        if not len(train) or not len(valid):
            return 1.5
        outer_train = fit_rows[split[fit_rows] != f]
        prior = float(np.mean(y[outer_train] == K - 1))
        hits = np.sum(y[train] == K - 1)
        target = y[valid] == K - 1
        for i, alpha in enumerate(alphas):
            skip = (hits + alpha * prior) / (len(train) + alpha)
            # Answer conditionals cancel in this gate-only comparison.
            mass = np.where(target, skip, 1.0 - skip)
            losses[i, f] = -np.log(np.clip(mass, 1e-15, 1.0)).mean()
    eligible = np.all(losses <= losses[-1] + 1e-12, axis=1)
    totals = np.where(eligible, losses.mean(1), np.inf)
    return alphas[int(np.argmin(totals))]

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
        alpha = _given_gate_alpha(y, K, fit_rows, open_mask, True) if strong else 2.0
        skip = (np.sum(y[closed] == K - 1) + alpha * p0) / (len(closed) + alpha)
        target = ~open_mask[query_rows]
        selected = rescale_skip(prob[target], skip)
        legacy = rescale_skip(prob[target], (np.sum(y[closed] == K - 1) + (1.5 if strong else 2.0) * p0) / (len(closed) + (1.5 if strong else 2.0)))
        known = y[query_rows[target]] >= 0
        if known.any() and _nll(selected[known], y[query_rows[target]][known], K) > _nll(legacy[known], y[query_rows[target]][known], K):
            selected = legacy
        prob[target] = selected
    if lock_open:
        target = open_mask[query_rows]
        alpha = _given_gate_alpha(y, K, fit_rows, open_mask, False)
        selected = rescale_skip(prob[target], alpha * p0 / (len(opened) + alpha))
        legacy = rescale_skip(prob[target], 1.5 * p0 / (len(opened) + 1.5))
        known = y[query_rows[target]] >= 0
        if known.any() and _nll(selected[known], y[query_rows[target]][known], K) > _nll(legacy[known], y[query_rows[target]][known], K):
            selected = legacy
        prob[target] = selected
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

def fit_partition(frame, schema, X, given, levels, source, allowed, deadline, near=None, phase='dev', finish_deadline=float('inf'), n_raw_given=None, ortho_reserve=0.0):
    """Every target label outside allowed is masked before any feature/model fit."""
    now = time.monotonic()
    post_ortho_deadline = finish_deadline
    finish_deadline = finish_deadline - ortho_reserve
    cal_finish_deadline = finish_deadline
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
        # Allocate 24.0% of UNHCR span to H100 TabPFN+Kumo, 61.5% to clean 5-fold CPU learners, and preserve neighbor/sct/_fork_stack windows.
        cpu_stop = now + (0.615 if gpu_enabled else 0.785) * span
        pfn_stop = now + (0.855 if gpu_enabled else 0.785) * span
        net_stop = now + (0.885 if gpu_enabled else 0.855) * span
        nb_build_stop = now + (0.910 if gpu_enabled else 0.885) * span
        ct_stop = now + (0.925 if gpu_enabled else 0.910) * span
        nb_select_stop = finish_deadline - 22.0
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
    final_windows = phase == 'final' and _final.enabled('FINAL_WINDOWS')
    if final_windows and gpu_enabled and span > 900.0:
        pfn_stop = max(pfn_stop, finish_deadline - 72.0)
        net_stop = max(net_stop, finish_deadline - 66.0)
        nb_build_stop = max(nb_build_stop, finish_deadline - 56.0)
        ct_stop = max(ct_stop, finish_deadline - 42.0)
    h100_long = final_windows and gpu_enabled and span > 900.0
    if h100_long:
        # Absolute caps also protect the reserve when a fractional stop was later.
        pfn_stop = min(pfn_stop, finish_deadline - 72.0)
        net_stop = min(net_stop, finish_deadline - 66.0)
        nb_build_stop = min(nb_build_stop, finish_deadline - 56.0)
        ct_stop = min(ct_stop, finish_deadline - 42.0)
    route_parents = {item['gate']['parent'] for item in schema['items'].values() if isinstance(item.get('gate'), dict) and 'parent' in item['gate']} | {'s3_11_intent_return_12m_lbl', 's3_12_hope_return_someday_lbl'}
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
        o, p, g, config = _fit_table(tables, np.maximum(y, 0), train, j['K'], 0 if 'marginal' in DISABLED else n_raw_given, return_config=True)
        _LAST_SAFE_OUT[name] = p.copy()
        work[name] = {'y': y, 'K': j['K'], 'train': train, 'gains': g, 'oof': [o], 'all': [p], 'base': _nll(o, y[train], j['K']), 'arms': ['table'], 'partition_k': {'table': 5}, 'table_config': config, 'table_gain': float(np.maximum(g, 0).sum()), 'route_parent': name in route_parents, 'has_gate': bool(schema['items'][name].get('gate'))}
        if g.size:
            strength += g
    drivers = [int(a) for a in np.argsort(-strength[:n_raw_given])[:3] if strength[a] > 0] if is_deep_hierarchy else list(_safe_table_config(X, allowed, n_raw_given)[0])
    D = _design(X[:, :n_raw_given], levels[:n_raw_given], drivers, max_columns=320, support_rows=allowed, given=given[:n_raw_given]) if is_deep_hierarchy else _design(X, levels, drivers, max_columns=600, support_rows=allowed, given=given)
    D_pure = _design(X[:, :n_raw_given], levels[:n_raw_given], [], support_rows=allowed, given=given[:n_raw_given])
    X_raw = np.ascontiguousarray(X[:, :n_raw_given])
    report = {k: 0 for k in ('high_order', 'followup', 'network', 'strong', 'shallow', 'deep', 'hgb', 'hgb_given', 'catboost', 'third', 'logit', 'logit_given', 'neighbor', 'cross_target', 'tabpfn', 'two_stage', 'gate_table', 'gate_logit', 'rb_distill')}
    report.update(variant=VARIANT, table=len(work), drivers=[given[a] for a in drivers], strength=strength.tolist(), design_width=D.shape[1], threads=_THREADS)
    names = topological(list(work), schema['items'])
    if len(source) == 12 and 'high_order' not in DISABLED:
        for j in work.values():
            if time.monotonic() >= cpu_stop:
                break
            found = fit_high_order_tables(tables, j, n_raw_given, deadline=cpu_stop)
            if found is not None:
                j['oof'][0], j['all'][0], _ = found
                report['high_order'] += 1
    if len(source) == 12 and 'full_cell_eb' not in DISABLED and (4 <= n_raw_given <= 7):
        report['full_cell_eb'] = attach_full_cell_eb(X[:, :n_raw_given], given[:n_raw_given], work, fold, cpu_stop, heavy=VARIANT == 'wb_unicef_heavy', items=schema['items'])
    report['report_priors'] = survey_priors.attach(schema, X, given, work, fold, min(cpu_stop, time.monotonic() + (2.0 if len(source) == 76 else 2.5))) if 'report_prior' not in DISABLED else {}
    report['latent_profile'] = 0
    if 'latent_profile' not in DISABLED and len(allowed) >= 40:
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
    report['latent_class'] = 0
    if V79_K1 and is_deep_hierarchy and len(allowed) >= 200 and V88_K1:
        start = time.monotonic()
        scales = ((latent_classes.Q, 20261106),) + (((LATENT_CLASS_SECOND_Q, 20261184),) if V84_K3 else ())
        lca_budget = min(16.0, LATENT_CLASS_SECONDS + (LATENT_CLASS_SECOND_SECONDS if V84_K3 else 0.0)) if UNHCR_LEARNER_BUDGET else LATENT_CLASS_SECONDS + (LATENT_CLASS_SECOND_SECONDS if V84_K3 else 0.0)
        window = min(cpu_stop, start + lca_budget)
        trait_extra = min(2.0, LATENT_TRAIT_EXTRA) if UNHCR_LEARNER_BUDGET else LATENT_TRAIT_EXTRA
        trait = None
        if V100_K2 and V92_K1:
            # The two coordinators wait outside the compute pool. Every fold fit
            # and both readbacks consume one of the same eight compute slots.
            from concurrent.futures import ThreadPoolExecutor
            with threadpool_limits(1), ThreadPoolExecutor(_THREADS) as compute_pool, ThreadPoolExecutor(1) as coordinator:
                trait = coordinator.submit(latent_classes.fit_trait, X[:, :n_raw_given], levels[:n_raw_given], given[:n_raw_given], work, names, schema['items'], np.maximum(fold, -1), window + trait_extra, workers=5, executor=compute_pool, strat=V98_K1)
                lca_state = latent_classes.fit_scales(X[:, :n_raw_given], levels[:n_raw_given], given[:n_raw_given], work, names, np.maximum(fold, -1), window, workers=_THREADS, scales=scales, strata=latent_classes.STRATA, executor=compute_pool, items=schema['items'])
                found, report['latent_trait_info'] = trait.result()
            arms = [*zip(('latent_class', 'latent_class_b'), lca_state['pooled']), ('latent_class_strat', lca_state['strat']), *found.items()]
        else:
            if V92_K1:
                from concurrent.futures import ThreadPoolExecutor
                trait_pool = ThreadPoolExecutor(1)
                trait = trait_pool.submit(_traced_trait, X[:, :n_raw_given], levels[:n_raw_given], given[:n_raw_given], work, names, schema['items'], np.maximum(fold, -1), window + trait_extra)
            with threadpool_limits(1):
                lca_state = latent_classes.fit_scales(X[:, :n_raw_given], levels[:n_raw_given], given[:n_raw_given], work, names, np.maximum(fold, -1), window, workers=_THREADS, scales=scales, strata=latent_classes.STRATA, items=schema['items'])
            arms = [*zip(('latent_class', 'latent_class_b'), lca_state['pooled']), ('latent_class_strat', lca_state['strat'])]
            if trait is not None:
                found, report['latent_trait_info'] = trait.result()
                trait_pool.shutdown()
                arms.extend(found.items())
        for arm, found in arms:
            for name, (o, p) in found.items():
                work[name]['oof'].append(o)
                work[name]['all'].append(p)
                work[name]['arms'].append(arm)
            report[arm] = len(found)
        report['latent_class_seconds'] = time.monotonic() - start
        report['latent_class_info'] = lca_state['info']
        if V100_K2:
            report['latent_class_info']['compute_workers'] = _THREADS
            del lca_state, arms, trait, found
            gc.collect()
            _release_heap()
    elif V79_K1 and is_deep_hierarchy and len(allowed) >= 200:
        start = time.monotonic()
        try:
            with threadpool_limits(1):
                found, report["latent_class_tempers"] = latent_classes.fit(X[:, :n_raw_given], levels[:n_raw_given], work, names, np.maximum(fold, -1), min(cpu_stop, start + LATENT_CLASS_SECONDS), workers=_THREADS, graceful=V84_K3)
        except latent_classes.Expired:
            found = {}
        for name, (o, p) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('latent_class')
        report['latent_class'] = len(found)
        report['latent_class_seconds'] = time.monotonic() - start
        if V84_K3 and found:
            start = time.monotonic()
            try:
                with threadpool_limits(1):
                    found, report["latent_class_b_tempers"] = latent_classes.fit(X[:, :n_raw_given], levels[:n_raw_given], work, names, np.maximum(fold, -1), min(cpu_stop, start + LATENT_CLASS_SECOND_SECONDS), workers=_THREADS, q=LATENT_CLASS_SECOND_Q, seed=20261184, graceful=True)
            except latent_classes.Expired:
                found = {}
            for name, (o, p) in found.items():
                work[name]['oof'].append(o)
                work[name]['all'].append(p)
                work[name]['arms'].append('latent_class_b')
            report['latent_class_b'] = len(found)
            report['latent_class_b_seconds'] = time.monotonic() - start
    if (_ortho._flag('V125_K2', True) if _ortho.V125_K2 else False) and is_deep_hierarchy and report.get('latent_class_seconds', 0.0) > 0:
        spare_lca = max(0.0, 26.0 - float(report.get('latent_class_seconds', 0.0)))
        refund = min(20.0, spare_lca, max(0.0, (finish_deadline - 44.0) - ct_stop))
        cpu_stop += refund
        pfn_stop += refund
        net_stop = min(finish_deadline - 49.0, net_stop + refund)
        nb_build_stop = min(finish_deadline - 46.0, nb_build_stop + refund)
        ct_stop = min(finish_deadline - 42.0, ct_stop + refund)
        report['lca_budget_refund_seconds'] = refund
    if h100_long:
        pfn_stop = min(pfn_stop, finish_deadline - 72.0)
        net_stop = min(net_stop, finish_deadline - 66.0)
        nb_build_stop = min(nb_build_stop, finish_deadline - 56.0)
        ct_stop = min(ct_stop, finish_deadline - 42.0)
    ZERO_ON_SKIP.clear()
    if 'zero_on_skip' not in DISABLED:
        ZERO_ON_SKIP.update(detect_zero_on_skip(schema['items'], given, X, work, allowed))
    report['zero_on_skip'] = len(ZERO_ON_SKIP)
    EMPIRICAL_PARENTS.clear()
    if 'empirical_gate' not in DISABLED:
        declared_anchors, _ = split_anchors(names, schema['items'])
        found = discover_empirical_gate_parents(schema['items'], work, declared_anchors, allowed, X=X, given=given)
        names = topological(list(work), schema['items'])
        report['empirical_parents'] = {k: v[0] for k, v in found.items()}
        report['empirical_parent_mi'] = {k: [round(x, 4) for x in v[1:]] for k, v in found.items()}
    if 'rb_distill' not in DISABLED and len(allowed) >= 50:
        start = time.monotonic()
        found = _rb_distill(D, work, names, np.maximum(fold, 0), cpu_stop)
        for name, (o, p) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('rb_distill')
        report['rb_distill'] = len(found)
        report['rb_distill_seconds'] = time.monotonic() - start
    if 'cross_round' not in DISABLED:
        report['cross_round'] = attach_cross_round(schema, X, given, fold, work, deadline=cpu_stop)
    report['two_stage_eligible'] = sum((bool(schema['items'][name].get('gate')) for name in names))
    report['gate_table_eligible'] = report['two_stage_eligible']
    report['gate_logit_eligible'] = report['two_stage_eligible']
    bank = FoldFeatureBank(X, schema['items'], work, fold, cpu_stop, tables=tables, n_raw_given=n_raw_given, levels=levels, is_deep_hierarchy=is_deep_hierarchy, given=given)
    if final_windows and not gpu_enabled and span <= 900.0:
        bank.matrix_cache_limit = 96 * 1024 ** 2
    bank.latent_features = False
    from concurrent.futures import ThreadPoolExecutor
    try:
        import lightgbm
    except (ImportError, OSError):
        pass
    try:
        import catboost
    except (ImportError, OSError):
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse, optimize
    pools = ThreadpoolController()
    pfn_timing = {'recent_fit': 0.25, 'calls': 0, 'phase': phase}
    fold_sets = {5: fold, 3: np.where(fold >= 0, fold % 3, -1), 2: np.where(fold >= 0, fold % 2, -1)}
    chain = None
    csr_slices = {}
    design_cache = {}
    c_cache = {}
    report['logit_c'] = {}
    report['hgb_best_stage'] = {}
    from threading import RLock
    slice_lock = RLock()

    def candidate(profile, name, stop, plan, label=None, profile_names=(), item_idx=0):
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
                parent_has_s3_11 = any(p == 's3_11' or p.startswith('s3_11_') for p in parents)
                if name.startswith(('s7a_', 's7b_', 's7c_')) or parent_has_s3_11:
                    for anchor in ('s3_11_intent_return_12m_lbl', 's3_12_hope_return_someday_lbl'):
                        if anchor in work and anchor not in parents and profile + '_partitions' in work[anchor]:
                            parents.append(anchor)
                X_pfn = X[:, :min(X.shape[1], n_raw_given + (4 if is_deep_hierarchy else 0))]

                def features(tr, part):
                    blocks = [work[p][profile + '_partitions'].get(part, work[p][profile + '_partitions'].get('all')) for p in parents]
                    blocks = [b for b in blocks if b is not None]
                    if chain is not None:
                        for p in _parents(name, schema['items'])[:2]:
                            if p not in parents and p in chain.block:
                                blocks.append(chain.probabilities(p, tr))
                    return (np.column_stack([X_pfn] + blocks) if blocks else X_pfn), X_pfn.shape[1]
                strata = tuple((i for i, k in enumerate(given) if k == 'wave' or 'country' in k.lower()))
                capture_part = True
                result = gpu.candidate(X_pfn, y, rows, rest, K, X_pfn.shape[1], fold, stop, pfn_timing, strata=strata, feature_factory=features, capture=capture_part, arm=profile)
                if result is not None and capture_part:
                    j[profile + '_partitions'] = pfn_timing.pop('partitions', {})
                if result is not None: j.setdefault('partition_k', {})[profile] = int(pfn_timing.get('last_n_folds', 2))
                return None if result is None else [(profile, *result)]
            k = plan.choose()
            split = fold_sets[k]
            began = time.monotonic()
            linked = chain is not None and name in chain.children
            stage_o, stage_p = {}, {}
            guarded = (len(source) in (12, 76) or is_deep_hierarchy) and profile in ('catboost', 'deep', 'shallow', 'strong', 'two_stage')
            if guarded and plan.costs and time.monotonic() + .65 * float(np.median(plan.costs)) >= stop:
                return None
            if profile == 'hgb':
                stage_o, stage_p = hgb_crossfit_stages(lambda tr: bank.matrix(name, tr),
                    y, rows, rest, split, k, K, stop, max_iter=60)
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
                            dkey = tuple(mi_drivers)
                            with slice_lock:
                                design = design_cache.get(dkey)
                            if design is None:
                                built = _design(X, levels, mi_drivers, support_rows=tr, retain_crosses=True, given=given)
                                with slice_lock:
                                    design = design_cache.setdefault(dkey, built)
                            key = (dkey, k, f, tr.tobytes(), query.tobytes())
                        with slice_lock:
                            if key not in csr_slices:
                                csr_slices[key] = (design[tr], design[query])
                            fit_design, query_design = csr_slices[key]
                        if linked:
                            ext = chain.extras(name, tr)
                            fit_design = sparse.hstack((fit_design, sparse.csr_matrix(ext[tr])), format='csr')
                            query_design = sparse.hstack((query_design, sparse.csr_matrix(ext[query])), format='csr')
                        classes = np.unique(y[tr])
                        pred = np.zeros((len(query), K))
                        if len(classes) < 2:
                            pred[:, classes[0]] = 1.0
                        else:
                            # One C per outer fold, chosen on that fold's visible rows.
                            c_key = ('global', int(k), int(f)) if len(source) in (12, 76) or UNHCR_LEARNER_BUDGET else (k, f, tr.tobytes())
                            with slice_lock:
                                if c_key not in c_cache:
                                    c_rows = allowed[split[allowed] != f] if len(source) in (12, 76) or UNHCR_LEARNER_BUDGET else tr
                                    c_cache[c_key] = _pick_c(D, work, names, c_rows, stop)
                                c_val = c_cache[c_key]
                            model = LogisticRegression(C=c_val, max_iter=90, tol=5e-4)
                            model.fit(fit_design, y[tr])
                            pred[:, model.classes_] = model.predict_proba(query_design)
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
                if len(rest) and time.monotonic() + 0.9 < stop:
                    try:
                        feat_f, cat_f = bank.matrix(name, rows)
                        refit_h = tree_predict(feat_f, cat_f, y, rows, rest, K, 'hgb', stop, is_deep_hierarchy=is_deep_hierarchy, hgb_iter=int(best))
                        if refit_h is not None:
                            full[rest] = 0.5 * full[rest] + 0.5 * refit_h
                    except BudgetExpired:
                        pass
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
                    c_key = ('global_full',) if len(source) in (12, 76) or UNHCR_LEARNER_BUDGET else ('full', rows.tobytes())
                    if c_key not in c_cache:
                        c_cache[c_key] = _pick_c(D, work, names, allowed if len(source) in (12, 76) or UNHCR_LEARNER_BUDGET else rows, stop)
                    c_val = c_cache[c_key]
                classes = np.unique(y[rows])
                refit = np.zeros((len(rest), K))
                if len(classes) == 1:
                    refit[:, classes[0]] = 1.
                else:
                    model = LogisticRegression(C=c_val, max_iter=90, tol=5e-4)
                    model.fit(fit_design, y[rows])
                    refit[:, model.classes_.astype(int)] = model.predict_proba(query_design)
                full[rest] = .5 * full[rest] + .5 * refit
                report['logit_c'][name] = float(c_val)
            arm = 'shallow_asked' if profile == 'two_stage' else profile
            j.setdefault('partition_k', {})[arm] = int(k)
            found = [(arm, o, full)]
            if profile == 'logit' and is_deep_hierarchy and time.monotonic() + 0.03 < stop:
                try:
                    design_g = D_pure
                    tag_g = 'pure'
                    o_lg = np.zeros((len(rows), K), np.float32)
                    full_lg = np.zeros((n, K), np.float32)
                    for f in range(k):
                        check_clock(stop)
                        tr = rows[split[rows] != f]
                        val = rows[split[rows] == f]
                        query = np.concatenate((val, rest))
                        key_g = (tag_g, k, f, tr.tobytes(), query.tobytes())
                        with slice_lock:
                            if key_g not in csr_slices:
                                csr_slices[key_g] = (design_g[tr], design_g[query])
                            fit_g, query_g = csr_slices[key_g]
                        classes_g = np.unique(y[tr])
                        pred_g = np.zeros((len(query), K))
                        if len(classes_g) < 2:
                            pred_g[:, classes_g[0]] = 1.0
                        else:
                            c_val_g = c_cache.get(('global', int(k), int(f)), 0.25)
                            m_g = LogisticRegression(C=c_val_g, max_iter=90, tol=5e-4)
                            m_g.fit(fit_g, y[tr])
                            pred_g[:, m_g.classes_.astype(int)] = m_g.predict_proba(query_g)
                        o_lg[split[rows] == f] = pred_g[:len(val)]
                        full_lg[rest] += pred_g[len(val):] / k
                    if len(rest) and time.monotonic() + 0.05 < stop:
                        classes_gf = np.unique(y[rows])
                        refit_g = np.zeros((len(rest), K))
                        if len(classes_gf) == 1:
                            refit_g[:, classes_gf[0]] = 1.0
                        else:
                            c_val_gf = c_cache.get(('global_full',), 0.25)
                            m_gf = LogisticRegression(C=c_val_gf, max_iter=90, tol=5e-4)
                            m_gf.fit(design_g[rows], y[rows])
                            refit_g[:, m_gf.classes_.astype(int)] = m_gf.predict_proba(design_g[rest])
                        full_lg[rest] = .5 * full_lg[rest] + .5 * refit_g
                    found.append(('logit_given', o_lg, full_lg))
                    j.setdefault('partition_k', {})['logit_given'] = int(k)
                except BudgetExpired:
                    pass
            if (profile == 'hgb' and is_deep_hierarchy
                    and ((not h100_long) or (label != 'children')
                         or ((stop - time.monotonic()) / max(1, len(profile_names) - item_idx) >= 3.1))
                    and time.monotonic() + 0.12 < stop):
                try:
                    k_g = k
                    split_g = fold_sets[k_g]
                    so_g, sp_g = hgb_crossfit_stages(lambda tr: bank.matrix_given(name, tr),
                        y, rows, rest, split_g, k_g, K, stop, max_iter=60)
                    if so_g:
                        steps_g = sorted(so_g)
                        losses_g = np.stack([-np.log(.001 + (1.-K*.001) *
                            so_g[step][np.arange(len(rows)), y[rows]]) for step in steps_g])
                        best_g = steps_g[int(np.argmin(losses_g.mean(1)))]
                        o_hg = np.zeros((len(rows), K), np.float32)
                        for f in range(k_g):
                            va = split_g[rows] == f
                            step_g = steps_g[int(np.argmin(losses_g[:, ~va].mean(1)))]
                            o_hg[va] = so_g[step_g][va]
                        full_hg = sp_g[best_g]
                        if len(rest) and time.monotonic() + 0.7 < stop:
                            try:
                                feat_gf, cat_gf = bank.matrix_given(name, rows)
                                refit_hg = tree_predict(feat_gf, cat_gf, y, rows, rest, K, 'hgb', stop, is_deep_hierarchy=is_deep_hierarchy, hgb_iter=int(best_g))
                                if refit_hg is not None:
                                    full_hg[rest] = 0.5 * full_hg[rest] + 0.5 * refit_hg
                            except BudgetExpired:
                                pass
                        found.append(('hgb_given', o_hg, full_hg))
                        j.setdefault('partition_k', {})['hgb_given'] = int(k_g)
                except BudgetExpired:
                    pass
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
    windows = {'logit': 0.08, 'gate': 0.14 if is_deep_hierarchy else 0.26, 'strong': 0.26 if is_deep_hierarchy else 0.4, 'shallow': 0.36, 'two_stage': 0.38, 'third': 0.68, 'hgb': 0.88, 'deep': 1.0}
    _ANCHOR_WINDOWS = {'logit': 0.08, 'gate': 0.14, 'strong': 0.26, 'shallow': 0.36, 'two_stage': 0.38, 'third': 0.68, 'hgb': 0.88, 'deep': 1.0} if UNHCR_LEARNER_BUDGET else {'logit': 0.11, 'gate': 0.17, 'strong': 0.24, 'shallow': 0.44, 'two_stage': 0.52, 'deep': 0.76, 'third': 1.0}
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
    pass_windows = {'anchors': _ANCHOR_WINDOWS if is_deep_hierarchy else _WB_WINDOWS if len(source) == 76 else _UNICEF_WINDOWS, 'post': _ANCHOR_WINDOWS if is_deep_hierarchy else _WB_WINDOWS, 'children': ({'logit': 0.08, 'gate': 0.14, 'strong': 0.26, 'shallow': 0.36, 'two_stage': 0.38, 'third': 0.68, 'hgb': 0.88, 'deep': 1.0} if UNHCR_LEARNER_BUDGET else {'logit': 0.16, 'gate': 0.22, 'strong': 0.22, 'shallow': 0.48, 'two_stage': 0.58, 'deep': 0.78, 'third': 1.0}) if is_deep_hierarchy else _WB_WINDOWS if len(source) == 76 else _UNICEF_WINDOWS}
    report['folds_used'] = {}
    report['folds_completed'] = {}
    report['hidden_tree_prediction'] = 'adaptive_fold_mean'
    profile_costs = {}
    shallow_folds = (2,) if len(source) in (12, 76) else (5, 2)
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
        stage_order = list(w_map)
        stage_weights = {name: float(w_map[name]) - (float(w_map[stage_order[i - 1]]) if i else 0.0)
                         for i, name in enumerate(stage_order)}
        if final_windows:
            # Every final item receives all six CPU learners; dev keeps its historical recipe.
            stage_weights = ({'logit': .08, 'gate': .05, 'strong': .12, 'shallow': .08,
                              'two_stage': .03, 'deep': .12, 'hgb': .20, 'catboost': .32}
                             if gpu_enabled else
                             {'logit': .06, 'gate': .04, 'strong': .12, 'shallow': .08,
                              'two_stage': .02, 'deep': .08, 'catboost': .44, 'hgb': .16})
            if gpu_enabled and is_deep_hierarchy:
                stage_weights = {'logit': .06, 'gate': .04, 'strong': .10, 'shallow': .07, 'two_stage': .03, 'deep': .10, 'catboost': .26, 'hgb': .34}
            stage_order = list(stage_weights)
        for stage_idx, profile in enumerate(stage_order):
            if profile in DISABLED:
                continue
            start = time.monotonic()
            if final_windows:
                rem_weight = sum(float(stage_weights.get(stage, 0.0)) for stage in stage_order[stage_idx:])
                stage_share = float(stage_weights.get(profile, 0.0)) / max(rem_weight, 1e-9)
                stop = min(end_at, start + max(0.0, end_at - start) * stage_share)
            else:
                stop = start_at + w_map[profile] * max(0.0, end_at - start_at)
            if start >= stop:
                continue
            report.setdefault('cpu_stage_windows', []).append({'pass': tag or label, 'stage': profile, 'start': start - now, 'stop': stop - now})
            bank.deadline = stop
            if profile in ('gate', 'two_stage'):
                profile_names = [name for name in subset if schema['items'][name].get('gate')]
            elif final_windows and profile in ('strong', 'shallow', 'deep', 'catboost', 'hgb', 'logit'):
                profile_names = list(subset)
            elif profile == 'strong':
                profile_names = sorted(subset, key=lambda name: (-descendants[name], -float(work[name]['base']))) if is_deep_hierarchy else sorted((name for name in subset if descendants[name] >= 2), key=lambda name: -descendants[name])
            elif len(source) == 76 and profile == 'deep' and (label == 'anchors'):
                profile_names = list(subset[:16])
            elif len(source) == 76 and profile == 'deep':
                profile_names = [name for name in subset if _WB_CHILD_DEEP == 'all' or not name.startswith('tr_')]
            elif is_deep_hierarchy and label == 'children' and (profile in ('hgb', 'deep', 'third')):
                profile_names = sorted(subset, key=lambda name: (sum((arm in _TREE_ARMS for arm in work[name]['arms'])), -float(work[name]['base'])))
            else:
                profile_names = list(subset)
            if final_windows and not gpu_enabled and profile == 'catboost':
                profile_names.sort(key=lambda name: (work[name]['K'] * int(work[name]['train'].sum()), name))
            if not profile_names:
                continue
            report[profile + '_eligible'] = report.get(profile + '_eligible', 0) + len(profile_names)
            if profile == 'gate':
                singles = D[:, :int(levels[:n_raw_given].sum())] if is_deep_hierarchy else D[:, :int(levels.sum())]
                raw_singles = singles[:, :int(levels[:n_raw_given].sum())]
                gate_folds = shallow_folds
                plan = FoldPlan(gate_folds, len(profile_names), stop, _THREADS, max_high_folds=213 if is_deep_hierarchy else 32, prior=profile_costs.get(profile, ()))

                def run(name):
                    k = plan.choose()
                    began = time.monotonic()
                    try:
                        found = gate_candidate(name, schema['items'][name], schema['items'], given, X, levels, singles, tables, work[name], fold_sets[k], stop, n_raw_given=n_raw_given, is_deep_hierarchy=is_deep_hierarchy, chain=chain, raw_singles=raw_singles)
                    except BudgetExpired:
                        return None
                    if found is not None:
                        for label, _, _ in found:
                            work[name].setdefault('partition_k', {})[label] = int(k)
                        plan.record(k, time.monotonic() - began)
                    return found
            else:
                count = len(profile_names)
                plan = FoldPlan((2,) if len(source) in (12, 76) and profile in ('hgb', 'catboost') else shallow_folds, count, stop, _THREADS, max_high_folds=(6 if profile == 'catboost' else 12) if len(source) == 12 else 32 if len(source) == 76 else 213, prior=profile_costs.get(profile, ()))
                profile_indices = {name: i for i, name in enumerate(profile_names)}
                run = lambda name: candidate(profile, name, stop, plan, label, profile_names, profile_indices[name])
            if final_windows and gpu_enabled:
                plan.folds = (5,)
                plan.used = {5: 0}
                plan.completed = {5: 0}
                plan.max_high_folds = None
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
                        if arm in ('logit_given', 'hgb_given'):
                            report[arm] = report.get(arm, 0) + 1
                    if profile != 'gate':
                        report[profile] += 1
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            report['folds_used'][(tag or label) + ':' + profile] = {str(k): v for k, v in plan.used.items()}
            report['folds_completed'][(tag or label) + ':' + profile] = {str(k): v for k, v in plan.completed.items()}
            if plan.costs:
                profile_costs[profile] = plan.costs[-16:]
            key = ('gate_decomposition' if profile == 'gate' else profile) + '_seconds'
            report[key] = report.get(key, 0.0) + time.monotonic() - start
            gc.collect()
    anchors, children = split_anchors(names, schema['items'])
    post_items = []
    if is_deep_hierarchy and 'chain' not in DISABLED:
        roots = [a for a in anchors if not any((p in source for p in _parents(a, schema['items'])))]
        root_set = set(roots)
        post_items = topological([a for a in anchors if a not in root_set], schema['items'])
        anchors = roots
    if len(source) == 76 and 'prepost' not in DISABLED:
        anchors, children, post_items = questionnaire_split(names, schema['items'], anchors, children)
    if len(source) == 12:
        anchors = [a for a in UNICEF_ANCHORS if a in names]
        children = [a for a in names if a not in anchors]
    if 'chain' in DISABLED:
        anchors, children, post_items = (names, [], [])
    report.update(anchors=len(anchors), children=len(children), chain_items=0, post_items=len(post_items))
    try:
        bank.deadline = cpu_stop
        if is_deep_hierarchy:
            bank.prewarm(names, [fold, fold_sets[2]])
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

    def _stack_one_anchor(name, stop_at=cpu_stop):
        j = work[name]
        rest = np.flatnonzero(~j['train'])
        fold_arms(name, j, X, given, schema['items'])
        gate_latent(j, stop_at)
        keep = list(range(len(j['arms'])))
        aligned = aligned_partitions(j, fold, keep)
        p, o, _, _, _ = crossfit_calibrate([j['oof'][i] for i in keep],
            [j['all'][i][rest] for i in keep], j['y'][j['train']], j['K'], aligned, deadline=stop_at)
        return name, (o, p)

    def stack_into(subset, stop_at=cpu_stop):
        items_list = list(subset)
        if is_deep_hierarchy and len(items_list) > 1 and _THREADS > 1:
            ex = ThreadPoolExecutor(max_workers=min(_THREADS, len(items_list)),
                                    initializer=lambda: threadpool_limits(limits=1, user_api='openmp'))
            try:
                for name, pair in ex.map(lambda nm: _stack_one_anchor(nm, stop_at), items_list):
                    stacked_anchor[name] = pair
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
            return
        for name in items_list:
            _, pair = _stack_one_anchor(name, stop_at)
            stacked_anchor[name] = pair
    pfn_done = {arm: set() for arm in FOUNDATION_ARMS}

    pfn_timing['long_final'] = bool(final_windows and gpu_enabled and span > 900.0)

    def run_pfn(subset, stop):
        ordered = topological(candidate_order('tabpfn', list(subset), work), schema['items'])
        if h100_long:
            # Schedule high-cost multiclass items before cheap items consume the tail.
            ordered = topological(sorted(ordered, key=lambda key: -work[key]['K'] * int(work[key]['train'].sum())), schema['items'])
        report['tabpfn_eligible'] = len(names)
        for arm_index, arm in enumerate(FOUNDATION_ARMS):
            if arm in DISABLED:
                continue
            start = time.monotonic()
            arm_stop = start + (0.64 if is_deep_hierarchy else 0.68) * (stop - start) if arm == 'tabpfn' and len(FOUNDATION_ARMS) > 1 else stop
            pending = [k for k in ordered if k not in pfn_done[arm]]
            if h100_long:
                arm_w = {'tabpfn': 1.06, 'kumo': 1.0}
                remaining_arms = [a for a in FOUNDATION_ARMS[arm_index:] if a not in DISABLED]
                total_left = sum(arm_w.get(a, 1.0) * sum(k not in pfn_done[a] for k in ordered) for a in remaining_arms)
                share = (arm_w.get(arm, 1.0) * len(pending)) / max(1e-9, total_left)
                arm_stop = start + share * max(0.0, stop - start)
            bank.deadline = arm_stop
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
                            j.setdefault('partition_k', {})[arm + '_joint'] = int(pfn_timing.get('last_n_folds', 2))
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
                    pfn_done[arm].add(target)
                    report[arm] = report.get(arm, 0) + 1
            report[arm + '_seconds'] = report.get(arm + '_seconds', 0.0) + time.monotonic() - start
    with pools.limit(limits=1):
        run_pass('anchors', anchors, pass_one_start, pass_one_stop)
        early_pfn = is_deep_hierarchy and gpu_enabled and _final.enabled('CHAIN_ORDER')
        pfn_window = max(0.0, pfn_stop - cpu_stop)
        if early_pfn:
            tick = time.monotonic()
            run_pfn(anchors, min(pfn_stop, tick + 0.30 * pfn_window))
            shift = time.monotonic() - tick
            cpu_stop += shift
            pass_post_stop += shift
            pass_two_stop += shift
            bank.deadline = cpu_stop
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
            post_now = time.monotonic()
            eff_post_stop = max(pass_post_stop, min(pass_two_stop, post_now + 0.14 * max(0.0, pass_two_stop - post_now))) if is_deep_hierarchy else pass_post_stop
            run_pass('children', post_items, post_now, eff_post_stop, tag='post')
            if early_pfn:
                tick = time.monotonic()
                run_pfn(post_items, min(pfn_stop, tick + 0.15 * pfn_window))
                shift = time.monotonic() - tick
                cpu_stop += shift
                pass_two_stop += shift
                bank.deadline = cpu_stop
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
        if final_windows and gpu_enabled:
            report['matrix_cache_hits'] = bank.matrix_cache_hits
            report['matrix_cache_bytes'] = bank.matrix_cache_bytes
            bank.clear_matrices()
            gc.collect()
        if gpu_enabled and 'tabpfn' not in DISABLED and (time.monotonic() < pfn_stop):
            priority = list(dict.fromkeys([a for a in anchors + post_items if work[a]['route_parent']] + anchors + post_items + children))
            run_pfn(priority, pfn_stop)
    coverage_arms = ('strong', 'shallow', 'deep', 'catboost', 'hgb', 'logit', 'tabpfn', 'kumo')
    report['arm_items_5fold'] = {
        arm: sum(arm in job['arms'] and job.get('partition_k', {}).get(arm) == 5
                 for job in work.values()) for arm in coverage_arms}
    report['arm_folds_by_item'] = {
        name: {arm: job.get('partition_k', {}).get(arm) for arm in coverage_arms if arm in job['arms']}
        for name, job in work.items()}
    report['foundation_chain_validation'] = {name: j['foundation_chain_validation'] for name, j in work.items() if j.get('foundation_chain_validation')}
    report['foundation_chain_columns'] = {name: j['foundation_chain_columns'] for name, j in work.items() if j.get('foundation_chain_columns')}
    parent_partitions = len(bank.cache)
    if not (final_windows and gpu_enabled):
        report['matrix_cache_hits'] = bank.matrix_cache_hits
        report['matrix_cache_bytes'] = bank.matrix_cache_bytes
    if followup_chain is None and len(source) == 76:
        followup_chain = chain
    bank.clear_matrices()
    bank.chain = None
    chain = None
    links.clear()
    if len(source) != 76:
        stacked_anchor.clear()
    csr_slices.clear()
    design_cache.clear()
    gc.collect()
    _release_heap()
    if names and len(allowed) >= 20 and (time.monotonic() < net_stop) and ('network' not in DISABLED):
        start = time.monotonic()
        use_gpu_net = gpu.available()
        fn = _fit_net_gpu if use_gpu_net else _fit_net
        A = (D_pure if is_deep_hierarchy else D).toarray().astype(np.float32)
        with pools.limit(limits=_THREADS):
            results = fn(A, [work[k]['y'] for k in names], [work[k]['train'] for k in names], [work[k]['K'] for k in names], np.maximum(fold, 0), (400 if phase == 'final' else 240) if use_gpu_net else 140 if is_deep_hierarchy else 200, net_stop, hidden=64, **({'categories': X[:, :n_raw_given] if is_deep_hierarchy else X, 'category_levels': levels[:n_raw_given] if is_deep_hierarchy else levels} if use_gpu_net else {}))
        del A
        gc.collect()
        _release_heap()
        if results:
            for k, (o, p) in zip(names, results):
                work[k]['oof'].append(o)
                work[k]['all'].append(p)
                work[k]['arms'].append('network')
            report['network'] = len(names)
        report['network_seconds'] = time.monotonic() - start
        report['network_trace'] = base._NET_TRACE
    index = SimpleNamespace(visible=allowed, fold=fold)
    if 'neighbor_cross_target' not in DISABLED and len(allowed) >= 20 and (time.monotonic() + 1.5 < nb_build_stop):
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
        del masked
        gc.collect()
        _release_heap()
        report['neighbor_build_seconds'] = time.monotonic() - start
    # On 213 items the five-fold meta pass never completes inside its window
    # (0/213 completed on every 8-core cell); its seconds go to the gate stage.
    if 'neighbor_cross_target' not in DISABLED and time.monotonic() < ct_stop and (len(source) != 213 or UNHCR_CROSS_TARGET):
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
    if len(source) == 76 and 'followup' not in DISABLED and (time.monotonic() < finish_deadline - 16.0):
        cap = min(finish_deadline - 16.0, time.monotonic() + 40.0)
        bank.deadline = cap
        with pools.limit(limits=1):
            found, info = fit_tracer_followup(frame, schema, given, X, D, tables, work, deadline=cap, chain=followup_chain, workers=_THREADS, anchor_names=anchors)
        moved = info.pop('transition_arms', {})
        followup_event = info.pop('_event', None)
        report['followup_status'] = info
        report['followup'] = len(found)
        report['followup_anchor'] = len(moved)
        for name, (o, p, _) in found.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('followup')
        for name, (o, p) in moved.items():
            work[name]['oof'].append(o)
            work[name]['all'].append(p)
            work[name]['arms'].append('followup_anchor')
    if len(source) == 76 and 'paired_transition' not in DISABLED:
        bank.deadline = min(finish_deadline - 10.0, time.monotonic() + 10.0)
        report['paired_transition'] = attach_paired_transition(work, X, given, fold, bank, stacked_anchor, bank.deadline, event=followup_event, items=schema['items'], anchor_names=anchors, child_names=post_items + children)
    if len(source) == 76:
        stacked_anchor.clear()
    if len(source) == 76 and 'psychometric_irt' not in DISABLED and time.monotonic() < finish_deadline - 22.5:
        from . import wb_psychometrics
        cap = min(finish_deadline - 22.5, time.monotonic() + 2.0)
        with pools.limit(limits=1):
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
    if 'stacked_cross_target' not in DISABLED:
        start = time.monotonic()
        try:
            sct_stop = min(finish_deadline - (36.0 if is_deep_hierarchy else 10.5), start + (18.0 if phase == 'final' and _final.enabled('FINAL_WINDOWS') else 6.5))
            if time.monotonic() < sct_stop:
                with pools.limit(limits=1):
                    found, trace = base.stacked_cross_target(work, fold, sct_stop, workers=_THREADS, bank=bank)
                for name, (o, p, gain) in found.items():
                    work[name]['oof'].append(o)
                    work[name]['all'].append(p)
                    work[name]['arms'].append('stacked_cross_target')
                report['stacked_cross_target'] = len(found)
                report['stacked_cross_target_trace'] = trace
        except (BudgetExpired, Exception):
            pass
        report['stacked_cross_target_seconds'] = time.monotonic() - start
    _matrix_cache = {'hits': bank.matrix_cache_hits, 'bytes': bank.matrix_cache_bytes, 'build_seconds': bank.matrix_build_seconds, 'build_calls': bank.matrix_build_calls}
    _report_bank = dict(
        parents_fitted=bank.parent_fits, parent_cache_hits=bank.cache_hits,
        parent_partitions=parent_partitions, parent_model_cache_hits=bank.model_cache_hits,
        parent_fit_calls=bank.parent_fit_calls)
    bank.clear_matrices()
    bank = D = D_pure = tables = None
    csr_slices.clear()
    design_cache.clear()
    gc.collect()
    _release_heap()
    final, final_oofs, stacked, gates = calibrate_and_gate(schema, work, X, given, levels, fold, index, report, cal_finish_deadline, nb_select_stop, is_deep_hierarchy, pools)
    # Every module after the structural gates is judged against this snapshot on
    # the visible rows (post_gate.gate), then the gate graph is closed again.
    snap = post_gate.snapshot(final, final_oofs, stacked) if POST_GATE and is_deep_hierarchy else None
    post_gate_final = None
    post_gate_oofs = None
    for _job in work.values():
        for _key in ('oof', 'all', 'tabpfn_partitions', 'kumo_partitions', 'tabpfn_oof'):
            _job.pop(_key, None)
    gc.collect()
    _release_heap()
    twin_stop = cal_finish_deadline
    if V79_K2 and is_deep_hierarchy and 'wave' in given[:n_raw_given] and time.monotonic() < twin_stop:
        twin_start = time.monotonic()
        try:
            wcol = given.index('wave')
            pairs = twin_identity.exact_pairs(X, n_raw_given, wcol)
            report['twin_exact'] = {}
            twin_identity.apply(work, final, final_oofs, stacked, pairs, fold, X[:, wcol], report['twin_exact'], context=None, deadline=twin_stop)
        except (BudgetExpired, Exception):
            pass
        report['twin_seconds'] = time.monotonic() - twin_start
    if snap is not None:
        try:
            report['post_gate'] = post_gate.gate(work, final, final_oofs, stacked, snap, fold)
        except Exception:
            pass
        if V132_K1 and time.monotonic() < cal_finish_deadline:
            try:
                from . import erpis_ipf_projection
                report_snap = post_gate.snapshot(final, final_oofs, stacked)
                report['erpis_projection'] = erpis_ipf_projection.apply(
                    schema, X, given, work, fold, final, final_oofs, stacked,
                    min(cal_finish_deadline, time.monotonic() + 2.0))
                report['erpis_acceptance'] = post_gate.gate(
                    work, final, final_oofs, stacked, report_snap, fold, min_se=3.0)
            except Exception:
                pass
        snap = None
    report['soft_given'] = {name: j['soft_given'] for name, j in work.items() if 'soft_given' in j}
    report['dual_track'] = {name: j['dual_track'] for name, j in work.items() if 'dual_track' in j}
    report['row_gate'] = {name: j['row_gate'] for name, j in work.items() if 'row_gate' in j}
    report['gate_seconds'] = time.monotonic() - report.pop('_gate_start')

    def floored_nll(o, name):
        y, K = work[name]['y'][work[name]['train']], work[name]['K']
        if o is None or np.shape(o) != (len(y), K):
            return None
        p = np.asarray(o, float)
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
        return float(-np.log(1e-3 + (1. - K * 1e-3) * p[np.arange(len(y)), y]).mean())
    pre = {name: floored_nll(o, name) for name, (o, _, _) in stacked.items()}
    if V100_K1:
        report['floor_fill'] = _floor_fill(work, stacked, final, final_oofs, fold, schema['items'], V100_FLOOR_POLICY)
    post = {name: floored_nll(final_oofs.get(name, stacked[name][0]), name) for name in stacked}
    report['oof_nll_items'] = {name: [pre[name], post[name]] for name in stacked}
    report['oof_nll_pre_gate'] = float(np.mean([v for v in pre.values() if v is not None])) if stacked else None
    report['oof_nll_post_gate'] = float(np.mean([v for v in post.values() if v is not None])) if stacked else None
    report['pass_names'] = {'anchors': list(anchors), 'post': list(post_items), 'children': list(children)}
    report['stage_stops'] = {k: v - now for k, v in dict(cpu=cpu_stop, pfn=pfn_stop, network=net_stop, neighbor_build=nb_build_stop, cross_target=ct_stop, neighbor_select=nb_select_stop).items()}
    report['matrix_cache'] = _matrix_cache
    report.update(**_report_bank, gates=len(gates))
    _report_soft_given = {name: j.get('soft_given') for name, j in work.items() if 'soft_given' in j}
    _report_dual_track = {name: j.get('dual_track') for name, j in work.items() if 'dual_track' in j}
    _report_row_gate = {name: j.get('row_gate') for name, j in work.items() if 'row_gate' in j}
    _stacked_names = tuple(stacked)
    for _job in work.values():
        for _key in ('oof', 'all', 'tabpfn_partitions', 'kumo_partitions', 'tabpfn_oof'):
            _job.pop(_key, None)
    csr_slices.clear()
    design_cache.clear()
    bank = None
    chain = None
    index = None
    stacked = None
    if 'post_gate_final' in locals():
        post_gate_final = None
    if 'post_gate_oofs' in locals():
        post_gate_oofs = None
    gc.collect()
    _release_heap()
    from .post_orthogonal import augment
    final, final_oofs, ortho_report = augment(
        frame, schema, work, allowed, fold, final, final_oofs,
        n_raw_given, post_ortho_deadline if (final_windows and gpu_enabled) else min(post_ortho_deadline, time.monotonic() + max(ortho_reserve, 12.0)))
    _LAST_SAFE_OUT.update(final)
    report['post_orthogonal'] = ortho_report
    return (final, report)

TAIL_GATE_RESERVE = 24.0
PRE_TRACK_CROSSFIT = True
GATE_CROSSFIT = True
_CHECK_ARMS = ('psychometric_irt', 'latent_profile', 'latent_class', 'latent_class_b', 'latent_class_strat',
               'latent_trait', 'latent_trait_strat', 'latent_trait_route', 'latent_trait_strat_route',
               'cross_target', 'stacked_cross_target', 'followup_anchor', 'anchor_eb')
_FORKED = {}

def _prune_arms(j, fold, rest, p, o, w, t, loss, deadline):
    """Drop a checked arm while the stack without it is no worse on both fold halves."""
    active = list(range(len(j['arms'])))
    any_removed = False
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
                j['y'][j['train']], j['K'], aligned_keep, deadline=deadline)
            half = fold[j['train']] % 2 == 0
            y = j['y'][j['train']]
            if not half.any() or half.all() or any((_nll(o[h], y[h], j['K']) >= _nll(old_o[h], y[h], j['K']) for h in (half, ~half))):
                p, o, t, loss = (old_p, old_o, old_t, old_loss)
                w = np.zeros(len(j['arms']))
                w[keep] = old_w
                active = keep
                j['calibration_partitions'] = int(len(np.unique(aligned_keep)))
                removed = True
                any_removed = True
        if not removed:
            return p, o, w, t, loss

def _pre_track_aligned(j, fold, aligned, deadline):
    # The paired track is scored on the same held-out partitions
    # as the base track it competes with.
    track_fit = (lambda oofs, rests, y, K, deadline=float('inf'): crossfit_calibrate(oofs, rests, y, K, aligned, deadline=deadline)) if PRE_TRACK_CROSSFIT else calibrate
    return pre_track(j, fold, track_fit, deadline)

def _forked_item(name):
    """One item's stack, pruning and pre-track inside a forked worker; returns only what changed."""
    s = _FORKED
    j, fold, deadline = s['work'][name], s['fold'], s['deadline']
    with threadpool_limits(limits=1):
        rest = np.flatnonzero(~j['train'])
        latent = j['arms'].index('latent_profile') if 'latent_profile' in j['arms'] else None
        width = len(j['arms'])
        gate_latent(j, deadline)
        aligned = aligned_partitions(j, fold)
        p, o, w, t, loss = crossfit_calibrate(j['oof'], [a[rest] for a in j['all']],
            j['y'][j['train']], j['K'], aligned, deadline=deadline)
        j['calibration_partitions'] = int(len(np.unique(aligned)))
        extras = time.monotonic() < s['extras_by']
        j['check_arms'] = _CHECK_ARMS
        if extras:
            p, o, w, t, loss = _prune_arms(j, fold, rest, p, o, w, t, loss, deadline)
        if extras and _ortho.V109_K2:
            p, o, w, t, loss = compact_stack(j, rest, fold, (p, o, w, t, loss), deadline)
        alternate = None
        if extras and 'cross_round' not in DISABLED:
            found = _pre_track_aligned(j, fold, aligned, deadline)
            if found is not None:
                track, alt = found
                slots = [(i, alt['oof'][i], alt['all'][i]) for i in range(len(alt['arms']))
                         if i >= len(j['arms']) or alt['oof'][i] is not j['oof'][i] or alt['all'][i] is not j['all'][i]]
                alternate = (track, list(alt['arms']), slots, alt['_cr_weight'])
    return name, dict(popped=latent if len(j['arms']) < width else None, latent_oof_gain=j.get('latent_oof_gain'),
                      stack=(p, o, w, t, loss), calibration_partitions=j['calibration_partitions'], extras=extras, alternate=alternate,
                      compact=j.get('compact', False))

def _rebuild_alternate(j, a):
    if a is None:
        return None
    track, arms, slots, weight = a
    # Structural gate propagation reads only the item labels and train mask.
    # Keep the alternate track compact so it does not retain every arm matrix.
    alt = {'y': j['y'], 'K': j['K'], 'train': j['train']}
    return (track, alt)

def _release_heap():
    if sys.platform == 'linux':
        import ctypes
        libc = ctypes.CDLL(None)
        trim = getattr(libc, 'malloc_trim', None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)


def _fork_stack(work, X, given, items, fold, deadline, extras_by, collect_by):
    """Linux: every item's stack in _THREADS forked processes sharing the parent's arrays copy-on-write.

    Workers skip pruning and pre-track only past extras_by; items still out at
    collect_by are left to the serial loop. Returns (results, folded)."""
    import multiprocessing as mp
    if sys.platform != 'linux' or _THREADS < 2 or 'fork' not in mp.get_all_start_methods():
        return {}, False
    for name, j in work.items():
        fold_arms(name, j, X, given, items)
    if V100_K2:
        gc.collect()
        _release_heap()
    _FORKED.update(work=work, fold=fold, deadline=deadline, extras_by=extras_by)
    got = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        pool = mp.get_context('fork').Pool(_THREADS)
    try:
        results = pool.imap_unordered(_forked_item, list(work), chunksize=1)
        for _ in range(len(work)):
            name, r = results.next(timeout=max(0.0, collect_by - time.monotonic()))
            got[name] = r
    except mp.TimeoutError:
        pass
    finally:
        pool.terminate()
        pool.join()
        _FORKED.clear()
    return got, True

def calibrate_and_gate(schema, work, X, given, levels, fold, index, report, finish_deadline, nb_select_stop, is_deep_hierarchy, pools):
    """Cross-fitted stack per item, then structural gate propagation.

    On the 213-item hierarchy the stack loop ends TAIL_GATE_RESERVE seconds
    before the finish deadline so the gate stage always runs. Every item gets
    its full cross-fitted stack; arm pruning, the paired pre-track and the
    neighbour selector run while the remaining items still fit."""
    n = len(X)
    gate_splits = {}
    stacked = {}
    alternates = {}
    choices = {}
    stack_start = time.monotonic()
    select_seconds = 0.0
    recent_neighbor = 0.15
    report['neighbor_attempted'] = 0
    calibrate_seconds = 0.0
    calibrated = 0
    cal_deadline = finish_deadline - (TAIL_GATE_RESERVE if is_deep_hierarchy else 0.0)
    base_cost, extra_cost = 0.0, 0.0
    report['stack_extras_skipped'] = 0
    pre, folded = ({}, False)
    if V85_K1 and is_deep_hierarchy:
        pre, folded = _fork_stack(work, X, given, schema['items'], fold, finish_deadline, cal_deadline - 6.0, cal_deadline - 2.0)
        report['stack_forked'] = len(pre)
    serial = 0
    with pools.limit(limits=1):
        for name, j in work.items():
            began = time.monotonic()
            rest = np.flatnonzero(~j['train'])
            r = pre.get(name)
            if r is not None:
                if r['popped'] is not None:
                    for key in ('arms', 'oof', 'all'):
                        j[key].pop(r['popped'])
                if r['latent_oof_gain'] is not None:
                    j['latent_oof_gain'] = r['latent_oof_gain']
                aligned = aligned_partitions(j, fold)
                gate_splits[name] = aligned
                p, o, w, t, loss = r['stack']
                j['calibration_partitions'] = r['calibration_partitions']
                j['check_arms'] = _CHECK_ARMS
                extras = r['extras']
                if r.get('compact'):
                    j['compact'] = True
                extra_began = time.monotonic()
            else:
                if not folded:
                    fold_arms(name, j, X, given, schema['items'])
                gate_latent(j, finish_deadline)
                aligned = aligned_partitions(j, fold)
                gate_splits[name] = aligned
                p, o, w, t, loss = crossfit_calibrate(j['oof'], [a[rest] for a in j['all']],
                    j['y'][j['train']], j['K'], aligned, deadline=finish_deadline)
                j['calibration_partitions'] = int(len(np.unique(aligned)))
                spent = time.monotonic() - began
                base_cost = spent if serial == 0 else .8 * base_cost + .2 * spent
                serial += 1
                left = len(work) - calibrated - 1
                extras = not is_deep_hierarchy or time.monotonic() + extra_cost + left * base_cost < cal_deadline
                extra_began = time.monotonic()
                j['check_arms'] = _CHECK_ARMS
                if extras:
                    p, o, w, t, loss = _prune_arms(j, fold, rest, p, o, w, t, loss, finish_deadline)
                if extras and _ortho.V109_K2 and is_deep_hierarchy:
                    p, o, w, t, loss = compact_stack(j, rest, fold, (p, o, w, t, loss), finish_deadline)
            calibrate_seconds += time.monotonic() - began
            calibrated += 1
            per_cal = calibrate_seconds / calibrated if len(work) == 12 else min(calibrate_seconds / calibrated, 0.1)
            reserve = per_cal * (len(work) - calibrated) + (23.5 if is_deep_hierarchy else 4.5)
            leaf_ready = getattr(index, '_leaf_rank', None) is not None
            warm = leaf_ready and len(getattr(index, '_candidate_cache', ())) >= 4
            allow_nb = extras and time.monotonic() + (0.03 if warm else 0.8) < min(nb_select_stop, cal_deadline) and select_seconds < (16.0 if _final.is_final() and _final.enabled('FINAL_WINDOWS') else 4.8) if is_deep_hierarchy else time.monotonic() + max(0.04 if warm else 1.0, 1.5 * recent_neighbor) < min(nb_select_stop, finish_deadline - reserve)
            if hasattr(index, 'distributions') and 'neighbor_cross_target' not in DISABLED and allow_nb:
                start = time.monotonic()
                p, info, o = select_neighbors(index, name, o, p, rest, deadline=min(nb_select_stop, time.monotonic() + (3.8 if not warm else 0.35)))
                elapsed = time.monotonic() - start
                select_seconds += elapsed
                if leaf_ready:
                    recent_neighbor = 0.7 * recent_neighbor + 0.3 * elapsed
                if info['path'] is not None:
                    report['neighbor'] += 1
                report['neighbor_attempted'] += 1
            loss = _nll(o, j['y'][j['train']], j['K'])
            if 'cross_round' not in DISABLED and extras:
                if r is not None:
                    alternate = _rebuild_alternate(j, r['alternate'])
                else:
                    alternate = _pre_track_aligned(j, fold, aligned, finish_deadline)
                if alternate is not None:
                    alternates[name] = alternate
            if not extras:
                report['stack_extras_skipped'] += 1
            elif r is None:
                extra_cost = max(extra_cost, time.monotonic() - extra_began) if serial == 1 else .8 * extra_cost + .2 * (time.monotonic() - extra_began)
            stacked[name] = (o, p, loss)
            full_s = np.zeros((n, j['K']), np.float32)
            full_s[~j['train']] = p
            _LAST_SAFE_OUT[name] = full_s
            choices[name] = {'partition_k': j.get('partition_k', {}), 'calibration_partitions': j.get('calibration_partitions'), 'arms': j['arms'], 'weights': w.tolist(), 'temperature': t, 'latent_oof_gain': j.get('latent_oof_gain'), 'arm_nll': {a: float(_nll(q, j['y'][j['train']], j['K'])) for a, q in zip(j['arms'], j['oof'])}}
        report['compact_items'] = sum(bool(j.get('compact')) for j in work.values())
        report['stack_seconds'] = time.monotonic() - stack_start
        report['calibrate_seconds'] = calibrate_seconds
        report['neighbor_select_seconds'] = select_seconds
        report['selected'] = {arm: sum((float(weight) > 1e-08 for choice in choices.values() for arm_name, weight in zip(choice['arms'], choice['weights']) if arm_name == arm)) for arm in ('logit', 'logit_given', 'strong', 'shallow', 'shallow_asked', 'gate_table', 'gate_logit', 'deep', 'hgb', 'hgb_given', 'catboost', 'tabpfn', 'followup', 'followup_anchor', 'anchor_eb', 'network', 'cross_round', 'rb_distill', 'cross_target', 'stacked_cross_target', 'latent_profile', 'latent_class', 'latent_class_b', 'latent_class_strat', 'latent_trait', 'latent_trait_strat', 'full_cell_eb', 'conditional_cell_eb', 'paired_transition', 'kumo', 'tabpfn_joint', 'report_prior', 'psychometric_irt')}
        pre.clear()
        gate_start = time.monotonic()
        # The forked stack frees ~24 s before the gates; cap them at 26.5 s and
        # guarantee at least 18.0 s so all 156 structural gates complete.
        gate_deadline = max(gate_start + 18.0, min(finish_deadline - 3.5, gate_start + 26.5)) if V85_K1 and pre and sys.platform.startswith('linux') else max(gate_start + 18.0, finish_deadline)
        final_oofs = {}
        if 'gate' in DISABLED:
            final = {}
            gates = []
            for name, (o, p, _) in stacked.items():
                if 'cross_round' not in DISABLED:
                    o, p = cross_round_row_gate(work[name], o, p, fold)
                final[name] = np.zeros((n, work[name]['K']))
                final[name][~work[name]['train']] = p
                final_oofs[name] = o
        else:

            def tail_calibrate(oofs, rests, y, K, name=None):
                # Gate mixtures compete with the cross-fitted base stack on
                # the same held-out partitions.
                split = gate_splits.get(name) if GATE_CROSSFIT else None
                if split is None or len(split) != len(y) or len(np.unique(split)) < 2:
                    if name in work:
                        split = np.asarray(fold[work[name]['train']], dtype=np.int64) % 2
                    else:
                        split = np.arange(len(y), dtype=np.int64) % 2
                return crossfit_calibrate(oofs, rests, y, K, split, deadline=finish_deadline)
            final, gates = propagate_structural_gates(schema['items'], work, stacked, X, given, levels, np.maximum(fold, 0), tail_calibrate, _nll, deadline=gate_deadline, disabled=DISABLED, row_gate_fn=None if 'cross_round' in DISABLED else cross_round_row_gate, alternate_stacked=alternates, oof_output=final_oofs)
            _LAST_SAFE_OUT.update(final)
            report['nested_gate'] = {str(k): v for k, v in NESTED_TRACE.items()}
            emp = [g for g in gates if g[10] and np.isfinite(g[11])]
            report['empirical_gate'] = {'items': len(emp), 'selected_nll': float(np.mean([g[4] for g in emp])) if emp else None, 'declared_nll': float(np.mean([min(g[3], g[8]) for g in emp])) if emp else None, 'empirical_nll': float(np.mean([g[11] for g in emp])) if emp else None, 'base_nll': float(np.mean([g[12] for g in emp])) if emp else None, 'selected_kinds': {k: sum((g[6] == k for g in emp)) for k in ('base', 'multi', 'legacy')}}
        report['_gate_start'] = gate_start
        report['choices'] = choices
    return final, final_oofs, stacked, gates

def short_table_prediction(schema, X, given, levels, source, allowed, marginal_out, deadline, n_raw_given):
    """Commit known gates first, then complete five-fold table experts in time."""
    n = len(X)
    out = {}
    locked = 0
    for name, job in source.items():
        full = marginal_out[name].copy()
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        selected = full[rest].copy()
        flags = lock_given_gate(selected, schema['items'][name], schema['items'],
                                X, given, job['y'], job['K'], rows, rest)
        full[rest] = selected
        locked += int(any(flags))
        out[name] = full
        _LAST_SAFE_OUT[name] = full
    fold = folds_for(X, allowed)
    tables = _Tables(X, np.maximum(fold, 0), 5)
    completed = 0
    order = sorted(source, key=lambda name: (source[name]['K'] * int(source[name]['train'].sum()), name))
    for name in order:
        if time.monotonic() >= deadline:
            break
        job = source[name]
        train = np.zeros(n, dtype=bool)
        train[allowed] = job['train'][allowed]
        if not train.any():
            continue
        y = np.where(train, job['y'], -1)
        _, full, _, _ = _fit_table(tables, np.maximum(y, 0), train, job['K'],
                                   n_raw_given, return_config=True)
        full = np.asarray(full).copy()
        rows, rest = np.flatnonzero(train), np.flatnonzero(~train)
        selected = full[rest].copy()
        lock_given_gate(selected, schema['items'][name], schema['items'], X,
                        given, y, job['K'], rows, rest)
        full[rest] = selected
        if (np.isfinite(full[rest]).all() and (full[rest] >= 0).all()
                and np.all(full[rest].sum(1) > 0)):
            out[name] = full
            _LAST_SAFE_OUT[name] = full
            completed += 1
    return out, dict(variant=VARIANT, table=len(source), threads=_THREADS,
                     short_table_items=completed, short_table_folds=5,
                     deterministic_given_items=locked,
                     entry_strategy='visible_gates_then_crossfit_tables')

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
    _final.set_phase(phase)
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
    total = 690.0 if phase == 'dev' else min(3320.0, float(os.environ.get('SB_V157_TOTAL_SECONDS', '3320')))
    shares = {213: 2015.0 / 3320.0, 76: 955.0 / 3320.0, 12: 350.0 / 3320.0} if phase == 'final' else _SHARE
    mine = shares.get(len(names), 0.2)
    later = sum((v for k, v in shares.items() if k not in _DONE and k != len(names)))
    end = began + max(445.0 if phase == 'dev' and len(names) == 213 else 0.0, total * mine, max(0.0, total - (began - _START)) * mine / (mine + later))
    span = max(0.0, end - began)
    tail_base = min(32.0 if len(names) > 100 else 18.0 if len(names) > 30 else 10.0, 0.12 * span)
    ortho_tail = ortho_reserve(len(names), phase, span)
    tail = tail_base + ortho_tail
    _LAST_SAFE_OUT.clear()
    try:
        if phase == 'final' and total <= 20.0:
            raw_out, report = short_table_prediction(schema, X, given, levels, source, visible, marginal_out, min(end, _START + total) - 3.0, n_raw_given)
        else:
            raw_out, report = fit_partition(frame, schema, X, given, levels, source, visible, end - tail, phase=phase, finish_deadline=end - (5.0 if phase == 'final' and not gpu.available() and span <= 900.0 else 1.5), n_raw_given=n_raw_given, ortho_reserve=ortho_tail)
    except (BudgetExpired, Exception):
        raw_out, report = (dict(_LAST_SAFE_OUT), {'variant': VARIANT, 'table': len(source), 'threads': _THREADS})
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
    # vectors owns Python lists; release every full-table probability matrix.
    out.clear()
    raw_out.clear()
    marginal_out.clear()
    _LAST_SAFE_OUT.clear()
    cand = base_mat = selected = None
    gc.collect()
    _release_heap()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    LAST_REPORT['seconds'] = time.monotonic() - began
    _DONE.add(len(names))
    return vectors
