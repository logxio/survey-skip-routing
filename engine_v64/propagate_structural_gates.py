"""Empirical two-stage gate propagation over visible survey answers.

Each target learns a skip probability and an answer distribution from distinct
visible subsets. Parent predictions are consumed in schema topological order.
The schema's observed_if values describe routing; observed answers, including
off-route answers, determine the transition probabilities.
"""

import time
import numpy as np
from .v64_config import V128_K2, V128_K3


# The labels below are literal values from data/unhcr.json.  This table is a
# description of additional *features* for an empirical transition, not a
# deterministic routing rule: the schema says observed_if is often violated.
_SAFETY = (
    "s3_23_safety_active_conflict_lbl",
    "s3_23_safety_law_enforcement_lbl",
    "s3_23_safety_tribal_conflict_lbl",
    "s3_23_safety_armed_actors_lbl",
    "s3_23_safety_explosives_lbl",
    "s3_23_safety_public_spaces_lbl",
    "s3_23_safety_no_concerns_lbl",
    "s3_23_safety_theft_lbl",
    "s3_23_safety_kidnapping_lbl",
    "s3_23_safety_other",
)
GATE_CORRECTIONS = {
    "s2_9_property_docs_lbl": (
        ("s2_7_own_house_syria_lbl", "s2_8_own_land_syria_lbl"),
        (("1. Yes",), ("1. Yes",)), "or"),
    "s6_12_children_schooling_lbl": (
        ("s2_3_boys_6_17", "s2_3_girls_6_17"),
        (tuple(("1", "2", "3", "4", "5", "6-10", "11+")),) * 2, "or"),
    "s3_23a_safety_rank1_lbl": (_SAFETY, (("1",),) * 10, "count>=1"),
    "s3_23b_safety_rank2_lbl": (_SAFETY, (("1",),) * 10, "count>=2"),
    **{
        name: (("s4_0_hh_anyone_working_lbl", "s4_1_is_main_earner_lbl"),
               (("1. Yes",), ("0. No",)), "and")
        for name in (
            "s4_3_main_earner_gender", "s4_4_main_earner_age",
            "s4_5_main_earner_relation_lbl", "s4_6_main_earner_education_lbl")
    },
    "s3_14_visit_importance_lbl": (
        ("s3_11_intent_return_12m_lbl",), (("0. No",),), "single"),
}
# Items whose first-round rows record a routing skip as "0" rather than the
# gate sentinel.  The schema names five such families; which items actually
# carry the convention is read off the visible rows by detect_zero_on_skip()
# and stored here per instrument (item -> empirical first-round skip rate).
ZERO_ON_SKIP = {}


def _is_zero_on_skip(name):
    return name in ZERO_ON_SKIP


def detect_zero_on_skip(items, given, X, work, rows, min_rows=50,
                        max_rate=0.005, min_round_two=0.01):
    """Gated items with a "0" level whose round-1 visible rows never skip.

    Fires on items where the visible round-1 rows hold the sentinel at under
    ``max_rate`` while round-2 rows hold it at ``min_round_two`` or more: the
    exact signature of a round that wrote 0 for respondents it skipped.
    Returns item -> smoothed round-1 skip rate (the cap kept on that mass).
    """
    if "wave" not in given or "wave" not in items:
        return {}
    values = [str(v) for v in items["wave"]["values"]]
    if "1" not in values:
        return {}
    first = values.index("1")
    wave = X[:, given.index("wave")]
    rows = np.asarray(rows, dtype=np.int64)
    found = {}
    for name, job in work.items():
        item = items[name]
        if not item.get("gate") or "0" not in [str(v) for v in item["values"]]:
            continue
        visible = rows[job["train"][rows]]
        one = visible[wave[visible] == first]
        two = visible[wave[visible] != first]
        if len(one) < min_rows or not len(two):
            continue
        skip_one = int((job["y"][one] == job["K"] - 1).sum())
        rate_one = skip_one / len(one)
        rate_two = float((job["y"][two] == job["K"] - 1).mean())
        if rate_one < max_rate and rate_two >= min_round_two:
            found[name] = (skip_one + 0.5) / (len(one) + 1.0)
    return found


def fold_wave_one_skip(prob, waves, name, items):
    """Move round-1 sentinel mass above the empirical cap onto level "0"."""
    cap = ZERO_ON_SKIP.get(name)
    if cap is None or "wave" not in items:
        return prob
    first = [str(v) for v in items["wave"]["values"]].index("1")
    zero = [str(v) for v in items[name]["values"]].index("0")
    rows = waves == first
    if not rows.any():
        return prob
    excess = np.clip(prob[rows, -1] - cap, 0.0, None)
    prob[rows, -1] -= excess
    prob[rows, zero] += excess
    return prob


# Empirical second gate parent per item, read off the visible rows by
# discover_empirical_gate_parents().  UNHCR's dataset note says the s7a and s7b
# branches follow s3_12 more closely than their declared parent s3_11.
EMPIRICAL_PARENTS = {}
# Items per depth that received the nested transition arms in the last call.
NESTED_TRACE = {}


def _declared_parents(name, items):
    if name in GATE_CORRECTIONS and name in items:
        parents = GATE_CORRECTIONS[name][0]
        if any(parent not in items for parent in parents):
            raise ValueError("missing corrected gate parent for %s" % name)
        return parents
    gate = items[name].get("gate")
    if not isinstance(gate, dict) or not gate.get("parent"):
        return ()
    return (gate["parent"],)


def _parents(name, items):
    declared = _declared_parents(name, items)
    extra = EMPIRICAL_PARENTS.get(name)
    return declared + (extra,) if extra and extra not in declared else declared


def _mi_columns(H, offsets, target, weights=None):
    """Bias-corrected MI between a binary target and each one-hot block of H."""
    n = float(len(target))
    hit = H.T @ target
    count = H.sum(0)
    out = []
    for a, b in offsets:
        c, h = count[a:b], hit[a:b]
        joint = np.stack((c - h, h), 1) / n
        px = joint.sum(1, keepdims=True)
        py = joint.sum(0, keepdims=True)
        v = joint > 0
        mi = float(np.sum(joint[v] * np.log(joint[v] / (px * py)[v])))
        out.append(mi - (int((c > 0).sum()) - 1) / (2.0 * n))
    return np.array(out)


def _joint_mi(state, P, target):
    n = float(len(target))
    c = np.bincount(state, minlength=P).astype(float)
    h = np.bincount(state, weights=target, minlength=P)
    joint = np.stack((c - h, h), 1) / n
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    v = joint > 0
    mi = float(np.sum(joint[v] * np.log(joint[v] / (px * py)[v])))
    return mi - (int((c > 0).sum()) - 1) / (2.0 * n)


def discover_empirical_gate_parents(items, work, anchors, rows, X=None,
                                    given=None, min_mi_gain=0.015):
    """Add the anchor that best explains who skipped an item as a second parent.

    For every gated PREDICT item, the skip indicator on the visible ``rows`` is
    scored against every candidate anchor's answer by mutual information.  The
    best anchor joins the declared parent when the joint (declared, anchor)
    state explains the skip by ``min_mi_gain`` nats more than the declared
    parent alone.  Candidates exclude the item's descendants (the graph stays
    acyclic), its declared siblings and corrected multi-parent gates.
    """
    EMPIRICAL_PARENTS.clear()
    rows = np.asarray(rows, dtype=np.int64)
    pool = [a for a in anchors if a in work and len(rows)
            and np.all(work[a]["y"][rows] >= 0)]
    if not pool or len(rows) < 100:
        return {}
    offsets, blocks, start = [], [], 0
    for a in pool:
        K = int(work[a]["K"])
        block = np.zeros((len(rows), K), np.float32)
        block[np.arange(len(rows)), work[a]["y"][rows]] = 1.0
        blocks.append(block)
        offsets.append((start, start + K))
        start += K
    H = np.concatenate(blocks, 1)
    wave_two = None
    if X is not None and given is not None and "wave" in given and "wave" in items:
        values = [str(v) for v in items["wave"]["values"]]
        if "2" in values:
            wave_two = X[rows, given.index("wave")] == values.index("2")

    def ancestors(name):
        seen, stack = set(), list(_parents(name, items))
        while stack:
            p = stack.pop()
            if p in seen or p not in items:
                continue
            seen.add(p)
            stack.extend(_parents(p, items))
        return seen

    found = {}
    for name in work:
        item = items[name]
        gate = item.get("gate")
        if not isinstance(gate, dict) or name in GATE_CORRECTIONS:
            continue
        declared = _declared_parents(name, items)
        if len(declared) != 1:
            continue
        job = work[name]
        sel = np.ones(len(rows), bool)
        if name in ZERO_ON_SKIP and wave_two is not None:
            sel = wave_two
        if sel.sum() < 100:
            continue
        target = (job["y"][rows][sel] == job["K"] - 1).astype(np.float32)
        rate = float(target.mean())
        if rate <= 0.0 or rate >= 1.0:
            continue
        Hs = H[sel] if not sel.all() else H
        scores = _mi_columns(Hs, offsets, target)
        parent = declared[0]
        observed = tuple(sorted(map(str, gate.get("observed_if", ()))))
        if not V128_K2:
            for i in np.argsort(-scores):
                a = pool[i]
                if a == name or a in declared or scores[i] <= 0:
                    continue
                other = items[a].get("gate")
                if (isinstance(other, dict) and other.get("parent") == parent and
                        tuple(sorted(map(str, other.get("observed_if", ())))) == observed):
                    continue  # a sibling on the same branch
                if name in ancestors(a):
                    continue
                if parent in work and np.all(work[parent]["y"][rows] >= 0):
                    Kp = int(work[parent]["K"])
                    ps = work[parent]["y"][rows][sel]
                else:
                    break
                Ka = int(work[a]["K"])
                single = _joint_mi(ps, Kp, target)
                joint = _joint_mi(ps * Ka + work[a]["y"][rows][sel], Kp * Ka, target)
                if joint - single > min_mi_gain or scores[i] > single:
                    found[name] = (a, float(single), float(joint), float(scores[i]))
                    EMPIRICAL_PARENTS[name] = a
                break
        else:
            if not (parent in work and np.all(work[parent]["y"][rows] >= 0)):
                continue
            Kp = int(work[parent]["K"])
            ps = work[parent]["y"][rows][sel]
            single = _joint_mi(ps, Kp, target)
            ranked = list(np.argsort(-scores)[:6])
            s312 = "s3_12_hope_return_someday_lbl"
            if (parent == "s3_11_intent_return_12m_lbl" or name.startswith(("s7a_", "s7b_", "s7c_"))) and s312 in pool:
                idx_312 = pool.index(s312)
                if idx_312 not in ranked:
                    ranked.append(idx_312)
            best_cand = None
            for i in ranked:
                a = pool[i]
                if a == name or a in declared or scores[i] <= 0:
                    continue
                other = items[a].get("gate")
                is_cross_section_hub = (a == s312 or a.split("_")[0] != name.split("_")[0])
                if (not is_cross_section_hub and isinstance(other, dict) and other.get("parent") == parent and
                        tuple(sorted(map(str, other.get("observed_if", ())))) == observed):
                    continue
                if name in ancestors(a):
                    continue
                Ka = int(work[a]["K"])
                joint = _joint_mi(ps * Ka + work[a]["y"][rows][sel], Kp * Ka, target)
                req_gain = min_mi_gain * 0.25 if (a == s312 and name.startswith(("s7a_", "s7b_", "s7c_"))) else min_mi_gain
                if joint - single > req_gain or scores[i] > single:
                    score_key = joint + (0.005 if a == s312 and name.startswith(("s7a_", "s7b_", "s7c_")) else 0.0)
                    if best_cand is None or score_key > best_cand[0]:
                        best_cand = (score_key, a, float(single), float(joint), float(scores[i]))
            if best_cand is not None:
                _, a, s_mi, j_mi, a_mi = best_cand
                found[name] = (a, s_mi, j_mi, a_mi)
                EMPIRICAL_PARENTS[name] = a
    return found


def gate_correction_codes(items):
    """Resolve literal schema labels to integer codes; reject schema drift."""
    resolved = {}
    for name, (parents, active, mode) in GATE_CORRECTIONS.items():
        if name not in items:
            continue
        if not items[name].get("gate"):
            raise ValueError("corrected target has no schema gate: %s" % name)
        if items[name]["gate"].get("parent") not in parents:
            raise ValueError("schema gate parent changed for %s" % name)
        codes = []
        for parent, labels in zip(parents, active):
            values = items[parent]["values"]
            if len(values) != len(set(values)):
                raise ValueError("duplicate schema values: %s" % parent)
            if not set(labels).issubset(values):
                raise ValueError("unknown active value for %s: %r" % (parent, labels))
            codes.append(tuple(values.index(label) for label in labels))
        if mode.startswith("count") and any(items[p]["values"] != ["0", "1"] for p in parents):
            raise ValueError("s3_23 count parents must be binary 0/1")
        resolved[name] = (parents, tuple(codes), mode)
    return resolved


def _depth(name, items, cache, visiting):
    if name in cache:
        return cache[name]
    if name in visiting:
        raise ValueError("cycle in gate graph: %s" % name)
    visiting.add(name)
    depth = 0
    for parent in _parents(name, items):
        if parent in items and items[parent]["class"] == "PREDICT":
            depth = max(depth, 1 + _depth(parent, items, cache, visiting))
        else:
            depth = max(depth, 1)
    visiting.remove(name)
    cache[name] = depth
    return depth


def _joint(distributions):
    out = distributions[0]
    for p in distributions[1:]:
        out = (out[:, :, None] * p[:, None, :]).reshape(len(out), -1)
    return out


def _coupled_joint_pair(d0_oof, d1_oof, d0_rest, d1_rest, tp0, tp1, W0, W1, fold_rows):
    """Empirical copula coupling of two parent distributions with Sinkhorn margin matching."""
    P = W0 * W1
    joint_code = tp0 * W1 + tp1

    def couple(p0, p1, fit_codes):
        cnt = np.bincount(fit_codes, minlength=P).reshape(W0, W1).astype(float) + 0.5
        c_mat = cnt / cnt.sum()
        m0 = np.maximum(c_mat.sum(1, keepdims=True), 1e-12)
        m1 = np.maximum(c_mat.sum(0, keepdims=True), 1e-12)
        psi = np.clip(c_mat / (m0 * m1), 0.02, 50.0)
        J = p0[:, :, None] * p1[:, None, :] * psi[None, :, :]
        for _ in range(2):
            s0 = np.maximum(J.sum(2, keepdims=True), 1e-12)
            J = J * (p0[:, :, None] / s0)
            s1 = np.maximum(J.sum(1, keepdims=True), 1e-12)
            J = J * (p1[:, None, :] / s1)
        J = J.reshape(len(p0), P)
        J = np.clip(J, 0.0, 1.0)
        return J / np.maximum(J.sum(1, keepdims=True), 1e-12)

    out_oof = np.zeros((len(d0_oof), P), dtype=float)
    for f in np.unique(fold_rows):
        va = fold_rows == f
        fit = ~va
        if not fit.any():
            fit = np.ones(len(fold_rows), bool)
        out_oof[va] = couple(d0_oof[va], d1_oof[va], joint_code[fit])
    out_rest = couple(d0_rest, d1_rest, joint_code) if len(d0_rest) else np.zeros((0, P), dtype=float)
    return out_oof, out_rest


def _count_012(distributions):
    """Poisson-binomial 0, 1, >=2 counts under conditional independence."""
    zero = np.ones(len(distributions[0]), dtype=float)
    one = np.zeros_like(zero)
    for dist in distributions:
        p0, p1 = dist[:, 0], dist[:, 1]
        one, zero = one * p0 + zero * p1, zero * p0
    return np.column_stack((zero, one, np.clip(1.0 - zero - one, 0.0, 1.0)))


COMPOSITE_COPULA = False


def _logits(columns):
    p = np.clip(np.column_stack(columns), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _copula_state(true_state, oof, rest, parent_oof, parent_rest, fold_rows, P):
    """Fold-excluded multinomial recalibration of a composite parent state.

    The composite law multiplies the parents' marginals as if independent;
    a logistic layer on the composite and per-parent logits restores their
    dependence. Fold f is scored by a layer fitted on the other folds."""
    from sklearn.linear_model import LogisticRegression
    feats_oof = np.column_stack((np.log(np.clip(oof, 1e-6, 1.0)), _logits(parent_oof)))
    feats_rest = np.column_stack((np.log(np.clip(rest, 1e-6, 1.0)), _logits(parent_rest)))
    out_oof = np.zeros_like(oof)

    def fit(features, target, query):
        classes = np.unique(target)
        pred = np.zeros((len(query), P))
        if len(classes) < 2:
            pred[:, classes[0]] = 1.0
            return pred
        model = LogisticRegression(C=1.0, max_iter=200, tol=1e-6)
        model.fit(features, target)
        pred[:, model.classes_.astype(int)] = model.predict_proba(query)
        return pred
    for f in np.unique(fold_rows):
        va = fold_rows == f
        out_oof[va] = fit(feats_oof[~va], true_state[~va], feats_oof[va])
    return out_oof, fit(feats_oof, true_state, feats_rest)


def _parent_state(name, parents, rows, rest, work, final_oof, final_rest,
                  X, given, levels, items, fold_rows=None, coupled=False):
    """Joint parent state; with ``fold_rows`` a corrected composite state is
    recalibrated by _copula_state (None when there is nothing to correct)."""
    distributions_oof, distributions_rest, true_parents, widths = [], [], [], []
    for parent in parents:
        if parent in work:
            if parent not in final_oof or parent not in final_rest:
                return None
            job = work[parent]
            if not np.array_equal(rows, np.flatnonzero(job["train"])):
                return None
            distributions_oof.append(final_oof[parent])
            distributions_rest.append(final_rest[parent])
            true_parents.append(job["y"][rows])
            widths.append(job["K"])
        elif parent in given:
            f = given.index(parent)
            eye = np.eye(int(levels[f]), dtype=float)
            distributions_oof.append(eye[X[rows, f]])
            distributions_rest.append(eye[X[rest, f]])
            true_parents.append(X[rows, f])
            widths.append(int(levels[f]))
        else:
            return None
    if fold_rows is not None and not coupled and (name not in GATE_CORRECTIONS or len(parents) < 2
                                  or not any(p in work for p in parents)):
        return None
    if name in GATE_CORRECTIONS:
        _, active_codes, mode = gate_correction_codes(items)[name]
        if mode.startswith("count"):
            if len(parents) != 10 or any(width != 2 for width in widths):
                raise ValueError("s3_23 rank gate requires ten binary parents")
            true_state = np.minimum(np.sum(true_parents, axis=0), 2).astype(np.int64)
            oof, rest_state = _count_012(distributions_oof), _count_012(distributions_rest)
            if fold_rows is not None:
                oof, rest_state = _copula_state(true_state, oof, rest_state,
                    [d[:, 1] for d in distributions_oof], [d[:, 1] for d in distributions_rest], fold_rows, 3)
            return (true_state, oof, rest_state, 3)
        actual = [np.isin(codes, selected) for codes, selected in zip(true_parents, active_codes)]

        def probabilities(distributions):
            active = [np.clip(d[:, selected].sum(1), 0.0, 1.0)
                      for d, selected in zip(distributions, active_codes)]
            if mode == "or":
                prob = 1.0 - np.prod([1.0 - p for p in active], axis=0)
            elif mode == "and":
                prob = np.prod(active, axis=0)
            else:
                prob = active[0]
            return np.column_stack((1.0 - prob, prob))

        if mode == "or":
            true_state = np.logical_or.reduce(actual).astype(np.int64)
        elif mode == "and":
            true_state = np.logical_and.reduce(actual).astype(np.int64)
        else:
            true_state = actual[0].astype(np.int64)
        oof, rest_state = probabilities(distributions_oof), probabilities(distributions_rest)
        if fold_rows is not None:
            def active(distributions):
                return [np.clip(d[:, s].sum(1), 0.0, 1.0) for d, s in zip(distributions, active_codes)]
            oof, rest_state = _copula_state(true_state, oof, rest_state, active(distributions_oof),
                                            active(distributions_rest), fold_rows, 2)
        return (true_state, oof, rest_state, 2)
    true_state = np.zeros(len(rows), dtype=np.int64)
    for codes, width in zip(true_parents, widths):
        true_state = true_state * width + codes
    if coupled and len(parents) == 2 and fold_rows is not None:
        j_oof, j_rest = _coupled_joint_pair(
            distributions_oof[0], distributions_oof[1],
            distributions_rest[0], distributions_rest[1],
            true_parents[0], true_parents[1], widths[0], widths[1], fold_rows)
        return (true_state, j_oof, j_rest, int(np.prod(widths)))
    return (true_state, _joint(distributions_oof),
            _joint(distributions_rest), int(np.prod(widths)))



JOINT_IPF = True
_IPF_CACHE = {}

def _rake(M, qa, qb, iters=30):
    J = np.broadcast_to(M, (len(qa),) + M.shape).copy()
    for _ in range(iters):
        J *= (qa / np.maximum(J.sum(2), 1e-300))[:, :, None]
        J *= (qb / np.maximum(J.sum(1), 1e-300))[:, None, :]
    J = J.reshape(len(qa), M.size)
    return J / np.maximum(J.sum(1, keepdims=True), 1e-300)

def _ipf_state(parents, rows, rest, work, final_oof, final_rest, fold_rows):
    key = (tuple(parents), len(rows), len(rest))
    if key in _IPF_CACHE:
        return _IPF_CACHE[key]
    a, b = parents
    ja, jb = work[a], work[b]
    state = None
    if (a in final_oof and b in final_oof and np.array_equal(rows, np.flatnonzero(ja["train"]))
            and np.array_equal(rows, np.flatnonzero(jb["train"]))):
        Ka, Kb = int(ja["K"]), int(jb["K"])
        true = ja["y"][rows] * Kb + jb["y"][rows]
        def table(sel):
            c = np.bincount(true[sel], minlength=Ka * Kb).astype(float) + 0.5
            return (c / c.sum()).reshape(Ka, Kb)
        qa_o, qb_o = np.asarray(final_oof[a], float), np.asarray(final_oof[b], float)
        oof = np.zeros((len(rows), Ka * Kb))
        for f in np.unique(fold_rows):
            m = fold_rows == f
            oof[m] = _rake(table(~m), qa_o[m], qb_o[m])
        hidden = _rake(table(np.ones(len(rows), bool)), np.asarray(final_rest[a], float),
                       np.asarray(final_rest[b], float))
        state = (true, oof, hidden, Ka * Kb)
    _IPF_CACHE[key] = state
    return state

def _rates(parent_ids, group_ids, fold, y, K, P, G, F, excluded=None, low_shrinkage=None):
    """Fold-excluded P(asked | parents, group) and P(answer | asked,...)."""
    select = np.ones(len(y), bool) if excluded is None else fold != excluded
    parent_ids, group_ids, y = parent_ids[select], group_ids[select], y[select]
    state = parent_ids * G + group_ids
    count = np.bincount(state, minlength=P * G).reshape(P, G).astype(float)
    skip = np.bincount(state, weights=(y == K - 1),
                       minlength=P * G).reshape(P, G)
    answers = np.bincount(state[y != K - 1] * (K - 1) + y[y != K - 1],
                          minlength=P * G * (K - 1)).reshape(P, G, K - 1).astype(float)
    global_asked = (count.sum() - skip.sum() + 0.5) / (count.sum() + 1.0)
    global_answers = (answers.sum((0, 1)) + 0.5)
    global_answers /= global_answers.sum()
    parent_count = count.sum(1)
    parent_asked = (parent_count - skip.sum(1) + 8.0 * global_asked) / (parent_count + 8.0)
    parent_answers = (answers.sum(1) + 8.0 * global_answers) / (answers.sum((1, 2))[:, None] + 8.0)
    asked = (count - skip + 12.0 * parent_asked[:, None]) / (count + 12.0)
    if low_shrinkage is not None:
        purity = np.divide(skip, count, out=np.full_like(skip, 0.5), where=count > 0)
        pure = (count >= 25) & ((purity >= 0.985) | (purity <= 0.015))
        pc_purity = np.divide(skip.sum(1), parent_count, out=np.full_like(parent_count, 0.5), where=parent_count > 0)
        parent_pure = (parent_count >= 25) & ((pc_purity >= 0.985) | (pc_purity <= 0.015))
        sharp_parent = (parent_count - skip.sum(1) + low_shrinkage * global_asked) / (parent_count + low_shrinkage)
        prior = np.where(parent_pure, sharp_parent, parent_asked)
        exact = (count >= 20) & ((skip == count) | (skip == 0))
        strength = np.where(exact, min(low_shrinkage, 0.02), low_shrinkage)
        sharp = (count - skip + strength * prior[:, None]) / (count + strength)
        asked = np.where(pure | exact, sharp, asked)
    answer = (answers + 12.0 * parent_answers[:, None, :]) / (answers.sum(2)[:, :, None] + 12.0)
    return asked, answer


def _apply(parent_prob, group, rates):
    asked, answer = rates
    conditional = asked[:, group, None] * answer[:, group, :]
    part = np.einsum("np,pnk->nk", parent_prob, conditional)
    out = np.empty((len(group), answer.shape[-1] + 1), dtype=float)
    out[:, :-1] = part
    out[:, -1] = 1.0 - part.sum(1)
    out = np.clip(out, 0.0, 1.0)
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def _wave_one_index(name, items, given):
    if not _is_zero_on_skip(name) or "wave" not in given:
        return None
    return next((i for i, value in enumerate(items["wave"]["values"])
                 if str(value) == "1"), None)


def _wave_two_source(name, items, X, given, rows):
    """Select genuine round-two training responses for zero-on-skip items."""
    if not _is_zero_on_skip(name):
        return np.ones(len(rows), dtype=bool)
    if "wave" not in given or _wave_one_index(name, items, given) is None:
        return None
    wave_two = next((i for i, value in enumerate(items["wave"]["values"])
                    if str(value) == "2"), None)
    if wave_two is None:
        return None
    return X[rows, given.index("wave")] == wave_two


def _source_rates(parent_ids, group_ids, fold, y, K, P, G, source,
                  excluded=None, low_shrinkage=None):
    """Fit every level of the transition on eligible, fold-excluded rows."""
    fit = source if excluded is None else source & (fold != excluded)
    if not fit.any():
        return None
    return _rates(parent_ids[fit], group_ids[fit], fold[fit], y[fit],
                  K, P, G, int(fold.max()) + 1, low_shrinkage=low_shrinkage)


def _encode_wave_one_zero(out, waves, wave_one):
    if wave_one is not None:
        first = waves == wave_one
        out[first, 0] += out[first, -1]
        out[first, -1] = 0.0
    return out


def _shape_apply(base, joint, group, rates, wave=None, wave_one=None):
    """Preserve wave-two answer shape; rebuild wave one from conditional law."""
    asked, _ = rates
    p_na = 1.0 - np.einsum("np,pn->n", joint, asked[:, group])
    p_na = np.clip(p_na, 0.0, 1.0)
    K = base.shape[1]
    answer = np.clip(base[:, :K - 1], 0.0, None)
    total = answer.sum(1, keepdims=True)
    answer = np.divide(answer, total, out=np.full_like(answer, 1.0 / (K - 1)),
                       where=total > 0)
    out = np.empty_like(base)
    out[:, :K - 1] = answer * (1.0 - p_na)[:, None]
    out[:, K - 1] = p_na
    if wave_one is not None:
        first = wave == wave_one
        if first.any():
            # The first-round base already has skips in class 0.  Its
            # conditional answer shape is not identifiable from that base.
            corrected = _apply(joint[first], group[first], rates)
            out[first] = _encode_wave_one_zero(corrected, wave[first], wave_one)
    return out


def _candidate(name, parents, rows, rest, work, final_oof, final_rest,
               X, given, levels, fold, items, copula=False, coupled=False, state=None, low_shrinkage=None):
    if state is None:
        state = _parent_state(name, parents, rows, rest, work, final_oof,
                              final_rest, X, given, levels, items,
                              fold_rows=fold[rows] if (copula or coupled) else None,
                              coupled=coupled)
    if state is None:
        return None
    true_joint, joint_oof, joint_rest, P = state
    job = work[name]
    K = job["K"]
    wave_one = _wave_one_index(name, items, given)
    source = _wave_two_source(name, items, X, given, rows)
    if source is None or not source.any():
        return None
    wave_feature = given.index("wave") if wave_one is not None else None
    features = np.argsort(job["gains"])[::-1] if job["gains"].size else ()
    g = next((int(f) for f in features if f != wave_feature), None)
    groups = X[:, g] if g is not None else np.zeros(len(X), dtype=np.int64)
    G = int(levels[g]) if g is not None else 1
    training_fold = fold[rows]
    F = int(training_fold.max()) + 1
    y = job["y"][rows]
    oof = np.zeros((len(rows), K), dtype=float)
    for f in range(F):
        mask = training_fold == f
        if mask.any():
            rates = _source_rates(true_joint, groups[rows], training_fold,
                                  y, K, P, G, source, excluded=f, low_shrinkage=low_shrinkage)
            if rates is None:
                return None
            oof[mask] = _apply(joint_oof[mask], groups[rows][mask], rates)
    rates = _source_rates(true_joint, groups[rows], training_fold,
                          y, K, P, G, source, low_shrinkage=low_shrinkage)
    if rates is None:
        return None
    hidden = _apply(joint_rest, groups[rest], rates)
    if wave_one is not None:
        _encode_wave_one_zero(oof, X[rows, wave_feature], wave_one)
        _encode_wave_one_zero(hidden, X[rest, wave_feature], wave_one)
    return oof, hidden


def _shape_candidate(name, parents, rows, rest, work, final_oof, final_rest,
                     X, given, levels, fold, items, base_oof, base_rest, copula=False, coupled=False, state=None, low_shrinkage=None):
    """Route only skip mass while preserving the child's fitted answer shape."""
    if state is None:
        state = _parent_state(name, parents, rows, rest, work, final_oof,
                              final_rest, X, given, levels, items,
                              fold_rows=fold[rows] if (copula or coupled) else None,
                              coupled=coupled)
    if state is None:
        return None
    true_joint, joint_oof, joint_rest, P = state
    job = work[name]
    K = job["K"]
    zero_on_skip = _is_zero_on_skip(name) and "wave" in given
    wave_feature = given.index("wave") if zero_on_skip else None
    features = np.argsort(job["gains"])[::-1] if job["gains"].size else ()
    g = next((int(f) for f in features if f != wave_feature), None)
    groups = X[:, g] if g is not None else np.zeros(len(X), dtype=np.int64)
    G = int(levels[g]) if g is not None else 1
    training_fold = fold[rows]
    F = int(training_fold.max()) + 1
    y = job["y"][rows]
    source = _wave_two_source(name, items, X, given, rows)
    if source is None or not source.any():
        return None
    wave_one = _wave_one_index(name, items, given)

    oof = np.zeros_like(base_oof)
    for f in range(F):
        mask = training_fold == f
        if mask.any():
            rates = _source_rates(true_joint, groups[rows], training_fold,
                                  y, K, P, G, source, excluded=f, low_shrinkage=low_shrinkage)
            if rates is None:
                return None
            wave = X[rows[mask], wave_feature] if zero_on_skip else None
            oof[mask] = _shape_apply(base_oof[mask], joint_oof[mask],
                                     groups[rows][mask], rates, wave, wave_one)
    rates = _source_rates(true_joint, groups[rows], training_fold,
                          y, K, P, G, source, low_shrinkage=low_shrinkage)
    if rates is None:
        return None
    wave = X[rest, wave_feature] if zero_on_skip else None
    hidden = _shape_apply(base_rest, joint_rest, groups[rest],
                          rates, wave, wave_one)
    return oof, hidden


def _nested_rates(parent_ids, codes, sizes, y, K, P, m):
    """P(asked) and P(answer | asked) on parent -> (parent, ) -> (parent,
    , ), each level shrunk toward the one above with strength ``m``."""
    count_all = len(y)
    skip_all = float((y == K - 1).sum())
    asked = np.array([(count_all - skip_all + 0.5) / (count_all + 1.0)])
    answers = np.bincount(y[y != K - 1], minlength=K - 1).astype(float) + 0.5
    answer = (answers / answers.sum())[None, :]
    state = np.zeros(len(y), dtype=np.int64)
    width = 1
    for level, size in enumerate([P] + list(sizes)):
        ids = parent_ids if level == 0 else codes[level - 1]
        up = state
        state = state * size + ids
        width *= size
        count = np.bincount(state, minlength=width).astype(float)
        skip = np.bincount(state, weights=(y == K - 1), minlength=width)
        hit = np.bincount(state[y != K - 1] * (K - 1) + y[y != K - 1],
                          minlength=width * (K - 1)).reshape(width, K - 1).astype(float)
        prior_asked = np.repeat(asked, size)
        prior_answer = np.repeat(answer, size, axis=0)
        asked = (count - skip + m * prior_asked) / (count + m)
        answer = (hit + m * prior_answer) / (hit.sum(1, keepdims=True) + m)
    G = width // P
    return asked.reshape(P, G), answer.reshape(P, G, K - 1)


def _nested_candidate(name, parents, rows, rest, work, final_oof, final_rest,
                      X, given, levels, fold, items, base_oof, base_rest,
                      depth=2, strengths=(4.0, 12.0, 36.0), max_cells=400000):
    """Parent-state transition over the top ``depth`` GIVEN features.

    Zero-on-skip items fit round two on round-two rows (skip is a class there)
    and round one on round-one rows (skip is folded into "0").  Returns the
    direct transition and a shape arm that routes only the round-two skip
    mass and keeps the base round-one distribution."""
    state = _parent_state(name, parents, rows, rest, work, final_oof,
                          final_rest, X, given, levels, items)
    if state is None:
        return None
    true_joint, joint_oof, joint_rest, P = state
    job = work[name]
    K = job["K"]
    wave_one = _wave_one_index(name, items, given)
    wave_feature = given.index("wave") if wave_one is not None else None
    if wave_one is None and depth < 2:
        return None
    features = np.argsort(job["gains"])[::-1] if job["gains"].size else ()
    picked = [int(f) for f in features if f != wave_feature][:depth]
    sizes = [int(levels[f]) for f in picked]
    if P * int(np.prod(sizes or [1])) * K > max_cells:
        picked, sizes = picked[:1], sizes[:1]
    if not picked and wave_one is None:
        return None
    G = int(np.prod(sizes or [1]))
    code_all = np.zeros(len(X), dtype=np.int64)
    for f, size in zip(picked, sizes):
        code_all = code_all * size + X[:, f]
    codes_rows = [X[rows, f] for f in picked]
    y = job["y"][rows]
    training_fold = fold[rows]
    F = int(training_fold.max()) + 1
    if wave_one is None:
        segments = [(np.ones(len(rows), bool), np.ones(len(rest), bool),
                     np.ones(len(rows), bool), False)]
    else:
        first_rows = X[rows, wave_feature] == wave_one
        first_rest = X[rest, wave_feature] == wave_one
        two = _wave_two_source(name, items, X, given, rows)
        if two is None or not two.any() or not first_rows.any():
            return None
        segments = [(~first_rows, ~first_rest, two, False),
                    (first_rows, first_rest, first_rows, True)]
    row_index = np.arange(len(rows))
    best = None
    for m in strengths:
        oof = np.zeros((len(rows), K), dtype=float)
        for seg_rows, _, source, is_first in segments:
            for f in range(F):
                fit = source & (training_fold != f)
                mask = seg_rows & (training_fold == f)
                if not mask.any():
                    continue
                if not fit.any():
                    return None
                rates = _nested_rates(true_joint[fit], [c[fit] for c in codes_rows],
                                      sizes, y[fit], K, P, m)
                oof[mask] = _apply(joint_oof[mask], code_all[rows][mask], rates)
            if is_first:
                oof[seg_rows] = _encode_wave_one_zero(
                    oof[seg_rows], np.full(int(seg_rows.sum()), wave_one), wave_one)
        q = 1e-3 + (1.0 - 1e-3 * K) * oof[row_index, y]
        loss = -float(np.mean(np.log(q)))
        if best is None or loss < best[0]:
            best = (loss, m, oof)
    _, m, oof = best
    hidden = np.zeros((len(rest), K), dtype=float)
    shape_oof = np.array(base_oof, dtype=float, copy=True)
    shape_rest = np.array(base_rest, dtype=float, copy=True)
    for seg_rows, seg_rest, source, is_first in segments:
        rates = _nested_rates(true_joint[source], [c[source] for c in codes_rows],
                              sizes, y[source], K, P, m)
        if seg_rest.any():
            hidden[seg_rest] = _apply(joint_rest[seg_rest], code_all[rest][seg_rest], rates)
        if is_first:
            hidden[seg_rest] = _encode_wave_one_zero(
                hidden[seg_rest], np.full(int(seg_rest.sum()), wave_one), wave_one)
            continue
        for f in range(F):
            mask = seg_rows & (training_fold == f)
            if mask.any():
                fold_rates = _nested_rates(
                    true_joint[source & (training_fold != f)],
                    [c[source & (training_fold != f)] for c in codes_rows],
                    sizes, y[source & (training_fold != f)], K, P, m)
                shape_oof[mask] = _shape_apply(base_oof[mask], joint_oof[mask],
                                               code_all[rows][mask], fold_rates)
        if seg_rest.any():
            shape_rest[seg_rest] = _shape_apply(base_rest[seg_rest], joint_rest[seg_rest],
                                                code_all[rest][seg_rest], rates)
    return (oof, hidden), (shape_oof, shape_rest), m


def _blend_gate(base_oof, gate_oof, base_rest, gate_rest, y, K, nll_fn):
    """The base stack's 200 EM updates, specialized to two candidates."""
    row = np.arange(len(y))
    b = base_oof[row, y]
    g = gate_oof[row, y]
    scale = 1.0 - K * 1e-3
    weight = 0.5
    for _ in range(200):
        q = 1e-3 + scale * (b * (1.0 - weight) + g * weight)
        gain_b = np.mean(scale * b / q)
        gain_g = np.mean(scale * g / q)
        next_weight = weight * gain_g / ((1.0 - weight) * gain_b + weight * gain_g)
        if abs(next_weight - weight) < 1e-6:
            weight = next_weight
            break
        weight = next_weight

    mixed_oof = base_oof + weight * (gate_oof - base_oof)
    mixed_rest = base_rest + weight * (gate_rest - base_rest)
    best_t, best_loss = 1.0, nll_fn(mixed_oof, y, K)
    if weight:
        log_oof = np.log(np.clip(mixed_oof, 1e-12, None))
        for temp in (0.8, 0.9, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0):
            scaled = log_oof * temp
            scaled -= scaled.max(1, keepdims=True)
            e = np.exp(scaled)
            proposal = e / e.sum(1, keepdims=True)
            loss = nll_fn(proposal, y, K)
            if loss < best_loss:
                best_t, best_loss = temp, loss
        if best_t != 1.0:
            log_rest = np.log(np.clip(mixed_rest, 1e-12, None)) * best_t
            log_rest -= log_rest.max(1, keepdims=True)
            e = np.exp(log_rest)
            mixed_rest = e / e.sum(1, keepdims=True)
            log_oof *= best_t
            log_oof -= log_oof.max(1, keepdims=True)
            e = np.exp(log_oof)
            mixed_oof = e / e.sum(1, keepdims=True)
    return mixed_rest, mixed_oof, (1.0 - weight, weight), best_t, best_loss


def _half_consistent(arm, ref, y, K, half, nll_fn, margin=1e-4):
    """Arm beats ref by ``margin`` overall and on each fold-parity half."""
    if half.all() or not half.any():
        return False
    if not nll_fn(arm, y, K) + margin < nll_fn(ref, y, K):
        return False
    return all(nll_fn(arm[h], y[h], K) < nll_fn(ref[h], y[h], K)
               for h in (half, ~half))


def _low_dominates(arm, ref, y, K, row_fold, nll_fn, pure_mask=None):
    """Test the affected cells, then protect every OOF fold and parity half."""
    active = np.max(np.abs(arm - ref), axis=1) > 1e-7
    if pure_mask is not None:
        if not np.array_equal(arm[~pure_mask], ref[~pure_mask]):
            return False
        active &= pure_mask
        delta = [nll_fn(arm[row_fold == f], y[row_fold == f], K)
                 - nll_fn(ref[row_fold == f], y[row_fold == f], K)
                 for f in np.unique(row_fold)]
        if sum(d < -1e-6 for d in delta) < 4:
            return False
        # A newly admitted graft must resolve paired row and fold uncertainty.
        # The original  selection path keeps its exact arithmetic.
        row = np.arange(len(y))
        scale = 1.0 - 0.001 * K
        loss_delta = (np.log(0.001 + scale * ref[row, y])
                      - np.log(0.001 + scale * arm[row, y]))
        row_se = float(loss_delta.std(ddof=1) / np.sqrt(len(y)))
        fold_se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
        if -loss_delta.mean() < 2.0 * max(row_se, fold_se):
            return False
    half = (row_fold % 2) == 0
    # Resolve the 1.5% impurity scale before selecting a local correction.
    min_active = int(np.ceil(1.0 / (1.0 - 0.985)))
    if active.sum() < min_active or len(np.unique(row_fold)) != 5 or half.all() or not half.any():
        return False
    if not nll_fn(arm[active], y[active], K) + 1e-4 < nll_fn(ref[active], y[active], K):
        return False
    if not nll_fn(arm, y, K) + 1e-6 < nll_fn(ref, y, K):
        return False
    if not all(nll_fn(arm[h], y[h], K) <= nll_fn(ref[h], y[h], K) - 1e-7
               for h in (half, ~half)):
        return False
    return all(nll_fn(arm[row_fold == f], y[row_fold == f], K)
               <= nll_fn(ref[row_fold == f], y[row_fold == f], K)
               for f in np.unique(row_fold))


def _battery_project(q, pi, mask):
    """Proposition 1: change skip mass while retaining conditional answer odds."""
    out = q.copy()
    answer = q[mask, :-1]
    mass = answer.sum(1, keepdims=True)
    conditional = np.divide(answer, mass, out=np.full_like(answer, 1.0 / answer.shape[1]),
                            where=mass > 0)
    out[mask, :-1] = conditional * (1.0 - pi[mask, None])
    out[mask, -1] = pi[mask]
    return out


def _conserve_battery_skips(items, work, final, final_oof, final_rest, X, given, fold, nll_fn, deadline=float('inf')):
    """Cross-fit shared skip laws from exactly identical observed skip vectors.

    Cohorts share visible rows and eligible wave masks. Each validation fold
    discovers its batteries and chooses its top-k mean using other folds only.
    Query batteries and weights use all visible labels, never query labels.
    All proposals read the same immutable input probabilities.
    """
    cohorts = {}
    for name in sorted(work):
        job = work[name]
        if not items[name].get('gate') or job['K'] < 2 or name not in final_oof:
            continue
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        eligible = _wave_two_source(name, items, X, given, rows)
        hidden = _wave_two_source(name, items, X, given, rest)
        if eligible is None or hidden is None or eligible.sum() < 40:
            continue
        key = (job['train'].tobytes(), eligible.tobytes(), hidden.tobytes())
        cohorts.setdefault(key, []).append((name, rows, rest, eligible, hidden))
    accepted, batteries = [], []
    proposals = []
    for records in cohorts.values():
        if time.monotonic()>=deadline:
            return dict(groups=[],accepted=[],offset_groups=[],offset_accepted=[])
        if len(records) < 4:
            continue
        names = [r[0] for r in records]
        _, rows, rest, eligible, hidden = records[0]
        split = fold[rows]
        labels = np.column_stack([work[n]['y'][rows] == work[n]['K'] - 1 for n in names])
        po = np.column_stack([final_oof[n][:, -1] for n in names])
        pr = np.column_stack([final_rest[n][:, -1] for n in names])
        co, cr = po.copy(), pr.copy()
        offered = np.zeros(len(names), dtype=bool)
        fold_groups = {}
        for f in list(np.unique(split)) + [None]:
            if time.monotonic()>=deadline:
                return dict(groups=[],accepted=[],offset_groups=[],offset_accepted=[])
            fit = eligible if f is None else eligible & (split != f)
            val = hidden if f is None else eligible & (split == f)
            if fit.sum() < 20 or not val.any():
                continue
            groups = {}
            for j in range(len(names)):
                groups.setdefault(np.packbits(labels[fit, j]).tobytes(), []).append(j)
            for group in groups.values():
                if len(group) < 4:
                    continue
                target = labels[fit, group[0]]
                source = po[fit][:, group]
                cost = -np.log(np.clip(np.where(target[:, None], source, 1.0 - source),
                                       1e-12, 1.0)).mean(0)
                ranked = np.argsort(cost, kind='stable')
                best = None
                for k in sorted({min(3, len(group)), min(5, len(group)), len(group)}):
                    chosen = [group[int(i)] for i in ranked[:k]]
                    mean = po[fit][:, chosen].mean(1)
                    loss = -np.log(np.clip(np.where(target, mean, 1.0 - mean), 1e-12, 1.0)).mean()
                    if best is None or loss < best[0]:
                        best = (float(loss), chosen)
                chosen = best[1]
                mean_fit = po[fit][:, chosen].mean(1)
                mean = (pr if f is None else po)[:, chosen].mean(1)
                alphas = {}
                for j in group:
                    original = po[fit, j]
                    base_loss = -np.log(np.clip(np.where(target, original, 1.0 - original),
                                               1e-12, 1.0)).mean()
                    best_loss, alpha_j = base_loss, 0.0
                    for alpha in (0.15, 0.30, 0.50):
                        mixed = (1.0 - alpha) * original + alpha * mean_fit
                        loss = -np.log(np.clip(np.where(target, mixed, 1.0 - mixed),
                                              1e-12, 1.0)).mean()
                        if loss < best_loss:
                            best_loss, alpha_j = loss, alpha
                    if base_loss - best_loss < 5e-5 or alpha_j == 0.0:
                        continue
                    alphas[names[j]] = alpha_j
                    if f is None:
                        cr[val, j] = (1.0 - alpha_j) * pr[val, j] + alpha_j * mean[val]
                    else:
                        co[val, j] = (1.0 - alpha_j) * po[val, j] + alpha_j * mean[val]
                        offered[j] = True
                info = {'members': [names[j] for j in group], 'top': [names[j] for j in chosen],
                        'alphas': alphas}
                if f is None:
                    batteries.append(info)
                else:
                    fold_groups.setdefault(str(int(f)), []).append(info)
        for j, name in enumerate(names):
            if not offered[j]:
                continue
            oo = _battery_project(final_oof[name], co[:, j], eligible)
            rr = _battery_project(final_rest[name], cr[:, j], hidden)
            job = work[name]
            y, K = job['y'][rows], job['K']
            gain = nll_fn(final_oof[name], y, K) - nll_fn(oo, y, K)
            fold_delta = [nll_fn(oo[split == f], y[split == f], K)
                          - nll_fn(final_oof[name][split == f], y[split == f], K)
                          for f in np.unique(split)]
            if (len(fold_delta) == 5 and gain >= 1e-5
                    and sum(d < -1e-6 for d in fold_delta) >= 4
                    and all(d <= 0.0 for d in fold_delta)
                    and _low_dominates(oo, final_oof[name], y, K, split, nll_fn)):
                proposals.append((name, oo, rr, rest))
                job['battery_skip'] = {'oof_gain': float(nll_fn(final_oof[name], job['y'][rows], job['K'])
                                                       - nll_fn(oo, job['y'][rows], job['K'])),
                                       'fold_groups': fold_groups}
                accepted.append(name)
    for name, oo, rr, rest in proposals:
        final_oof[name], final_rest[name] = oo, rr
        final[name][rest] = rr
    # Exact batteries keep their original arithmetic and selection. Additional
    # modules use fold-fit complete-link agreement and target-specific offsets.
    high_groups, high_accepted = [], []
    for records in cohorts.values():
        if time.monotonic()>=deadline:
            return dict(groups=[],accepted=[],offset_groups=[],offset_accepted=[])
        if len(records) < 4:
            continue
        names = [r[0] for r in records]
        _, rows, rest, eligible, hidden = records[0]
        split = fold[rows]
        labels = np.column_stack([work[n]['y'][rows] == work[n]['K'] - 1 for n in names])
        po = np.column_stack([final_oof[n][:, -1] for n in names])
        pr = np.column_stack([final_rest[n][:, -1] for n in names])
        co, cr = po.copy(), pr.copy()
        offered = np.zeros(len(names), dtype=bool)
        fold_groups = {}

        def align(probability, offsets):
            clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
            logits = np.log(clipped) - np.log1p(-clipped)
            return 1.0 / (1.0 + np.exp(-(logits + offsets)))

        for f in list(np.unique(split)) + [None]:
            if time.monotonic()>=deadline:
                return dict(groups=[],accepted=[],offset_groups=[],offset_accepted=[])
            fit = eligible if f is None else eligible & (split != f)
            val = hidden if f is None else eligible & (split == f)
            if fit.sum() < 67 or not val.any():
                continue
            exact = {}
            for j in range(len(names)):
                exact.setdefault(np.packbits(labels[fit, j]).tobytes(), []).append(j)
            locked = {j for group in exact.values() if len(group) >= 4 for j in group}
            rates = labels[fit].mean(0)
            remaining = [j for j in range(len(names)) if j not in locked and 0.02 < rates[j] < 0.98]
            groups = []
            for j in remaining:
                for group in groups:
                    if all(np.mean(labels[fit, j] == labels[fit, m]) >= 0.992 for m in group):
                        group.append(j)
                        break
                else:
                    groups.append([j])
            for group in groups:
                if len(group) < 4:
                    continue
                info = {'members': [names[j] for j in group], 'targets': {}}
                for j in group:
                    target = labels[fit, j]
                    offsets = np.clip(np.log(rates[j] / (1.0 - rates[j]))
                                      - np.log(rates[group] / (1.0 - rates[group])), -0.5, 0.5)
                    source = align(po[fit][:, group], offsets)
                    cost = -np.log(np.clip(np.where(target[:, None], source, 1.0 - source), 1e-12, 1.0)).mean(0)
                    ranked = np.argsort(cost, kind='stable')
                    best = None
                    for k in sorted({3, min(5, len(group)), len(group)}):
                        chosen = ranked[:k]
                        mean_fit = source[:, chosen].mean(1)
                        loss = -np.log(np.clip(np.where(target, mean_fit, 1.0 - mean_fit), 1e-12, 1.0)).mean()
                        if best is None or loss < best[0]:
                            best = (float(loss), chosen, mean_fit)
                    _, chosen, mean_fit = best
                    original = po[fit, j]
                    base_loss = -np.log(np.clip(np.where(target, original, 1.0 - original), 1e-12, 1.0)).mean()
                    best_loss, alpha_j = base_loss, 0.0
                    for alpha in (0.15, 0.30, 0.50):
                        mixed = (1.0 - alpha) * original + alpha * mean_fit
                        loss = -np.log(np.clip(np.where(target, mixed, 1.0 - mixed), 1e-12, 1.0)).mean()
                        if loss < best_loss:
                            best_loss, alpha_j = float(loss), alpha
                    if base_loss - best_loss < 5e-5 or alpha_j == 0.0:
                        continue
                    members = [group[int(m)] for m in chosen]
                    mean = align((pr if f is None else po)[val][:, members], offsets[chosen]).mean(1)
                    if f is None:
                        cr[val, j] = (1.0 - alpha_j) * pr[val, j] + alpha_j * mean
                    else:
                        co[val, j] = (1.0 - alpha_j) * po[val, j] + alpha_j * mean
                        offered[j] = True
                    info['targets'][names[j]] = {'top': [names[m] for m in members],
                        'offsets': offsets[chosen].tolist(), 'alpha': alpha_j,
                        'fit_binary_gain': float(base_loss - best_loss)}
                if f is None:
                    high_groups.append(info)
                else:
                    fold_groups.setdefault(str(int(f)), []).append(info)
        for j, name in enumerate(names):
            if not offered[j]:
                continue
            before = final_oof[name]
            oo = _battery_project(before, co[:, j], eligible)
            rr = _battery_project(final_rest[name], cr[:, j], hidden)
            job = work[name]
            y, K = job['y'][rows], job['K']
            gain = nll_fn(before, y, K) - nll_fn(oo, y, K)
            delta = [nll_fn(oo[split == f], y[split == f], K)
                     - nll_fn(before[split == f], y[split == f], K) for f in np.unique(split)]
            if (len(delta) == 5 and gain >= 1e-5 and sum(d < -1e-6 for d in delta) >= 4
                    and all(d <= 0.0 for d in delta)
                    and _low_dominates(oo, before, y, K, split, nll_fn)):
                final_oof[name], final_rest[name] = oo, rr
                final[name][rest] = rr
                job['battery_offset'] = {'oof_gain': float(gain), 'fold_groups': fold_groups}
                high_accepted.append(name)
    NESTED_TRACE['battery_offset_groups'] = len(high_groups)
    NESTED_TRACE['battery_offset_accepted'] = len(high_accepted)
    NESTED_TRACE['battery_groups'] = len(batteries)
    NESTED_TRACE['battery_accepted'] = len(accepted)
    return {'groups': batteries, 'accepted': accepted, 'offset_groups': high_groups, 'offset_accepted': high_accepted}


def _legacy_route(parent_oof, parent_rest, parent_true, group, fold, y, K,
                  base_oof, base_rest):
    """The Series4 single-parent skip model, retained as an OOF reserve."""
    na = (y == K - 1).astype(float)
    Kp = parent_oof.shape[1]
    gt, gr = group
    Gl = int(max(gt.max(), gr.max() if gr.size else 0)) + 1
    F = int(fold.max()) + 1
    index = (fold * Kp + parent_true) * Gl + gt
    cnt = np.bincount(index, minlength=F * Kp * Gl).reshape(F, Kp, Gl).astype(float)
    hit = np.bincount(index, weights=na, minlength=F * Kp * Gl).reshape(F, Kp, Gl)
    r0 = na.mean()

    def rates(c, a):
        rv = (a.sum(1) + r0) / (c.sum(1) + 1.0)
        return (a + 2.0 * rv[:, None]) / (c + 2.0)

    tc, ta = cnt.sum(0), hit.sum(0)
    R = np.stack([rates(tc - cnt[f], ta - hit[f]) for f in range(F)])
    p_na = (parent_oof * R[fold, :, gt]).sum(1)
    p_na_rest = (parent_rest * rates(tc, ta)[:, gr].T).sum(1)

    def assemble(q, pn):
        out = np.zeros_like(q)
        ans = q[:, :K - 1]
        s = ans.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        out[:, :K - 1] = ans / s * (1.0 - pn)[:, None]
        out[:, K - 1] = pn
        return out

    return assemble(base_oof, p_na), assemble(base_rest, p_na_rest)


def _answer_dominates(arm, ref, y, K, split, mask, nll_fn,
                      se_multiplier=7.5):
    """Paired high-support dominance against the current best distribution."""
    if mask.sum() < 67 or len(np.unique(split[mask])) != 5:
        return False
    row = np.arange(len(y))
    floor = 0.001
    gain = (np.log(floor + (1.0 - floor * K) * arm[row, y])
            - np.log(floor + (1.0 - floor * K) * ref[row, y]))
    if gain[mask].mean() < 1e-4 or gain.mean() < 1.5e-5:
        return False
    folds = np.array([gain[split == f].mean() for f in np.unique(split)])
    if len(folds) != 5 or np.any(folds < 0.0) or (folds > 1e-6).sum() < 4:
        return False
    if any(gain[split % 2 == h].mean() <= 0.0 for h in (0, 1)):
        return False
    se_row = gain.std(ddof=1) / np.sqrt(len(gain))
    se_fold = folds.std(ddof=1) / np.sqrt(len(folds))
    return bool(gain.mean() >= max(2.2, se_multiplier) * max(se_row, se_fold)
                and nll_fn(ref, y, K) - nll_fn(arm, y, K) >= 1.5e-5)


def _answer_project(best, arm, mask, alpha, gated, zero_rows=None):
    """Copy the complement exactly and preserve every encoded skip mass."""
    out = best.copy()
    if not gated:
        out[mask] = (1.0 - alpha) * best[mask] + alpha * arm[mask]
        return out
    for zero in (False, True):
        active = mask.copy()
        if zero_rows is None:
            if zero:
                continue
        else:
            active &= zero_rows if zero else ~zero_rows
        if not active.any():
            continue
        # Wave one encodes routing nonresponse in class zero. Keep both
        # class zero and the separate sentinel unchanged on these rows.
        start = 1 if zero else 0
        a, b = best[active, start:-1], arm[active, start:-1]
        mass = a.sum(1, keepdims=True)
        a = np.divide(a, mass, out=np.full_like(a, 1.0 / a.shape[1]), where=mass > 0)
        total = b.sum(1, keepdims=True)
        b = np.divide(b, total, out=a.copy(), where=total > 0)
        out[np.ix_(active, np.arange(start, best.shape[1] - 1))] = mass * ((1.0 - alpha) * a + alpha * b)
    return out


def _answer_route_arms(name, items, work, fo, fr, X, given, levels, fold):
    """Existing parent transition arms with fold-fitted answered support."""
    job = work[name]
    if not items[name].get('gate') or job['K'] < 3:
        return
    rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
    source = _wave_two_source(name, items, X, given, rows)
    if source is None:
        return
    parents, joint = _declared_parents(name, items), _parents(name, items)
    routes = [parents] + ([joint] if joint != parents else [])
    for route in routes:
        state = _parent_state(name, route, rows, rest, work, fo, fr, X, given, levels, items)
        if state is None:
            continue
        true, po, pr, P = state
        if P > 4096:
            continue
        features = np.argsort(job['gains'])[::-1] if job['gains'].size else ()
        wave_one = _wave_one_index(name, items, given)
        wave_feature = given.index('wave') if wave_one is not None else None
        g = next((int(f) for f in features if f != wave_feature), None)
        group = X[:, g] if g is not None else np.zeros(len(X), dtype=np.int64)
        G = int(levels[g]) if g is not None else 1
        keys = [('parent', np.zeros(len(X), dtype=np.int64), 1), ('parent_given', group, G)]
        if 'wave' in given:
            f = given.index('wave')
            keys.append(('parent_wave', X[:, f], int(levels[f])))
        y, split, K = job['y'][rows], fold[rows], job['K']
        for label, groups, width in keys:
            oo, rr = np.zeros((len(rows), K)), np.zeros((len(rest), K))
            mo, mr = np.zeros(len(rows), bool), np.zeros(len(rest), bool)
            for held in list(np.unique(split)) + [None]:
                fit = source & (split != held) if held is not None else source
                target = split == held if held is not None else np.ones(len(rest), bool)
                rates = _rates(true[fit], groups[rows][fit], split[fit], y[fit], K, P, width, 5)
                states = true[fit] * width + groups[rows][fit]
                count = np.bincount(states, minlength=P * width)
                answered = np.bincount(states, weights=y[fit] != K - 1, minlength=P * width)
                support = ((answered >= 45) & (answered >= .25 * count)).reshape(P, width)
                pp = po[target] if held is not None else pr
                gg = groups[rows][target] if held is not None else groups[rest]
                pred = _apply(pp, gg, rates)
                # Every parent cell carrying nontrivial posterior mass must
                # have sufficient observed answers; no hidden labels enter.
                mask = np.sum(pp * support[:, gg].T, axis=1) >= .999
                if held is not None:
                    oo[target], mo[target] = pred, mask
                else:
                    rr, mr = pred, mask
            yield label + (':joint' if route != parents else ''), (oo, rr), (mo, mr)


def _stratum_answer_arms(name, items, work, X, given, levels, fold,
                          base_oof=None, base_rest=None):
    """Select top one/two discrete GIVEN columns inside each fit fold only."""
    job = work[name]
    rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
    y, K, split = job['y'][rows], job['K'], fold[rows]
    gated = bool(items[name].get('gate'))
    if K < 2 or len(rows) < 100:
        return
    source = _wave_two_source(name, items, X, given, rows) if gated else np.ones(len(rows), bool)
    if source is None:
        return
    features = [f for f in range(len(given)) if 1 < int(levels[f]) <= 128]
    if not features:
        return
    for depth in (1, 2):
        oo, rr = np.zeros((len(rows), K)), np.zeros((len(rest), K))
        mo, mr = np.zeros(len(rows), bool), np.zeros(len(rest), bool)
        selected = []
        residual_oof = None if base_oof is None else base_oof.copy()
        residual_rest = None if base_rest is None else base_rest.copy()
        for held in list(np.unique(split)) + [None]:
            fit = source & (split != held) if held is not None else source
            fit_rows, yy = rows[fit], y[fit]
            if len(yy) < 50:
                break
            scores = []
            # Rank using asked-conditional labels for a gated target.
            answered = yy != K - 1 if gated else np.ones(len(yy), bool)
            for f in features:
                counts = np.bincount(X[fit_rows, f][answered] * K + yy[answered],
                    minlength=int(levels[f]) * K).reshape(int(levels[f]), K).astype(float)
                total = counts.sum()
                joint = counts / max(total, 1.0)
                den = joint.sum(1)[:, None] * joint.sum(0)[None, :]
                nz = joint > 0
                mi = float(np.sum(joint[nz] * np.log(joint[nz] / den[nz])))
                # The finite sample correction penalizes high-cardinality noise.
                mi -= ((counts.sum(1) > 0).sum() - 1) * ((counts.sum(0) > 0).sum() - 1) / (2.0 * max(total, 1.0))
                scores.append((mi, -f, f))
            chosen = [v[2] for v in sorted(scores, reverse=True)[:depth]]
            selected.append([given[f] for f in chosen])
            key = np.zeros(len(X), dtype=np.int64)
            width = 1
            for f in chosen:
                key = key * int(levels[f]) + X[:, f]
                width *= int(levels[f])
            counts = np.bincount(key[fit_rows] * K + yy, minlength=width * K).reshape(width, K).astype(float)
            prior = counts.sum(0) + .5
            prior /= prior.sum()
            strength = 16.0
            eb = (counts + strength * prior) / (counts.sum(1, keepdims=True) + strength)
            support = counts.sum(1) >= 50
            if gated:
                n_answered = counts[:, :-1].sum(1)
                support &= (n_answered >= 45) & (n_answered >= .25 * counts.sum(1))
            if held is None:
                rr, mr = eb[key[rest]], support[key[rest]]
            else:
                va = split == held
                oo[va], mo[va] = eb[key[rows[va]]], support[key[rows[va]]]
            if base_oof is not None and base_rest is not None:
                # Correct fold-fit cell calibration error rather than replacing
                # the within-cell conditional ranking learned by the base.
                width_y = K - 1 if gated else K
                conditional = base_oof[fit, :width_y].copy()
                conditional /= np.maximum(conditional.sum(1, keepdims=True), 1e-12)
                eligible = yy != K - 1 if gated else np.ones(len(yy), bool)
                expected = np.column_stack([
                    np.bincount(key[fit_rows][eligible],
                        weights=conditional[eligible, c], minlength=width)
                    for c in range(width_y)])
                observed = counts[:, :width_y]
                residual_prior = observed.sum(0) + .5
                residual_prior /= residual_prior.sum()
                offset = np.clip(np.log((observed + strength * residual_prior)
                    / (expected + strength * residual_prior)), -.5, .5)
                dest = base_rest if held is None else base_oof[va]
                dest_key = key[rest] if held is None else key[rows[va]]
                corrected = dest.copy()
                answer = dest[:, :width_y] * np.exp(offset[dest_key])
                answer *= dest[:, :width_y].sum(1, keepdims=True) / np.maximum(answer.sum(1, keepdims=True), 1e-12)
                corrected[:, :width_y] = answer
                if held is None:
                    residual_rest = corrected
                else:
                    residual_oof[va] = corrected
        else:
            yield 'stratum_top' + str(depth), (oo, rr), (mo, mr), selected
            if base_oof is not None and base_rest is not None:
                yield 'stratum_residual_top' + str(depth), (residual_oof, residual_rest), (mo, mr), selected


def _graft_conditional_answers(items, work, final, fo, fr, X, given, levels,
                                fold, nll_fn, deadline):
    """Orthogonal answer grafts run strictly after all existing gate stages."""
    for name, job in work.items():
        if time.monotonic() >= deadline:
            break
        rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
        K, y, split = job['K'], job['y'][rows], fold[rows]
        gated = bool(items[name].get('gate'))
        if (gated and K < 3) or len(np.unique(split)) != 5:
            continue
        wave_one = _wave_one_index(name, items, given)
        zo = X[rows, given.index('wave')] == wave_one if wave_one is not None else None
        zr = X[rest, given.index('wave')] == wave_one if wave_one is not None else None
        arms = [(label, arm, masks, None) for label, arm, masks in
                _answer_route_arms(name, items, work, fo, fr, X, given, levels, fold)]
        arms.extend(_stratum_answer_arms(name, items, work, X, given, levels, fold,
                                         fo[name], fr[name]))
        for label, arm, (mo, mr), columns in arms:
            if time.monotonic() >= deadline:
                break
            if mo.sum() < 67:
                continue
            for alpha in (.10, .20, .35):
                before, before_rest = fo[name], fr[name]
                oo = _answer_project(before, arm[0], mo, alpha, gated, zo)
                if _answer_dominates(oo, before, y, K, split, mo, nll_fn,
                        se_multiplier=2.2 if 'residual' in label else 7.5):
                    rr = _answer_project(before_rest, arm[1], mr, alpha, gated, zr)
                    assert np.array_equal(oo[~mo], before[~mo])
                    assert np.array_equal(rr[~mr], before_rest[~mr])
                    if gated:
                        assert np.array_equal(oo[:, -1], before[:, -1])
                        assert np.array_equal(rr[:, -1], before_rest[:, -1])
                        if zo is not None:
                            assert np.array_equal(oo[zo, 0], before[zo, 0])
                            assert np.array_equal(rr[zr, 0], before_rest[zr, 0])
                    fo[name], fr[name] = oo, rr
                    final[name][rest] = rr
                    job.setdefault('answer_graft', []).append(dict(kind=label,
                        alpha=alpha, n_active=int(mo.sum()), columns=columns,
                        oof_gain=float(nll_fn(before, y, K) - nll_fn(oo, y, K))))


def _residual_keys(name, items, work, fo, fr, X, given, levels, shared):
    """Discrete observed GIVEN and matched OOF-predicted parent cells."""
    job = work[name]
    rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
    keys = list(shared)
    if items[name].get('gate'):
        for parent in _parents(name, items):
            if parent in given:
                p = X[:, given.index(parent)]
            elif parent in work and np.array_equal(job['train'], work[parent]['train']):
                p = np.zeros(len(X), dtype=np.int64)
                p[rows], p[rest] = fo[parent].argmax(1), fr[parent].argmax(1)
            else:
                continue
            for f in range(len(given)):
                if not 1 < int(levels[f]) <= 128 or given[f] == parent:
                    continue
                parts = [p, X[:, f]]
                for with_wave in (False, True):
                    if with_wave:
                        if 'wave' not in given or given[f] == 'wave':
                            continue
                        parts = parts + [X[:, given.index('wave')]]
                    _, key = np.unique(np.column_stack(parts), axis=0, return_inverse=True)
                    keys.append(('parent:' + parent + ':' + given[f]
                                 + (':wave' if with_wave else ''), key, True))
    # Duplicate encodings carry no additional information.
    unique, seen = [], set()
    for label, key, interaction in keys:
        token = key.tobytes()
        if token not in seen:
            seen.add(token)
            unique.append((label, key, interaction))
    return unique


def _residual_project(best, shifts, mask, gamma, binary=False):
    """Multiplicative cell tilting; the inactive complement is exact."""
    out = best.copy()
    if not mask.any():
        return out
    q = best[mask]
    if binary:
        rho = np.clip(q[:, -1], 1e-12, 1.0 - 1e-12)
        eta = np.log(rho / (1.0 - rho)) + gamma * shifts[mask, 0]
        new_rho = 1.0 / (1.0 + np.exp(-eta))
        return _residual_skip_project(best, mask, new_rho)
    answer = q[:, :-1] * np.exp(gamma * shifts[mask])
    answer *= q[:, :-1].sum(1, keepdims=True) / np.maximum(answer.sum(1, keepdims=True), 1e-300)
    out[np.ix_(mask, np.arange(best.shape[1] - 1))] = answer
    return out


def _residual_skip_project(best, mask, rho):
    out = best.copy()
    q = best[mask, :-1]
    mass = q.sum(1, keepdims=True)
    out[np.ix_(mask, np.arange(best.shape[1] - 1))] = np.divide(
        q, mass, out=np.full_like(q, 1.0 / q.shape[1]), where=mass > 0) * (1.0 - rho[:, None])
    out[mask, -1] = rho
    return out


def _residual_ranked_shifts(keys, rows, rest, y, split, best, query,
                            source, gated, tau, binary=False, deadline=float('inf')):
    """Fit and rank by realized fit NLL reduction, never target MI."""
    K = best.shape[1]
    ranks = 2 if binary else 3
    width = 1 if binary else K - 1
    shifts_o = [np.zeros((len(rows), width)) for _ in range(ranks)]
    shifts_r = [np.zeros((len(rest), width)) for _ in range(ranks)]
    masks_o = [np.zeros(len(rows), bool) for _ in range(ranks)]
    masks_r = [np.zeros(len(rest), bool) for _ in range(ranks)]
    trace = []
    for held in list(np.unique(split)) + [None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        yy, base = y[fit], best[fit]
        if len(yy) < 50:
            continue
        answered = yy != K - 1
        conditional = base[:, :-1] / np.maximum(base[:, :-1].sum(1, keepdims=True), 1e-300)
        ranked = []
        for label, key, interaction in keys:
            if time.monotonic() >= deadline:
                return None
            if binary and not interaction:
                continue
            codes = key[rows[fit]]
            cells = int(key.max()) + 1
            n = np.bincount(codes, minlength=cells)
            n_ans = np.bincount(codes[answered], minlength=cells)
            if binary:
                observed = n - n_ans
                support = (n >= 50) & (observed > .03 * n) & (observed < .97 * n)
                if not support.any():
                    continue
                expected = np.bincount(codes, weights=base[:, -1], minlength=cells)
                p0 = float(base[:, -1].mean())
                o = np.clip((observed + tau * p0) / (n + tau), 1e-12, 1.0 - 1e-12)
                e = np.clip((expected + tau * p0) / (n + tau), 1e-12, 1.0 - 1e-12)
                offset = np.clip(np.log(o / (1.0-o)) - np.log(e / (1.0-e)), -.45, .45)[:, None]
                rho = np.clip(base[:, -1], 1e-12, 1.0 - 1e-12)
                new = 1.0 / (1.0 + np.exp(-(np.log(rho/(1.0-rho)) + offset[codes, 0])))
                target = ~answered
                gain = np.log(np.where(target, new, 1.0-new)) - np.log(np.where(target, rho, 1.0-rho))
            else:
                support = n_ans >= 45
                if gated:
                    support &= n_ans >= .25 * n
                if not support.any():
                    continue
                obs = np.bincount(codes[answered] * width + yy[answered], minlength=cells * width).reshape(cells, width)
                expected = np.column_stack([np.bincount(codes[answered], weights=conditional[answered, c], minlength=cells) for c in range(width)])
                mean = expected / np.maximum(n_ans[:, None], 1)
                fallback = conditional[answered].mean(0) if answered.any() else np.full(width, 1.0/width)
                mean[n_ans == 0] = fallback
                prior = np.maximum(mean, 1e-12)
                offset = np.clip(np.log((obs + tau * prior) / (expected + tau * prior)), -.45, .45)
                tilted = conditional * np.exp(offset[codes])
                tilted /= np.maximum(tilted.sum(1, keepdims=True), 1e-300)
                gain = np.zeros(len(yy))
                ix = np.flatnonzero(answered)
                gain[ix] = np.log(np.maximum(tilted[ix, yy[ix]], 1e-300)) - np.log(np.maximum(conditional[ix, yy[ix]], 1e-300))
            # An ineligible cell contributes exactly zero to ranking.
            divergence = float(gain[support[codes]].sum() / len(yy))
            if divergence > 0:
                ranked.append((divergence, label, key, offset, support))
        ranked.sort(key=lambda v: (-v[0], v[1]))
        dest = rest if held is None else rows[split == held]
        chosen = []
        for rank, (score, label, key, offset, support) in enumerate(ranked[:ranks]):
            values, mask = offset[key[dest]], support[key[dest]]
            if held is None:
                shifts_r[rank], masks_r[rank] = values, mask
            else:
                shifts_o[rank][split == held], masks_o[rank][split == held] = values, mask
            chosen.append(dict(key=label, fit_divergence=score))
        trace.append(dict(held=None if held is None else int(held), selected=chosen))
    return shifts_o, shifts_r, masks_o, masks_r, trace


def _residual_divergence_raking(items, work, final, fo, fr, X, given, levels,
                                 fold, nll_fn, deadline):
    """Third orthogonal stage after every  selection, with paired guards."""
    if time.monotonic() >= deadline:
        return
    features = [f for f in range(len(given)) if 1 < int(levels[f]) <= 128]
    shared = []
    for ix, a in enumerate(features):
        shared.append((given[a], X[:, a], False))
        for b in features[ix + 1:]:
            _, key = np.unique(X[:, [a, b]], axis=0, return_inverse=True)
            shared.append((given[a] + ':' + given[b], key, True))
    for name, job in work.items():
        if time.monotonic() >= deadline:
            break
        rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
        y, K, split = job['y'][rows], job['K'], fold[rows]
        if K < 2 or len(np.unique(split)) != 5:
            continue
        gated = bool(items[name].get('gate'))
        source = _wave_two_source(name, items, X, given, rows)
        query_source = _wave_two_source(name, items, X, given, rest)
        if source is None or query_source is None:
            continue
        keys = _residual_keys(name, items, work, fo, fr, X, given, levels, shared)
        for binary in (False, True):
            if (binary and not gated) or (not binary and K < 3):
                continue
            for tau in (12.0, 20.0):
                if time.monotonic() >= deadline:
                    break
                fitted = _residual_ranked_shifts(keys, rows, rest,
                    y, split, fo[name], fr[name], source, gated, tau, binary, deadline)
                if fitted is None:
                    return
                so, sr, mo, mr, trace = fitted
                for rank in range(len(so)):
                    active, hidden = mo[rank] & source, mr[rank] & query_source
                    if active.sum() < 67:
                        continue
                    for gamma in ((.20, .35, .50) if binary else (.20, .35, .55)):
                        if time.monotonic() >= deadline:
                            return
                        before, before_r = fo[name], fr[name]
                        oo = _residual_project(before, so[rank], active, gamma, binary)
                        if not _answer_dominates(oo, before, y, K, split, active,
                                                nll_fn, se_multiplier=2.4):
                            continue
                        rr = _residual_project(before_r, sr[rank], hidden, gamma, binary)
                        assert np.array_equal(oo[~active], before[~active])
                        assert np.array_equal(rr[~hidden], before_r[~hidden])
                        if not binary:
                            assert np.array_equal(oo[:, -1], before[:, -1])
                            assert np.array_equal(rr[:, -1], before_r[:, -1])
                        fo[name], fr[name] = oo, rr
                        final[name][rest] = rr
                        row = np.arange(len(y))
                        gain = np.log(.001+(1-.001*K)*oo[row,y])-np.log(.001+(1-.001*K)*before[row,y])
                        fg = np.array([gain[split == f].mean() for f in np.unique(split)])
                        se = max(gain.std(ddof=1)/np.sqrt(len(gain)), fg.std(ddof=1)/np.sqrt(5))
                        job.setdefault('residual_raking', []).append(dict(
                            se_ratio=float(gain.mean()/max(se,1e-300)),
                            fold_gains=fg.tolist(),active_gain=float(gain[active].mean()),
                            operator='C2' if binary else 'C1', tau=tau,
                            gamma=gamma, rank=rank + 1, n_active=int(active.sum()),
                            oof_gain=float(nll_fn(before, y, K)-nll_fn(oo, y, K)),
                            selected_keys=trace))


def _peer_states(stacked, work, items, n):
    """Freeze integer OOF/Query features before any structural stage; no labels."""
    peers = {}
    for name, (oof, query, _) in stacked.items():
        train = work[name]['train']
        k = oof.shape[1]
        mode = np.empty(n, dtype=np.int32)
        for dest, prob in ((train, oof), (~train, query)):
            state = prob.argmax(1).astype(np.int32)
            state[prob.max(1) < .45] = k
            mode[dest] = state
        quantile = None
        if items[name].get('gate') or k <= 3:
            score = np.empty(n)
            score[train], score[~train] = oof[:, -1] if items[name].get('gate') else oof[:, 0], query[:, -1] if items[name].get('gate') else query[:, 0]
            quantile = np.searchsorted(np.quantile(score, [.25, .50, .75]), score, side='right').astype(np.int32)
        peers[name] = (mode, quantile)
    return peers


def _peer_mi(key, target, k):
    """Vectorized empirical MI on the current fit rows only."""
    cells = int(key.max()) + 1
    counts = np.bincount(key * k + target, minlength=cells*k).reshape(cells, k)
    nz = counts > 0
    denom = counts.sum(1)[:, None] * counts.sum(0)[None, :]
    return float((counts[nz] * np.log(counts[nz] * len(target) / denom[nz])).sum() / len(target))


def _peer_key_factory(name, items, work, peers, X, given, levels, rows, y):
    """Each held fold independently selects Top-6 peers and Top-4 GIVEN."""
    cache = {}
    def keys(fit):
        token = fit.tobytes()
        if token in cache:
            return cache[token]
        rr, yy, k = rows[fit], y[fit], work[name]['K']
        selected = sorted(((_peer_mi(v[0][rr], yy, k), m) for m, v in peers.items() if m != name), key=lambda v: (-v[0], v[1]))[:6]
        demographic = sorted(((_peer_mi(X[rr,f], yy, k), f) for f in range(len(given)) if 1 < int(levels[f]) <= 128), key=lambda v: (-v[0], v[1]))[:4]
        result = []
        def add(label, a, b=None):
            key = a if b is None else a * (int(b.max()) + 1) + b
            result.append((label, key, True))
        for rank, (_, m) in enumerate(selected):
            mode, quantile = peers[m]
            add('peer:' + m, mode)
            if quantile is not None:
                add('peer_q:' + m, quantile)
            if rank < 4:
                for _, f in demographic:
                    add('peer:' + m + ':given:' + given[f], mode, X[:, f])
                if items[name].get('gate'):
                    for parent in _declared_parents(name, items):
                        state = X[:, given.index(parent)] if parent in given else peers[parent][0] if parent in peers else None
                        if state is not None:
                            add('parent:' + parent + ':peer:' + m, state, mode)
        for i, (_, m) in enumerate(selected[:4]):
            for _, other in selected[:i]:
                add('peer:' + m + ':peer:' + other, peers[m][0], peers[other][0])
        cache[token] = result
        return result
    return keys


def _peer_project(best, shifts, mask, gamma, gated, binary=False):
    """Non-gated answers use every class; gated answers retain the skip column."""
    if gated:
        return _residual_project(best, shifts, mask, gamma, binary)
    out = best.copy()
    q = best[mask] * np.exp(gamma * shifts[mask])
    q *= best[mask].sum(1, keepdims=True) / np.maximum(q.sum(1, keepdims=True), 1e-300)
    out[mask] = q
    return out


def _peer_ranked_shifts(keys, rows, rest, y, split, best, query,
                            source, gated, tau, binary=False, deadline=float('inf')):
    """Fit and rank by realized fit NLL reduction, never target MI."""
    K = best.shape[1]
    ranks = 2 if binary else 3
    width = 1 if binary else K - 1 if gated else K
    shifts_o = [np.zeros((len(rows), width)) for _ in range(ranks)]
    shifts_r = [np.zeros((len(rest), width)) for _ in range(ranks)]
    masks_o = [np.zeros(len(rows), bool) for _ in range(ranks)]
    masks_r = [np.zeros(len(rest), bool) for _ in range(ranks)]
    trace = []
    for held in list(np.unique(split)) + [None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        yy, base = y[fit], best[fit]
        if len(yy) < 50:
            continue
        answered = yy != K - 1 if gated else np.ones(len(yy), bool)
        conditional = base[:, :width] / np.maximum(base[:, :width].sum(1, keepdims=True), 1e-300)
        ranked = []
        for label, key, interaction in keys(fit):
            if time.monotonic() >= deadline:
                return None
            if binary and not interaction:
                continue
            codes = key[rows[fit]]
            cells = int(key.max()) + 1
            n = np.bincount(codes, minlength=cells)
            n_ans = np.bincount(codes[answered], minlength=cells)
            if binary:
                observed = n - n_ans
                support = (n >= 50) & (observed > .03 * n) & (observed < .97 * n)
                if not support.any():
                    continue
                expected = np.bincount(codes, weights=base[:, -1], minlength=cells)
                p0 = float(base[:, -1].mean())
                o = np.clip((observed + tau * p0) / (n + tau), 1e-12, 1.0 - 1e-12)
                e = np.clip((expected + tau * p0) / (n + tau), 1e-12, 1.0 - 1e-12)
                offset = np.clip(np.log(o / (1.0-o)) - np.log(e / (1.0-e)), -.45, .45)[:, None]
                rho = np.clip(base[:, -1], 1e-12, 1.0 - 1e-12)
                new = 1.0 / (1.0 + np.exp(-(np.log(rho/(1.0-rho)) + offset[codes, 0])))
                target = ~answered
                gain = np.log(np.where(target, new, 1.0-new)) - np.log(np.where(target, rho, 1.0-rho))
            else:
                support = n_ans >= 45
                if gated:
                    support &= n_ans >= .25 * n
                if not support.any():
                    continue
                obs = np.bincount(codes[answered] * width + yy[answered], minlength=cells * width).reshape(cells, width)
                expected = np.column_stack([np.bincount(codes[answered], weights=conditional[answered, c], minlength=cells) for c in range(width)])
                mean = expected / np.maximum(n_ans[:, None], 1)
                fallback = conditional[answered].mean(0) if answered.any() else np.full(width, 1.0/width)
                mean[n_ans == 0] = fallback
                prior = np.maximum(mean, 1e-12)
                offset = np.clip(np.log((obs + tau * prior) / (expected + tau * prior)), -.45, .45)
                tilted = conditional * np.exp(offset[codes])
                tilted /= np.maximum(tilted.sum(1, keepdims=True), 1e-300)
                gain = np.zeros(len(yy))
                ix = np.flatnonzero(answered)
                gain[ix] = np.log(np.maximum(tilted[ix, yy[ix]], 1e-300)) - np.log(np.maximum(conditional[ix, yy[ix]], 1e-300))
            # An ineligible cell contributes exactly zero to ranking.
            divergence = float(gain[support[codes]].sum() / len(yy))
            if divergence > 0:
                ranked.append((divergence, label, key, offset, support))
        ranked.sort(key=lambda v: (-v[0], v[1]))
        dest = rest if held is None else rows[split == held]
        chosen = []
        for rank, (score, label, key, offset, support) in enumerate(ranked[:ranks]):
            values, mask = offset[key[dest]], support[key[dest]]
            if held is None:
                shifts_r[rank], masks_r[rank] = values, mask
            else:
                shifts_o[rank][split == held], masks_o[rank][split == held] = values, mask
            chosen.append(dict(key=label, fit_divergence=score))
        trace.append(dict(held=None if held is None else int(held), selected=chosen))
    return shifts_o, shifts_r, masks_o, masks_r, trace


def _peer_divergence_raking(items, work, final, fo, fr, X, given, levels,
                                 fold, nll_fn, deadline, peers):
    """Fourth orthogonal stage after /, using frozen peer predictions."""
    if time.monotonic() >= deadline:
        return
    for name, job in work.items():
        if time.monotonic() >= deadline:
            break
        rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
        y, K, split = job['y'][rows], job['K'], fold[rows]
        if K < 2 or len(np.unique(split)) != 5:
            continue
        gated = bool(items[name].get('gate'))
        source = _wave_two_source(name, items, X, given, rows)
        query_source = _wave_two_source(name, items, X, given, rest)
        if source is None or query_source is None:
            continue
        keys = _peer_key_factory(name, items, work, peers, X, given, levels, rows, y)
        for binary in (False, True):
            # GIVEN-parent skip masses are already locked by the observed routing arm.
            if (binary and (not gated or any(p in given for p in _declared_parents(name, items)))) or (not binary and K < (3 if gated else 2)):
                continue
            for tau in (12.0, 20.0):
                if time.monotonic() >= deadline:
                    break
                fitted = _peer_ranked_shifts(keys, rows, rest,
                    y, split, fo[name], fr[name], source, gated, tau, binary, deadline)
                if fitted is None:
                    return
                so, sr, mo, mr, trace = fitted
                for rank in range(len(so)):
                    active, hidden = mo[rank] & source, mr[rank] & query_source
                    if active.sum() < 67:
                        continue
                    for gamma in ((.20, .35, .50) if binary else (.20, .35, .55)):
                        if time.monotonic() >= deadline:
                            return
                        before, before_r = fo[name], fr[name]
                        oo = _peer_project(before, so[rank], active, gamma, gated, binary)
                        if not _answer_dominates(oo, before, y, K, split, active,
                                                nll_fn, se_multiplier=2.4):
                            continue
                        rr = _peer_project(before_r, sr[rank], hidden, gamma, gated, binary)
                        assert np.array_equal(oo[~active], before[~active])
                        assert np.array_equal(rr[~hidden], before_r[~hidden])
                        if not binary and gated:
                            assert np.array_equal(oo[:, -1], before[:, -1])
                            assert np.array_equal(rr[:, -1], before_r[:, -1])
                        fo[name], fr[name] = oo, rr
                        final[name][rest] = rr
                        row = np.arange(len(y))
                        gain = np.log(.001+(1-.001*K)*oo[row,y])-np.log(.001+(1-.001*K)*before[row,y])
                        fg = np.array([gain[split == f].mean() for f in np.unique(split)])
                        se = max(gain.std(ddof=1)/np.sqrt(len(gain)), fg.std(ddof=1)/np.sqrt(5))
                        job.setdefault('peer_raking', []).append(dict(
                            se_ratio=float(gain.mean()/max(se,1e-300)),
                            fold_gains=fg.tolist(),active_gain=float(gain[active].mean()),
                            operator='D2' if binary else 'D1', tau=tau,
                            gamma=gamma, rank=rank + 1, n_active=int(active.sum()),
                            oof_gain=float(nll_fn(before, y, K)-nll_fn(oo, y, K)),
                            selected_keys=trace))


def _ef_project(best, shift, mask, gamma, gated, beta=1.0):
    """Conditional exponential tilting with an exact inactive complement."""
    out = best.copy()
    width = best.shape[1] - int(gated)
    p = best[mask, :width]
    mass = p.sum(1, keepdims=True)
    conditional = p / np.maximum(mass, 1e-300)
    logits = beta * np.log(np.maximum(conditional, 1e-300)) + gamma * shift[mask]
    logits -= logits.max(1, keepdims=True)
    q = np.exp(logits)
    out[np.ix_(mask, np.arange(width))] = q * mass / q.sum(1, keepdims=True)
    return out


def _ef_dominates(arm, ref, y, split, mask, all_rows=False, kappa=2.4):
    """Five-fold, two-half and paired-SE hard dominance on training rows."""
    if mask.sum() < 67 or len(np.unique(split[mask])) != 5:
        return False
    K = ref.shape[1]
    row = np.arange(len(y))
    gain = (np.log(.001 + (1.0 - .001*K)*arm[row, y])
            - np.log(.001 + (1.0 - .001*K)*ref[row, y]))
    if gain[mask].mean() < (1.5e-5 if all_rows else 1e-4) or gain.mean() < 1.5e-5:
        return False
    fg = np.array([gain[split == f].mean() for f in np.unique(split)])
    if np.any(fg < 0.0) or (fg > 1e-6).sum() < 4:
        return False
    if any(gain[split % 2 == h].mean() <= 0.0 for h in (0, 1)):
        return False
    se = max(gain.std(ddof=1)/np.sqrt(len(gain)), fg.std(ddof=1)/np.sqrt(5))
    return bool(gain.mean() >= kappa * se)


def _ef_ranked_shifts(keys, rows, rest, y, split, best, query, source,
                      gated, tau, deadline):
    """: one Newton parameter per cell, three fit-ranked key/basis arms."""
    width = best.shape[1] - int(gated)
    so = [np.zeros((len(rows), width)) for _ in range(3)]
    sr = [np.zeros((len(rest), width)) for _ in range(3)]
    mo = [np.zeros(len(rows), bool) for _ in range(3)]
    mr = [np.zeros(len(rest), bool) for _ in range(3)]
    trace = []
    for held in list(np.unique(split)) + [None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        answered = y[fit] < width
        yy = y[fit][answered]
        rr = rows[fit][answered]
        base = best[fit][answered, :width]
        p = base / np.maximum(base.sum(1, keepdims=True), 1e-300)
        if len(yy) < 40:
            return None
        ordinal = np.linspace(-1.0, 1.0, width)
        entropy = np.log(p.mean(0) + 1e-3)
        entropy = np.clip(entropy - entropy.mean(), -1.5, 1.5)
        entropy /= max(float(np.max(np.abs(entropy))), 1e-12)
        ranked = []
        for label, key, _ in keys(fit):
            if time.monotonic() >= deadline:
                return None
            codes = key[rr]
            cells = int(key.max()) + 1
            n = np.bincount(key[rows[fit]], minlength=cells)
            n_ans = np.bincount(codes, minlength=cells)
            support = (n_ans >= 40) & (n_ans >= .25*n if gated else True)
            if not support.any():
                continue
            expected = np.column_stack([np.bincount(codes, weights=p[:, k], minlength=cells) for k in range(width)])
            modal = expected.argmax(1)
            # Subtracting each row's p_modal from every basis coordinate is
            # constant across classes; O-E, V and normalized tilts are identical.
            contrast = np.eye(width)[modal]
            contrast -= (expected * contrast).sum(1, keepdims=True) / np.maximum(n_ans[:, None], 1)
            for basis, u in (('ordinal', ordinal), ('surprisal', entropy), ('modal', contrast)):
                values = np.broadcast_to(u, (len(yy), width)) if u.ndim == 1 else u[codes]
                mu = (p * values).sum(1)
                observed = np.bincount(codes, weights=values[np.arange(len(yy)), yy], minlength=cells)
                mean = np.bincount(codes, weights=mu, minlength=cells)
                variance = np.bincount(codes, weights=(p * (values - mu[:, None])**2).sum(1), minlength=cells)
                theta = np.clip((observed - mean)/(variance + tau), -.60, .60)
                offset = theta[:, None] * np.broadcast_to(u, (cells, width))
                logits = offset[codes]
                tilted = p * np.exp(logits)
                tilted /= tilted.sum(1, keepdims=True)
                mass = base.sum(1)
                floor_scale = 1.0 - .001*best.shape[1]
                gain = np.log(.001 + floor_scale*mass*tilted[np.arange(len(yy)), yy]) - np.log(.001 + floor_scale*base[np.arange(len(yy)), yy])
                score = float(gain[support[codes]].sum()/max(int(fit.sum()), 1))
                if score > 0:
                    ranked.append((score, label, basis, key, offset, support))
        ranked.sort(key=lambda v: (-v[0], v[1], v[2]))
        dest = rest if held is None else rows[split == held]
        chosen = []
        for rank, (score, label, basis, key, offset, support) in enumerate(ranked[:3]):
            if held is None:
                sr[rank], mr[rank] = offset[key[dest]], support[key[dest]]
            else:
                so[rank][split == held], mo[rank][split == held] = offset[key[dest]], support[key[dest]]
            chosen.append(dict(key=label, basis=basis, fit_gain=score))
        trace.append(dict(held=None if held is None else int(held), selected=chosen))
    return so, sr, mo, mr, trace


def _ef_macro_shifts(key, rows, rest, y, split, best, source, gated, beta, deadline, held_folds=None):
    """: cross-fitted temperature and Dirichlet-shrunk marginal raking."""
    width = best.shape[1] - int(gated)
    so, sr = np.zeros((len(rows), width)), np.zeros((len(rest), width))
    mo, mr = np.zeros(len(rows), bool), np.zeros(len(rest), bool)
    cells = int(key.max()) + 1
    for held in (list(np.unique(split)) + [None] if held_folds is None else held_folds):
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        answered = y[fit] < width
        yy, codes = y[fit][answered], key[rows[fit]][answered]
        n_ans = np.bincount(codes, minlength=cells)
        n = np.bincount(key[rows[fit]], minlength=cells)
        support = (n_ans >= 200) & (n_ans >= .25*n if gated else True)
        if not support.any():
            continue
        p = best[fit][answered, :width]
        p = np.maximum(p / np.maximum(p.sum(1, keepdims=True), 1e-300), 1e-300)**beta
        p /= p.sum(1, keepdims=True)
        obs = np.bincount(codes * width + yy, minlength=cells*width).reshape(cells, width)
        expected = np.column_stack([np.bincount(codes, weights=p[:, k], minlength=cells) for k in range(width)])
        prior = np.maximum(expected / np.maximum(n_ans[:, None], 1), 1e-12)
        offset = np.clip(np.log((obs + 20.0*prior)/(expected + 20.0*prior)), -.45, .45)
        dest = rest if held is None else rows[split == held]
        if held is None:
            sr, mr = offset[key[dest]], support[key[dest]]
        else:
            so[split == held], mo[split == held] = offset[key[dest]], support[key[dest]]
    return so, sr, mo, mr


def _exponential_family_raking(items, work, final, fo, fr, X, given, levels,
                                fold, nll_fn, deadline, peers):
    """Fifth orthogonal stage, preserving every  arm and skip invariant."""
    if time.monotonic() >= deadline:
        return
    features = [f for f in range(len(given)) if 1 < int(levels[f]) <= 128]
    shared = []
    for ix, a in enumerate(features):
        shared.append((given[a], X[:, a], False))
        for b in features[ix + 1:]:
            key = X[:, a] * int(levels[b]) + X[:, b]
            _, key = np.unique(key, return_inverse=True)
            shared.append((given[a] + ':' + given[b], key, True))
    for name, job in work.items():
        if time.monotonic() >= deadline:
            return
        rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
        y, split = job['y'][rows], fold[rows]
        gated = bool(items[name].get('gate'))
        width = job['K'] - int(gated)
        if width < 2 or len(np.unique(split)) != 5:
            continue
        source = _wave_two_source(name, items, X, given, rows)
        hidden_source = _wave_two_source(name, items, X, given, rest)
        if source is None or hidden_source is None:
            continue
        static = _residual_keys(name, items, work, fo, fr, X, given, levels, shared) if width >= 3 else []
        key_cache = {}
        def keys(fit):
            token = fit.tobytes()
            if token not in key_cache:
                rr, yy = rows[fit], y[fit]
                selected = sorted(((_peer_mi(v[0][rr], yy, job['K']), m) for m, v in peers.items() if m != name), key=lambda v: (-v[0], v[1]))[:4]
                key_cache[token] = static + [('peer_mode:' + m, peers[m][0], True) for _, m in selected]
            return key_cache[token]
        def accept(so, sr, mo, mr, gamma, operator, beta=1.0, all_rows=False, **metadata):
            active, hidden = mo & source, mr & hidden_source
            before, before_r = fo[name], fr[name]
            oo = _ef_project(before, so, active, gamma, gated, beta)
            if not _ef_dominates(oo, before, y, split, active, all_rows=all_rows,
                                  kappa=2.4 if operator == 'E1' else 3.2):
                return
            rr = _ef_project(before_r, sr, hidden, gamma, gated, beta)
            assert np.array_equal(oo[~active], before[~active]) and np.array_equal(rr[~hidden], before_r[~hidden])
            if gated:
                assert np.array_equal(oo[:, -1], before[:, -1]) and np.array_equal(rr[:, -1], before_r[:, -1])
            fo[name], fr[name], final[name][rest] = oo, rr, rr
            gain = np.log(.001 + (1-.001*job['K'])*oo[np.arange(len(y)), y]) - np.log(.001 + (1-.001*job['K'])*before[np.arange(len(y)), y])
            fg = np.array([gain[split == f].mean() for f in np.unique(split)])
            se = max(gain.std(ddof=1)/np.sqrt(len(gain)), fg.std(ddof=1)/np.sqrt(5))
            job.setdefault('ef_raking', []).append(dict(operator=operator, kappa=2.4 if operator == 'E1' else 3.2, gamma=gamma, beta=beta, n_active=int(active.sum()), active_gain=float(gain[active].mean()), oof_gain=float(gain.mean()), fold_gains=fg.tolist(), se_ratio=float(gain.mean()/max(se, 1e-300)), **metadata))
        if width >= 3:
            for tau in (6.0, 14.0):
                fitted = _ef_ranked_shifts(keys, rows, rest, y, split, fo[name], fr[name], source, gated, tau, deadline)
                if fitted is None:
                    return
                so, sr, mo, mr, trace = fitted
                for rank in range(3):
                    for gamma in (.35, .65, 1.0):
                        if time.monotonic() >= deadline:
                            return
                        accept(so[rank], sr[rank], mo[rank], mr[rank], gamma, 'E1', tau=tau, rank=rank+1, selected_keys=trace)
        macro = [('all_rows', np.zeros(len(X), dtype=np.int32))]
        macro_features = [f for f in features if given[f] in ('wave', 'country', 'group')]
        eligible_fit = source & (y < width)
        if eligible_fit.any() and features:
            # Top GIVEN is selected inside every fit fold below, not on held labels.
            macro.append(('top_given', None))
        macro.extend((given[f], X[:, f]) for f in macro_features)
        for label, fixed_key in macro:
            # For fold-selected GIVEN, emit fold-matched predictions by assembling
            # independent complete fits, each with its own key factory.
            for beta in (.92, .96, 1.0, 1.04, 1.08):
                if fixed_key is None:
                    fitted = [np.zeros((len(rows), width)), np.zeros((len(rest), width)), np.zeros(len(rows), bool), np.zeros(len(rest), bool)]
                    for held in list(np.unique(split)) + [None]:
                        fit = source & (split != held) if held is not None else source.copy()
                        answered = fit & (y < width)
                        if not answered.any():
                            continue
                        f = max(features, key=lambda f: _peer_mi(X[rows[answered], f], y[answered], job['K']))
                        part = _ef_macro_shifts(X[:, f], rows, rest, y, split, fo[name], source, gated, beta, deadline, (held,))
                        if part is None:
                            return
                        if held is None:
                            fitted[1], fitted[3] = part[1], part[3]
                        else:
                            v = split == held
                            fitted[0][v], fitted[2][v] = part[0][v], part[2][v]
                else:
                    fitted = _ef_macro_shifts(fixed_key, rows, rest, y, split, fo[name], source, gated, beta, deadline)
                if fitted is None:
                    return
                for gamma in (.25, .50):
                    if time.monotonic() >= deadline:
                        return
                    accept(*fitted, gamma, 'E2', beta=beta, all_rows=label == 'all_rows', tau=20.0, key=label)


def _frozen_peer_probabilities(stacked, work, n):
    """Immutable continuous peer features, assembled only from entry OOF/Query."""
    snapshot = {}
    for name, (oof, query, _) in stacked.items():
        p = np.empty((n, oof.shape[1]), dtype=float)
        p[work[name]['train']], p[~work[name]['train']] = oof, query
        p.setflags(write=False)
        snapshot[name] = p
    return snapshot


def _f1_ranked_shifts(rows, rest, y, split, best, source, gated,
                       snapshot, peer_keys, tau, deadline, skip=False):
    """: fold-local rank-one residual covariance or ridge skip residual."""
    width = best.shape[1] - int(gated)
    ranks, out_width = (1, 1) if skip else (2, width)
    so = [np.zeros((len(rows), out_width)) for _ in range(ranks)]
    sr = [np.zeros((len(rest), out_width)) for _ in range(ranks)]
    mo = [np.zeros(len(rows), bool) for _ in range(ranks)]
    mr = [np.zeros(len(rest), bool) for _ in range(ranks)]
    trace = []
    for held in list(np.unique(split)) + [None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        answered = fit & (y < width)
        if answered.sum() < 40 or (gated and answered.sum() < .20*fit.sum()):
            continue
        selected = sorted(((_peer_mi(v[0][rows[fit]], y[fit], best.shape[1]), m)
                           for m, v in peer_keys.items() if m in snapshot),
                          key=lambda v: (-v[0], v[1]))[:6]
        candidates = [(m, snapshot[m]) for _, m in selected]
        if len(selected) >= 3:
            candidates.append(('top3_concat', np.concatenate([snapshot[m] for _, m in selected[:3]], axis=1)))
        if skip:
            candidates = candidates[-1:] if len(selected) >= 3 else candidates[:1]
        ranked = []
        fit_use = fit if skip else answered
        rr, yy = rows[fit_use], y[fit_use]
        base = best[fit_use]
        p = base[:, :width]/np.maximum(base[:, :width].sum(1, keepdims=True), 1e-300)
        residual = (yy == width).astype(float)-base[:, -1] if skip else np.eye(width)[yy]-p
        dest = rest if held is None else rows[split == held]
        for label, prob in candidates:
            if time.monotonic() >= deadline:
                return None
            center = prob[rr].mean(0)
            xf = prob[rr]-center
            if skip:
                w = np.linalg.solve(xf.T@xf + 50.0*np.eye(xf.shape[1]), xf.T@residual)
                # Ridge predicts the probability residual; project along log odds.
                z = np.clip(xf@w, -.20, .20)
                rho = np.clip(base[:, -1], 1e-6, 1-1e-6)
                value = np.clip(z/np.maximum(rho*(1-rho), .04), -.50, .50)[:, None]
                new = _residual_skip_project(base, np.ones(len(base), bool), np.clip(base[:, -1]+z, 1e-9, 1-1e-9))
                zd = np.clip((prob[dest]-center)@w, -.20, .20)
                # Destination variance is supplied by the current best below.
                offsets = zd[:, None]
            else:
                left, _, right = np.linalg.svd(xf.T@residual/len(rr), full_matrices=False)
                w, u = left[:, 0], right[0]
                u = u/max(float(np.abs(u).max()), 1e-12)
                scale = float((xf@w).std())+1e-6
                z = np.clip((xf@w)/scale, -2.0, 2.0)
                mu = p@u
                var = (p*(u[None, :]-mu[:, None])**2).sum(1)
                theta = float(np.clip(np.sum(z*(u[yy]-mu))/(np.sum(z*z*var)+tau), -.50, .50))
                value = theta*z[:, None]*u
                new = _ef_project(base, value, np.ones(len(base), bool), 1.0, gated)
                offsets = theta*np.clip(((prob[dest]-center)@w)/scale, -2.0, 2.0)[:, None]*u
            row = np.arange(len(yy))
            gain = np.log(.001+(1-.001*best.shape[1])*new[row, yy])-np.log(.001+(1-.001*best.shape[1])*base[row, yy])
            score = float(gain.sum()/max(int(fit.sum()), 1))
            if score > 0:
                ranked.append((score, label, offsets))
        ranked.sort(key=lambda v: (-v[0], v[1]))
        chosen = []
        for rank, (score, label, offsets) in enumerate(ranked[:ranks]):
            if held is None:
                sr[rank], mr[rank] = offsets, np.ones(len(rest), bool)
            else:
                v = split == held
                so[rank][v], mo[rank][v] = offsets, True
            chosen.append(dict(peer=label, fit_gain=score))
        trace.append(dict(held=None if held is None else int(held), selected=chosen))
    return so, sr, mo, mr, trace


def _f2_ranked_shifts(keys, rows, rest, y, split, best, source, gated, tau, deadline):
    """: heteroscedastic binary Fisher, nominal residual SVD and Top-3 OVR."""
    width = best.shape[1]-int(gated)
    so = [np.zeros((len(rows), width)) for _ in range(3)]
    sr = [np.zeros((len(rest), width)) for _ in range(3)]
    mo = [np.zeros(len(rows), bool) for _ in range(3)]
    mr = [np.zeros(len(rest), bool) for _ in range(3)]
    trace = []
    for held in list(np.unique(split)) + [None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        ans = fit & (y < width)
        rr, yy, base = rows[ans], y[ans], best[ans, :width]
        if len(yy) < 40:
            continue
        p = base/np.maximum(base.sum(1, keepdims=True), 1e-300)
        top3 = np.argsort(-np.bincount(yy, minlength=width), kind='stable')[:3]
        ranked = []
        for label, key, _ in keys(fit):
            if time.monotonic() >= deadline:
                return None
            cells, codes = int(key.max())+1, key[rr]
            n = np.bincount(key[rows[fit]], minlength=cells)
            na = np.bincount(codes, minlength=cells)
            support = (na >= 40) & ((na >= .20*n) if gated else True)
            if not support.any():
                continue
            obs = np.bincount(codes*width+yy, minlength=cells*width).reshape(cells, width)
            exp = np.column_stack([np.bincount(codes, weights=p[:, k], minlength=cells) for k in range(width)])
            if width == 2:
                bases = [('binary_fisher', np.array([0.0, 1.0]))]
            else:
                matrix = (obs-exp)[support]/np.sqrt(na[support, None]+12.0)
                u = np.linalg.svd(matrix, full_matrices=False)[2][0]
                u = u/max(float(np.abs(u).max()), 1e-12)
                bases = [('residual_svd', u)]
                bases += [('ovr:'+str(int(k)), np.where(np.arange(width)==k, 1.0, -1.0)) for k in top3]
            for basis, u in bases:
                row = np.arange(len(yy))
                if basis == 'residual_svd':
                    mu = p@u
                    var = (p*(u[None, :]-mu[:, None])**2).sum(1)
                else:
                    category = 1 if width == 2 else int(basis.split(':')[1])
                    var = p[:, category]*(1.0-p[:, category])*(1.0 if width == 2 else 4.0)
                score_sum = (obs-exp)@u
                variance = np.bincount(codes, weights=var, minlength=cells)
                theta = np.clip(score_sum/(variance+tau), -.55, .55)
                offset = theta[:, None]*u
                if basis == 'residual_svd':
                    tilted = p*np.exp(offset[codes])
                    tilted /= tilted.sum(1, keepdims=True)
                    observed_prob = tilted[row, yy]
                else:
                    # Exact binary/OVR normalizer: cancel the common exp(-theta).
                    step = theta[codes]*(1.0 if width == 2 else 2.0)
                    observed_prob = p[row, yy]*np.exp(step*(yy == category))/(1.0+p[:, category]*np.expm1(step))
                mass = base.sum(1)
                scale = 1-.001*best.shape[1]
                gain = np.log(.001+scale*mass*observed_prob)-np.log(.001+scale*base[row, yy])
                score = float(gain[support[codes]].sum()/max(int(fit.sum()), 1))
                if score > 0:
                    ranked.append((score, label, basis, key, offset, support))
        ranked.sort(key=lambda v: (-v[0], v[1], v[2]))
        dest = rest if held is None else rows[split == held]
        chosen = []
        for rank, (score, label, basis, key, offset, support) in enumerate(ranked[:3]):
            if held is None:
                sr[rank], mr[rank] = offset[key[dest]], support[key[dest]]
            else:
                v = split == held
                so[rank][v], mo[rank][v] = offset[key[dest]], support[key[dest]]
            chosen.append(dict(key=label, basis=basis, fit_gain=score))
        trace.append(dict(held=None if held is None else int(held), selected=chosen))
    return so, sr, mo, mr, trace


def _fisher_peer_raking(items, work, final, fo, fr, X, given, levels,
                         fold, nll_fn, deadline, peers, base_snapshot):
    """Sixth orthogonal stage after all  stages; no peer labels are read."""
    features = [f for f in range(len(given)) if 1 < int(levels[f]) <= 128]
    shared = []
    for ix, a in enumerate(features):
        shared.append((given[a], X[:, a], False))
        for b in features[ix+1:]:
            _, key = np.unique(X[:, a]*int(levels[b])+X[:, b], return_inverse=True)
            shared.append((given[a]+':'+given[b], key, True))
    for name, job in work.items():
        if time.monotonic() >= deadline:
            return
        rows, rest = np.flatnonzero(job['train']), np.flatnonzero(~job['train'])
        y, split = job['y'][rows], fold[rows]
        gated = bool(items[name].get('gate'))
        width = job['K']-int(gated)
        if width < 1 or len(np.unique(split)) != 5:
            continue
        source = _wave_two_source(name, items, X, given, rows)
        hidden_source = _wave_two_source(name, items, X, given, rest)
        if source is None or hidden_source is None:
            continue
        peer_keys = {m:v for m,v in peers.items() if m != name}
        snapshot = {m:v for m,v in base_snapshot.items() if m != name}
        static = _residual_keys(name, items, work, fo, fr, X, given, levels, shared)
        key_cache = {}
        def keys(fit):
            token = fit.tobytes()
            if token not in key_cache:
                selected = sorted(((_peer_mi(v[0][rows[fit]], y[fit], job['K']), m) for m,v in peer_keys.items()), key=lambda v: (-v[0], v[1]))[:4]
                key_cache[token] = static + [('peer_mode:'+m, peer_keys[m][0], True) for _,m in selected]
            return key_cache[token]
        def accept(so, sr, mo, mr, gamma, operator, **metadata):
            before, before_r = fo[name], fr[name]
            active, hidden = mo & source, mr & hidden_source
            if gated:
                active &= 1-before[:, -1] >= .20
                hidden &= 1-before_r[:, -1] >= .20
            if operator == 'F1_skip':
                # Preserve pure gates and GIVEN-parent routing, plus Wave-1 rows.
                active &= (before[:, -1] > .015) & (before[:, -1] < .985)
                hidden &= (before_r[:, -1] > .015) & (before_r[:, -1] < .985)
                oo = _residual_skip_project(before, active, np.clip(before[active, -1]+gamma*so[active, 0], 1e-9, 1-1e-9))
            else:
                oo = _ef_project(before, so, active, gamma, gated)
            kappa = 2.7 if operator == 'F2' and width >= 3 else 2.4
            if not _ef_dominates(oo, before, y, split, active, all_rows=operator=='F1', kappa=kappa):
                return
            if operator == 'F1_skip':
                rr = _residual_skip_project(before_r, hidden, np.clip(before_r[hidden, -1]+gamma*sr[hidden, 0], 1e-9, 1-1e-9))
            else:
                rr = _ef_project(before_r, sr, hidden, gamma, gated)
            assert np.array_equal(oo[~active], before[~active]) and np.array_equal(rr[~hidden], before_r[~hidden])
            if gated and operator != 'F1_skip':
                assert np.array_equal(oo[:, -1], before[:, -1]) and np.array_equal(rr[:, -1], before_r[:, -1])
            fo[name], fr[name], final[name][rest] = oo, rr, rr
            gain = np.log(.001+(1-.001*job['K'])*oo[np.arange(len(y)), y])-np.log(.001+(1-.001*job['K'])*before[np.arange(len(y)), y])
            fg = np.array([gain[split==f].mean() for f in np.unique(split)])
            se = max(gain.std(ddof=1)/np.sqrt(len(gain)), fg.std(ddof=1)/np.sqrt(5))
            job.setdefault('f_raking', []).append(dict(operator=operator, kappa=kappa, gamma=gamma, n_active=int(active.sum()), active_gain=float(gain[active].mean()), oof_gain=float(gain.mean()), fold_gains=fg.tolist(), se_ratio=float(gain.mean()/max(se, 1e-300)), **metadata))
        for operator in ('F1', 'F1_skip', 'F2'):
            if width < 2 and operator != 'F1_skip':
                continue
            if operator == 'F1_skip' and (not gated or any(p in given for p in _declared_parents(name, items))):
                continue
            for tau in ((15.0, 40.0) if operator=='F1' else (50.0,) if operator=='F1_skip' else (6.0, 14.0)):
                if time.monotonic() >= deadline:
                    return
                fitted = (_f2_ranked_shifts(keys, rows, rest, y, split, fo[name], source, gated, tau, deadline)
                          if operator=='F2' else _f1_ranked_shifts(rows, rest, y, split, fo[name], source, gated, snapshot, peer_keys, tau, deadline, operator=='F1_skip'))
                if fitted is None:
                    return
                so, sr, mo, mr, trace = fitted
                for rank in range(len(so)):
                    for gamma in (.35, .65, 1.0):
                        if time.monotonic() >= deadline:
                            return
                        accept(so[rank], sr[rank], mo[rank], mr[rank], gamma, operator, tau=tau, rank=rank+1, selected_keys=trace)


def _g_conditional(best, width):
    return best[:, :width]/np.maximum(best[:, :width].sum(1, keepdims=True), 1e-300)


def _g_basis(p):
    """Row-centered bounded log probability, with exact model expectation zero."""
    logs = np.clip(np.log(p+1e-4), -6.5, 0.0)
    return logs-(p*logs).sum(1, keepdims=True)


def _g_fit_gain(base, y, shift, gated):
    # Exact observed-class normalizer avoids a full projected-copy and log pass.
    width = base.shape[1]-int(gated)
    p = base[:, :width]
    row = np.arange(len(y))
    if width == 2:
        odds = np.exp(shift[:, 1]-shift[:, 0])
        prob = p[row, y]*np.where(y == 1, odds, 1.0)*p.sum(1)/np.maximum(p[:, 0]+p[:, 1]*odds, 1e-300)
    else:
        weighted = p*np.exp(shift-shift.max(1, keepdims=True))
        prob = weighted[row, y]*p.sum(1)/np.maximum(weighted.sum(1), 1e-300)
    scale = 1.0-.001*base.shape[1]
    return np.log(.001+scale*prob)-np.log(.001+scale*base[row, y])


def _g1_ranked_shifts(rows, rest, y, split, best, query, source, gated,
                       snapshot, peer_keys, given_matrix, deadline, skip=False):
    """Fold-local ridge whitening; direct and rank-one/two residual Fisher heads."""
    width = best.shape[1]-int(gated)
    out_width = 1 if skip else width
    so = [np.zeros((len(rows), out_width)) for _ in range(2)]
    sr = [np.zeros((len(rest), out_width)) for _ in range(2)]
    mo = [np.zeros(len(rows), bool) for _ in range(2)]
    mr = [np.zeros(len(rest), bool) for _ in range(2)]
    trace = []
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        use = fit if skip else fit & (y < width)
        if use.sum() < 40:
            continue
        rr, yy, base = rows[use], y[use], best[use]
        p = _g_conditional(base, width)
        selected = sorted(((_peer_mi(v[0][rows[fit]], y[fit], best.shape[1]), m)
                           for m,v in peer_keys.items() if m in snapshot),
                          key=lambda v: (-v[0], v[1]))[:8]
        if not selected:
            continue
        dest = rest if held is None else rows[split == held]
        dest_base = query if held is None else best[split == held]
        pd = _g_conditional(dest_base, width)
        residual = ((yy == width).astype(float)-base[:, -1] if skip
                    else np.eye(width)[yy]-p)
        group_sizes = (8,) if skip else (4, 8, 9)
        ranked = []
        for group in group_sizes:
            names = [m for _,m in selected[:min(group, 8)]]
            xf = np.concatenate([snapshot[m][rr] for m in names]+([given_matrix[rr]] if group==9 or skip else []), axis=1)
            xd = np.concatenate([snapshot[m][dest] for m in names]+([given_matrix[dest]] if group==9 or skip else []), axis=1)
            center = xf.mean(0)
            xf, xd = xf-center, xd-center
            gram, cross = xf.T@xf, xf.T@residual
            for lam in ((40.0, 100.0) if skip else (30.0, 90.0)):
                if time.monotonic() >= deadline:
                    return None
                B = np.linalg.solve(gram+lam*np.eye(gram.shape[0]), cross)
                sf, sd = xf@B, xd@B
                projections = [('direct', None)]
                if not skip:
                    # SVD is over fitted residual predictions, not unwhitened X^T R.
                    right = np.linalg.svd(sf, full_matrices=False)[2]
                    projections += [('rank'+str(rank), right[:rank]) for rank in (1,2) if rank <= width]
                for label, vectors in projections:
                    a = sf if vectors is None else (sf@vectors.T)@vectors
                    b = sd if vectors is None else (sd@vectors.T)@vectors
                    if skip:
                        rho = np.clip(base[:, -1], 1e-6, 1-1e-6)
                        rd = np.clip(dest_base[:, -1], 1e-6, 1-1e-6)
                        a = np.clip(a/np.maximum(rho*(1-rho), .04), -.60, .60)
                        b = np.clip(b/np.maximum(rd*(1-rd), .04), -.60, .60)
                        observed = float(a@residual)
                        variance = float(np.sum(rho*(1-rho)*a*a))
                    else:
                        a, b = np.clip(a, -.60, .60), np.clip(b, -.60, .60)
                        a -= (p*a).sum(1, keepdims=True)
                        b -= (pd*b).sum(1, keepdims=True)
                        observed = float(a[np.arange(len(yy)), yy].sum())
                        variance = float(np.sum(p*a*a))
                    for tau in (15.0, 40.0):
                        theta = float(np.clip(observed/(variance+tau), 0.0, 1.20))
                        if skip:
                            new_rho = 1/(1+np.exp(-(np.log(rho/(1-rho))+theta*a)))
                            new = _residual_skip_project(base, np.ones(len(base), bool), new_rho)
                            scale = 1-.001*base.shape[1]
                            row = np.arange(len(yy))
                            gain = np.log(.001+scale*new[row,yy])-np.log(.001+scale*base[row,yy])
                            offsets = (theta*b)[:,None]
                        else:
                            gain = _g_fit_gain(base, yy, theta*a, gated)
                            offsets = theta*b
                        score = float(gain.sum()/max(int(fit.sum()),1))
                        if score > 0:
                            ranked.append((score, 'top'+str(group)+'_'+label, lam, tau, offsets, theta))
        ranked.sort(key=lambda v: (-v[0], v[1], v[2], v[3]))
        chosen = []
        for rank,(score,label,lam,tau,offsets,theta) in enumerate(ranked[:2]):
            if held is None:
                sr[rank], mr[rank] = offsets, np.ones(len(rest), bool)
            else:
                v = split == held
                so[rank][v], mo[rank][v] = offsets, True
            chosen.append(dict(features=label, lam=lam, tau=tau, theta=theta, fit_gain=score))
        trace.append(dict(held=None if held is None else int(held), peers=[m for _,m in selected], selected=chosen))
    return so,sr,mo,mr,trace


def _g2_ranked_shifts(keys, rows, rest, y, split, best, query, source,
                       gated, peer_keys, tau, deadline, global_step=False):
    """Row temperature steps, globally modulated or fit-selected cell-specific."""
    width = best.shape[1]-int(gated)
    p, pq = _g_conditional(best, width), _g_conditional(query, width)
    basis, bq = _g_basis(p), _g_basis(pq)
    variance = (p*basis*basis).sum(1)
    so = [np.zeros_like(p) for _ in range(2)]
    sr = [np.zeros_like(pq) for _ in range(2)]
    mo = [np.zeros(len(rows), bool) for _ in range(2)]
    mr = [np.zeros(len(rest), bool) for _ in range(2)]
    trace = []
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        ans = fit & (y < width)
        rr, yy, pp, ss = rows[ans], y[ans], p[ans], basis[ans]
        if len(yy) < 40:
            continue
        dest = rest if held is None else rows[split==held]
        ds = bq if held is None else basis[split==held]
        ranked = []
        observed = ss[np.arange(len(yy)), yy]
        if global_step:
            entropy = -(pp*np.log(np.maximum(pp,1e-300))).sum(1)
            dp = pq if held is None else p[split==held]
            de = -(dp*np.log(np.maximum(dp,1e-300))).sum(1)
            candidates = [('constant',np.ones(len(yy)),np.ones(len(dest)))]
            center,scale = float(entropy.mean()), max(float(entropy.std()), .05)
            candidates.append(('entropy_linear',np.clip((entropy-center)/scale,-2,2),np.clip((de-center)/scale,-2,2)))
            selected = sorted(((_peer_mi(v[0][rows[fit]],y[fit],best.shape[1]),m) for m,v in peer_keys.items()),key=lambda v:(-v[0],v[1]))[:1]
            for _,m in selected:
                # Frozen mode keys supply a fit-only one-dimensional peer score.
                score = peer_keys[m][0]
                z = score[rr].astype(float)
                center,scale = float(z.mean()), max(float(z.std()),1.0)
                candidates.append(('peer_linear:'+m,np.clip((z-center)/scale,-2,2),np.clip((score[dest]-center)/scale,-2,2)))
            for label,z,zd in candidates:
                theta = float(np.clip(np.sum(z*observed)/(np.sum(z*z*variance[ans])+tau),-.30,.30))
                shift = theta*z[:,None]*ss
                gain = float(_g_fit_gain(best[ans],yy,shift,gated).sum()/max(int(fit.sum()),1))
                if gain > 0:
                    ranked.append((gain,label,theta*zd[:,None]*ds,np.ones(len(dest),bool)))
        else:
            for label,key,_ in keys(fit):
                if time.monotonic() >= deadline:
                    return None
                cells,codes = int(key.max())+1,key[rr]
                na = np.bincount(codes,minlength=cells)
                support = na>=40
                if not support.any():
                    continue
                obs = np.bincount(codes,weights=observed,minlength=cells)
                var = np.bincount(codes,weights=variance[ans],minlength=cells)
                theta = np.clip(obs/(var+tau),-.35,.35)
                shift = theta[codes,None]*ss
                gain = _g_fit_gain(best[ans],yy,shift,gated)
                score = float(gain[support[codes]].sum()/max(int(fit.sum()),1))
                if score > 0:
                    ranked.append((score,label,theta[key[dest],None]*ds,support[key[dest]]))
        ranked.sort(key=lambda v:(-v[0],v[1]))
        chosen=[]
        for rank,(gain,label,offsets,support) in enumerate(ranked[:2]):
            if held is None:
                sr[rank],mr[rank]=offsets,support
            else:
                v=split==held
                so[rank][v],mo[rank][v]=offsets,support
            chosen.append(dict(key=label,fit_gain=gain))
        trace.append(dict(held=None if held is None else int(held),selected=chosen))
    return so,sr,mo,mr,trace


def _canonical_temperature_raking(items,work,final,fo,fr,X,given,levels,
                                    fold,nll_fn,deadline,peers,base_snapshot):
    """Seventh orthogonal stage; all existing  defenses precede this call."""
    features=[f for f in range(len(given)) if 1<int(levels[f])<=128]
    # Precomputed immutable discrete GIVEN one-hot block, shared by all heads.
    given_matrix=np.concatenate([(X[:,f,None]==np.arange(int(levels[f]))).astype(float)
                                 for f in features],axis=1) if features else np.empty((len(X),0))
    shared=[]
    for ix,a in enumerate(features):
        shared.append((given[a],X[:,a],False))
        for b in features[ix+1:]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((given[a]+':'+given[b],key,True))
    for name,job in work.items():
        if time.monotonic()>=deadline:
            return
        rows,rest=np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
        y,split=job['y'][rows],fold[rows]
        gated=bool(items[name].get('gate'))
        width=job['K']-int(gated)
        if width<1 or len(np.unique(split))!=5:
            continue
        source=_wave_two_source(name,items,X,given,rows)
        hidden_source=_wave_two_source(name,items,X,given,rest)
        if source is None or hidden_source is None:
            continue
        # The G-stage Wave-1 lock applies to every item, including items that
        # are not recognized by the legacy zero-on-skip detector.
        if 'wave' in given:
            wave_one = next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'), None)
            if wave_one is not None:
                source &= X[rows, given.index('wave')] != wave_one
                hidden_source &= X[rest, given.index('wave')] != wave_one
        peer_keys={m:v for m,v in peers.items() if m!=name}
        snapshot={m:v for m,v in base_snapshot.items() if m!=name}
        static=_residual_keys(name,items,work,fo,fr,X,given,levels,shared)
        key_cache={}
        def keys(fit):
            token=fit.tobytes()
            if token not in key_cache:
                selected=sorted(((_peer_mi(v[0][rows[fit]],y[fit],job['K']),m) for m,v in peer_keys.items()),key=lambda v:(-v[0],v[1]))[:4]
                key_cache[token]=static+[('peer_mode:'+m,peer_keys[m][0],True) for _,m in selected]
            return key_cache[token]
        def accept(so,sr,mo,mr,gamma,operator,**metadata):
            before,before_r=fo[name],fr[name]
            active,hidden=mo&source,mr&hidden_source
            if gated:
                active &= 1-before[:,-1]>=.20
                hidden &= 1-before_r[:,-1]>=.20
            if operator=='G1_skip':
                active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                hidden &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[active,-1],1e-9,1-1e-9)
                updated=1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*so[active,0])))
                oo=_residual_skip_project(before,active,updated)
            else:
                oo=_ef_project(before,so,active,gamma,gated)
            kappa=5.0 if operator in ('G2_global','G2_cell') else 2.4
            if not _ef_dominates(oo,before,y,split,active,all_rows=operator in ('G1','G2_global'),kappa=kappa):
                return
            if operator=='G1_skip':
                rho=np.clip(before_r[hidden,-1],1e-9,1-1e-9)
                updated=1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*sr[hidden,0])))
                rr=_residual_skip_project(before_r,hidden,updated)
            else:
                rr=_ef_project(before_r,sr,hidden,gamma,gated)
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(rr[~hidden],before_r[~hidden])
            if gated and operator!='G1_skip':
                assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(rr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,rr,rr
            row=np.arange(len(y))
            gain=np.log(.001+(1-.001*job['K'])*oo[row,y])-np.log(.001+(1-.001*job['K'])*before[row,y])
            fg=np.array([gain[split==f].mean() for f in np.unique(split)])
            se=max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
            job.setdefault('g_raking',[]).append(dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),active_gain=float(gain[active].mean()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata))
        for operator in ('G1','G1_skip','G2_global','G2_cell'):
            if width<2 and operator!='G1_skip':
                continue
            if operator=='G1_skip' and (not gated or any(p in given for p in _declared_parents(name,items))):
                continue
            for tau in ((None,) if operator.startswith('G1') else (6.0,16.0)):
                if time.monotonic()>=deadline:
                    return
                fitted=(_g1_ranked_shifts(rows,rest,y,split,fo[name],fr[name],source,gated,snapshot,peer_keys,given_matrix,deadline,operator=='G1_skip')
                        if operator.startswith('G1') else _g2_ranked_shifts(keys,rows,rest,y,split,fo[name],fr[name],source,gated,peer_keys,tau,deadline,operator=='G2_global'))
                if fitted is None:
                    return
                so,sr,mo,mr,trace=fitted
                for rank in range(2):
                    for gamma in (.35,.65,1.0):
                        if time.monotonic()>=deadline:
                            return
                        accept(so[rank],sr[rank],mo[rank],mr[rank],gamma,operator,tau=tau,rank=rank+1,selected_keys=trace)


# additive eighth stage, all  functions and parameters are immutable.
def _h_ridge_press(xf, xd, residual, lam):
    """Exact intercept-aware ridge PRESS; the intercept is unpenalized."""
    center, mean = xf.mean(0), residual.mean(0)
    xf, xd = xf-center, xd-center
    inverse = np.linalg.solve(xf.T@xf+lam*np.eye(xf.shape[1]), np.eye(xf.shape[1]))
    B = inverse@(xf.T@(residual-mean))
    fitted, dest = mean+xf@B, mean+xd@B
    # Including 1/n requires including the fitted intercept in predictions.
    leverage = 1.0/len(xf)+np.sum((xf@inverse)*xf, axis=1)
    h = np.clip(leverage, 0.0, .90)
    loo = (fitted-h[:, None]*residual)/(1.0-h[:, None])
    return loo, dest, h


def _h_skip_gain(base, y, shift):
    rho = np.clip(base[:, -1], 1e-9, 1-1e-9)
    updated = 1/(1+np.exp(-(np.log(rho/(1-rho))+shift)))
    new = _residual_skip_project(base, np.ones(len(base), bool), updated)
    scale, row = 1-.001*base.shape[1], np.arange(len(y))
    return np.log(.001+scale*new[row, y])-np.log(.001+scale*base[row, y])


def _h2_ranked_shifts(rows, rest, y, split, best, query, source, gated,
                       snapshot, peer_keys, given_matrix, deadline, skip=False):
    """Rank direct/rank-1/rank-2 ridge heads using PRESS rather than self-fit."""
    width = best.shape[1]-int(gated)
    so = [np.zeros((len(rows), 1 if skip else width)) for _ in range(2)]
    sr = [np.zeros((len(rest), 1 if skip else width)) for _ in range(2)]
    mo = [np.zeros(len(rows), bool) for _ in range(2)]
    mr = [np.zeros(len(rest), bool) for _ in range(2)]
    trace = []
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        use = fit if skip else fit & (y < width)
        if use.sum() < 40:
            return None
        rr, yy, base = rows[use], y[use], best[use]
        p = _g_conditional(base, width)
        selected = sorted(((_peer_mi(v[0][rows[fit]], y[fit], best.shape[1]), m)
                           for m,v in peer_keys.items() if m in snapshot), key=lambda v:(-v[0],v[1]))[:8]
        dest = rest if held is None else rows[split == held]
        dest_base = query if held is None else best[split == held]
        pd = _g_conditional(dest_base, width)
        residual = ((yy == width).astype(float)-base[:, -1])[:, None] if skip else np.eye(width)[yy]-p
        ranked = []
        for group in ((9,) if skip else (4,8,9)):
            names = [m for _,m in selected[:min(group,8)]]
            if not names:
                continue
            xf = np.concatenate([snapshot[m][rr] for m in names]+([given_matrix[rr]] if group == 9 else []), axis=1)
            xd = np.concatenate([snapshot[m][dest] for m in names]+([given_matrix[dest]] if group == 9 else []), axis=1)
            for lam in (25.0,80.0,200.0):
                if time.monotonic() >= deadline:
                    return None
                sf, sd, h = _h_ridge_press(xf, xd, residual, lam)
                projections = [('direct',None)]
                if not skip:
                    right = np.linalg.svd(sf, full_matrices=False)[2]
                    projections += [('rank'+str(rank),right[:rank]) for rank in (1,2) if rank <= width]
                for label,vectors in projections:
                    a = sf.copy() if vectors is None else (sf@vectors.T)@vectors
                    b = sd.copy() if vectors is None else (sd@vectors.T)@vectors
                    if skip:
                        rho = np.clip(base[:, -1],1e-6,1-1e-6)
                        rd = np.clip(dest_base[:, -1],1e-6,1-1e-6)
                        a = np.clip(a[:,0]/np.maximum(rho*(1-rho),.04),-.60,.60)
                        b = np.clip(b[:,0]/np.maximum(rd*(1-rd),.04),-.60,.60)
                        observed, variance = float(a@residual[:,0]), float(np.sum(rho*(1-rho)*a*a))
                    else:
                        a,b = np.clip(a,-.60,.60),np.clip(b,-.60,.60)
                        a -= (p*a).sum(1,keepdims=True)
                        b -= (pd*b).sum(1,keepdims=True)
                        observed,variance = float(a[np.arange(len(yy)),yy].sum()),float(np.sum(p*a*a))
                    for tau in (15.0,40.0):
                        theta = float(np.clip(observed/(variance+tau),0.0,1.0))
                        gain = _h_skip_gain(base,yy,theta*a) if skip else _g_fit_gain(base,yy,theta*a,gated)
                        score = float(gain.sum()/max(int(fit.sum()),1))
                        if score > 0:
                            offsets = (theta*b)[:,None] if skip else theta*b
                            ranked.append((score,'top'+str(group)+'_'+label,lam,tau,offsets,theta,float(h.max())))
        ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]))
        chosen = []
        for rank,(score,label,lam,tau,offsets,theta,hmax) in enumerate(ranked[:2]):
            if held is None:
                sr[rank],mr[rank] = offsets,np.ones(len(rest),bool)
            else:
                v = split == held
                so[rank][v],mo[rank][v] = offsets,True
            chosen.append(dict(features=label,lam=lam,tau=tau,theta=theta,loo_gain=score,leverage_max=hmax))
        trace.append(dict(held=None if held is None else int(held),peers=[m for _,m in selected],selected=chosen))
    return so,sr,mo,mr,trace


def _h_expert_basis(best, expert, gated):
    width = best.shape[1]-int(gated)
    p,q = _g_conditional(best,width),_g_conditional(expert,width)
    d = np.clip(np.log(q+1e-3)-np.log(p+1e-3),-1.5,1.5)
    return d-(p*d).sum(1,keepdims=True)


def _h3_ranked_shifts(keys, rows, rest, y, split, best, query, source,
                       gated, expert_oof, expert_query, deadline, mode):
    """Geometric expert pooling; fit-label routing, symmetric OOF/Query bases."""
    width = best.shape[1]-int(gated)
    skip = mode == 'H3_skip'
    so = [np.zeros((len(rows),1 if skip else width)) for _ in range(2)]
    sr = [np.zeros((len(rest),1 if skip else width)) for _ in range(2)]
    mo = [np.zeros(len(rows),bool) for _ in range(2)]
    mr = [np.zeros(len(rest),bool) for _ in range(2)]
    arms = []
    for m,(qo,qr) in enumerate(zip(expert_oof,expert_query)):
        if qo.shape != best.shape or qr.shape != query.shape or not np.isfinite(qo).all() or not np.isfinite(qr).all():
            continue
        if max(float(np.max(abs(qo-best))),float(np.max(abs(qr-query))) if qr.size else 0.0) < 1e-6:
            continue
        if skip:
            rho, rhq = np.clip(best[:,-1],1e-6,1-1e-6),np.clip(query[:,-1],1e-6,1-1e-6)
            qo,qr = np.clip(qo[:,-1],1e-6,1-1e-6),np.clip(qr[:,-1],1e-6,1-1e-6)
            u = np.clip(np.log(qo/(1-qo))-np.log(rho/(1-rho)),-1.5,1.5)[:,None]
            uq = np.clip(np.log(qr/(1-qr))-np.log(rhq/(1-rhq)),-1.5,1.5)[:,None]
        else:
            u,uq = _h_expert_basis(best,qo,gated),_h_expert_basis(query,qr,gated)
        arms.append((m,u,uq))
    trace = []
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        use = fit if skip else fit & (y < width)
        if use.sum() < 40:
            return None
        yy,base,rr = y[use],best[use],rows[use]
        dest = rest if held is None else rows[split == held]
        p = _g_conditional(base,width)
        score_context=_s136_score_context(base,yy,gated)
        ranked,fit_arms,dest_arms = [],[],[]
        for m,u,uq in arms:
            if time.monotonic() >= deadline:
                return None
            a,b = u[use],uq if held is None else u[split == held]
            if skip:
                rho = np.clip(base[:,-1],1e-6,1-1e-6)
                observed = a[:,0]*((yy==width)-rho)
                variance = rho*(1-rho)*a[:,0]**2
            else:
                observed = a[np.arange(len(yy)),yy]
                variance = (p*a*a).sum(1)
            fit_arms.append(a);dest_arms.append(b)
            routes = [('global',None,False)] if mode=='H3_global' else keys(fit)
            if skip:
                routes = [('global',None,False)]+routes
            for label,key,_ in routes:
                if time.monotonic() >= deadline:
                    return None
                codes = np.zeros(len(yy),int) if key is None else key[rr]
                dcodes = np.zeros(len(dest),int) if key is None else key[dest]
                cells = 1 if key is None else int(key.max())+1
                count = np.bincount(codes,minlength=cells)
                # Skip routing requires 40 answered fit rows, too.
                na = np.bincount(codes,weights=(yy<width).astype(float),minlength=cells)
                support = (count>=40)&(na>=40)
                if not support.any():
                    continue
                obs = np.bincount(codes,weights=observed,minlength=cells)
                var = np.bincount(codes,weights=variance,minlength=cells)
                for tau in (8.0,20.0):
                    theta = np.clip(obs/(var+tau),-.45,.45)
                    shift = theta[codes,None]*a
                    gain = _h_skip_gain(base,yy,shift[:,0]) if skip else None
                    score = (float(gain[support[codes]].sum()/max(int(fit.sum()),1)) if skip else
                             _s136_geodesic_score(base,yy,a,theta[codes],support[codes],max(int(fit.sum()),1),gated,score_context))
                    if score > 0:
                        ranked.append((score,label,m,tau,theta[dcodes,None]*b,support[dcodes]))
                        ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]))
                        del ranked[2:]
        if mode=='H3_global' and len(fit_arms)>1:
            # Row Fisher matrices are rank K-1, not rank one. Exact deletion
            # solves (G-V_i) theta_-i = b-g_i, including cross-arm covariance.
            U = np.stack(fit_arms,axis=2)
            UD = np.stack(dest_arms,axis=2)
            covariance = np.einsum('nk,nka,nkb->nab',p,U,U,optimize=True)
            gradient = U[np.arange(len(yy)),yy,:]
            for tau in (8.0,20.0):
                if time.monotonic() >= deadline:
                    return None
                gram = covariance.sum(0)+tau*np.eye(len(fit_arms))
                rhs = gradient.sum(0)
                theta = np.clip(np.linalg.solve(gram,rhs),-.45,.45)
                loo = np.empty_like(gradient)
                # Bound temporary memory to 256 small M x M systems.
                for begin in range(0,len(yy),256):
                    if time.monotonic() >= deadline:
                        return None
                    end = min(begin+256,len(yy))
                    loo[begin:end] = np.clip(_s136_delete_fisher(gram,rhs,covariance[begin:end],gradient[begin:end],U[begin:end],p[begin:end]),-.45,.45)
                a = np.einsum('nka,na->nk',U,loo)
                b = np.einsum('nka,a->nk',UD,theta)
                # One conservative LOO Fisher rescaling is shared by all arms.
                scale = float(np.clip(a[np.arange(len(yy)),yy].sum()/((p*a*a).sum()+tau),0,1))
                score = float(_g_fit_gain(base,yy,scale*a,gated).sum()/max(int(fit.sum()),1))
                if score>0:
                    ranked.append((score,'joint_loo',-1,tau,scale*b,np.ones(len(dest),bool)))
                    ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]))
                    del ranked[2:]
        ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]))
        chosen = []
        for rank,(score,label,m,tau,offsets,support) in enumerate(ranked[:2]):
            if held is None:
                sr[rank],mr[rank] = offsets,support
            else:
                v = split == held
                so[rank][v],mo[rank][v] = offsets,support
            chosen.append(dict(key=label,arm=m,tau=tau,fit_gain=score))
        trace.append(dict(held=None if held is None else int(held),arms=len(arms),selected=chosen))
    return so,sr,mo,mr,trace


def _loo_specialist_raking(items,work,final,fo,fr,X,given,levels,
                             fold,nll_fn,deadline,peers,base_snapshot):
    """Eighth orthogonal stage after the byte-preserved  prefix."""
    if time.monotonic() >= deadline:
        return
    features = [f for f in range(len(given)) if 1<int(levels[f])<=128]
    given_matrix = np.concatenate([(X[:,f,None]==np.arange(int(levels[f]))).astype(float)
                                   for f in features],axis=1) if features else np.empty((len(X),0))
    shared = []
    for ix,a in enumerate(features):
        shared.append((given[a],X[:,a],False))
        for b in features[ix+1:]:
            _,key = np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((given[a]+':'+given[b],key,True))
    # Freeze parent mode keys as well as peer probabilities across the stage.
    key_fo,key_fr = dict(fo),dict(fr)
    # Complete the cheap dimension-aware sweep before any ridge/expert heads.
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{}) if np.isfinite(deadline) else _s136_order(work,items,fo,X,given,levels,deadline))
    for _s136_oi,operator in enumerate( ('H1','H2','H2_skip','H3_global','H3_cell','H3_skip')):
        _s136_stop=deadline if not np.isfinite(deadline) else time.monotonic()+max(0.,deadline-time.monotonic())/(6-_s136_oi)
        for name,job in ordered:
            if time.monotonic() >= _s136_stop:
                break
            rows,rest = np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
            y,split = job['y'][rows],fold[rows]
            gated = bool(items[name].get('gate'))
            width = job['K']-int(gated)
            if width<1 or len(np.unique(split))!=5:
                continue
            source = _wave_two_source(name,items,X,given,rows)
            hidden_source = _wave_two_source(name,items,X,given,rest)
            if source is None or hidden_source is None:
                continue
            if 'wave' in given:
                wave_one = next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if wave_one is not None:
                    source &= X[rows,given.index('wave')] != wave_one
                    hidden_source &= X[rest,given.index('wave')] != wave_one
            if source.sum()<67:
                continue
            peer_keys = {m:v for m,v in peers.items() if m!=name}
            snapshot = {m:v for m,v in base_snapshot.items() if m!=name}
            static = None
            key_cache = {}
            def keys(fit):
                nonlocal static
                if static is None:
                    static = _residual_keys(name,items,work,key_fo,key_fr,X,given,levels,shared)
                token = fit.tobytes()
                if token not in key_cache:
                    selected = sorted(((_peer_mi(v[0][rows[fit]],y[fit],job['K']),m) for m,v in peer_keys.items()),key=lambda v:(-v[0],v[1]))[:4]
                    key_cache[token] = static+[('peer_mode:'+m,peer_keys[m][0],True) for _,m in selected]
                return key_cache[token]
            def accept(so,sr,mo,mr,gamma,operator,**metadata):
                if time.monotonic() >= _s136_stop:
                    return
                before,before_r = fo[name],fr[name]
                active,hidden = mo&source,mr&hidden_source
                skip = operator.endswith('_skip')
                if gated:
                    active &= 1-before[:,-1]>=.20
                    hidden &= 1-before_r[:,-1]>=.20
                if skip:
                    active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                    hidden &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                    rho = np.clip(before[active,-1],1e-9,1-1e-9)
                    oo = _residual_skip_project(before,active,1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*so[active,0]))))
                else:
                    oo = _ef_project(before,so,active,gamma,gated)
                kappa = {'H1':3.9,'H2':2.9,'H2_skip':2.4,'H3_global':3.5,'H3_cell':5.4,'H3_skip':2.6}[operator]
                if not _ef_dominates(oo,before,y,split,active,all_rows=operator in ('H1','H2','H3_global'),kappa=kappa):
                    return
                if skip:
                    rho = np.clip(before_r[hidden,-1],1e-9,1-1e-9)
                    rr = _residual_skip_project(before_r,hidden,1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*sr[hidden,0]))))
                else:
                    rr = _ef_project(before_r,sr,hidden,gamma,gated)
                assert np.array_equal(oo[~active],before[~active]) and np.array_equal(rr[~hidden],before_r[~hidden])
                if gated and not skip:
                    assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(rr[:,-1],before_r[:,-1])
                if time.monotonic() >= _s136_stop:
                    return
                fo[name],fr[name],final[name][rest] = oo,rr,rr
                row = np.arange(len(y))
                gain = np.log(.001+(1-.001*job['K'])*oo[row,y])-np.log(.001+(1-.001*job['K'])*before[row,y])
                fg = np.array([gain[split==f].mean() for f in np.unique(split)])
                se = max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
                job.setdefault('h_raking',[]).append(dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),active_gain=float(gain[active].mean()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata))
            if (operator=='H1' and width<3) or (not operator.endswith('_skip') and width<2):
                continue
            if operator.endswith('_skip') and (not gated or any(p in given for p in _declared_parents(name,items))):
                continue
            for tau in ((6.0,16.0) if operator=='H1' else (None,)):
                if time.monotonic() >= _s136_stop:
                    return
                if operator=='H1':
                    fitted = _g2_ranked_shifts(keys,rows,rest,y,split,fo[name],fr[name],source,gated,peer_keys,tau,_s136_stop,global_step=True)
                elif operator.startswith('H2'):
                    fitted = _h2_ranked_shifts(rows,rest,y,split,fo[name],fr[name],source,gated,snapshot,peer_keys,given_matrix,_s136_stop,skip=operator.endswith('_skip'))
                else:
                    fitted = _h3_ranked_shifts(keys,rows,rest,y,split,fo[name],fr[name],source,gated,job['oof'],[a[rest] for a in job['all']],_s136_stop,operator)
                if fitted is None:
                    return
                so,sr,mo,mr,trace = fitted
                for rank in range(2):
                    for gamma in (.35,.65,1.0):
                        if time.monotonic() >= _s136_stop:
                            return
                        accept(so[rank],sr[rank],mo[rank],mr[rank],gamma,operator,tau=tau,rank=rank+1,selected_keys=trace)


# additive ninth stage, complete  code and parameters preserved.
def _i_residual_peers(snapshot, rr, residual, count):
    """No peer labels; centered prediction covariance with fit target residuals."""
    scores=[]
    for name,prob in snapshot.items():
        x=prob[rr];x=x-x.mean(0)
        score=float(np.sum((x.T@residual)**2)/(np.sqrt(np.sum(x*x))*len(rr)+1e-6))
        scores.append((score,name))
    return [m for _,m in sorted(scores,key=lambda v:(-v[0],v[1]))[:count]]


def _i2_ranked_shifts(rows, rest, y, split, best, query, source, gated,
                       snapshot, peer_keys, given_matrix, deadline, skip=False):
    """Fit-only residual covariance selects peers before bilinear/PRESS heads."""
    width = best.shape[1]-int(gated)
    so = [np.zeros((len(rows), 1 if skip else width)) for _ in range(2)]
    sr = [np.zeros((len(rest), 1 if skip else width)) for _ in range(2)]
    mo = [np.zeros(len(rows), bool) for _ in range(2)]
    mr = [np.zeros(len(rest), bool) for _ in range(2)]
    trace = []
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit = source & (split != held) if held is not None else source.copy()
        use = fit if skip else fit & (y < width)
        if use.sum() < 40:
            return None
        rr, yy, base = rows[use], y[use], best[use]
        p = _g_conditional(base, width)
        dest = rest if held is None else rows[split == held]
        dest_base = query if held is None else best[split == held]
        pd = _g_conditional(dest_base, width)
        residual = ((yy == width).astype(float)-base[:, -1])[:, None] if skip else np.eye(width)[yy]-p
        ranked = []
        selected = _i_residual_peers(snapshot, rr, residual, 4 if skip else 6)
        mi = {m for _,m in sorted(((_peer_mi(v[0][rows[fit]],y[fit],best.shape[1]),m)
                                  for m,v in peer_keys.items() if m in snapshot),key=lambda v:(-v[0],v[1]))[:4]}
        groups = [(m,[m],False) for m in selected] if not skip else []
        if skip:
            groups.append(('skip_top4',selected[:4],False))
        else:
            for size in (3,6):
                names=selected[:size]
                if len(names)>=size and not set(names).issubset(mi):
                    groups.append(('residual_top'+str(size),names,False))
                    groups.append(('entropy_top'+str(size),names,True))
        for group,names,entropy in groups:
            if not names:
                continue
            xf = np.concatenate([snapshot[m][rr] for m in names],axis=1)
            xd = np.concatenate([snapshot[m][dest] for m in names],axis=1)
            if entropy:
                xf = np.column_stack([xf]+[-np.sum(snapshot[m][rr]*np.log(np.maximum(snapshot[m][rr],1e-12)),axis=1) for m in names])
                xd = np.column_stack([xd]+[-np.sum(snapshot[m][dest]*np.log(np.maximum(snapshot[m][dest],1e-12)),axis=1) for m in names])
            if time.monotonic() >= deadline:
                return None
            if not skip:
                center=xf.mean(0); xc=xf-center
                left,_,right=np.linalg.svd(xc.T@residual/len(rr),full_matrices=False)
                w,u=left[:,0],right[0];u=u/max(float(abs(u).max()),1e-12)
                scale=float((xc@w).std())+1e-6
                z=np.clip(xc@w/scale,-2,2);zd=np.clip((xd-center)@w/scale,-2,2)
                mu=p@u;var=(p*(u[None]-mu[:,None])**2).sum(1)
                for tau in (15.0,40.0):
                    theta=float(np.clip(np.sum(z*(u[yy]-mu))/(np.sum(z*z*var)+tau),-.50,.50))
                    score=float(_g_fit_gain(base,yy,theta*z[:,None]*u,gated).sum()/max(int(fit.sum()),1))
                    if score>0:
                        ranked.append((score,group+'_bilinear',0.0,tau,theta*zd[:,None]*u,theta,0.0))
            for lam in (30.0,100.0):
                if time.monotonic() >= deadline:
                    return None
                sf, sd, h = _h_ridge_press(xf, xd, residual, lam)
                projections = [('direct',None)]
                if not skip:
                    right = np.linalg.svd(sf, full_matrices=False)[2]
                    projections += [('rank'+str(rank),right[:rank]) for rank in (1,2) if rank <= width]
                for label,vectors in projections:
                    a = sf.copy() if vectors is None else (sf@vectors.T)@vectors
                    b = sd.copy() if vectors is None else (sd@vectors.T)@vectors
                    if skip:
                        rho = np.clip(base[:, -1],1e-6,1-1e-6)
                        rd = np.clip(dest_base[:, -1],1e-6,1-1e-6)
                        a = np.clip(a[:,0]/np.maximum(rho*(1-rho),.04),-.60,.60)
                        b = np.clip(b[:,0]/np.maximum(rd*(1-rd),.04),-.60,.60)
                        observed, variance = float(a@residual[:,0]), float(np.sum(rho*(1-rho)*a*a))
                    else:
                        a,b = np.clip(a,-.60,.60),np.clip(b,-.60,.60)
                        a -= (p*a).sum(1,keepdims=True)
                        b -= (pd*b).sum(1,keepdims=True)
                        observed,variance = float(a[np.arange(len(yy)),yy].sum()),float(np.sum(p*a*a))
                    for tau in (15.0,40.0):
                        theta = float(np.clip(observed/(variance+tau),0.0,1.0))
                        gain = _h_skip_gain(base,yy,theta*a) if skip else _g_fit_gain(base,yy,theta*a,gated)
                        score = float(gain.sum()/max(int(fit.sum()),1))
                        if score > 0:
                            offsets = (theta*b)[:,None] if skip else theta*b
                            ranked.append((score,group+'_'+label,lam,tau,offsets,theta,float(h.max())))
        ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]))
        chosen = []
        for rank,(score,label,lam,tau,offsets,theta,hmax) in enumerate(ranked[:2]):
            if held is None:
                sr[rank],mr[rank] = offsets,np.ones(len(rest),bool)
            else:
                v = split == held
                so[rank][v],mo[rank][v] = offsets,True
            chosen.append(dict(features=label,lam=lam,tau=tau,theta=theta,loo_gain=score,leverage_max=hmax))
        trace.append(dict(held=None if held is None else int(held),peers=selected,selected=chosen))
    return so,sr,mo,mr,trace


def _i_triad_keys(X, given, levels, rows, y, best, fit, gated, snapshot, peer_keys):
    """Rank pairs on fit residual divergence; form at most sixteen unique triads."""
    width=best.shape[1]-int(gated)
    use=fit & (y<width);rr=rows[use];yy=y[use]
    if len(rr)<45:
        return []
    p=_g_conditional(best[use],width);residual=np.eye(width)[yy]-p
    features=[i for i in range(len(given)) if 1<int(levels[i])<=128]
    def cross(a,b):
        return np.unique(a.astype(np.int64)*(int(b.max())+1)+b,return_inverse=True)[1]
    def divergence(key):
        codes=key[rr];cells=int(key.max())+1
        n=np.bincount(codes,minlength=cells)
        r=np.column_stack([np.bincount(codes,weights=residual[:,k],minlength=cells) for k in range(width)])
        return float(np.sum(r[n>=25]**2/(n[n>=25,None]+24)))
    singles=sorted([(divergence(X[:,i]),i) for i in features],key=lambda v:(-v[0],v[1]))[:4]
    pairs=[]
    for pos,a in enumerate(features):
        for b in features[pos+1:]:
            key=cross(X[:,a],X[:,b]);pairs.append((divergence(key),a,b,key))
    pairs.sort(key=lambda v:(-v[0],v[1],v[2]));pairs=pairs[:4]
    peers=_i_residual_peers(snapshot,rr,residual,2)
    thirds=[(given[i],X[:,i],i) for _,i in singles]+[('peer_mode:'+m,peer_keys[m][0],None) for m in peers if m in peer_keys]
    result=[];seen=set()
    # Round-robin extensions retain both GIVEN and residual peer candidates.
    for ix in range(len(thirds)):
        for _,a,b,key in pairs:
            label,third,i=thirds[ix]
            if i in (a,b):
                continue
            token=tuple(sorted((given[a],given[b],label)))
            if token in seen:
                continue
            seen.add(token);result.append((':'.join(token),cross(key,third),True))
            if len(result)==16:
                return result
    return result


def _i3_ranked_shifts(keys, rows, rest, y, split, best, query, source, gated, deadline, mode):
    """Third-order EB ratios and scalar Fisher steps, fit-only selected strata."""
    width=best.shape[1]-int(gated);skip=mode=='I3_skip'
    so=[np.zeros((len(rows),1 if skip else width)) for _ in range(2)]
    sr=[np.zeros((len(rest),1 if skip else width)) for _ in range(2)]
    mo=[np.zeros(len(rows),bool) for _ in range(2)];mr=[np.zeros(len(rest),bool) for _ in range(2)]
    trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:
            return None
        fit=source & (split!=held) if held is not None else source.copy()
        use=fit if skip else fit & (y<width)
        rr,yy,base=rows[use],y[use],best[use]
        if len(rr)<45:
            return None
        p=_g_conditional(base,width)
        dest=rest if held is None else rows[split==held]
        db=query if held is None else best[split==held];pd=_g_conditional(db,width)
        prior=(np.bincount(yy[yy<width],minlength=width)+1.0);prior/=prior.sum()
        ranked=[]
        for label,key,_ in keys(fit):
            if time.monotonic()>=deadline:
                return None
            cells=int(key.max())+1;codes=key[rr];dc=key[dest]
            n=np.bincount(key[rows[fit]],minlength=cells)
            na=np.bincount(key[rows[fit & (y<width)]],minlength=cells)
            support=(na>=45)&((na>=.25*n) if gated else True)
            proposals=[]
            if skip:
                rho=np.clip(base[:,-1],1e-6,1-1e-6)
                obs=np.bincount(codes,weights=(yy==width).astype(float),minlength=cells)
                expected=np.bincount(codes,weights=rho,minlength=cells)
                rate=obs/np.maximum(n,1)
                support=(n>=50)&(rate>.03)&(rate<.97)
                for tau in (14.0,24.0):
                    offset=np.clip(np.log((obs+tau)/(n-obs+tau))-np.log((expected+tau)/(n-expected+tau)),-.45,.45)
                    proposals.append(('skip',tau,offset[codes,None],offset[dc,None]))
            elif mode=='I3_mult':
                obs=np.column_stack([np.bincount(codes,weights=(yy==k).astype(float),minlength=cells) for k in range(width)])
                expected=np.column_stack([np.bincount(codes,weights=p[:,k],minlength=cells) for k in range(width)])
                for tau in (14.0,24.0):
                    offset=np.clip(np.log((obs+tau*prior)/(expected+tau*prior)),-.45,.45)
                    proposals.append(('ratio',tau,offset[codes],offset[dc]))
            else:
                residual=np.eye(width)[yy]-p
                directions=[('ordinal',np.linspace(-1,1,width)),('information',np.log(prior))]
                directions.append(('svd',np.linalg.svd(residual,full_matrices=False)[2][0]))
                if width==2:
                    value=np.clip(residual[:,1]/np.maximum(p[:,1]*(1-p[:,1]),.04),-.6,.6)
                    obs=np.bincount(codes,weights=residual[:,1],minlength=cells)
                    var=np.bincount(codes,weights=p[:,1]*(1-p[:,1]),minlength=cells)
                    for tau in (8.0,18.0):
                        theta=np.clip(obs/(var+tau),-.45,.45)
                        a=np.column_stack((np.zeros(len(rr)),theta[codes]));b=np.column_stack((np.zeros(len(dest)),theta[dc]))
                        proposals.append(('binary_fisher',tau,a,b))
                else:
                    for basis,u in directions:
                        u=(u-u.mean())/max(float(abs(u-u.mean()).max()),1e-12)
                        mu=p@u;var=(p*(u[None]-mu[:,None])**2).sum(1)
                        obs=np.bincount(codes,weights=u[yy]-mu,minlength=cells)
                        variance=np.bincount(codes,weights=var,minlength=cells)
                        for tau in (8.0,18.0):
                            theta=np.clip(obs/(variance+tau),-.45,.45)
                            proposals.append((basis,tau,theta[codes,None]*u,theta[dc,None]*u))
            if not support.any():
                continue
            for basis,tau,a,b in proposals:
                gain=_h_skip_gain(base,yy,a[:,0]) if skip else _g_fit_gain(base,yy,a,gated)
                score=float(gain[support[codes]].sum()/max(int(fit.sum()),1))
                if score>0:
                    ranked.append((score,label,basis,tau,b,support[dc]))
                    ranked.sort(key=lambda v:(-v[0],v[1],v[2],v[3]));del ranked[2:]
        chosen=[]
        for rank,(score,label,basis,tau,offsets,support) in enumerate(ranked):
            if held is None:
                sr[rank],mr[rank]=offsets,support
            else:
                v=split==held;so[rank][v],mo[rank][v]=offsets,support
            chosen.append(dict(key=label,basis=basis,tau=tau,fit_gain=score))
        trace.append(dict(held=None if held is None else int(held),selected=chosen))
    return so,sr,mo,mr,trace


def _residual_peer_and_triad_raking(items,work,final,fo,fr,X,given,levels,
                             fold,nll_fn,deadline,peers,base_snapshot):
    """Ninth orthogonal stage after the byte-preserved  prefix."""
    if time.monotonic() >= deadline:
        return
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{}) if np.isfinite(deadline) else _s136_order(work,items,fo,X,given,levels,deadline))
    for _s136_oi,operator in enumerate( ('I1','I2','I2_skip','I3_mult','I3_fisher','I3_skip')):
        _s136_stop=deadline if not np.isfinite(deadline) else time.monotonic()+max(0.,deadline-time.monotonic())/(6-_s136_oi)
        for name,job in ordered:
            if time.monotonic() >= _s136_stop:
                break
            rows,rest = np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
            y,split = job['y'][rows],fold[rows]
            gated = bool(items[name].get('gate'))
            width = job['K']-int(gated)
            if width<1 or len(np.unique(split))!=5:
                continue
            source = _wave_two_source(name,items,X,given,rows)
            hidden_source = _wave_two_source(name,items,X,given,rest)
            if source is None or hidden_source is None:
                continue
            if 'wave' in given:
                wave_one = next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if wave_one is not None:
                    source &= X[rows,given.index('wave')] != wave_one
                    hidden_source &= X[rest,given.index('wave')] != wave_one
            if source.sum()<67:
                continue
            peer_keys = {m:v for m,v in peers.items() if m!=name}
            snapshot = {m:v for m,v in base_snapshot.items() if m!=name}
            key_cache={}
            def keys(fit):
                token=fit.tobytes()
                if token not in key_cache:
                    key_cache[token]=_i_triad_keys(X,given,levels,rows,y,fo[name],fit,gated,snapshot,peer_keys)
                return key_cache[token]
            def accept(so,sr,mo,mr,gamma,operator,**metadata):
                if time.monotonic() >= _s136_stop:
                    return
                before,before_r = fo[name],fr[name]
                active,hidden = mo&source,mr&hidden_source
                skip = operator.endswith('_skip')
                if gated:
                    active &= 1-before[:,-1]>=.20
                    hidden &= 1-before_r[:,-1]>=.20
                if skip:
                    active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                    hidden &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                    rho = np.clip(before[active,-1],1e-9,1-1e-9)
                    oo = _residual_skip_project(before,active,1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*so[active,0]))))
                else:
                    oo = _ef_project(before,so,active,gamma,gated)
                kappa = {'I1':3.4,'I2':2.9,'I2_skip':2.4,'I3_mult':2.4,'I3_fisher':2.4,'I3_skip':2.4}[operator]
                if not _ef_dominates(oo,before,y,split,active,all_rows=operator in ('I1','I2','I2_skip'),kappa=kappa):
                    return
                if skip:
                    rho = np.clip(before_r[hidden,-1],1e-9,1-1e-9)
                    rr = _residual_skip_project(before_r,hidden,1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*sr[hidden,0]))))
                else:
                    rr = _ef_project(before_r,sr,hidden,gamma,gated)
                assert np.array_equal(oo[~active],before[~active]) and np.array_equal(rr[~hidden],before_r[~hidden])
                if gated and not skip:
                    assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(rr[:,-1],before_r[:,-1])
                if time.monotonic() >= _s136_stop:
                    return
                fo[name],fr[name],final[name][rest] = oo,rr,rr
                row = np.arange(len(y))
                gain = np.log(.001+(1-.001*job['K'])*oo[row,y])-np.log(.001+(1-.001*job['K'])*before[row,y])
                fg = np.array([gain[split==f].mean() for f in np.unique(split)])
                se = max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
                job.setdefault('i_raking',[]).append(dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),active_gain=float(gain[active].mean()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata))
            if (operator=='I1' and (width<3 or np.sort(np.bincount(y[source & (y<width)],minlength=width))[-3]<80)) or (not operator.endswith('_skip') and width<2):
                continue
            if operator.endswith('_skip') and (not gated or any(p in given for p in _declared_parents(name,items))):
                continue
            for tau in ((6.0,16.0) if operator=='I1' else (None,)):
                if time.monotonic() >= _s136_stop:
                    return
                if operator=='I1':
                    fitted = _g2_ranked_shifts(keys,rows,rest,y,split,fo[name],fr[name],source,gated,peer_keys,tau,_s136_stop,global_step=True)
                elif operator.startswith('I2'):
                    fitted = _i2_ranked_shifts(rows,rest,y,split,fo[name],fr[name],source,gated,snapshot,peer_keys,None,_s136_stop,skip=operator.endswith('_skip'))
                else:
                    fitted = _i3_ranked_shifts(keys,rows,rest,y,split,fo[name],fr[name],source,gated,_s136_stop,operator)
                if fitted is None:
                    return
                so,sr,mo,mr,trace = fitted
                for rank in range(2):
                    for gamma in (.35,.65,1.0):
                        if time.monotonic() >= _s136_stop:
                            return
                        accept(so[rank],sr[rank],mo[rank],mr[rank],gamma,operator,tau=tau,rank=rank+1,selected_keys=trace)


# additive tenth stage; all  bytes outside this block are preserved.
def _j_consensus(fits):
    """A single complete specification, top two in >=4 folds and full fit."""
    if len(fits) != 6:
        return []
    tops = [sorted(v, key=lambda k: (-v[k][0], repr(k)))[:2] for v in fits]
    candidates = []
    for spec in tops[-1]:
        votes = sum(spec in top for top in tops[:-1])
        if votes >= 4 and all(spec in fit for fit in fits):
            candidates.append((sum(f[spec][0] for f in fits[:-1])/5, spec, votes))
    return sorted(candidates, key=lambda t: (-t[0], repr(t[1])))


def _j_collect(fits, split, rows, rest):
    result = []
    for score, spec, votes in _j_consensus(fits):
        shape = fits[-1][spec][1].shape[1:]
        oo = np.zeros((len(rows),)+shape); mask = np.zeros(len(rows), bool)
        for held, fit in zip(np.unique(split), fits[:-1]):
            use = split == held
            oo[use], mask[use] = fit[spec][1], fit[spec][2]
        result.append((oo, fits[-1][spec][1], mask, fits[-1][spec][2],
                       dict(spec=list(spec), consensus_folds=votes, fit_mean_gain=score,
                            fit_scores=[float(f[spec][0]) for f in fits])))
    return result


def _j_fold_context(rows, rest, y, split, best, query, source, gated, held, skip):
    width = best.shape[1]-int(gated)
    fit = source & (split != held) if held is not None else source.copy()
    use = fit if skip else fit & (y < width)
    dest = rest if held is None else rows[split == held]
    db = query if held is None else best[split == held]
    base = best[use]; yy = y[use]
    p = _g_conditional(base, width); pd = _g_conditional(db, width)
    residual = ((yy == width).astype(float)-base[:, -1])[:, None] if skip else np.eye(width)[yy]-p
    return fit, use, rows[use], yy, base, p, dest, db, pd, residual


def _j_peer_shifts(rows, rest, y, split, best, query, source, gated,
                   snapshot, X, given, levels, deadline, interaction=False, skip=False):
    """Fixed peer identities; centered Kronecker blocks and exact ridge PRESS."""
    width = best.shape[1]-int(gated); fits=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic() >= deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual = _j_fold_context(rows,rest,y,split,best,query,source,gated,held,skip)
        if len(rr)<45:
            return []
        selected = _i_residual_peers(snapshot,rr,residual,4 if interaction else 6)
        groups=[]
        if interaction:
            demographic=[]
            for f in range(len(given)):
                if not 2<=int(levels[f])<=16:
                    continue
                z=(X[rr,f,None]==np.arange(int(levels[f]))).astype(float)
                z-=z.mean(0)
                demographic.append((float(np.sum((z.T@residual)**2)/(np.linalg.norm(z)+1e-6)),f))
            dem=[f for _,f in sorted(demographic,key=lambda t:(-t[0],t[1]))[:3]]
            # Fit-only lists form named specifications; identity is voted across fits.
            for count in (3,4):
                names=sorted(selected[:count])
                if len(names)<count:
                    continue
                for ng in (2,3):
                    fs=sorted(dem[:ng]); xf=[];xd=[]
                    def peer(m):
                        a=snapshot[m][rr];b=snapshot[m][dest];mean=a.mean(0)
                        a,b=a-mean,b-mean
                        # Probabilities have one redundant column. Fit-only PCA
                        # caps each peer at two dimensions before outer products.
                        v=np.linalg.svd(a,full_matrices=False)[2][:min(2,a.shape[1]-1)]
                        return a@v.T,b@v.T
                    pp={m:peer(m) for m in names}
                    for m in names:
                        a,b=pp[m]
                        for f in fs:
                            u=(X[rr,f,None]==np.arange(int(levels[f]))).astype(float)[:,:-1]
                            v=(X[dest,f,None]==np.arange(int(levels[f]))).astype(float)[:,:-1]
                            mean=u.mean(0);u,v=u-mean,v-mean
                            xf.append((u[:,:,None]*a[:,None,:]).reshape(len(rr),-1))
                            xd.append((v[:,:,None]*b[:,None,:]).reshape(len(dest),-1))
                    for i,m in enumerate(names):
                        for other in names[:i]:
                            a,b=pp[m];u,v=pp[other]
                            xf.append((a[:,:,None]*u[:,None,:]).reshape(len(rr),-1))
                            xd.append((b[:,:,None]*v[:,None,:]).reshape(len(dest),-1))
                    if not xf:
                        continue
                    a=np.concatenate(xf,axis=1);b=np.concatenate(xd,axis=1)
                    # Select at most 96 columns using only fit residual covariance.
                    scale=np.maximum(a.std(0),.03);a,b=a/scale,b/scale
                    keep=np.argsort(-np.sum(((a-a.mean(0)).T@residual)**2,axis=1),kind='stable')[:96]
                    groups.append(('kron:'+','.join(names)+'|'+','.join(given[f] for f in fs),a[:,keep],b[:,keep]))
        else:
            groups=[('peer:'+m,snapshot[m][rr],snapshot[m][dest]) for m in selected]
        ranked={}
        for label,xf,xd in groups:
            if time.monotonic() >= deadline:
                return None
            if not interaction and not skip:
                center=xf.mean(0);xc=xf-center
                left,_,right=np.linalg.svd(xc.T@residual/len(rr),full_matrices=False)
                w,u=left[:,0],right[0];u=u/max(float(abs(u).max()),1e-12)
                sd=float((xc@w).std())+1e-6
                z=np.clip(xc@w/sd,-2,2);zd=np.clip((xd-center)@w/sd,-2,2)
                mu=p@u;variance=(p*(u[None]-mu[:,None])**2).sum(1)
                for tau in (15.0,40.0):
                    theta=float(np.clip(np.sum(z*(u[yy]-mu))/(np.sum(z*z*variance)+tau),-.5,.5))
                    score=float(_g_fit_gain(base,yy,theta*z[:,None]*u,gated).sum()/fit.sum())
                    if score>0:
                        ranked[(label,'bilinear',0.0,tau)]=(score,theta*zd[:,None]*u,np.ones(len(dest),bool))
            for lam in ((40.,120.,300.) if interaction else (30.,100.)):
                if time.monotonic() >= deadline:
                    return None
                sf,sd,h=_h_ridge_press(xf,xd,residual,lam)
                projections=[('direct',None)]
                if not skip:
                    v=np.linalg.svd(sf,full_matrices=False)[2]
                    projections += [('rank'+str(r),v[:r]) for r in (1,2) if r<=min(width,len(v))]
                for projection,v in projections:
                    a,b=(sf.copy(),sd.copy()) if v is None else ((sf@v.T)@v,(sd@v.T)@v)
                    if skip:
                        rho=np.clip(base[:,-1],1e-6,1-1e-6);rd=np.clip(db[:,-1],1e-6,1-1e-6)
                        a=np.clip(a[:,0]/np.maximum(rho*(1-rho),.04),-.6,.6)
                        b=np.clip(b[:,0]/np.maximum(rd*(1-rd),.04),-.6,.6)
                        obs=float(a@residual[:,0]);var=float(np.sum(rho*(1-rho)*a*a))
                    else:
                        a,b=np.clip(a,-.6,.6),np.clip(b,-.6,.6)
                        a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                        obs=float(a[np.arange(len(yy)),yy].sum());var=float(np.sum(p*a*a))
                    for tau in (15.,40.):
                        theta=float(np.clip(obs/(var+tau),0,1))
                        gain=_h_skip_gain(base,yy,theta*a) if skip else _g_fit_gain(base,yy,theta*a,gated)
                        score=float(gain.sum()/fit.sum())
                        if score>0:
                            ranked[(label,projection,lam,tau)]=(score,(theta*b)[:,None] if skip else theta*b,np.ones(len(dest),bool))
        fits.append(ranked)
    return _j_collect(fits,split,rows,rest)


def _j_keys(name, items, work, key_fo, key_fr, X, given, levels,
            rows, y, best, fit, gated, snapshot, peer_keys, skip=False):
    """Fit-residual selected demographics, parent cells and peer quantile bins."""
    width=best.shape[1]-int(gated);use=fit if skip else fit & (y<width);rr=rows[use]
    residual=((y[use]==width).astype(float)-best[use,-1])[:,None] if skip else np.eye(width)[y[use]]-_g_conditional(best[use],width)
    features=[f for f in range(len(given)) if 2<=int(levels[f])<=16]
    def divergence(key):
        c=key[rr];n=np.bincount(c,minlength=int(key.max())+1)
        r=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(residual.shape[1])])
        return float(np.sum(r[n>=45]**2/(n[n>=45,None]+40)))
    selected=sorted(features,key=lambda f:(-divergence(X[:,f]),f))[:4]
    shared=[]
    for i,a in enumerate(selected):
        shared.append((given[a],X[:,a],False))
        for b in selected[:i]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((':'.join(sorted((given[a],given[b]))),key,True))
    keys=_residual_keys(name,items,work,key_fo,key_fr,X,given,levels,shared)
    for m in _i_residual_peers(snapshot,rr,residual,3):
        if m in peer_keys:
            mode=peer_keys[m][0]
            keys.append(('peer_mode:'+m,mode,True))
            for f in selected[:3]:
                _,key=np.unique(mode*int(levels[f])+X[:,f],return_inverse=True)
                keys.append(('peer_mode:'+m+':'+given[f],key,True))
        # Score prediction coordinates fitted without target labels on held rows.
        prob=snapshot[m];x=prob[rr]-prob[rr].mean(0)
        w=np.linalg.svd(x.T@residual,full_matrices=False)[0][:,0]
        score=prob@w;cuts=np.unique(np.quantile(score[rr],[.2,.4,.6,.8]))
        keys.append(('peer_quantile:'+m,np.searchsorted(cuts,score).astype(int),True))
    return keys


def _j_kkt_step(base, direction, y, codes, cells, maximum=.4):
    """Exact one-dimensional floor-NLL convex KKT solution per supported cell."""
    scale=1-.001*base.shape[1];r=np.arange(len(y))
    a=.001+scale*base[r,y];d=scale*direction[r,y]
    lo=np.zeros(cells);hi=np.full(cells,maximum)
    for _ in range(28):
        mid=(lo+hi)/2
        grad=np.bincount(codes,weights=d/np.maximum(a+mid[codes]*d,1e-12),minlength=cells)
        lo=np.where(grad>0,mid,lo);hi=np.where(grad>0,hi,mid)
    return (lo+hi)/2


def _j_simplex_project(base, target, mask, gamma, gated, skip=False):
    out=base.copy()
    if skip:
        rho=(1-gamma)*base[mask,-1]+gamma*target[mask,0]
        return _residual_skip_project(base,mask,rho)
    width=base.shape[1]-int(gated)
    p=_g_conditional(base[mask],width)
    q=(1-gamma)*p+gamma*target[mask,:width]
    q/=np.maximum(q.sum(1,keepdims=True),1e-12)
    out[mask,:width]=q*(1-base[mask,-1,None]) if gated else q
    return out


def _j_simplex_shifts(keys,rows,rest,y,split,best,query,source,gated,deadline,skip=False):
    """Dirichlet deletion q_-i and floor-aware exact convex line minimization."""
    width=best.shape[1]-int(gated);fits=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,skip)
        if len(rr)<45:
            return []
        ranked={}
        target=(yy==width).astype(int) if skip else yy
        classes=2 if skip else width
        counts=np.bincount(target,minlength=classes)+1.
        prior=counts/counts.sum();prior_loo=(counts[None]-np.eye(classes)[target])/(counts.sum()-1)
        for label,key,_ in keys(fit):
            if time.monotonic()>=deadline:
                return None
            codes=key[rr];dc=key[dest];cells=int(key.max())+1
            n=np.bincount(codes,minlength=cells)
            na=np.bincount(key[rows[fit & (y<width)]],minlength=cells)
            support=(na>=45)&(n>=45)
            if not support.any():
                continue
            obs=np.column_stack([np.bincount(codes,weights=(target==k).astype(float),minlength=cells) for k in range(classes)])
            for tau in (24.,64.):
                qloo=(obs[codes]-np.eye(classes)[target]+tau*prior_loo)/(n[codes,None]-1+tau)
                qdest=(obs[dc]+tau*prior)/(n[dc,None]+tau)
                a=qloo[:,1,None] if skip else qloo
                b=qdest[:,1,None] if skip else qdest
                mask=support[codes];dm=support[dc]
                for step in (.08,.15,.25,.40,'kkt'):
                    if step=='kkt':
                        if skip:
                            trial=_residual_skip_project(base,np.ones(len(base),bool),a[:,0]);direction=trial-base
                        else:
                            trial=_j_simplex_project(base,a,np.ones(len(base),bool),1.,gated);direction=trial-base
                        gamma=_j_kkt_step(base,direction,yy,codes,cells)
                        aq=(1-gamma[codes,None])*(base[:,-1,None] if skip else p)+gamma[codes,None]*a
                        bq=(1-gamma[dc,None])*(db[:,-1,None] if skip else pd)+gamma[dc,None]*b
                    else:
                        aq=(1-step)*(base[:,-1,None] if skip else p)+step*a
                        bq=(1-step)*(db[:,-1,None] if skip else pd)+step*b
                    candidate=_j_simplex_project(base,aq,mask,1.,gated,skip)
                    row=np.arange(len(yy));scale=1-.001*best.shape[1]
                    gain=np.log(.001+scale*candidate[row,yy])-np.log(.001+scale*base[row,yy])
                    score=float(gain.sum()/fit.sum())
                    if score>0:
                        ranked[(label,'simplex',tau,step)]=(score,bq,dm)
        fits.append(ranked)
    return _j_collect(fits,split,rows,rest)


def _j_expert_shifts(keys, rows, rest, y, split, best, query, source,
                      gated, job, X, given, deadline):
    """Consensus key/arm/tau; country-prior arms cannot rerake country gates."""
    width=best.shape[1]-int(gated);fits=[];arms=[];full_specs={}
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        b=b[rest]
        if a.shape!=best.shape or b.shape!=query.shape:
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        if max(float(np.max(abs(a-best))),float(np.max(abs(b-query))) if b.size else 0.)<1e-6:
            continue
        arm=job.get('arms',[])[m] if m<len(job.get('arms',[])) else 'unknown'
        arms.append((m,arm,_h_expert_basis(best,a,gated),_h_expert_basis(query,b,gated)))
    if not arms:
        return []
    for held in [None]+list(np.unique(split)):
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<45:
            return []
        score_context=_s136_score_context(base,yy,gated)
        ranked={};routes=keys(fit)
        for m,arm,u,uq in arms:
            a=u[use];b=uq if held is None else u[split==held]
            # Remove the country-only component already present in the base
            # before fitting a gated expert correction. This is a fit-only
            # orthogonal projection, shared by held and Query rows.
            country=next((i for i,f in enumerate(given) if f.lower()=='country'),None)
            if gated and country is not None:
                codes=X[rr,country];dc=X[dest,country];cells=int(X[:,country].max())+1
                count=np.bincount(codes,minlength=cells)
                mean=np.column_stack([np.bincount(codes,weights=a[:,k],minlength=cells) for k in range(width)])/np.maximum(count[:,None],1)
                a=a-mean[codes];b=b-mean[dc]
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
            allowed=[r for r in routes if not (gated and any(f.lower()=='country' and f in r[0].split(':') for f in given))]
            selected=_s136_axis_rank(allowed,rr,dest,base,p,yy,a,b,fit.sum(),gated,m,-.45,full_specs.get(m,set()))
            ranked.update(selected)
            if held is None:
                full_specs[m]=set(selected)
        fits.append(ranked)
    return _j_collect(fits[1:]+fits[:1],split,rows,rest)


def _consensus_interaction_and_simplex_raking(items,work,final,fo,fr,X,given,levels,
                                             fold,nll_fn,deadline,peers,base_snapshot):
    """Tenth stage, atomic five-fold proposals against current best."""
    if time.monotonic()>=deadline:
        return
    if np.isfinite(deadline):
        _s138_screen(_s139_screen_work(work,items,fo,X,given,levels,deadline),items,fo,X,given,levels,fold,base_snapshot)
        _s137_finite_stage('J',items,work,final,fo,fr,X,given,levels,fold,time.monotonic()+max(0.,deadline-time.monotonic())*.20)
    _s136_budget_expert(items,work,fo,fr,X,given,levels,fold,deadline)
    if np.isfinite(deadline) and any(j.get('j_raking') for j in work.values()):return  #  finite accepted-write handoff
    if np.isfinite(deadline):
        if time.monotonic()<deadline:
            _s138_screen(_s139_screen_work(work,items,fo,X,given,levels,deadline),items,fo,X,given,levels,fold,base_snapshot,True)
    key_fo,key_fr=dict(fo),dict(fr)
    features=[f for f in range(len(given)) if 1<int(levels[f])<=128]
    shared=[]
    for i,a in enumerate(features):
        shared.append((given[a],X[:,a],False))
        for b in features[:i]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((':'.join(sorted((given[a],given[b]))),key,True))
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{}) if np.isfinite(deadline) else _s136_order(work,items,fo,X,given,levels,deadline))
    _s136_operators=_s136_plan(('J1','J2','J2_skip','J3','J3_skip','J4'),deadline)
    if np.isfinite(deadline):
        if max((j.get('_s138_scores',{}).get('J4',0.) for j in work.values()),default=0.)<4.8:
            _s136_operators=('J3','J4','J1','J3_skip','J2','J2_skip')
    for _s136_oi,operator in enumerate(_s136_operators):
        _s136_stop=_s136_channel_stop(_s136_operators,_s136_oi,deadline)
        for name,job in _s137_order(work,items,fo,X,given,levels,fold,deadline,operator,base_snapshot):
            if time.monotonic()>=_s136_stop:
                break
            rows,rest=np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
            y,split=job['y'][rows],fold[rows];gated=bool(items[name].get('gate'))
            width=job['K']-int(gated);skip=operator.endswith('_skip')
            if len(np.unique(split))!=5 or (not skip and width<2) or (skip and not gated):
                continue
            if skip and any(p in given for p in _declared_parents(name,items)):
                continue
            source=_wave_two_source(name,items,X,given,rows)
            hidden_source=_wave_two_source(name,items,X,given,rest)
            if source is None or hidden_source is None:
                continue
            if 'wave' in given:
                w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if w is not None:
                    source &= X[rows,given.index('wave')]!=w
                    hidden_source &= X[rest,given.index('wave')]!=w
            if source.sum()<67:
                continue
            before,before_r=fo[name],fr[name]
            snapshot={m:v for m,v in base_snapshot.items() if m!=name}
            peer_keys={m:v for m,v in peers.items() if m!=name}
            cache={}
            def keys(fit):
                token=fit.tobytes()
                if token not in cache:
                    cache[token]=_j_keys(name,items,work,key_fo,key_fr,X,given,levels,rows,y,before,fit,gated,snapshot,peer_keys,skip)
                return cache[token]
            if operator in ('J1','J2','J2_skip'):
                proposals=_j_peer_shifts(rows,rest,y,split,before,before_r,source,gated,snapshot,X,given,levels,_s136_stop,operator!='J1',skip)
            elif operator in ('J3','J3_skip'):
                proposals=_j_simplex_shifts(keys,rows,rest,y,split,before,before_r,source,gated,_s136_stop,skip)
            else:
                static=_residual_keys(name,items,work,key_fo,key_fr,X,given,levels,shared)
                def expert_keys(fit):
                    selected=sorted(((_peer_mi(v[0][rows[fit]],y[fit],job['K']),m) for m,v in peer_keys.items()),key=lambda t:(-t[0],t[1]))[:4]
                    return static+[('peer_mode:'+m,peer_keys[m][0],True) for _,m in selected]
                proposals=_j_expert_shifts(expert_keys,rows,rest,y,split,before,before_r,source,gated,_s136_small_arm_job(job,_s136_stop),X,given,_s136_stop)
            if proposals is None:
                if not np.isfinite(deadline):
                    return
                break
            accepted=[]
            for so,sr,mo,mr,metadata in proposals:
                for gamma in ((1.,) if operator.startswith('J3') else ((.35,) if operator=='J4' else (.35,.65,1.))):
                    if time.monotonic()>=_s136_stop:
                        break
                    active,hidden=mo&source,mr&hidden_source
                    if gated:
                        active &= 1-before[:,-1]>=.20;hidden &= 1-before_r[:,-1]>=.20
                    if skip:
                        active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                        hidden &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                    if operator.startswith('J3'):
                        oo=_j_simplex_project(before,so,active,gamma,gated,skip)
                        qr=_j_simplex_project(before_r,sr,hidden,gamma,gated,skip)
                    elif skip:
                        def project(base,s,mask):
                            rho=np.clip(base[mask,-1],1e-9,1-1e-9)
                            rho=1/(1+np.exp(-(np.log(rho/(1-rho))+gamma*s[mask,0])))
                            return _residual_skip_project(base,mask,rho)
                        oo,qr=project(before,so,active),project(before_r,sr,hidden)
                    else:
                        oo=_ef_project(before,so,active,gamma,gated)
                        qr=_ef_project(before_r,sr,hidden,gamma,gated)
                    kappa={'J1':2.4,'J2':2.4,'J2_skip':2.4,'J3':3.6,'J3_skip':3.2,'J4':4.8}[operator]
                    if not _ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa):
                        continue
                    assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hidden],before_r[~hidden])
                    if gated and not skip:
                        assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
                    scale=1-.001*job['K'];row=np.arange(len(y))
                    gain=np.log(.001+scale*oo[row,y])-np.log(.001+scale*before[row,y])
                    fg=np.array([gain[split==f].mean() for f in np.unique(split)])
                    se=max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
                    record=dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata)
                    accepted.append((float(gain.mean()),oo,qr,record))
            # One specification per subchannel, no stacking on a stale fit base.
            if accepted and time.monotonic()<_s136_stop:
                _,oo,qr,record=max(accepted,key=lambda t:t[0])
                fo[name],fr[name],final[name][rest]=oo,qr,qr
                job.setdefault('j_raking',[]).append(record)
                if np.isfinite(deadline):return  #  finite accepted-write handoff


# additive eleventh stage; all  bytes outside this block are preserved.
# exact active-set scoring and immutable fit-route memoization.
def _s136_small_arm_job(job,deadline,operator='J4'):
    if not np.isfinite(deadline) or len(job['oof'])<=1:
        return job
    ranked=job.get('_s136_k_arm_order' if operator.startswith('K') else '_s136_arm_order',list(range(len(job['oof']))))
    if operator=='K2':
        ranked=[m for m in ranked if m>=1]
    chosen=ranked[:2]
    if operator=='K4':
        chosen=list(dict.fromkeys([0]+chosen))
    return dict(job,oof=[a if m in chosen else a[:0] for m,a in enumerate(job['oof'])])


def _s136_k_order(work,items,fo,X,given,levels,fold,deadline):
    if not np.isfinite(deadline):
        return list(work.items())
    ranked=[]
    for name,job in work.items():
        rows=np.flatnonzero(job['train']);y=job['y'][rows];split=fold[rows]
        gated=bool(items[name].get('gate'));width=job['K']-int(gated)
        source=_wave_two_source(name,items,X,given,rows)
        if width<2 or source is None or len(np.unique(split))!=5:
            ranked.append((0.,name,job));continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        use=source & (y<width)
        if gated:use &= 1-fo[name][:,-1]>=.2
        if use.sum()<67:
            ranked.append((0.,name,job));continue
        base=fo[name][use];p=_g_conditional(base,width);scores=[]
        choices=job.get('_s136_arm_order',list(range(len(job['oof']))))[:2]
        for m in choices:
            arm=job['oof'][m]
            if arm.shape!=fo[name].shape:continue
            u=_h_expert_basis(base,arm[use],gated)
            country=next((f for f,v in enumerate(given) if 'country' in v.lower()),None)
            if gated and country is not None:
                c=X[rows[use],country];nc=int(X[:,country].max())+1;n=np.bincount(c,minlength=nc)
                mean=np.column_stack([np.bincount(c,weights=u[:,k],minlength=nc) for k in range(width)])/np.maximum(n[:,None],1)
                u-=mean[c];u-=(p*u).sum(1,keepdims=True)
            gain=np.zeros(len(rows));gain[use]=_g_fit_gain(base,y[use],.05*u,gated)
            fg=np.array([gain[split==f].mean() for f in np.unique(split)])
            se=max(gain.std(ddof=1)/np.sqrt(len(rows)),fg.std(ddof=1)/np.sqrt(5))
            confidence=float(gain.mean()/max(se,1e-300)) if gain.mean()>1e-5 and (fg>=0).all() else 0.
            scores.append((confidence,m))
        scores.sort(key=lambda t:(-t[0],t[1]));job['_s136_k_arm_order']=[m for _,m in scores]
        confidence=scores[0][0] if scores else 0.
        ranked.append((confidence+1e-6*job.get('_s136_priority',0.),name,job))
    return [(name,job) for _,name,job in sorted(ranked,key=lambda t:(-t[0],t[1]))]


def _s136_budget_expert(items,work,fo,fr,X,given,levels,fold,deadline):
    # A real fit-only residual expert closes the finite CPU path from the
    # current stack. No historical or Query labels enter it.
    if not np.isfinite(deadline):
        return
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{}) if np.isfinite(deadline) else _s136_order(work,items,fo,X,given,levels,deadline))
    started=time.monotonic();stop=started+max(0.,deadline-started)*.22
    added=0
    for name,job in ordered:
        if time.monotonic()>=stop:
            break
        if job.get('_s136_budget_expert'):
            continue
        rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train'])
        y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
        if width<2 or len(np.unique(split))!=5:
            continue
        source=_wave_two_source(name,items,X,given,rows)
        hidden=_wave_two_source(name,items,X,given,rest)
        if source is None or hidden is None:
            continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:
                source &= X[rows,given.index('wave')]!=w
                hidden &= X[rest,given.index('wave')]!=w
        if source.sum()<120:
            continue
        oo=fo[name].copy();qr=fr[name].copy();complete=True;specifications=[]
        for held in list(np.unique(split))+[None]:
            if time.monotonic()>=stop:
                complete=False;break
            fit=source & (split!=held) if held is not None else source.copy()
            use=fit & (y<width);rr=rows[use];yy=y[use]
            if len(rr)<120:
                complete=False;break
            p=_g_conditional(fo[name][use],width);residual=np.eye(width)[yy]-p
            ranked=[]
            for f in range(len(given)):
                if not 1<int(levels[f])<=16 or 'country' in given[f].lower():
                    continue
                c=X[rr,f];n=np.bincount(c,minlength=int(levels[f]));s=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(width)])
                ranked.append((float((s[n>=45]**2/(n[n>=45,None]+64.)).sum()),f))
            selected=[f for _,f in sorted(ranked,key=lambda t:(-t[0],t[1]))[:2]]
            if not selected:
                complete=False;break
            key=X[:,selected[0]].astype(np.int64)
            if len(selected)>1:
                key=key*int(levels[selected[1]])+X[:,selected[1]]
            cells=int(key.max())+1;c=key[rr];n=np.bincount(c,minlength=cells)
            prior=(np.bincount(yy,minlength=width)+1)/(len(yy)+width)
            counts=np.column_stack([np.bincount(c,weights=(yy==k).astype(float),minlength=cells) for k in range(width)])
            expected=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
            offsets=np.clip(np.log((counts+64.*prior)/(expected+64.*prior)),-.4,.4)
            centered=offsets[c]-(p*offsets[c]).sum(1,keepdims=True)
            fisher_scale=max(float(np.sqrt((p*centered*centered).sum(1).mean())),.02)
            offsets=np.clip(.5*offsets/fisher_scale,-1.5,1.5)
            target=hidden if held is None else source & (split==held)
            dest=rest if held is None else rows
            target &= n[key[dest]]>=45
            base=qr if held is None else oo
            mass=base[target,:width].sum(1,keepdims=True)
            weighted=base[target,:width]*np.exp(offsets[key[dest[target]]])
            base[target,:width]=mass*weighted/np.maximum(weighted.sum(1,keepdims=True),1e-300)
            specifications.append([given[f] for f in selected])
        if not complete:
            continue
        full=np.zeros((len(X),job['K']));full[rest]=qr
        job['oof'].append(oo);job['all'].append(full)
        if 'arms' in job:
            job['arms']=list(job['arms'])+['budget_demographic_eb']
        job['_s136_budget_expert']=dict(folds=5,fit_only=True,shrinkage=64.,routes=specifications)
        job.pop('_s136_priority',None);job.pop('_s136_arm_order',None)
        added+=1


def _s136_plan(operators,deadline):
    if not np.isfinite(deadline):
        return operators
    if operators[0]=='J1':
        return ('J4','J3','J3_skip','J1','J2','J2_skip')
    if operators[0]=='K1':
        return ('K4','K1','K2','K3')
    return operators


def _s136_channel_stop(operators,index,deadline):
    if not np.isfinite(deadline):
        return deadline
    weights={'J4':11.,'J3':6.,'J3_skip':1.,'J1':1.,'J2':1.,'J2_skip':1.,
             'K4':5.,'K1':3.,'K2':1.,'K3':1.,
             'L1_signed':7.,'L1_macro':2.,'L2_pair':5.,'L2_specialist':1.,'L3_answer':1.,'L3_skip':1.}
    if operators[0]=='J3':weights.update(J3=11.,J4=6.)
    if operators[0]=='K1':weights.update(K1=5.,K4=3.)
    now=time.monotonic();remaining=max(0.,deadline-now)
    return now+remaining*weights.get(operators[index],1.)/sum(weights.get(op,1.) for op in operators[index:])


def _s136_delete_fisher(gram,rhs,covariance,gradient,basis,probabilities):
    # A binary conditional simplex has a rank-one Fisher matrix, irrespective
    # of the number of expert axes. Delete it by Sherman-Morrison instead of
    # solving thousands of identical-size dense systems. A scalar expert uses
    # exact element-wise division for every class count.
    if gram.shape[0]==1:
        return (rhs[0]-gradient[:,0])[:,None]/(gram[0,0]-covariance[:,0,0])[:,None]
    if probabilities.shape[1]==2 and min(probabilities.dtype.itemsize,basis.dtype.itemsize,covariance.dtype.itemsize)>=8:
        inverse=np.linalg.solve(gram,np.eye(len(rhs)))
        v=(basis[:,1]-basis[:,0])*np.sqrt(probabilities[:,0]*probabilities[:,1])[:,None]
        iv=v@inverse.T;ib=(rhs[None]-gradient)@inverse.T
        denominator=1-(v*iv).sum(1)
        out=ib+iv*((v*ib).sum(1)/np.maximum(denominator,1e-300))[:,None]
        bad=(denominator<=1e-8)|(~np.isfinite(out).all(1))
        if bad.any():
            out[bad]=np.linalg.solve(gram[None]-covariance[bad],(rhs[None]-gradient[bad])[...,None])[...,0]
        return out
    return np.linalg.solve(gram[None]-covariance,(rhs[None]-gradient)[...,None])[...,0]


def _s136_order(work, items, fo, X, given, levels, deadline):
    # Infinite replay retains the original order. Finite execution prioritizes
    # fit-visible demographic residual energy per unit of matrix work.
    if not np.isfinite(deadline):
        return list(work.items())
    result=[]
    for name,job in work.items():
        if '_s136_priority' in job:
            result.append((job['_s136_priority'],name,job));continue
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')))
        y=job['y'][rows];use=y<width
        source=_wave_two_source(name,items,X,given,rows)
        if source is not None:
            use &= source
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:
                use &= X[rows,given.index('wave')]!=w
        if width<2 or use.sum()<67:
            result.append((0.,name,job));continue
        rr=rows[use];yy=y[use];base=fo[name][use];p=_g_conditional(base,width)
        residual=np.eye(width)[yy]-p
        score=0.
        for f in range(len(given)):
            if not 1<int(levels[f])<=16:
                continue
            c=X[rr,f];n=np.bincount(c,minlength=int(levels[f]));ok=n>=45
            s=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(width)])
            score=max(score,float((s[ok]**2/(n[ok,None]+64.)).sum()/len(rows)))
        names=job.get('arms',[])
        preferred=('full_cell_eb','conditional_cell_eb','budget_demographic_eb','catboost','hgb','deep','tabpfn','kumo','table')
        def arm_rank(m):
            label=names[m] if m<len(names) else ('full_cell_eb' if m==1 else 'table' if m==0 else 'other')
            return (len(preferred)+1 if label=='table' else preferred.index(label) if label in preferred else len(preferred),m)
        job['_s136_arm_order']=sorted(range(len(job['oof'])),key=arm_rank)
        job['_s136_priority']=score/max(1.,width/4.)
        result.append((job['_s136_priority'],name,job))
    return [(name,job) for _,name,job in sorted(result,key=lambda t:(-t[0],t[1]))]


def _s136_cached_keys(factory):
    cache = {}
    def keys(fit):
        token = fit.tobytes()
        if token not in cache:
            cache[token] = factory(fit)
        return cache[token]
    return keys


def _s136_score_context(base,yy,gated):
    width=base.shape[1]-int(gated);p=base[:,:width];row=np.arange(len(yy))
    scale=1-.001*base.shape[1]
    observed=base[row,yy]
    return p,p.sum(1),observed,np.log(.001+scale*observed),scale


def _s136_geodesic_score(base, yy, a, step, support, denominator, gated, context=None):
    # Unsupported rows are absent from the old reduction. Zero steps have
    # mathematically zero gain; retaining their zero in the reduction preserves
    # the summation order (and avoids exp/log on the inactive complement).
    chosen = support & ((np.abs(step) > 1e-12) if base.dtype.itemsize>=8 else True)
    gains = np.zeros(int(support.sum()))
    if chosen.any():
        p,mass,observed,logs,scale=(_s136_score_context(base,yy,gated) if context is None else context)
        if chosen.all():
            use=slice(None);dest=slice(None)
        else:
            use=chosen;dest=np.flatnonzero(chosen[support])
        q=p[use];ss=step[use];aa=a[use];target=yy[use]
        if p.shape[1]==2:
            dtype=np.result_type(ss.dtype,aa.dtype)
            odds=np.exp(ss*(aa[:,1].astype(dtype)-aa[:,0].astype(dtype)))
            prob=observed[use]*np.where(target==1,odds,1.)*mass[use]/np.maximum(q[:,0]+q[:,1]*odds,1e-300)
        else:
            shift=ss[:,None]*aa
            weighted=q*np.exp(shift-shift.max(1,keepdims=True))
            prob=weighted[np.arange(len(target)),target]*mass[use]/np.maximum(weighted.sum(1),1e-300)
        gains[dest]=np.log(.001+scale*prob)-logs[use]
    return float(gains.sum()/denominator)


def _s136_axis_rank(routes,rr,dest,base,p,yy,a,b,denominator,gated,arm,lower,forced):
    """Exact top-three deletion ranking with a rigorous softmax upper bound.

    A four-route quadratic shortlist supplies an initial cutoff. The remaining
    routes are skipped only if a Taylor lower bound on the softmax normalizer
    proves their exact gain cannot reach that cutoff. Full-fit candidates are
    always evaluated in every held fit, even outside its top three.
    """
    obs=a[np.arange(len(yy)),yy];variance=(p*a*a).sum(1)
    context=_s136_score_context(base,yy,gated);raw, mass, observed, logs, scale=context
    weights=raw.astype(np.float64)/mass.astype(np.float64)[:,None]
    precise=a.astype(np.float64)
    total=weights.sum(1);mean=(weights*precise).sum(1);bound_variance=(weights*precise*precise).sum(1);radius=np.abs(precise).max(1)
    candidates=[]
    for label,key,_ in routes:
        cells=int(key.max())+1;c=key[rr];dc=key[dest]
        n=np.bincount(c,minlength=cells);support=n>=45
        if not support.any():continue
        grad=np.bincount(c,weights=obs,minlength=cells)
        var=np.bincount(c,weights=variance,minlength=cells)
        for tau in (8.,20.):
            loo=np.clip((grad[c]-obs)/(var[c]-variance+tau),lower,.45)
            approximation=float((loo*obs-.5*loo*loo*variance)[support[c]].sum()/denominator)
            candidates.append((approximation,label,tau,c,dc,support,grad,var))
    candidates.sort(key=lambda v:(-v[0],v[1],v[2]))
    initial=[]
    for value in candidates:
        if value[1] not in initial:initial.append(value[1])
        if len(initial)==4:break
    evaluated={};lookup={}
    for candidate in candidates:
        _,label,tau,c,dc,support,grad,var=candidate;spec=(label,arm,tau)
        lookup[spec]=candidate
        loo=np.clip((grad[c]-obs)/(var[c]-variance+tau),lower,.45)
        mask=support[c]
        exact=(p.shape[1]==2 or label in initial or spec in forced or len(evaluated)<3)
        if not exact:
            cutoff=sorted(evaluated.values(),reverse=True)[2]
            # exp(x)>=1+x+exp(-r)*x^2/2, for |x|<=r. The margin covers
            # normalization roundoff and makes pruning conservative.
            z=total+loo*mean+.5*loo*loo*bound_variance*np.exp(-np.abs(loo)*radius)-1e-12
            probability=observed*np.exp(loo*obs)/np.maximum(z,1e-300)
            upper=float((np.log(.001+scale*probability)-logs)[mask].sum()/denominator)+1e-12
            exact=upper>=cutoff
        if exact:
            score=_s136_geodesic_score(base,yy,a,loo,mask,denominator,gated,context)
            if score>0:evaluated[spec]=score
    best=sorted(evaluated,key=lambda s:(-evaluated[s],repr(s)))[:3]
    keep=set(best)|{s for s in forced if s in evaluated}
    out={}
    for spec in keep:
        _,label,tau,c,dc,support,grad,var=lookup[spec]
        theta=np.clip(grad/(var+tau),lower,.45)
        out[spec]=(evaluated[spec],theta[dc,None]*b,support[dc])
    return out


def _k_collect(fits, split, rows, rest):
    """Require the same specification in every fit and 5 top-2 or 4 top-3 votes."""
    if len(fits)!=6:
        return []
    tops=[sorted(f,key=lambda k:(-f[k][0],repr(k))) for f in fits]
    result=[]
    for spec in tops[-1][:3]:
        votes=sum(spec in t[:3] for t in tops[:-1])
        tight=sum(spec in t[:2] for t in tops[:-1])
        if not (tight==5 or votes>=4) or not all(spec in f for f in fits):
            continue
        shape=fits[-1][spec][1].shape[1:]
        oo=np.zeros((len(rows),)+shape);mask=np.zeros(len(rows),bool)
        for held,fit in zip(np.unique(split),fits[:-1]):
            use=split==held;oo[use],mask[use]=fit[spec][1],fit[spec][2]
        result.append((oo,fits[-1][spec][1],mask,fits[-1][spec][2],dict(spec=list(spec),consensus_folds=votes,top2_folds=tight,fit_mean_gain=float(np.mean([f[spec][0] for f in fits[:-1]])),fit_scores=[float(f[spec][0]) for f in fits])))
    return result


def _k_geodesic_shifts(keys, rows, rest, y, split, best, query, source,
                      gated, job, X, given, deadline):
    """Consensus key/arm/tau; country-prior arms cannot rerake country gates."""
    width=best.shape[1]-int(gated);fits=[];arms=[];full_specs={}
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        b=b[rest]
        if a.shape!=best.shape or b.shape!=query.shape:
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        if max(float(np.max(abs(a-best))),float(np.max(abs(b-query))) if b.size else 0.)<1e-6:
            continue
        arm=job.get('arms',[])[m] if m<len(job.get('arms',[])) else 'unknown'
        arms.append((m,arm,_h_expert_basis(best,a,gated),_h_expert_basis(query,b,gated)))
    if not arms:
        return []
    for held in [None]+list(np.unique(split)):
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<45:
            return []
        score_context=_s136_score_context(base,yy,gated)
        ranked={};routes=keys(fit)
        for m,arm,u,uq in arms:
            a=u[use];b=uq if held is None else u[split==held]
            # Remove the country-only component already present in the base
            # before fitting a gated expert correction. This is a fit-only
            # orthogonal projection, shared by held and Query rows.
            country=next((i for i,f in enumerate(given) if 'country' in f.lower()),None)
            if gated and country is not None:
                codes=X[rr,country];dc=X[dest,country];cells=int(X[:,country].max())+1
                count=np.bincount(codes,minlength=cells)
                mean=np.column_stack([np.bincount(codes,weights=a[:,k],minlength=cells) for k in range(width)])/np.maximum(count[:,None],1)
                a=a-mean[codes];b=b-mean[dc]
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
            allowed=[r for r in routes if not (gated and any('country' in f.lower() and f in r[0].split(':') for f in given))]
            selected=_s136_axis_rank(allowed,rr,dest,base,p,yy,a,b,fit.sum(),gated,m,0.,full_specs.get(m,set()))
            ranked.update(selected)
            if held is None:
                full_specs[m]=set(selected)
        fits.append(ranked)
    fits=fits[1:]+fits[:1]
    result=[]
    for group in sorted({spec[1] for f in fits for spec in f}):
        separate=[{spec:value for spec,value in f.items() if spec[1]==group} for f in fits]
        candidates=_k_collect(separate,split,rows,rest)
        if candidates:
            result.append(max(candidates,key=lambda v:(v[4]['fit_mean_gain'],repr(v[4]['spec']))))
    return result


def _k_loo_kkt(a,d,codes,cells,deadline,maximum=.35):
    """Original 28/26 bisections; identical midpoint derivatives share a sum.

    Every deletion root starts from the same dyadic interval. Most of its
    midpoint values are shared with other rows. The cached value is the exact
    same contiguous NumPy reduction as the original 64-row block, so no root
    approximation, polynomial expansion or stopping tolerance is introduced.
    """
    full=np.zeros(cells);loo=np.zeros(len(a))
    for cell in np.unique(codes):
        if time.monotonic()>=deadline:return None
        ix=np.flatnonzero(codes==cell);aa,dd=a[ix],d[ix]
        cache={0.:float(np.sum(dd/aa)),maximum:float(np.sum(dd/np.maximum(aa+maximum*dd,1e-15)))}
        def derivatives(values):
            if len(aa)<256:
                return (dd[None]/np.maximum(aa[None]+values[:,None]*dd[None],1e-15)).sum(1)
            unique,inverse=np.unique(values,return_inverse=True)
            missing=[v for v in unique if v not in cache]
            for begin in range(0,len(missing),64):
                ts=np.asarray(missing[begin:begin+64])
                sums=(dd[None]/np.maximum(aa[None]+ts[:,None]*dd[None],1e-15)).sum(1)
                cache.update(zip(ts,sums))
            return np.asarray([cache[v] for v in unique])[inverse]
        lo,hi=0.,maximum
        for _ in range(28):
            mid=(lo+hi)/2
            if derivatives(np.array([mid]))[0]>0:lo=mid
            else:hi=mid
        full[cell]=(lo+hi)/2
        for start in range(0,len(ix),64):
            if time.monotonic()>=deadline:return None
            jj=np.arange(start,min(start+64,len(ix)))
            g0=cache[0.]-dd[jj]/aa[jj]
            gm=cache[maximum]-dd[jj]/np.maximum(aa[jj]+maximum*dd[jj],1e-15)
            loo[ix[jj[gm>=0]]]=maximum
            jj=jj[(g0>0)&(gm<0)]
            if not len(jj):continue
            left=np.zeros(len(jj));right=np.full(len(jj),maximum)
            for _ in range(26):
                mid=(left+right)/2
                grad=derivatives(mid)-dd[jj]/np.maximum(aa[jj]+mid*dd[jj],1e-15)
                left=np.where(grad>0,mid,left);right=np.where(grad>0,right,mid)
            loo[ix[jj]]=(left+right)/2
    return full,loo

def _k_specialist_shifts(keys,rows,rest,y,split,best,query,source,gated,job,deadline):
    """Separate expert arms, full-fit shortlist then exact held/deletion scoring."""
    width=best.shape[1]-int(gated);contexts=[];arms=[]
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        if m<1 or a.shape!=best.shape or b[rest].shape!=query.shape:
            continue
        if np.isfinite(a).all() and np.isfinite(b[rest]).all():
            arms.append((m,_g_conditional(a,width),_g_conditional(b[rest],width)))
    if not arms:
        return []
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<45:
            return []
        pre={};idx=np.arange(len(yy));scale=1-.001*best.shape[1]
        a0=.001+scale*base[idx,yy];mass=(1-base[:,-1]) if gated else np.ones(len(yy))
        for m,qo,qr in arms:
            q,qd=qo[use],qr if held is None else qo[split==held]
            d=scale*mass*(q[idx,yy]-p[idx,yy]);obs=d/a0;var=obs*obs
            for label,key,_ in keys(fit):
                codes=key[rr];dc=key[dest];cells=int(key.max())+1
                n=np.bincount(codes,minlength=cells);support=n>=45
                if not support.any():
                    continue
                grad=np.bincount(codes,weights=obs,minlength=cells);v=np.bincount(codes,weights=var,minlength=cells)
                theta=np.clip((grad[codes]-obs)/(v[codes]-var+8.),0,.35)
                score=float(np.log1p(theta*obs)[support[codes]].sum()/fit.sum())
                pre[(label,m,'linear_kkt')]=(score,a0,d,codes,dc,cells,pd,qd,support,fit.sum())
        contexts.append(pre)
    candidates=[]
    for m,_,_ in arms:
        top=sorted((spec for spec in contexts[-1] if spec[1]==m),key=lambda spec:(-contexts[-1][spec][0],repr(spec)))[:2]
        candidates.extend(spec for spec in top if all(spec in c and c[spec][0]>0 for c in contexts))
    fits=[]
    for context in contexts:
        ranked={}
        for spec in candidates:
            score,a0,d,codes,dc,cells,pd,qd,support,n=context[spec]
            solution=_k_loo_kkt(a0,d,codes,cells,deadline)
            if solution is None:
                return None
            theta,loo=solution;gain=np.log1p(loo*d/a0)
            exact=float(gain[support[codes]].sum()/n)
            if exact>0:
                ranked[spec]=(exact,(1-theta[dc,None])*pd+theta[dc,None]*qd,support[dc])
        fits.append(ranked)
    return _k_macro_collect(fits,split,rows,rest)


def _k_risk_shifts(rows,rest,y,split,best,query,source,gated,snapshot,X,given,levels,deadline):
    """Continuous Top-4 peer risk, fit-only quantiles and deletion shrinkage."""
    width=best.shape[1]-int(gated);fits=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<120:
            return []
        selected=_i_residual_peers(snapshot,rr,residual,4)
        if not selected:
            return []
        xf=np.concatenate([snapshot[m][rr] for m in selected],axis=1)
        xd=np.concatenate([snapshot[m][dest] for m in selected],axis=1)
        mu=xf.mean(0);xf=xf-mu;xd=xd-mu
        # PRESS predictions exclude each fitted row; held/full predictions use beta.
        loo,pred,h=_h_ridge_press(xf,xd,residual,120.)
        direction=np.linalg.svd(residual.T@xf,full_matrices=False)[0][:,0]
        score=loo@direction;sd=pred@direction
        features=[f for f in range(len(given)) if 2<=int(levels[f])<=16]
        def strength(f):
            c=X[rr,f];n=np.bincount(c,minlength=int(levels[f]));s=np.bincount(c,weights=residual@direction,minlength=len(n))
            return float(np.sum(s*s/(n+40)))
        top=sorted(features,key=lambda f:(-strength(f),f))[:1]
        ranked={}
        for bins in (3,4):
            cuts=np.unique(np.quantile(score,np.arange(1,bins)/bins))
            codes=np.searchsorted(cuts,score);dc=np.searchsorted(cuts,sd)
            for f in [None]+top:
                label='risk'+str(bins)+('' if f is None else ':'+given[f])
                c=codes if f is None else codes*int(levels[f])+X[rr,f]
                cd=dc if f is None else dc*int(levels[f])+X[dest,f]
                cells=bins if f is None else bins*int(levels[f]);n=np.bincount(c,minlength=cells);support=n>=45
                grad=np.bincount(c,weights=(residual@direction),minlength=cells)
                variance=(p*direction[None]**2).sum(1)-(p@direction)**2
                var=np.bincount(c,weights=variance,minlength=cells)
                for tau in (40.,120.):
                    theta=np.clip(grad/(var+tau),-.35,.35)
                    deleted=np.clip((grad[c]-residual@direction)/(var[c]-variance+tau),-.35,.35)
                    gain=_g_fit_gain(base,yy,deleted[:,None]*direction,gated)
                    sg=float(gain[support[c]].sum()/fit.sum())
                    if sg>0:
                        ranked[(label,'top4',tau)]=(sg,theta[cd,None]*direction,support[cd])
        fits.append(ranked)
    return _k_collect(fits,split,rows,rest)


def _k_macro_collect(fits, split, rows, rest):
    """Require the same specification in every fit and 5 top-2 or 4 top-3 votes."""
    if len(fits)!=6:
        return []
    tops=[sorted(f,key=lambda k:(-f[k][0],repr(k))) for f in fits]
    result=[]
    for spec in tops[-1][:3]:
        votes=sum(spec in t[:3] for t in tops[:-1])
        tight=sum(spec in t[:2] for t in tops[:-1])
        if not all(spec in f for f in fits):
            continue
        shape=fits[-1][spec][1].shape[1:]
        oo=np.zeros((len(rows),)+shape);mask=np.zeros(len(rows),bool)
        for held,fit in zip(np.unique(split),fits[:-1]):
            use=split==held;oo[use],mask[use]=fit[spec][1],fit[spec][2]
        result.append((oo,fits[-1][spec][1],mask,fits[-1][spec][2],dict(spec=list(spec),consensus_folds=votes,top2_folds=tight,fit_mean_gain=float(np.mean([f[spec][0] for f in fits[:-1]])),fit_scores=[float(f[spec][0]) for f in fits])))
    return result


def _k_joint_shifts(keys,rows,rest,y,split,best,query,source,gated,job,deadline):
    """Macro-cell joint log pooling, exact full Fisher row deletion, strong ridge."""
    width=best.shape[1]-int(gated);arms=[];fits=[];candidates=None
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        if a.shape!=best.shape or b[rest].shape!=query.shape:
            continue
        if np.isfinite(a).all() and np.isfinite(b[rest]).all():
            arms.append((m,_h_expert_basis(best,a,gated),_h_expert_basis(query,b[rest],gated)))
    if not arms:
        return []
    for held in [None]+list(np.unique(split)):
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<120:
            return []
        # At most eight expert axes in their original deterministic order.
        chosen=arms[:8]
        u=np.stack([a[1][use] for a in chosen],axis=2)
        ud=np.stack([a[2] if held is None else a[1][split==held] for a in chosen],axis=2)
        cov=np.einsum('nk,nka,nkb->nab',p,u,u,optimize=True);grad=u[np.arange(len(yy)),yy]
        ranked={};ident=tuple(a[0] for a in chosen)
        for label,key,interaction in keys(fit):
            if interaction and int(key.max())+1>32:
                continue
            c=key[rr];dc=key[dest];cells=int(key.max())+1;n=np.bincount(c,minlength=cells);support=n>=120
            if not support.any():
                continue
            for tau in (20.,80.):
                if time.monotonic()>=deadline:
                    return None
                if candidates is not None and (label,'joint_experts',tau) not in candidates:
                    continue
                offsets=np.zeros((len(dest),width));loo=np.zeros((len(yy),width))
                for cell in np.flatnonzero(support):
                    ix=np.flatnonzero(c==cell);di=np.flatnonzero(dc==cell)
                    gram=cov[ix].sum(0)+tau*np.eye(len(chosen));rhs=grad[ix].sum(0)
                    theta=np.clip(np.linalg.solve(gram,rhs),-.35,.35)
                    theta*=min(1.,.45/max(np.abs(theta).sum(),1e-12))
                    offsets[di]=np.einsum('nka,a->nk',ud[di],theta)
                    for start in range(0,len(ix),128):
                        if time.monotonic()>=deadline:
                            return None
                        ii=ix[start:start+128]
                        t=np.clip(_s136_delete_fisher(gram,rhs,cov[ii],grad[ii],u[ii],p[ii]),-.35,.35)
                        t*=np.minimum(1.,.45/np.maximum(np.abs(t).sum(1),1e-12))[:,None]
                        loo[ii]=np.einsum('nka,na->nk',u[ii],t)
                gain=_g_fit_gain(base,yy,loo,gated);score=float(gain[support[c]].sum()/fit.sum())
                if score>0:
                    ranked[(label,'joint_experts',tau)]=(score,offsets,support[dc])
        fits.append(ranked)
        if held is None:
            candidates=set(sorted(ranked,key=lambda k:(-ranked[k][0],repr(k)))[:3])
            if not candidates:
                return []
    return _k_macro_collect(fits[1:]+fits[:1],split,rows,rest)


def _nonnegative_geodesic_and_specialist_raking(items,work,final,fo,fr,X,given,levels,
                                               fold,nll_fn,deadline,peers,base_snapshot):
    """Eleventh stage: independent nonnegative expert geometries and risk cells."""
    if time.monotonic()>=deadline:
        return
    if np.isfinite(deadline):
        _s137_finite_stage('K',items,work,final,fo,fr,X,given,levels,fold,time.monotonic()+max(0.,deadline-time.monotonic())*.35)
        if any(j.get('k_raking') for j in work.values()):return  #  finite accepted-write handoff
    key_fo,key_fr=dict(fo),dict(fr)
    features=[f for f in range(len(given)) if 1<int(levels[f])<=128]
    shared=[]
    for i,a in enumerate(features):
        shared.append((given[a],X[:,a],False))
        for b in features[:i]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((':'.join(sorted((given[a],given[b]))),key,True))
    route_cache={}
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'K1',base_snapshot) if np.isfinite(deadline) else _s136_k_order(work,items,fo,X,given,levels,fold,deadline))
    _s136_operators=_s136_plan(('K1','K2','K3','K4'),deadline)
    if np.isfinite(deadline):
        _s137_order(work,items,fo,X,given,levels,fold,deadline,'K1',base_snapshot)
        if max((j.get('_s137_screen',(None,0.,0.,0.,[],0.))[5] for j in work.values()),default=0.)>=4.:
            _s136_operators=('K1','K4','K2','K3')
    for oi,operator in enumerate(_s136_operators):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            return
        channel_deadline=_s136_channel_stop(_s136_operators,oi,deadline)
        for name,job in _s137_order(work,items,fo,X,given,levels,fold,deadline,operator,base_snapshot):
            if time.monotonic()>=channel_deadline:
                break
            rows,rest=np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
            y,split=job['y'][rows],fold[rows];gated=bool(items[name].get('gate'))
            width=job['K']-int(gated)
            if len(np.unique(split))!=5 or width<2:
                continue
            source=_wave_two_source(name,items,X,given,rows)
            hidden_source=_wave_two_source(name,items,X,given,rest)
            if source is None or hidden_source is None:
                continue
            if 'wave' in given:
                w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if w is not None:
                    source &= X[rows,given.index('wave')]!=w
                    hidden_source &= X[rest,given.index('wave')]!=w
            if source.sum()<67:
                continue
            before,before_r=fo[name],fr[name]
            snapshot={m:v for m,v in base_snapshot.items() if m!=name}
            peer_keys={m:v for m,v in peers.items() if m!=name}
            if name not in route_cache:
                static=_residual_keys(name,items,work,key_fo,key_fr,X,given,levels,shared)
                def keys(fit,static=static,peer_keys=peer_keys,rows=rows,y=y,job=job):
                    selected=sorted(((_peer_mi(v[0][rows[fit]],y[fit],job['K']),m) for m,v in peer_keys.items()),key=lambda t:(-t[0],t[1]))[:4]
                    routes=static+[('peer_mode:'+m,peer_keys[m][0],True) for _,m in selected]
                    return [r for r in routes if not any('country' in f.lower() and f in r[0].split(':') for f in given)]
                route_cache[name]=_s136_cached_keys(keys)
            keys=route_cache[name]
            if operator=='K1':
                proposals=_k_geodesic_shifts(keys,rows,rest,y,split,before,before_r,source,gated,_s136_small_arm_job(job,channel_deadline,operator),X,given,channel_deadline)
            elif operator=='K2':
                proposals=_k_specialist_shifts(keys,rows,rest,y,split,before,before_r,source,gated,_s136_small_arm_job(job,channel_deadline,operator),channel_deadline)
            elif operator=='K3':
                proposals=_k_risk_shifts(rows,rest,y,split,before,before_r,source,gated,snapshot,X,given,levels,channel_deadline)
            else:
                proposals=_k_joint_shifts(keys,rows,rest,y,split,before,before_r,source,gated,_s136_small_arm_job(job,channel_deadline,operator),channel_deadline)
            if proposals is None:
                break
            accepted=[]
            for so,sr,mo,mr,metadata in proposals:
                for gamma in ((1.,) if operator=='K2' else (.35,)):
                    if time.monotonic()>=deadline:
                        return
                    active,hidden=mo&source,mr&hidden_source
                    if gated:
                        active &= 1-before[:,-1]>=.20;hidden &= 1-before_r[:,-1]>=.20
                    if operator=='K2':
                        oo=_j_simplex_project(before,so,active,gamma,gated)
                        qr=_j_simplex_project(before_r,sr,hidden,gamma,gated)
                    else:
                        oo=_ef_project(before,so,active,gamma,gated)
                        qr=_ef_project(before_r,sr,hidden,gamma,gated)
                    kappa={'K1':3.4,'K2':3.8,'K3':3.2,'K4':3.2}[operator]
                    if operator=='K4' and len(job['oof'])>1:
                        kappa=4.4
                    if operator=='K1':
                        if metadata['spec'][1]==0:
                            kappa=4.3
                        else:
                            substantive=[v for v in items[name].get('values',[]) if str(v).strip().lower() not in ('not answered','no response','na_gated')]
                            if width==2 or len(substantive)<=2:
                                kappa=4.6
                            elif ':' not in metadata['spec'][0]:
                                kappa=3.8
                            if metadata['top2_folds']<4:
                                continue
                    if not _ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa):
                        continue
                    assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hidden],before_r[~hidden])
                    if gated:
                        assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
                    scale=1-.001*job['K'];idx=np.arange(len(y))
                    gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y])
                    fg=np.array([gain[split==f].mean() for f in np.unique(split)])
                    se=max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
                    # Complete fit/OOF transfer: reject a correction whose held
                    # gain retains less than half the nominal-step fit signal.
                    if gain.mean()<.5*gamma*metadata['fit_mean_gain']:
                        continue
                    record=dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata)
                    accepted.append((float(gain.mean()),oo,qr,record))
            if accepted and time.monotonic()<deadline:
                _,oo,qr,record=max(accepted,key=lambda t:(t[3]['fit_mean_gain'],t[0]))
                fo[name],fr[name],final[name][rest]=oo,qr,qr
                job.setdefault('k_raking',[]).append(record)
                if np.isfinite(deadline):return  #  finite accepted-write handoff


# additive twelfth stage, training-label-only signed demographic heads.
def _l_macro_shifts(routes,rows,rest,y,split,best,query,source,gated,job,deadline):
    """Exact one-axis  ridge, independently for every original arm."""
    result=[];width=best.shape[1]-int(gated)
    def keys(fit):
        rr=rows[fit & (y<width)]
        return [(label,key,interaction) for label,key,interaction in routes
                if interaction and 'country' not in label.lower() and int(key.max())+1<=16
                and (np.bincount(key[rr],minlength=int(key.max())+1)>=120).sum()>=3]
    keys=_s136_cached_keys(keys)
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        if m not in job.get('_s136_arm_limit',range(len(job['oof']))):
            continue
        if time.monotonic()>=deadline:
            return None
        proposals=_k_joint_shifts(keys,rows,rest,y,split,best,query,source,gated,dict(oof=[a],all=[b]),deadline)
        if proposals is None:
            return None
        for v in proposals:
            v[4]['arm']=m;v[4]['fit_positive_folds']=5
            result.append(v)
    return result


def _l_expert_shifts(routes, rows, rest, y, split, best, query, source,
                     gated, job, deadline, mode, country_key=None):
    """Independent signed Fisher axes; no country centering on other routes.

    Route screening is defined on fit labels only. Each arm keeps four routes
    by quadratic deletion score, then ranks exact floor-likelihood gains.
    Macro cells use the exact scalar Fisher deletion of the single arm.
    """
    width=best.shape[1]-int(gated)
    arms=[]
    for m,(a,b) in enumerate(zip(job['oof'],job['all'])):
        if m not in job.get('_s136_arm_limit',range(len(job['oof']))):
            continue
        if mode in ('specialist','pair') and m==0:
            continue
        if a.shape!=best.shape or b[rest].shape!=query.shape:
            continue
        if not np.isfinite(a).all() or not np.isfinite(b[rest]).all():
            continue
        arms.append((m,_h_expert_basis(best,a,gated),_h_expert_basis(query,b[rest],gated)))
    fits=[];pair_routes=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:
            return None
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,best,query,source,gated,held,False)
        if len(rr)<120:
            return []
        ranked={};route_sets={}
        fold_arms=[(m,u[use],uq if held is None else u[split==held]) for m,u,uq in arms]
        if '_s137_endogenous' in job:
            fold_arms=_s137_endogenous_axes(routes,rr,dest,yy,p,pd,fit.sum(),job['_s137_endogenous'])
        fold_routes=routes
        if '_s137_endogenous' in job:
            screened=[]
            for label,key,interaction in routes:
                c=key[rr];n=np.bincount(c,minlength=int(key.max())+1);ok=n>=67
                if not ok.any():continue
                sums=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(width)])
                screened.append((float((sums[ok]**2/(n[ok,None]+64.)).sum()),label,key,interaction))
            screened.sort(key=lambda t:(-t[0],t[1]))
            fold_routes=[(label,key,interaction) for _,label,key,interaction in screened[:8]]
        for m,a,b in fold_arms:
            if time.monotonic()>=deadline:
                return None
            if gated and country_key is not None and mode in ('signed','macro'):
                cc=country_key[rr];dd=country_key[dest];nc=int(country_key.max())+1
                nn=np.bincount(cc,minlength=nc)
                mean=np.column_stack([np.bincount(cc,weights=a[:,k],minlength=nc) for k in range(width)])/np.maximum(nn[:,None],1)
                a=a-mean[cc];b=b-mean[dd]
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
            obs=a[np.arange(len(yy)),yy];variance=(p*a*a).sum(1)
            country_a,country_b=a,b
            if gated and country_key is not None and mode=='pair':
                cc=country_key[rr];dd=country_key[dest];nc=int(country_key.max())+1
                nn=np.bincount(cc,minlength=nc)
                mean=np.column_stack([np.bincount(cc,weights=a[:,k],minlength=nc) for k in range(width)])/np.maximum(nn[:,None],1)
                country_a=a-mean[cc];country_b=b-mean[dd]
                country_a-=(p*country_a).sum(1,keepdims=True);country_b-=(pd*country_b).sum(1,keepdims=True)
            screen=[]
            for label,key,interaction in fold_routes:
                if ('country' in label.lower() and mode not in ('pair','specialist')) or label.startswith(('peer_mode:','parent:')):
                    continue
                cells=int(key.max())+1
                if mode=='macro' and cells>16:
                    continue
                codes=key[rr];dc=key[dest];n=np.bincount(codes,minlength=cells)
                scalar_macro=(mode=='signed' and m==0 and gated and cells<=16)
                support=n>=(120 if mode=='macro' or scalar_macro else 45)
                if not support.any() or (mode=='macro' and support.sum()<3):
                    continue
                if scalar_macro and support.sum()<3:
                    continue
                aa,bb=(country_a,country_b) if 'country' in label.lower() else (a,b)
                observed=aa[np.arange(len(yy)),yy];vv=(p*aa*aa).sum(1)
                grad=np.bincount(codes,weights=observed,minlength=cells)
                var=np.bincount(codes,weights=vv,minlength=cells)
                trials=[]
                for tau in ((20.,80.) if mode=='macro' else ((20.,) if mode=='specialist' else (8.,) if mode=='pair' else (8.,20.))):
                    loo=np.clip((grad[codes]-observed)/(var[codes]-vv+tau),-.45,.45)
                    screening=np.clip(grad[codes]/(var[codes]+tau),-.45,.45) if mode in ('pair','specialist') else loo
                    approx=float((screening*observed-.5*screening*screening*vv)[support[codes]].sum()/fit.sum())
                    trials.append((approx,tau,loo))
                screen.append((max(t[0] for t in trials),label,key,dc,support,grad,var,trials,codes,aa,bb))
            screen.sort(key=lambda t:(-t[0],t[1]))
            exact=[]
            shortlist=screen[:4]
            if mode=='macro':
                shortlist=screen
            if mode=='pair' and country_key is not None:
                shortlist=([v for v in screen if 'country' in v[1].lower()][:4]
                           +[v for v in screen if 'country' not in v[1].lower()][:4])
            for _,label,key,dc,support,grad,var,trials,codes,aa,bb in shortlist:
                for _,tau,loo in trials:
                    score=_s136_geodesic_score(base,yy,aa,loo,support[codes],fit.sum(),gated)
                    theta=np.clip(grad/(var+tau),-.45,.45)
                    if mode in ('pair','specialist'):
                        # Restore the fit-ranked ordering before the cell-wise
                        # deletion penalty. This is an exact optimism correction:
                        # score_LOO + (score_plugin-score_LOO), computed on fit
                        # labels only, followed by the independent OOF gates.
                        score=_s136_geodesic_score(base,yy,aa,theta[codes],support[codes],fit.sum(),gated)
                    if score<=0:
                        continue
                    exact.append((score,label,tau,theta[dc,None]*bb,support[dc]))
            if mode=='pair':
                # Pick distinct routes, so two tau values cannot impersonate
                # two routes. Pool the same arm, never different expert axes.
                unique={}
                for v in sorted(exact,key=lambda t:(-t[0],t[1],t[2])):
                    unique.setdefault(v[1],v)
                ordered=list(unique.values())
                country=[v for v in ordered if 'country' in v[1].lower()]
                local=[v for v in ordered if 'country' not in v[1].lower()]
                top=(country[:1]+local[:1]) if country_key is not None else ordered[:2]
                if len(top)==2:
                    spec=('top2_pool',m)
                    mask=top[0][4]&top[1][4]
                    ranked[spec]=(sum(t[0] for t in top)/2,(top[0][3]+top[1][3])*.5,mask)
                    route_sets[m]={t[1] for t in top}
            else:
                for score,label,tau,offset,mask in exact:
                    ranked[(label,m,tau)]=(score,offset,mask)
        hook=globals().get("_L_FITS")
        if hook is not None: hook(mode,held,ranked)
        fits.append(ranked)
        pair_routes.append(route_sets)
    result=[]
    groups=sorted({spec[-1] if mode=='pair' else spec[1] for f in fits for spec in f})
    for m in groups:
        fs=[{s:v for s,v in f.items() if (s[-1] if mode=='pair' else s[1])==m} for f in fits]
        candidates=_k_collect(fs,split,rows,rest)
        if mode=='macro':
            # A macro ridge specification has consensus when its exact deletion
            # score is positive in every fit. Its final OOF gate still requires
            # all five folds and both parity halves to improve.
            candidates=_k_macro_collect(fs,split,rows,rest)
        if mode=='pair':
            common=set.intersection(*(f.get(m,set()) for f in pair_routes))
            if not common:
                candidates=[]
            for v in candidates:
                v[4]['anchor_routes']=sorted(common)
                v[4]['pooled_routes']=[sorted(f.get(m,set())) for f in pair_routes]
        candidates=[v for v in candidates if mode=='macro' or (v[4]['consensus_folds']==5 if mode in ('specialist','pair') or m==0 else v[4]['top2_folds']>=4)]
        if candidates:
            if mode=='signed' and m==0 and gated:
                for v in candidates:v[4]['arm']=m
                result.extend(candidates)
                continue
            if mode=='macro':
                for v in candidates:
                    v[4]['arm']=m;v[4]['fit_positive_folds']=5
                result.extend(candidates)
                continue
            chosen=max(candidates,key=lambda v:(v[4]['fit_mean_gain'],repr(v[4]['spec'])))
            chosen[4]['arm']=m
            result.append(chosen)
    return result


def _s137_demographic_routes(X,rr,residual,given,levels):
    """Exact fit-row pair energies, one packed bincount per outcome class."""
    cached=globals().get('_S137_DEMOGRAPHIC_KEYS')
    if cached is None or cached[0] is not X:
        features=[f for f in range(len(given)) if 2<=int(levels[f])<=16]
        keys=[]
        for i,a in enumerate(features):
            keys.append((given[a],X[:,a],False))
            for b in features[:i]:
                keys.append((':'.join(sorted((given[a],given[b]))),X[:,a]*int(levels[b])+X[:,b],True))
        offsets=np.r_[0,np.cumsum([int(key.max())+1 for _,key,_ in keys])]
        packed=np.column_stack([key+offsets[j] for j,(_,key,_) in enumerate(keys)]) if keys else np.empty((len(X),0),int)
        cached=(X,keys,packed,offsets);globals()['_S137_DEMOGRAPHIC_KEYS']=cached
    _,keys,packed,offsets=cached
    if not keys:return []
    codes=packed[rr].ravel();n=np.bincount(codes,minlength=int(offsets[-1]));energy=np.zeros(len(n));valid=n>=67
    for k in range(residual.shape[1]):
        sums=np.bincount(codes,weights=np.repeat(residual[:,k],len(keys)),minlength=len(n))
        energy+=sums*sums/(n+64.)
    energy[~valid]=0.;scores=np.add.reduceat(energy,offsets[:-1])
    order=sorted(range(len(keys)),key=lambda j:(-scores[j],keys[j][0]))
    return [keys[j] for j in order if np.any(valid[offsets[j]:offsets[j+1]])]


def _s138_finite_stage(stage,items,work,final,fo,fr,X,given,levels,fold,deadline,record_stage=None):
    """Distinct floor-simplex, nonnegative and signed fit-only EB geometries.

    Every route and coefficient is selected inside its five fit partitions.
    Tiny convex/geodesic moves leave the current high-confidence stack intact.
    """
    if time.monotonic()>=deadline:return
    order=_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{})
    cached=globals().get('_S138_SCREEN_KEYS')
    if cached is None:return
    _,routes,offsets,packed,hall=cached
    record_key={'J':'j_raking','K':'k_raking','L':'l_raking'}[record_stage or stage]
    for name,job in order:
        if time.monotonic()>=deadline:return
        rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        gated=bool(items[name].get('gate'));width=job['K']-int(gated)
        source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
        if width<2 or source is None or hidden is None or len(np.unique(split))!=5:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
        before,before_r=fo[name],fr[name];fits=[]
        for held in list(np.unique(split))+[None]:
            if time.monotonic()>=deadline:return
            fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
            if len(rr)<120:break
            h=hall[:,rr];n=np.asarray(h.sum(1)).ravel();sums=h@residual
            energy=np.sum(sums*sums,axis=1)/(n+24.);energy[n<67]=0
            totals=np.add.reduceat(energy,offsets[:-1]);top=np.argsort(-totals,kind='stable')[:8]
            one=np.eye(width)[yy];obs=h@one;exp=h@p
            prior=(one.sum(0)+1)/(len(yy)+width);pl=(one.sum(0)[None]+1-one)/(len(yy)+width-1)
            ranked={}
            for j in top:
                if time.monotonic()>=deadline:return
                c=packed[rr,j];dc=packed[dest,j];support=n>=67;mask=support[c];dm=support[dc]
                for tau in (24.,64.):
                    q=(obs[c]-one+tau*pl)/(n[c,None]-1+tau);qd=(obs[dc]+tau*prior)/(n[dc,None]+tau)
                    if stage=='J':
                        step=.08;target=(1-step)*p+step*q;out=(1-step)*pd+step*qd
                        candidate=_j_simplex_project(base,target,mask,1.,gated)
                    else:
                        if stage=='K':
                            a=np.log(np.maximum(q,1e-9)/np.maximum(p,1e-9));b=np.log(np.maximum(qd,1e-9)/np.maximum(pd,1e-9))
                        else:
                            a=np.log((obs[c]-one+tau*pl)/(exp[c]-p+tau*pl));b=np.log((obs[dc]+tau*prior)/(exp[dc]+tau*prior))
                        a=np.clip(a,-1.5,1.5);b=np.clip(b,-1.5,1.5)
                        step=.08;candidate=_ef_project(base,a,mask,step,gated);out=step*b
                    scale=1-.001*job['K'];idx=np.arange(len(yy));gain=np.log(.001+scale*candidate[idx,yy])-np.log(.001+scale*base[idx,yy])
                    score=float(gain.sum()/fit.sum())
                    if score>0:ranked[(routes[j][0],tau)]=(score,out,dm)
            fits.append(ranked)
        if len(fits)!=6:continue
        for so,sr,mo,mr,metadata in _j_collect(fits,split,rows,rest):
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            if stage=='J':
                oo=_j_simplex_project(before,so,active,1.,gated);qr=_j_simplex_project(before_r,sr,hm,1.,gated)
            else:
                oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
            kappa={'J':3.6,'K':3.4,'L':4.}[stage]
            passed=_ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y])
            probe=globals().get('_FAST_PROBE')
            if probe is not None:probe(name,stage,oo,qr,passed,True,metadata)
            if not passed:continue
            if time.monotonic()>=deadline:return
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if gated:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fg=np.array([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            spec={'J':'population_eb_simplex','K':'nonnegative_population_geodesic','L':'signed_residual_geodesic'}[stage]
            record=dict(operator=(record_stage or stage)+'_fit_'+spec,estimator=spec,kappa=kappa,folds=5,fit_only=True,n_active=int(active.sum()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata)
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault(record_key,[]).append(record)
            return


def _s137_finite_stage(stage,items,work,final,fo,fr,X,given,levels,fold,deadline,record_stage=None):
    """Atomic five-fold residual experts for the short-budget CPU path.

    Each stage has a distinct estimator: population EB, residual EB geodesic,
    or full-rank probability PRESS. Every statistic is recomputed on fit rows.
    """
    if np.isfinite(deadline):
        return _s138_finite_stage(stage,items,work,final,fo,fr,X,given,levels,fold,deadline,record_stage)
    if time.monotonic()>=deadline:return
    key={'J':'j_raking','K':'k_raking','L':'l_raking'}[record_stage or stage]
    order=_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{})
    for name,job in order:
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        gated=bool(items[name].get('gate'));width=job['K']-int(gated)
        source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
        if width<2 or source is None or hidden is None or len(np.unique(split))!=5:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:
                source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
        if source.sum()<150:continue
        before,before_r=fo[name],fr[name];fits=[]
        for held in list(np.unique(split))+[None]:
            if time.monotonic()>=deadline:break
            fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
            if len(rr)<120:break
            if stage in ('K','L'):
                # Conditional probability calibration closes a direction absent
                # from peer-only residual heads: the item's own p by GIVEN.
                xx=[];zz=[]
                for f in range(len(given)):
                    if not 2<=int(levels[f])<=32:continue
                    a0=(X[rr,f,None]==np.arange(int(levels[f])-1)).astype(float)
                    b0=(X[dest,f,None]==np.arange(int(levels[f])-1)).astype(float)
                    xx.append(a0);zz.append(b0)
                xf=np.concatenate(xx,axis=1) if xx else np.ones((len(rr),1))
                xd=np.concatenate(zz,axis=1) if zz else np.ones((len(dest),1))
                if stage=='L':
                    xf=(xf[:,:,None]*p[:,None,:-1]).reshape(len(rr),-1)
                    xd=(xd[:,:,None]*pd[:,None,:-1]).reshape(len(dest),-1)
                    xf=np.concatenate((p[:,:-1],p[:,:-1]**2,xf),axis=1)
                    xd=np.concatenate((pd[:,:-1],pd[:,:-1]**2,xd),axis=1)
                # Fit-only covariance selection bounds exact PRESS to 64 axes.
                std=np.maximum(xf.std(0),.05);xf=xf/std;xd=xd/std
                covariance=(xf-xf.mean(0)).T@residual
                keep=np.argsort(-np.sum(covariance*covariance,axis=1),kind='stable')[:64]
                af,ad,_=_h_ridge_press(xf[:,keep],xd[:,keep],residual,120.)
                a=np.clip(af/np.maximum(p,.05),-.6,.6);b=np.clip(ad/np.maximum(pd,.05),-.6,.6)
                mask=np.ones(len(dest),bool);spec='given_press' if stage=='K' else 'conditional_probability_press'
            else:
                if np.isfinite(deadline):
                    ranked=_s137_demographic_routes(X,rr,residual,given,levels)
                    if not ranked:break
                    code=ranked[0][1]
                else:
                    ranked=[]
                    for f in range(len(given)):
                        if not 1<int(levels[f])<=128:continue
                        c=X[rr,f];n=np.bincount(c,minlength=int(levels[f]));ok=n>=67
                        sums=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(width)])
                        ranked.append((float((sums[ok]**2/(n[ok,None]+64.)).sum()),f))
                    ranked.sort(key=lambda t:(-t[0],t[1]));fs=[f for _,f in ranked[:2]]
                    if not fs:break
                    code=X[:,fs[0]].astype(np.int64)
                    if len(fs)>1:code=code*int(levels[fs[1]])+X[:,fs[1]]
                c=code[rr];dc=code[dest];cells=int(code.max())+1;n=np.bincount(c,minlength=cells);support=n>=67
                one=np.eye(width)[yy];obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
                prior=(one.sum(0)+1)/(len(yy)+width);pl=(one.sum(0)[None]+1-one)/(len(yy)+width-1)
                if stage=='J':
                    q=(obs[c]-one+64*pl)/(n[c,None]-1+64);qd=(obs[dc]+64*prior)/(n[dc,None]+64)
                    a=np.log(np.maximum(q,1e-8)/np.maximum(p,1e-8));b=np.log(np.maximum(qd,1e-8)/np.maximum(pd,1e-8));spec='population_eb'
                else:
                    exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
                    a=np.log((obs[c]-one+64*pl)/(exp[c]-p+64*pl));b=np.log((obs[dc]+64*prior)/(exp[dc]+64*prior));spec='residual_eb'
                a=np.clip(a,-1.5,1.5);b=np.clip(b,-1.5,1.5);a[~support[c]]=0;b[~support[dc]]=0;mask=support[dc]
            if stage=='J':
                # The population EB branch uses the exact official-floor
                # convex KKT line, with the same simplex gate as .
                trial=_j_simplex_project(base,q,np.ones(len(base),bool),1.,gated)
                t=_j_kkt_step(base,trial-base,yy,np.zeros(len(yy),np.int64),1)[0]
                mixed=(1-t)*pd+t*qd
                offset=np.log(np.maximum(mixed,1e-12)/np.maximum(pd,1e-12))
                candidate=_j_simplex_project(base,(1-t)*p+t*q,np.ones(len(base),bool),1.,gated)
                fl=1-.001*job['K'];ix=np.arange(len(yy))
                score=float((np.log(.001+fl*candidate[ix,yy])-np.log(.001+fl*base[ix,yy])).sum()/fit.sum())
                fits.append({(spec,):(score,offset,mask)} if score>0 else {})
                continue
            a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
            # Fisher units give the ridge the same meaning for weak and strong
            # experts. The scale uses fit probabilities, not destination labels.
            scale=max(float(np.sqrt((p*a*a).sum(1).mean())),.02);a=a/scale;b=b/scale
            obs=a[np.arange(len(yy)),yy];var=(p*a*a).sum(1)
            theta=float(np.clip(obs.sum()/(var.sum()+20.),0.,.45))
            loo=np.clip((obs.sum()-obs)/(var.sum()-var+20.),0.,.45)
            score=_s136_geodesic_score(base,yy,a,loo,np.ones(len(yy),bool),fit.sum(),gated)
            fits.append({(spec,): (score,theta*b,mask)} if score>0 else {})
        if len(fits)!=6:continue
        proposals=_j_collect(fits,split,rows,rest)
        for so,sr,mo,mr,metadata in proposals:
            if time.monotonic()>=deadline:break
            active=mo&source;hm=mr&hidden
            if gated:
                active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            gamma=1. if stage=='J' else .35
            oo=_ef_project(before,so,active,gamma,gated);qr=_ef_project(before_r,sr,hm,gamma,gated)
            passed=_ef_dominates(oo,before,y,split,active,all_rows=True,kappa=4.)
            idx=np.arange(len(y));fl=1-.001*job['K'];gain=float(np.mean(np.log(.001+fl*oo[idx,y])-np.log(.001+fl*before[idx,y])))
            transfer=stage=='J' or gain>=.5*.35*metadata['fit_mean_gain']
            probe=globals().get('_FAST_PROBE')
            if probe is not None:probe(name,stage,oo,qr,passed,transfer,metadata)
            if not passed or not transfer:continue
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if gated:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            record=dict(operator=(record_stage or stage)+'_fit_'+spec,estimator=spec,folds=5,fit_only=True,n_active=int(active.sum()),oof_gain=gain,**metadata)
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault(key,[]).append(record)
            if np.isfinite(deadline):return


def _s138_screen(work,items,fo,X,given,levels,fold,snapshot,expert_mode=False):
    """Training-only, shared sparse sufficient statistics; no exp/log ranking loop.

    Ranking is not an acceptance test. Every proposal is rebuilt from each
    fold's fit rows and must pass the original five-fold and two-half gate.
    """
    from scipy.sparse import csr_matrix
    cached=globals().get('_S138_SCREEN_KEYS')
    if cached is None or cached[0] is not X:
        fs=[i for i,n in enumerate(levels) if 2<=int(n)<=128]
        keys=[('all',np.zeros(len(X),int),False)]
        for pos,a in enumerate(fs):
            keys.append((given[a],X[:,a],False))
            for b in fs[:pos]:
                if int(levels[a])*int(levels[b])<=1024:
                    keys.append((':'.join(sorted((given[a],given[b]))),X[:,a]*int(levels[b])+X[:,b],True))
        offsets=np.r_[0,np.cumsum([int(c.max())+1 for _,c,_ in keys])]
        packed=np.column_stack([c+offsets[j] for j,(_,c,_) in enumerate(keys)])
        h=csr_matrix((np.ones(packed.size),(packed.ravel(),np.repeat(np.arange(len(X)),len(keys)))),shape=(int(offsets[-1]),len(X)))
        cached=(X,keys,offsets,packed,h);globals()['_S138_SCREEN_KEYS']=cached
    _,keys,offsets,packed,hall=cached
    country=next((i for i,n in enumerate(given) if 'country' in n.lower()),None)
    for name,job in work.items():
        token=(id(fo[name]),len(job['oof']),expert_mode)
        prev=job.get('_s138_token')
        if prev is not None and prev[:2]==token[:2] and (prev[2] or not expert_mode):continue
        job['_s138_token']=token
        rows=np.flatnonzero(job['train']);yy=job['y'][rows]
        gated=bool(items[name].get('gate'));width=job['K']-int(gated)
        src=_wave_two_source(name,items,X,given,rows)
        scores=dict.fromkeys(('J1','J2','J3','J4','K1','K2','K3','K4','L1_signed','L1_macro','L2_pair','L2_specialist','L3_answer','L3_skip','J3_skip','J2_skip'),0.)
        job['_s138_scores']=scores
        if width<2 or src is None:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:src &= X[rows,given.index('wave')]!=w
        use=src & (yy<width)
        if gated:use &= 1-fo[name][:,-1]>=.2
        ix=np.flatnonzero(use)
        if len(ix)<120:continue
        rr=rows[ix];y=yy[ix];base=fo[name][ix];p=_g_conditional(base,width)
        residual=np.eye(width)[y]-p;n=len(rr);sp=fold[rr]
        h=hall[:,rr];counts=np.asarray(h.sum(1)).ravel();support=counts>=67
        sums=h@residual
        energy=np.sum(sums*sums,axis=1)/(counts+64.)
        energy[~support]=0
        demo=np.add.reduceat(energy,offsets[:-1])/len(rows)
        demo_order=np.argsort(-demo,kind='stable')[:4]
        obs_counts=h@np.eye(width)[y]
        confidence=0.
        for j in demo_order:
            c=packed[rr,j];nn=counts[c]
            prior=(np.bincount(y,minlength=width)+1)/(n+width)
            q=(obs_counts[c]-np.eye(width)[y]+24*prior)/(nn[:,None]-1+24)
            shift=np.log(np.maximum(q,1e-8)/np.maximum(p,1e-8))
            gain=_g_fit_gain(base,y,.4*shift,gated);gain[~support[c]]=0
            fg=np.array([gain[sp==f].mean() for f in np.unique(sp)])
            se=max(float(gain.std(ddof=1))/np.sqrt(n),float(fg.std(ddof=1))/np.sqrt(len(fg)),1e-12)
            if gain.sum()/len(rows)>1.5e-5 and (fg>=0).all():confidence=max(confidence,float(gain.mean()/se))
        scores.update(J3=confidence,K3=confidence,L3_answer=confidence)
        peer=0.
        if snapshot:
            pc=globals().get('_S138_PEER_SCREEN')
            if pc is None or pc[0] is not X:
                names=list(snapshot);matrix=np.column_stack([snapshot[m][:,0] for m in names]);pc=(X,names,matrix);globals()['_S138_PEER_SCREEN']=pc
            _,names,matrix=pc
            panel=np.linspace(0,n-1,min(n,192),dtype=int);rp=residual[panel];rp-=rp.mean(0)
            a=matrix[rr[panel]];a=a-a.mean(0)
            cov=a.T@rp;norm=np.sum(a*a,axis=0)
            value=np.maximum(0,np.sum(cov*cov,axis=1)-norm*np.mean(np.sum(rp*rp,axis=1)))/np.maximum(norm,1e-12)/len(panel)
            if name in names:value[names.index(name)]=0
            peer=float(value.max(initial=0.))
        scores.update(J1=peer,J2=peer*float(demo.max(initial=0.)))
        if not expert_mode:
            job['_s137_screen']=(token,confidence,0.,float(demo.max(initial=0.)),[],0.,[])
            continue
        # A bounded panel ranks expert arms; complete full-row fits follow.
        panel=np.arange(n) if len(work)<=32 else np.linspace(0,n-1,min(n,768),dtype=int);ratio=n/len(panel)
        ex=ix[panel];er=rr[panel];ep=p[panel];ey=y[panel]
        eh=hall[:,er];ec=np.asarray(eh.sum(1)).ravel();esup=ec*ratio>=45
        signed=[];positive=[]
        for m,arm in enumerate(job['oof']):
            if arm.shape!=fo[name].shape:continue
            # Probability contrasts supply Fisher screening without log/exp.
            a=_h_expert_basis(base[panel],arm[ex],gated) if len(work)<=32 else (_g_conditional(arm[ex],width)-ep)/np.maximum(ep,.05)
            a-=(ep*a).sum(1,keepdims=True)
            if gated and country is not None:
                c=X[er,country];nc=np.bincount(c)
                mean=np.column_stack([np.bincount(c,weights=a[:,k],minlength=len(nc)) for k in range(width)])/np.maximum(nc[:,None],1)
                a-=mean[c];a-=(ep*a).sum(1,keepdims=True)
            obs=a[np.arange(len(panel)),ey];var=(ep*a*a).sum(1)
            grad=(eh@obs)*ratio;fisher=(eh@var)*ratio
            # Fisher upper scores for all keys at once, including interactions.
            gs=np.where(esup,grad,0.);vf=np.where(esup,fisher,0.)
            allowed=np.array([not(gated and country is not None and given[country] in label.split(':')) for label,_,_ in keys])
            best=posbest=0.
            for positive_only in (False,True):
                t=np.clip(gs/(vf+8.),0. if positive_only else -.45,.45)
                gain=t*gs-.5*t*t*vf-t*gs/np.maximum(counts,1.)
                variance=t*t*vf
                totals=np.add.reduceat(gain,offsets[:-1]);vars=np.add.reduceat(variance,offsets[:-1])
                z=totals/np.sqrt(np.maximum(vars,1e-16));z[~allowed]=0
                value=float(z.max(initial=0.))
                if positive_only:posbest=value
                else:best=value
            if len(work)<=32 and m<=2:
                best=posbest=0.
                for positive_only in (False,True):
                    t=np.clip(gs/(vf+8.),0. if positive_only else -.45,.45)
                    energy=np.where(esup,gs*gs/(vf+8.),0.)
                    score=np.add.reduceat(energy,offsets[:-1]);score[~allowed]=-np.inf
                    for j in np.argsort(-score,kind='stable')[:4]:
                        c=packed[er,j]
                        for tau in (8.,20.):
                            t=np.clip((grad[c]-obs)/(fisher[c]-var+tau),0. if positive_only else -.45,.45)
                            gain=_g_fit_gain(base[panel],ey,.35*t[:,None]*a,gated);gain[~esup[c]]=0
                            fg=np.array([gain[fold[er]==f].mean() for f in np.unique(fold[er])])
                            se=max(float(gain.std(ddof=1))/np.sqrt(len(er)),float(fg.std(ddof=1))/np.sqrt(len(fg)),1e-12)
                            value=float(gain.mean()/se) if gain.sum()/len(rows)>1.5e-5 and (fg>=0).all() else 0.
                            if positive_only:posbest=max(posbest,value)
                            else:best=max(best,value)
            if len(work)<=32 and m>2:best/=np.sqrt(ratio);posbest/=np.sqrt(ratio)
            signed.append((best,m));positive.append((posbest,m))
        signed.sort(key=lambda v:(int(len(work)<=32 and v[1]>2),-v[0],v[1]));positive.sort(key=lambda v:(int(len(work)<=32 and v[1]>2),-v[0],v[1]))
        job['_s136_arm_order']=[m for _,m in signed];job['_s136_k_arm_order']=[m for _,m in positive]
        ss=max([v for v,m in signed if len(work)>32 or m<=2] or [0.]);ps=max([v for v,m in positive if len(work)>32 or m<=2] or [0.])
        scores.update(J4=ss,K1=ps,K2=ps,K4=ss,L1_signed=ss,L1_macro=ss,L2_pair=ss,L2_specialist=ss)
        job['_s137_screen']=(token,confidence,ss,float(demo.max(initial=0.)),job['_s136_arm_order'],ps,job['_s136_k_arm_order'])


def _s137_order(work,items,fo,X,given,levels,fold,deadline,operator,snapshot):
    if not np.isfinite(deadline):return list(work.items())
    if time.monotonic()>=deadline:return []
    _s138_screen(_s139_screen_work(work,items,fo,X,given,levels,deadline),items,fo,X,given,levels,fold,snapshot,operator not in ('J1','J2','J2_skip','J3','J3_skip','K3','L3_answer','L3_skip'))
    ranked=[]
    for name,job in work.items():
        score=job.get('_s138_scores',{}).get(operator,0.)
        if score>1.5e-5:ranked.append((score,name,job))
    return [(n,j) for _,n,j in sorted(ranked,key=lambda t:(-t[0],t[1]))]


def _s137_endogenous_axes(routes,rr,dest,yy,p,pd,fit_size,snapshot=None):
    """Nested fit-only axes: Dirichlet deletion and residual ridge PRESS.

    A held fold never contributes to counts, route screening, priors or PRESS.
    The fit direction deletes that respondent's label and fitted probability.
    """
    width=p.shape[1];one=np.eye(width)[yy];residual=one-p
    prior=(one.sum(0)+1)/(len(yy)+width)
    prior_loo=(one.sum(0)[None]+1-one)/(len(yy)+width-1)
    # Cheap Fisher energy screens routes before constructing dense directions.
    screen=[]
    for label,key,interaction in routes:
        codes=key[rr];cells=int(key.max())+1;n=np.bincount(codes,minlength=cells);ok=n>=67
        if not ok.any():continue
        sums=np.column_stack([np.bincount(codes,weights=residual[:,k],minlength=cells) for k in range(width)])
        screen.append((float((sums[ok]**2/(n[ok,None]+64.)).sum()),label,key))
    screen.sort(key=lambda t:(-t[0],t[1]));ranked=[]
    for _,label,key in screen[:4]:
        codes=key[rr];dc=key[dest];cells=int(key.max())+1
        n=np.bincount(codes,minlength=cells);ok=n>=67
        obs=np.column_stack([np.bincount(codes,weights=one[:,k],minlength=cells) for k in range(width)])
        exp=np.column_stack([np.bincount(codes,weights=p[:,k],minlength=cells) for k in range(width)])
        af=np.clip(np.log((obs[codes]-one+64*prior_loo)/(exp[codes]-p+64*prior_loo)),-.6,.6)
        ad=np.clip(np.log((obs[dc]+64*prior)/(exp[dc]+64*prior)),-.6,.6)
        af[~ok[codes]]=0;ad[~ok[dc]]=0
        af-=(p*af).sum(1,keepdims=True);ad-=(pd*ad).sum(1,keepdims=True)
        observed=af[np.arange(len(yy)),yy];var=(p*af*af).sum(1)
        score=float((observed-.5*var).sum()/max(fit_size,1))
        ranked.append((score,label,af,ad))
    ranked.sort(key=lambda t:(-t[0],t[1]));axes=[]
    for i,v in enumerate(ranked[:2]):axes.append((1000+i,v[2],v[3]))
    if len(ranked)>=2:
        axes.append((1002,.5*(ranked[0][2]+ranked[1][2]),.5*(ranked[0][3]+ranked[1][3])))
    if snapshot:
        selected=_i_residual_peers(snapshot,rr,residual,3)
        if selected:
            xf=np.concatenate([snapshot[m][rr][:,:-1] for m in selected],axis=1)
            xd=np.concatenate([snapshot[m][dest][:,:-1] for m in selected],axis=1)
            sf,sd,_=_h_ridge_press(xf,xd,residual,100.)
            af=np.clip(sf/np.maximum(p,.05),-.6,.6);ad=np.clip(sd/np.maximum(pd,.05),-.6,.6)
            af-=(p*af).sum(1,keepdims=True);ad-=(pd*ad).sum(1,keepdims=True)
            axes.append((1003,af,ad))
    # Full-rank probability calibration and ordered class contrasts.
    xf=np.concatenate((p[:,:-1],p[:,:-1]**2),axis=1)
    xd=np.concatenate((pd[:,:-1],pd[:,:-1]**2),axis=1)
    sf,sd,_=_h_ridge_press(xf,xd,residual,20.)
    af=np.clip(sf/np.maximum(p,.05),-.6,.6);ad=np.clip(sd/np.maximum(pd,.05),-.6,.6)
    af-=(p*af).sum(1,keepdims=True);ad-=(pd*ad).sum(1,keepdims=True)
    axes.append((1004,af,ad))
    axes=[(m,a/max(float(np.sqrt((p*a*a).sum(1).mean())),.02),
             b/max(float(np.sqrt((p*a*a).sum(1).mean())),.02)) for m,a,b in axes]
    return axes


def _s137_endogenous_raking(items,work,final,fo,fr,X,given,levels,fold,
                            nll_fn,deadline,peers,base_snapshot):
    """Self-contained  directions with the existing signed/pool gates."""
    features=[f for f in range(len(given)) if 1<int(levels[f])<=128]
    routes=[]
    for i,a in enumerate(features):
        routes.append((given[a],X[:,a],False))
        for b in features[:i]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            routes.append((':'.join(sorted((given[a],given[b]))),key,True))
    channels=('L3_endo_signed','L3_endo_pair')
    for ci,operator in enumerate(channels):
        stop=deadline if not np.isfinite(deadline) else time.monotonic()+max(0.,deadline-time.monotonic())/(len(channels)-ci)
        for name,job in _s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',base_snapshot):
            if time.monotonic()>=stop:break
            rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train'])
            y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
            if width<2 or len(np.unique(split))!=5:continue
            source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
            if source is None or hidden is None:continue
            if 'wave' in given:
                w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if w is not None:
                    source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
            if source.sum()<150:continue
            before,before_r=fo[name],fr[name]
            head=dict(job,_s137_endogenous={k:v for k,v in base_snapshot.items() if k!=name})
            mode='pair' if operator.endswith('pair') else 'signed'
            proposals=_l_expert_shifts(routes,rows,rest,y,split,before,before_r,source,gated,head,stop,mode)
            if proposals is None:break
            accepted=[]
            for so,sr,mo,mr,metadata in proposals:
                active=mo&source;hm=mr&hidden
                if gated:
                    active &= 1-before[:,-1]>=.20;hm &= 1-before_r[:,-1]>=.20
                oo=_ef_project(before,so,active,.35,gated);qr=_ef_project(before_r,sr,hm,.35,gated)
                passed=_ef_dominates(oo,before,y,split,active,all_rows=True,kappa=4.0)
                scale=1-.001*job['K'];idx=np.arange(len(y))
                gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y])
                transfer=gain.mean()>=.5*.35*metadata['fit_mean_gain']
                record=dict(operator=operator,kappa=4.,gamma=.35,n_active=int(active.sum()),oof_gain=float(gain.mean()),endogenous=True,**metadata)
                probe=globals().get('_L_PROBE')
                if probe is not None:probe(name,record,oo,qr,passed and transfer)
                if not passed or not transfer:continue
                assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
                if gated:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
                accepted.append((metadata['fit_mean_gain'],oo,qr,record))
            if accepted and time.monotonic()<stop:
                _,oo,qr,record=max(accepted,key=lambda t:t[0])
                fo[name],fr[name],final[name][rest]=oo,qr,qr
                job.setdefault('l_raking',[]).append(record)
    if not np.isfinite(deadline):
        for mode in ('J','K','L'):
            _s137_finite_stage(mode,items,work,final,fo,fr,X,given,levels,fold,deadline,record_stage='L')


def _signed_demographic_and_consensus_raking(items,work,final,fo,fr,X,given,levels,
                                           fold,nll_fn,deadline,peers,base_snapshot):
    """ signed multiclass,  same-arm consensus,  residual answer/skip."""
    if time.monotonic()>=deadline:
        return
    if np.isfinite(deadline):
        _s137_finite_stage('L',items,work,final,fo,fr,X,given,levels,fold,time.monotonic()+max(0.,deadline-time.monotonic())*(.65 if len(work)>100 else .20))
        if any(j.get('l_raking') for j in work.values()):return  #  finite accepted-write handoff
    key_fo,key_fr=dict(fo),dict(fr)
    features=[f for f in range(len(given)) if 1<int(levels[f])<=128]
    shared=[]
    for i,a in enumerate(features):
        shared.append((given[a],X[:,a],False))
        for b in features[:i]:
            _,key=np.unique(X[:,a]*int(levels[b])+X[:,b],return_inverse=True)
            shared.append((':'.join(sorted((given[a],given[b]))),key,True))
    channels=('L1_signed','L1_macro','L2_pair','L2_specialist','L3_answer','L3_skip')
    _s137_l_deadline=deadline
    if np.isfinite(deadline):deadline=time.monotonic()+max(0.,deadline-time.monotonic())*.85
    ordered=(_s137_order(work,items,fo,X,given,levels,fold,deadline,'J3',{}) if np.isfinite(deadline) else _s136_order(work,items,fo,X,given,levels,deadline))
    for ci,operator in enumerate(channels):
        now=time.monotonic()
        if now>=deadline:
            break
        stop=_s136_channel_stop(channels,ci,deadline)
        for name,job in _s137_order(work,items,fo,X,given,levels,fold,deadline,operator,base_snapshot):
            if time.monotonic()>=stop:
                break
            rows,rest=np.flatnonzero(job['train']),np.flatnonzero(~job['train'])
            y,split=job['y'][rows],fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
            skip=operator=='L3_skip'
            if width<(1 if skip else 2) or len(np.unique(split))!=5:
                continue
            source=_wave_two_source(name,items,X,given,rows)
            hidden_source=_wave_two_source(name,items,X,given,rest)
            if source is None or hidden_source is None:
                continue
            if 'wave' in given:
                w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                if w is not None:
                    source &= X[rows,given.index('wave')]!=w
                    hidden_source &= X[rest,given.index('wave')]!=w
            if source.sum()<67:
                continue
            counts=np.sort(np.bincount(y[source & (y<width)],minlength=width))
            substantive=width>=3 and counts[-3]>=80
            if operator.startswith('L2') and width>=3 and counts[-3]<120:
                continue
            if operator.startswith('L1') and not substantive:
                continue
            if operator.startswith('L1') and gated and any('country' in f.lower() for f in given) and any(p not in given for p in _declared_parents(name,items)):
                continue
            # Explicit response refusal is a separate outcome process. Its
            # already fitted mass is outside the substantive two-route layer.
            if operator.startswith('L2') and any(str(v).strip().lower() in ('no response','not answered','prefer not to answer') for v in items[name].get('values',[])):
                continue
            if skip and (not gated or any(p in given for p in _declared_parents(name,items))):
                continue
            before,before_r=fo[name],fr[name]
            head_job=job
            if np.isfinite(deadline):
                ranked=job.get('_s136_arm_order',list(range(len(job['oof']))))
                eligible=[m for m in ranked if not operator.startswith('L2') or m>=1]
                limit=([0] if operator=='L1_macro' else eligible[:1])
                head_job=dict(job,_s136_arm_limit=limit)
            if operator.startswith(('L1','L2')):
                mode={'L1_signed':'signed','L1_macro':'macro','L2_specialist':'specialist','L2_pair':'pair'}[operator]
                country=next((i for i,f in enumerate(given) if 'country' in f.lower()),None)
                if operator.startswith('L2') and gated and any(p not in given for p in _declared_parents(name,items)):
                    continue
                routes=shared if operator=='L2_pair' else [r for r in shared if 'country' not in r[0].lower()]
                if mode=='macro':
                    proposals=_l_macro_shifts(routes,rows,rest,y,split,before,before_r,source,gated,head_job,stop)
                else:
                    proposals=_l_expert_shifts(routes,rows,rest,y,split,before,before_r,source,gated,head_job,stop,mode,None if country is None else X[:,country])
            else:
                # Residual routing is fold-local; all old answer and skip
                # conservation operators remain shared with the first stages.
                def residual_keys(fit):
                    use=fit if skip else fit & (y<width)
                    rr=rows[use]
                    residual=((y[use]==width).astype(float)-before[use,-1])[:,None] if skip else np.eye(width)[y[use]]-_g_conditional(before[use],width)
                    scores=[]
                    for label,key,interaction in shared:
                        c=key[rr];n=np.bincount(c,minlength=int(key.max())+1);ok=n>=67
                        if not ok.any():
                            continue
                        r=np.column_stack([np.bincount(c,weights=residual[:,k],minlength=len(n)) for k in range(residual.shape[1])])
                        scores.append((float((r[ok]**2/(n[ok,None]+64.)).sum()),label,key,interaction))
                    scores.sort(key=lambda t:(-t[0],t[1]))
                    return [(label,key,interaction) for _,label,key,interaction in scores[:4]]
                keyfn=_s136_cached_keys(residual_keys)
                proposals=_j_simplex_shifts(keyfn,rows,rest,y,split,before,before_r,source,gated,stop,skip)
            if proposals is None:
                break
            accepted=[]
            for so,sr,mo,mr,metadata in proposals:
                if time.monotonic()>=stop:
                    break
                gamma=1. if operator.startswith('L3') else .35
                active,hidden=mo&source,mr&hidden_source
                if gated:
                    active &= 1-before[:,-1]>=.20
                    hidden &= 1-before_r[:,-1]>=.20
                if skip:
                    active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                    hidden &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                if skip:
                    oo=_residual_skip_project(before,active,(1-gamma)*before[active,-1]+gamma*so[active,0])
                    qr=_residual_skip_project(before_r,hidden,(1-gamma)*before_r[hidden,-1]+gamma*sr[hidden,0])
                elif operator=='L3_answer':
                    oo=_j_simplex_project(before,so,active,gamma,gated)
                    qr=_j_simplex_project(before_r,sr,hidden,gamma,gated)
                else:
                    oo=_ef_project(before,so,active,gamma,gated)
                    qr=_ef_project(before_r,sr,hidden,gamma,gated)
                kappa={'L1_signed':4.0,'L1_macro':4.0,'L2_specialist':4.2,'L2_pair':3.8,'L3_answer':3.6,'L3_skip':3.8}[operator]
                if operator.startswith('L1') and metadata['arm']==0 and gated:
                    if operator=='L1_signed' and metadata['top2_folds']<4:
                        continue
                    kappa=4.3
                    if operator=='L1_macro':
                        kappa=3.8
                    elif metadata['consensus_folds']==5 and metadata['top2_folds']==5:
                        label=metadata['spec'][0]
                        key=next((r[1] for r in shared if r[0]==label),None)
                        if key is not None and int(key.max())+1<=16:
                            kappa=3.8
                if operator=='L1_macro' and metadata['arm']>=1:
                    kappa=4.5
                if operator=='L1_macro' and metadata['arm']==0 and not gated and metadata['top2_folds']<4:
                    continue
                if operator=='L1_signed' and metadata['arm']==0 and not gated:
                    kappa=4.1 if ':' not in metadata['spec'][0] else 3.9
                if operator=='L1_signed' and metadata['arm']>=1:
                    kappa=4.2 if ':' in metadata['spec'][0] else 4.0
                passed=_ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa)
                idx=np.arange(len(y));scale=1-.001*job['K']
                gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y])
                fg=np.array([gain[split==f].mean() for f in np.unique(split)])
                se=max(gain.std(ddof=1)/np.sqrt(len(gain)),fg.std(ddof=1)/np.sqrt(5))
                transfer=gain.mean()>=.5*gamma*metadata['fit_mean_gain']
                record=dict(operator=operator,kappa=kappa,gamma=gamma,n_active=int(active.sum()),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),se_ratio=float(gain.mean()/max(se,1e-300)),**metadata)
                # Optional local probe receives predictions, never Query labels.
                probe=globals().get('_L_PROBE')
                if probe is not None:
                    probe(name,record,oo,qr,passed and transfer)
                if not passed or not transfer:
                    continue
                assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hidden],before_r[~hidden])
                if gated and not skip:
                    assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
                accepted.append((metadata['fit_mean_gain'],float(gain.mean()),oo,qr,record))
            if accepted and time.monotonic()<stop:
                _,_,oo,qr,record=max(accepted,key=lambda t:(t[0],t[1]))
                fo[name],fr[name],final[name][rest]=oo,qr,qr
                job.setdefault('l_raking',[]).append(record)
                if np.isfinite(deadline):return  #  finite accepted-write handoff
    _s137_endogenous_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,_s137_l_deadline,peers,base_snapshot)


def _s139_screen_work(work,items,fo,X,given,levels,deadline):
    """Grow the finite screen after actual stage writes, sharing cached rows."""
    if not np.isfinite(deadline) or len(work)<=100:return work
    limit=len(work) if any(j.get('k_raking') for j in work.values()) else 128 if any(j.get('j_raking') for j in work.values()) else 64
    return dict(_s136_order(work,items,fo,X,given,levels,deadline)[:limit])


def _m_routes(name,items,work,fo,fr,X,given,levels,peers):
    """Only marginal axes: no Cartesian splitting of respondent support."""
    routes=[(given[f],X[:,f]) for f in range(len(given)) if 2<=int(levels[f])<=128]
    job=work[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train'])
    for parent in _parents(name,items):
        if parent in given:continue
        if parent in fo and parent in work and np.array_equal(job['train'],work[parent]['train']):
            key=np.zeros(len(X),int);key[rows]=fo[parent].argmax(1);key[rest]=fr[parent].argmax(1)
            routes.append(('parent:'+parent,key))
    for key,(mode,_) in peers.items():
        if key!=name:routes.append(('peer:'+key,mode))
    return routes


def _m_axes(routes,rr,dest,yy,p,pd,tau=60.):
    """Fit-only marginal log ratios, with each fit response deleted exactly."""
    width=p.shape[1];one=np.eye(width)[yy];residual=one-p
    total=one.sum(0);prior=(total+1)/(len(yy)+width)
    pl=(total[None]+1-one)/(len(yy)+width-1);ranked=[]
    for label,key in routes:
        c=key[rr];dc=key[dest];cells=int(key.max())+1
        n=np.bincount(c,minlength=cells);ok=n>=67
        if not ok.any():continue
        obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
        exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
        sums=obs-exp;energy=float((sums[ok]**2/(exp[ok]+tau*prior)).sum())
        ranked.append((energy,label,c,dc,n,obs,exp,ok))
    ranked.sort(key=lambda v:(-v[0],v[1]));axes=[]
    for energy,label,c,dc,n,obs,exp,ok in ranked[:4]:
        a=np.clip(np.log((obs[c]-one+tau*pl)/(exp[c]-p+tau*pl)),-1.2,1.2)
        b=np.clip(np.log((obs[dc]+tau*prior)/(exp[dc]+tau*prior)),-1.2,1.2)
        a[~ok[c]]=0;b[~ok[dc]]=0
        a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        norm=max(float(np.sqrt(np.mean(np.sum(p*a*a,axis=1)))),.02)
        axes.append((label,a/norm,b/norm,ok[dc]))
    return axes


def _m_joint_fisher(axes,p,pd,yy,lam=60.):
    """Joint marginal coefficients and exact deletion of each Fisher block."""
    a=np.stack([v[1] for v in axes],axis=2);b=np.stack([v[2] for v in axes],axis=2)
    gram=np.einsum('nc,ncm,nck->nmk',p,a,a,optimize=True)
    rhs=a[np.arange(len(yy)),yy]-np.einsum('nc,ncm->nm',p,a,optimize=True)
    total=gram.sum(0)+lam*np.eye(a.shape[2]);r=rhs.sum(0)
    coef=np.linalg.solve(total,r)
    delete=np.linalg.solve(total[None]-gram,(r[None]-rhs)[...,None])[...,0]
    af=np.einsum('ncm,nm->nc',a,delete);ad=b@coef
    return np.clip(af,-1.,1.),np.clip(ad,-1.,1.)


def _m_marginal_simplex(routes,axes,rr,dest,yy,p,pd):
    """Pool supported marginal EB distributions on the answer simplex."""
    one=np.eye(p.shape[1])[yy];total=one.sum(0);prior=(total+1)/(len(yy)+p.shape[1])
    pl=(total[None]+1-one)/(len(yy)+p.shape[1]-1);keys=dict(routes);targets=[]
    for label,_,_,_ in axes:
        key=keys[label];c=key[rr];dc=key[dest];cells=int(key.max())+1;n=np.bincount(c,minlength=cells)
        obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(p.shape[1])])
        q=(obs[c]-one+64*pl)/(n[c,None]-1+64);qd=(obs[dc]+64*prior)/(n[dc,None]+64)
        q[n[c]<67]=p[n[c]<67];qd[n[dc]<67]=pd[n[dc]<67]
        targets.append((q,qd))
    return targets


def _m_features(axes,p,pd,rr,dest,residual,snapshot,skip=False,parent=None):
    """At most 48 joint continuous and marginal residual features."""
    xf=[];xd=[]
    selected=_i_residual_peers(snapshot,rr,residual,4)
    for name in selected:
        v=snapshot[name]
        # Rank the coordinates on fit residuals; never inspect held labels.
        centered=v[rr]-v[rr].mean(0)
        order=np.argsort(-np.sum((centered.T@residual)**2,axis=1),kind='stable')[:4]
        xf.append(v[rr][:,order]);xd.append(v[dest][:,order])
    pp=p if p.shape[1]<=6 else p[:,:6];dd=pd if pd.shape[1]<=6 else pd[:,:6]
    xf.extend([pp,pp**2,(-np.sum(p*np.log(np.maximum(p,1e-12)),axis=1))[:,None]])
    xd.extend([dd,dd**2,(-np.sum(pd*np.log(np.maximum(pd,1e-12)),axis=1))[:,None]])
    for _,a,b,_ in axes[:3]:
        # A fixed class coordinate basis, limited before concatenation.
        xf.append(a[:,:min(4,a.shape[1])]);xd.append(b[:,:min(4,b.shape[1])])
    if parent is not None:
        xf.append(parent[rr][:,:4]);xd.append(parent[dest][:,:4])
    return np.concatenate(xf,axis=1)[:,:48],np.concatenate(xd,axis=1)[:,:48]


def _m_ordinal(p,pd,af,ad):
    """Proportional-odds tangent from cumulative, ordered cutpoint shifts."""
    def convert(q,a):
        dq=q*(a-(q*a).sum(1,keepdims=True))
        c=np.cumsum(q,axis=1)[:,:-1];dc=np.cumsum(dq,axis=1)[:,:-1]
        weight=np.maximum(c*(1-c),.03)
        shift=np.sum(dc,axis=1)/np.maximum(weight.sum(1),.03)
        ordered=np.clip(c+shift[:,None]*c*(1-c),0.,1.)
        ordered=np.maximum.accumulate(ordered,axis=1)
        out=np.diff(np.column_stack((np.zeros(len(q)),ordered,np.ones(len(q)))),axis=1)
        return np.clip((out-q)/np.maximum(q,.03),-1.,1.)
    return convert(p,af),convert(pd,ad)


def _m_multi_arm(job,use,held,rest,split,base,p,pd,yy,gated,deadline):
    """Positive-gradient joint convex floor fit followed by deletion KKT."""
    arms=[];idx=np.arange(len(yy));scale=1-.001*base.shape[1]
    mass=base[:,:p.shape[1]].sum(1);a=.001+scale*base[idx,yy]
    for m,(oo,full) in enumerate(zip(job['oof'],job['all'])):
        if m==0 or oo.shape[1]!=base.shape[1]:continue
        q=_g_conditional(oo[use],p.shape[1]);qd=_g_conditional(full[rest] if held is None else oo[split==held],p.shape[1])
        d=scale*mass*(q[idx,yy]-p[idx,yy]);gradient=float(np.sum(d/a))
        if gradient>0 and np.isfinite(q).all() and np.isfinite(qd).all():arms.append((gradient,m,q,qd,d))
    arms.sort(key=lambda v:(-v[0],v[1]));out=[]
    for count in (2,3):
        chosen=arms[:count]
        if len(chosen)<2:continue
        D=np.column_stack([v[4] for v in chosen]);alpha=np.zeros(len(chosen))
        # Coordinate KKT on the capped convex simplex, official floor objective.
        for sweep in range(5):
            for m in range(len(chosen)):
                offset=a+D@alpha-D[:,m]*alpha[m];lo=0.;hi=.35-(alpha.sum()-alpha[m])
                for _ in range(22):
                    mid=(lo+hi)/2
                    if np.sum(D[:,m]/np.maximum(offset+mid*D[:,m],1e-15))>0:lo=mid
                    else:hi=mid
                alpha[m]=(lo+hi)/2
        weight=alpha/max(alpha.sum(),1e-12)
        target=sum(w*v[2] for w,v in zip(weight,chosen));td=sum(w*v[3] for w,v in zip(weight,chosen))
        d=scale*mass*(target[idx,yy]-p[idx,yy])
        solution=_k_loo_kkt(a,d,np.zeros(len(a),int),1,deadline,maximum=.35)
        if solution is None:return out
        full,loo=solution
        score=float(np.log1p(loo*d/a).mean())
        if score>0:out.append((count,score,(1-full[0])*pd+full[0]*td,[v[1] for v in chosen],weight.tolist()))
    return out


def _m_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
    before,before_r=fo[name],fr[name];routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers)
    snapshot={k:v for k,v in snapshot.items() if k!=name};fits=[]
    parent=None
    for key in _parents(name,items):
        if key in snapshot:parent=snapshot[key];break
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<150:return []
        selected_peers=set(_i_residual_peers(snapshot,rr,residual,4))
        local_routes=[v for v in routes if not v[0].startswith('peer:') or v[0][5:] in selected_peers]
        if np.isfinite(deadline):local_routes=[v for v in routes if not v[0].startswith('peer:')]
        axes=_m_axes(local_routes,rr,dest,yy,p,pd);ranked={}
        pooled=_m_marginal_simplex(local_routes,axes,rr,dest,yy,p,pd) if 'M1_marginal_simplex' in channels else []
        for count in (2,3,4):
            if count>len(axes):continue
            if time.monotonic()>=deadline:return []
            af,ad=_m_joint_fisher(axes[:count],p,pd,yy)
            if pooled:
                q=np.mean([v[0] for v in pooled[:count]],axis=0);qd=np.mean([v[1] for v in pooled[:count]],axis=0)
                candidate=_j_simplex_project(base,q,np.ones(len(base),bool),.08,gated)
                scale=1-.001*job['K'];idx=np.arange(len(yy))
                score=float((np.log(.001+scale*candidate[idx,yy])-np.log(.001+scale*base[idx,yy])).sum()/max(fit.sum(),1))
                if score>0:ranked[('M1_marginal_simplex',count)]=(score,.92*pd+.08*qd,np.ones(len(dest),bool))
            for op,a,b in [('M1_additive_eb',af,ad)]+([('M1_ordinal',*_m_ordinal(p,pd,af,ad))] if width>=3 else []):
                if op not in channels:continue
                candidate=_ef_project(base,a,np.ones(len(base),bool),.35,gated)
                score=float(_g_fit_gain(base,yy,.35*a,gated).sum()/max(fit.sum(),1))
                if score>0:ranked[(op,count)]=(score,.35*b,np.ones(len(dest),bool))
        if 'M2_joint_press' in channels:
            xf,xd=_m_features(axes,p,pd,rr,dest,residual,snapshot)
            if np.isfinite(deadline):
                xf,xd=_m_features([],p,pd,rr,dest,residual,snapshot)
                keys=dict(local_routes)
                for label,_,_,_ in axes[:3]:
                    key=keys[label];cells=int(key.max())+1
                    if cells>16 or xf.shape[1]+cells>48:continue
                    eye=np.eye(cells)
                    xf=np.column_stack((xf,eye[key[rr]]));xd=np.column_stack((xd,eye[key[dest]]))
            for lam in (60.,180.):
                af,ad,_=_h_ridge_press(xf,xd,residual,lam)
                af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                score=float(_g_fit_gain(base,yy,.35*af,gated).sum()/max(fit.sum(),1))
                if score>0:ranked[('M2_joint_press',lam)]=(score,.35*ad,np.ones(len(dest),bool))
        if 'M2_multi_arm_kkt' in channels:
            for count,score,target,arms,weights in _m_multi_arm(job,use,held,rest,split,base,p,pd,yy,gated,deadline):
                ranked[('M2_multi_arm_kkt',count)]=(score,target,np.ones(len(dest),bool))
        if gated and parent is not None and 'M3_skip_answer' in channels:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rr)>=150:
                # Marginal binary skip effects use a separate residual response.
                rho=base[:,-1];rd=db[:,-1];binary=np.column_stack((1-rho,rho));bd=np.column_stack((1-rd,rd))
                axes=_m_axes(local_routes,rr,dest,(yy==width).astype(int),binary,bd)
                xf,xd=_m_features(axes,binary,bd,rr,dest,residual,snapshot,True,parent)
                for lam in (60.,180.):
                    af,ad,_=_h_ridge_press(xf,xd,residual,lam)
                    a=np.clip(af[:,0]/np.maximum(rho*(1-rho),.04),-1.,1.)
                    b=np.clip(ad[:,0]/np.maximum(rd*(1-rd),.04),-1.,1.)
                    score=float(_h_skip_gain(base,yy,.35*a).sum()/max(fit.sum(),1))
                    if score>0:ranked[('M3_skip_answer',lam)]=(score,(.35*b)[:,None],(rd>.015)&(rd<.985))
        observe=globals().get('_M_FITS')
        if observe is not None:observe(held,ranked)
        fits.append(ranked)
    proposals=[]
    # Consensus is applied within each orthogonal operator, not across operators.
    for op in channels:
        ff=[{k:v for k,v in f.items() if k[0]==op} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            active=mo&source;hm=mr&hidden
            if op=='M3_skip_answer':
                active &= (before[:,-1]>.015)&(before[:,-1]<.985)
                if any(p in given for p in _parents(name,items)):
                    # GIVEN gate conservation includes its exact pure complement.
                    active &= before[:,-1]>1e-12;hm &= before_r[:,-1]>1e-12
                rho=1/(1+np.exp(-np.clip(np.log(np.clip(before[:,-1],1e-12,1-1e-12)/(1-np.clip(before[:,-1],1e-12,1-1e-12)))+so[:,0],-30,30)))
                rd=1/(1+np.exp(-np.clip(np.log(np.clip(before_r[:,-1],1e-12,1-1e-12)/(1-np.clip(before_r[:,-1],1e-12,1-1e-12)))+sr[:,0],-30,30)))
                oo=_residual_skip_project(before,active,rho[active]);qr=_residual_skip_project(before_r,hm,rd[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if op in ('M2_multi_arm_kkt','M1_marginal_simplex') else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            gain=np.log(.001+(1-.001*job['K'])*oo[np.arange(len(y)),y])-np.log(.001+(1-.001*job['K'])*before[np.arange(len(y)),y])
            fg=np.array([gain[split==f].mean() for f in np.unique(split)])
            se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=op,se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True)
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _additive_multi_effect_and_multi_arm_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Thirteenth stage: additive marginals, joint experts, conditional skip."""
    channels=('M1_additive_eb','M1_ordinal','M2_multi_arm_kkt','M2_joint_press','M3_skip_answer')
    thresholds={'M1_additive_eb':9.7,'M1_marginal_simplex':2.4,'M1_ordinal':3.8,'M2_multi_arm_kkt':3.6,'M2_joint_press':3.6,'M3_skip_answer':3.6}
    if np.isfinite(deadline):channels=('M1_marginal_simplex','M1_additive_eb','M1_ordinal','M2_joint_press','M3_skip_answer')
    key_fo,key_fr=dict(fo),dict(fr)
    order=_s136_order(work,items,fo,X,given,levels,deadline)
    if np.isfinite(deadline) and any('_s138_scores' in j for j in work.values()):
        order=sorted(order,key=lambda v:(-v[1].get('_s138_scores',{}).get('J3',0.),v[0]))
    for name,job in order:
        if time.monotonic()>=deadline:return
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        candidates=_m_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,channels,key_fo,key_fr)
        accepted=[]
        for oo,qr,active,hm,record in candidates:
            kappa=thresholds[record['operator']]
            passed=_ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa)
            record['kappa']=kappa
            probe=globals().get('_M_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted and time.monotonic()<deadline:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0])
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and record['operator']!='M3_skip_answer':assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('m_raking',[]).append(record)
            if np.isfinite(deadline):return


def _n_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot,thresholds):
    thresholds=dict(thresholds,N2_expert_subspace=2.4,N3_entropy_skip=3.4)
    precise=deadline-time.monotonic()>=8.
    for job in work.values():job['_n_finite_precise']=precise
    key_fo,key_fr=dict(fo),dict(fr)
    queue=[];deferred=[]
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        plans=([('N1_ortho_additive',),('N1_ortho_additive',),('N1_peer_pair',)] if precise else [('N1_ortho_additive',),('N1_peer_pair',)])
        # Keep slow entropy/skip fits atomic without blocking completed  heads.
        deferred.append((name,job,[('N3_entropy_skip',)]))
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')))
        if len(job['oof'])>2 and len(rows)*width*len(job['oof'])<150000:
            source=_wave_two_source(name,items,X,given,rows)
            if source is not None:
                source=source.copy()
                if 'wave' in given:
                    wave_one=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
                    if wave_one is not None:source &= X[rows,given.index('wave')]!=wave_one
                # Defer large deletion blocks so they cannot consume a short
                # window before already-complete compact candidates are tested.
                if int((source&(job['y'][rows]<width)).sum())<=4000:plans.append(('N2_expert_subspace',))
                else:deferred.append((name,job,[('N2_expert_subspace',)]))
        queue.append((name,job,plans))
    for name,job,plans in queue+deferred:
        if time.monotonic()>=deadline:return
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        for plan_index,channels in enumerate(plans):
            job['_n_finite_precise']=precise and not (plan_index==0 and channels==('N1_ortho_additive',))
            accepted=[]
            local_deadline=min(deadline,time.monotonic()+8.) if channels==('N2_expert_subspace',) else deadline
            for oo,qr,active,hm,record in _n_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,local_deadline,channels,key_fo,key_fr):
                kappa=thresholds[record['operator']];record['kappa']=kappa
                record['full_fit_amplification']=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15)
                passed=_n_dominates(oo,before,y,split,active,kappa)
                probe=globals().get('_N_PROBE')
                if probe is not None:probe(name,record,oo,qr,passed)
                if passed:
                    alpha=min(1.,1.5/max(record['full_fit_amplification'],1e-15));record['query_full_fit_shrinkage']=alpha
                    candidate=before_r.copy();candidate[hm]=before_r[hm]+alpha*(qr[hm]-before_r[hm]);qr=candidate
                    if record['skip']:qr=_residual_skip_project(before_r,hm,qr[hm,-1])
                    accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
            if accepted and time.monotonic()<deadline:
                _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0])
                assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
                if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
                fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('n_raking',[]).append(record);return


def _n_expert_pca(aa,bb,p):
    A=np.stack(aa,2);B=np.stack(bb,2);H=np.einsum('nc,ncm,nck->mk',p,A,A,optimize=True)/len(p)
    values,vectors=np.linalg.eigh(H);order=np.argsort(-values,kind='stable');order=[int(k) for k in order if values[k]>1e-10][:3]
    axes=[]
    for rank,k in enumerate(order):
        v=vectors[:,k]/np.sqrt(values[k]);axes.append(('pc'+str(rank),A@v,B@v,np.ones(len(B),bool)))
    return axes,[v[0] for v in axes]


def _n_finite_order(order,items,fo,deadline):
    ranked=[];start=time.monotonic();limit=min(deadline-.20,start+.30)
    for index,(name,job) in enumerate(order[:32]):
        if time.monotonic()>=limit:break
        base=fo[name];width=job['K']-int(bool(items[name].get('gate')))
        if width<1:continue
        p=_g_conditional(base,width);score=0.
        for arm in job['oof']:
            q=_g_conditional(arm,width);value=np.sum((q-p)**2/(p+.04),1)
            if items[name].get('gate'):
                rho=base[:,-1];mask=(rho>.015)&(rho<.985)
                value+=mask*(arm[:,-1]-rho)**2/(rho*(1-rho)+.04)
            score=max(score,float(value.mean()))
        ranked.append((score,name,job))
    included={v[1] for v in ranked}
    return [(name,job) for _,name,job in sorted(ranked,key=lambda v:(-v[0],v[1]))]+[(name,job) for name,job in order if name not in included]


def _n_macro_expert(axes,p,pd,yy,c,dc,lam):
    """Exact cellwise Fisher block deletion on a three-expert subspace."""
    a=np.stack([v[1] for v in axes],2);b=np.stack([v[2] for v in axes],2)
    gram=np.einsum('nc,ncm,nck->nmk',p,a,a,optimize=True)
    rhs=a[np.arange(len(yy)),yy]-np.einsum('nc,ncm->nm',p,a,optimize=True)
    cells=int(max(c.max(),dc.max()))+1;count=np.bincount(c,minlength=cells);ok=count>=67
    af=np.zeros_like(p);ad=np.zeros_like(pd)
    for cell in np.flatnonzero(ok):
        fit=c==cell;dest=dc==cell;G=gram[fit].sum(0)+lam*np.eye(a.shape[2]);r=rhs[fit].sum(0)
        beta=np.linalg.solve(G,r)
        deleted=_s136_delete_fisher(G,r,gram[fit],rhs[fit],a[fit],p[fit])
        af[fit]=np.einsum('ncm,nm->nc',a[fit],deleted);ad[dest]=b[dest]@beta
    return np.clip(af,-1.,1.),np.clip(ad,-1.,1.),ok[dc]


def _n_dominates(arm,ref,y,split,mask,kappa):
    # The inactive complement is exactly unchanged; test the genuine subproblem.
    if int(mask.sum())<67:return False
    return _ef_dominates(arm[mask],ref[mask],y[mask],split[mask],np.ones(int(mask.sum()),bool),all_rows=True,kappa=kappa)


def _n_collect_finite(fits,split,rows,rest):
    """Vote separately for distinct estimators, before their dominance test."""
    def family(spec):
        if spec[0]=='N1_ortho_additive':return spec[1]
        if spec[0]=='N1_peer_pair':return spec[-1]
        if spec[0]=='N2_expert_subspace':return (str(spec[2])+'_' if spec[1]=='macro' else 'global_')+('kkt' if spec[-1]=='kkt' else 'fisher')
        if spec[0]=='N3_entropy_skip':return spec[1]
        return 'press'
    output=[]
    for variant in sorted({family(spec) for spec in fits[-1]}):
        output.extend(_j_collect([{k:v for k,v in fit.items() if family(k)==variant} for fit in fits],split,rows,rest))
    return output


def _n_collect(fits,split,rows,rest):
    """Vote separately for distinct estimators, before their dominance test."""
    def family(spec):
        if spec[0]=='N1_ortho_additive':return spec[1]
        if spec[0]=='N1_peer_pair':return spec[-1]
        if spec[0]=='N2_expert_subspace':return 'kkt' if spec[-1]=='kkt' else 'fisher'
        if spec[0]=='N3_entropy_skip':return spec[1]
        return 'press'
    output=[]
    for variant in sorted({family(spec) for spec in fits[-1]}):
        output.extend(_j_collect([{k:v for k,v in fit.items() if family(k)==variant} for fit in fits],split,rows,rest))
    return output


def _n_raw_axes(routes,rr,dest,yy,p,pd,tau=60.):
    """Fit-only marginal log ratios, with each fit response deleted exactly."""
    width=p.shape[1];one=np.eye(width)[yy];residual=one-p
    total=one.sum(0);prior=(total+1)/(len(yy)+width)
    pl=(total[None]+1-one)/(len(yy)+width-1);ranked=[]
    for label,key in routes:
        c=key[rr];dc=key[dest];cells=int(key.max())+1
        n=np.bincount(c,minlength=cells);ok=n>=67
        if not ok.any():continue
        obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
        exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
        sums=obs-exp;energy=float((sums[ok]**2/(exp[ok]+tau*prior)).sum())
        ranked.append((energy,label,c,dc,n,obs,exp,ok))
    ranked.sort(key=lambda v:(-v[0],v[1]));axes=[]
    for energy,label,c,dc,n,obs,exp,ok in ranked[:8]:
        a=np.clip(np.log((obs[c]-one+tau*pl)/(exp[c]-p+tau*pl)),-1.2,1.2)
        b=np.clip(np.log((obs[dc]+tau*prior)/(exp[dc]+tau*prior)),-1.2,1.2)
        a[~ok[c]]=0;b[~ok[dc]]=0
        a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        norm=max(float(np.sqrt(np.mean(np.sum(p*a*a,axis=1)))),.02)
        axes.append((label,a,b,ok[dc]))
    return axes



def _n_sum_orthogonal(axes,p,pd):
    selected=[]
    for label,a,b,mask in axes[:3]:
        a=a.copy();b=b.copy()
        for _,u,v,_ in selected:
            coefficient=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15))
            a-=coefficient*u;b-=coefficient*v
        if np.sum(p*a*a)>1e-12:selected.append((label,a,b,mask))
    return selected


def _n_floor_direction(base,p,pd,af,ad,yy,deadline):
    q=p*np.exp(np.clip(.5*af,-1.,1.));q/=q.sum(1,keepdims=True)
    qd=pd*np.exp(np.clip(.5*ad,-1.,1.));qd/=qd.sum(1,keepdims=True)
    idx=np.arange(len(yy));scale=1-.001*base.shape[1];mass=base[:,:p.shape[1]].sum(1)
    a=.001+scale*base[idx,yy];d=scale*mass*(q[idx,yy]-p[idx,yy])
    sol=_k_loo_kkt(a,d,np.zeros(len(yy),int),1,deadline,maximum=.5)
    if sol is None:return None
    alpha,loo=sol;score=float(np.log1p(loo*d/a).mean())
    return score,(1-alpha[0])*pd+alpha[0]*qd


def _n_axis_pool(routes,rr,dest,yy,p,pd,tau=60.):
    """Fit-only marginal log ratios, with each fit response deleted exactly."""
    width=p.shape[1];one=np.eye(width)[yy];residual=one-p
    total=one.sum(0);prior=(total+1)/(len(yy)+width)
    pl=(total[None]+1-one)/(len(yy)+width-1);ranked=[]
    for label,key in routes:
        c=key[rr];dc=key[dest];cells=int(key.max())+1
        n=np.bincount(c,minlength=cells);ok=n>=67
        if not ok.any():continue
        obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
        exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
        sums=obs-exp;energy=float((sums[ok]**2/(exp[ok]+tau*prior)).sum())
        ranked.append((energy,label,c,dc,n,obs,exp,ok))
    ranked.sort(key=lambda v:(-v[0],v[1]));axes=[]
    for energy,label,c,dc,n,obs,exp,ok in ranked[:8]:
        a=np.clip(np.log((obs[c]-one+tau*pl)/(exp[c]-p+tau*pl)),-1.2,1.2)
        b=np.clip(np.log((obs[dc]+tau*prior)/(exp[dc]+tau*prior)),-1.2,1.2)
        a[~ok[c]]=0;b[~ok[dc]]=0
        a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        norm=max(float(np.sqrt(np.mean(np.sum(p*a*a,axis=1)))),.02)
        axes.append((label,a/norm,b/norm,ok[dc]))
    return axes



def _n_select_axes(pool,p,pd,yy):
    """Choose three independent effects by deletion-scored conditional evidence."""
    chosen=[];available=list(pool);idx=np.arange(len(yy))
    for _ in range(3):
        candidates=[]
        for j,(label,a,b,mask) in enumerate(available):
            a=a.copy();b=b.copy()
            for _,u,v,_ in chosen:
                coef=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15));a-=coef*u;b-=coef*v
            norm=float(np.sqrt(np.mean(np.sum(p*a*a,axis=1))))
            if norm<1e-7:continue
            a/=norm;b/=norm;gram=np.sum(p*a*a,1);rhs=a[idx,yy]
            theta=np.clip((rhs.sum()-rhs)/(gram.sum()-gram+40.),-1.,1.)
            weighted=p*np.exp(np.clip(theta[:,None]*a,-2.,2.));weighted/=weighted.sum(1,keepdims=True)
            score=float(np.mean(np.log(np.maximum(weighted[idx,yy],1e-12)/np.maximum(p[idx,yy],1e-12))))
            candidates.append((score,label,j,a,b,mask))
        if not candidates:break
        score,label,j,a,b,mask=max(candidates,key=lambda v:(v[0],v[1]))
        if score<=0:break
        chosen.append((label,a,b,mask));available.pop(j)
    return chosen


def _n_gram_schmidt(axes,p,pd):
    """Exact Fisher Gram-Schmidt, carrying the same transformation to queries."""
    out=[]
    for label,a,b,mask in axes[:3]:
        a=a.copy();b=b.copy()
        for _,u,v,_ in out:
            coef=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15))
            a-=coef*u;b-=coef*v
        norm=float(np.sqrt(np.mean(np.sum(p*a*a,axis=1))))
        if norm>1e-7:out.append((label,a/norm,b/norm,mask))
    return out


def _n_subspace(snapshot,rr,dest,residual,routes,axes,p,pd):
    """Fit-only rank-three battery projection plus supported GIVEN interactions."""
    names=_i_residual_peers(snapshot,rr,residual,8)
    if not names:return None
    xf=np.concatenate([snapshot[k][rr] for k in names],axis=1)
    xd=np.concatenate([snapshot[k][dest] for k in names],axis=1)
    center=xf.mean(0);xf-=center;xd-=center
    _,sing,right=np.linalg.svd(xf,full_matrices=False)
    rank=min(3,int((sing>1e-8).sum()))
    if rank==0:return None
    basis=right[:rank].T;z=xf@basis;zd=xd@basis
    scale=np.maximum(z.std(0),.05);z/=scale;zd/=scale
    fs=[z];ds=[zd];keys=dict(routes)
    for label,_,_,_ in [axis for axis in axes if not axis[0].startswith(('parent:','peer:'))][:2]:
        if label.startswith(('parent:','peer:')):continue
        key=keys[label];cells=int(key.max())+1
        count=np.bincount(key[rr],minlength=cells);supported=np.flatnonzero(count>=67)
        if len(supported)>16:continue
        one=(key[rr,None]==supported).astype(float);od=(key[dest,None]==supported).astype(float)
        one-=one.mean(0);od-=((key[rr,None]==supported).mean(0))
        fs.extend([one,(z[:,:,None]*one[:,None,:]).reshape(len(rr),-1)])
        ds.extend([od,(zd[:,:,None]*od[:,None,:]).reshape(len(dest),-1)])
    return np.concatenate(fs,axis=1),np.concatenate(ds,axis=1)


def _n_pair_routes(name,peers,routes,rr,yy,width):
    ranked=sorted(((_peer_mi(v[0][rr],yy,width),k,v[0]) for k,v in peers.items() if k!=name),key=lambda v:(-v[0],v[1]))[:2]
    pairs=[]
    if len(ranked)==2:pairs.append((ranked[0][1]+'&'+ranked[1][1],ranked[0][2],ranked[1][2]))
    given_routes=[r for r in routes if not r[0].startswith(('peer:','parent:'))]
    if ranked and given_routes:
        best=max(given_routes,key=lambda v:_peer_mi(v[1][rr],yy,width))
        pairs.append((ranked[0][1]+'&'+best[0],ranked[0][2],best[1]))
    out=[]
    for label,a,b in pairs:
        _,key=np.unique(a*(int(b.max())+1)+b,return_inverse=True)
        out.append(('pair:'+label,key))
    return out


def _n_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
    before,before_r=fo[name],fr[name]
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers)
    snapshot={k:v for k,v in snapshot.items() if k!=name};fits=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<150:return []
        if np.isfinite(deadline):
            local=[r for r in routes if not r[0].startswith('peer:')];raw_axes=_n_axis_pool(local,rr,dest,yy,p,pd,tau=60.);axes=(_n_select_axes(raw_axes,p,pd,yy) if job.get('_n_finite_precise',True) else _n_gram_schmidt(raw_axes[:3],p,pd))
        else:
            selected=set(_i_residual_peers(snapshot,rr,residual,4))
            local=[r for r in routes if not r[0].startswith('peer:') or r[0][5:] in selected]
            raw_axes=_n_axis_pool(local,rr,dest,yy,p,pd,tau=60.)
            axes=_n_select_axes(raw_axes,p,pd,yy)
        ranked={}
        if 'N1_ortho_additive' in channels and len(axes)>=2:
            for count in ((min(3,len(axes)),) if np.isfinite(deadline) and not job.get('_n_finite_precise',True) else (2,3)):
                if count>len(axes):continue
                for lam in ((120.,) if np.isfinite(deadline) and not job.get('_n_finite_precise',True) else (40.,120.)):
                    af,ad=_m_joint_fisher(axes[:count],p,pd,yy,lam)
                    for step in ((.65,) if np.isfinite(deadline) and not job.get('_n_finite_precise',True) else (.35,.65)):
                        score=float(_g_fit_gain(base,yy,step*af,gated).sum()/max(fit.sum(),1))
                        if score>0:ranked[('N1_ortho_additive','fisher',(('fit_selected',) if np.isfinite(deadline) else tuple(v[0] for v in axes[:count])),lam,step)]=(score,step*ad,np.ones(len(dest),bool))
        if 'N1_ortho_additive' in channels and (not np.isfinite(deadline) or job.get('_n_finite_precise',True)):
            for tau in (40.,120.):
                raw=_n_raw_axes(local,rr,dest,yy,p,pd,tau)
                ortho=_n_sum_orthogonal(raw,p,pd)
                for count in (2,3):
                    if count>len(ortho):continue
                    af=sum(v[1] for v in ortho[:count]);ad=sum(v[2] for v in ortho[:count])
                    identity=('fit_selected',) if np.isfinite(deadline) else tuple(v[0] for v in ortho[:count])
                    for step in (.35,.65,1.):
                        score=float(_g_fit_gain(base,yy,step*af,gated).sum()/max(fit.sum(),1))
                        if score>0:ranked[('N1_ortho_additive','gram_sum',identity,tau,step)]=(score,step*ad,np.ones(len(dest),bool))
        if 'N1_peer_pair' in channels:
            pairs=_n_pair_routes(name,peers,routes,rr,yy,width)
            pair_axes=_m_axes(pairs,rr,dest,yy,p,pd,tau=40.)
            if np.isfinite(deadline):pair_alias={v[0]:'fit_selected_pair'+str(i) for i,v in enumerate(pair_axes)}
            for label,a,b,mask in pair_axes:
                # Signed Fisher and simplex KKT share the same supported pair cells.
                key=dict(pairs)[label];codes=key[rr];dc=key[dest];cells=int(key.max())+1
                count=np.bincount(codes,minlength=cells);ok=count>=67
                one=np.eye(width)[yy];total=one.sum(0);prior=(total+1)/(len(yy)+width)
                pl=(total[None]+1-one)/(len(yy)+width-1)
                obs=np.column_stack([np.bincount(codes,weights=one[:,k],minlength=cells) for k in range(width)])
                q=(obs[codes]-one+40*pl)/(count[codes,None]-1+40);qd=(obs[dc]+40*prior)/(count[dc,None]+40)
                q[~ok[codes]]=p[~ok[codes]];qd[~ok[dc]]=pd[~ok[dc]]
                mass=base[:,:width].sum(1);scale=1-.001*job['K'];idx=np.arange(len(yy))
                aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                sol=_k_loo_kkt(aa,dd,codes,cells,deadline,maximum=.4)
                if sol is None:return []
                alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fit.sum(),1))
                if score>0:ranked[('N1_peer_pair',(pair_alias[label] if np.isfinite(deadline) else label),'kkt')]=(score,(1-alpha[dc,None])*pd+alpha[dc,None]*qd,ok[dc])
                for _,u,v,_ in (_n_gram_schmidt(raw_axes[:3],p,pd) if np.isfinite(deadline) else axes):
                    coef=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15));a-=coef*u;b-=coef*v
                af,ad=_m_joint_fisher([(label,a,b,mask)],p,pd,yy,40.)
                score=float(_g_fit_gain(base,yy,.5*af,gated).sum()/max(fit.sum(),1))
                if score>0:ranked[('N1_peer_pair',(pair_alias[label] if np.isfinite(deadline) else label),'signed')]=(score,.5*ad,ok[dc])
        if 'N2_battery_subspace' in channels:
            sub=_n_subspace(snapshot,rr,dest,residual,local,raw_axes,p,pd)
            if sub is not None:
                xf,xd=sub
                for lam in (80.,240.):
                    af,ad,_=_h_ridge_press(xf,xd,residual,lam)
                    af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                    for step in (.35,.65):
                        score=float(_g_fit_gain(base,yy,step*af,gated).sum()/max(fit.sum(),1))
                        if score>0:ranked[('N2_battery_subspace',lam,step)]=(score,step*ad,np.ones(len(dest),bool))
        if 'N2_expert_subspace' in channels and len(job['oof'])>2 and (not np.isfinite(deadline) or len(rr)*width*len(job['oof'])<150000):
            aa=[];bb=[]
            for oo,full in zip(job['oof'],job['all']):
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else oo[split==held],width)
                a=np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6));b=np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6))
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                aa.append(a);bb.append(b)
            energy=[float(np.sum(a[np.arange(len(yy)),yy]))**2/(float(np.sum(p*a*a))+80.) for a in aa]
            if np.isfinite(deadline):
                ids=sorted(range(len(aa)),key=lambda k:(-energy[k],k))[:3];ax=_n_gram_schmidt([(str(k),aa[k],bb[k],np.ones(len(dest),bool)) for k in ids],p,pd);ranked_arms=['fit_selected']
            else:
                ranked_arms=sorted(range(len(aa)),key=lambda k:(-energy[k],k))[:3]
                ax=_n_gram_schmidt([(str(k),aa[k],bb[k],np.ones(len(dest),bool)) for k in ranked_arms],p,pd)
            for lam in ((80.,240.) if ax else ()):
                af,ad=_m_joint_fisher(ax,p,pd,yy,lam)
                score=float(_g_fit_gain(base,yy,.5*af,gated).sum()/max(fit.sum(),1))
                if score>0:ranked[('N2_expert_subspace',tuple(ranked_arms),lam)]=(score,.5*ad,np.ones(len(dest),bool))
                convex=_n_floor_direction(base,p,pd,af,ad,yy,deadline)
                if convex is not None and convex[0]>0:ranked[('N2_expert_subspace',tuple(ranked_arms),lam,'kkt')]=(convex[0],convex[1],np.ones(len(dest),bool))
            given_routes=[v for v in local if not v[0].startswith(('peer:','parent:'))]
            chosen=sorted(given_routes,key=lambda v:(-_peer_mi(v[1][rr],yy,width),v[0]))[:2]
            if np.isfinite(deadline):given_alias={v[0]:'fit_selected_given'+str(i) for i,v in enumerate(chosen)}
            for label,key in chosen:
                c=key[rr];dc=key[dest];cells=int(max(c.max(),dc.max()))+1;n=np.bincount(c,minlength=cells);ok=n>=67
                if not ok.any():continue
                conditional=[]
                for a in aa:
                    r=np.bincount(c,weights=a[np.arange(len(yy)),yy],minlength=cells)
                    f=np.bincount(c,weights=np.sum(p*a*a,1),minlength=cells)
                    conditional.append(float(np.sum(r[ok]**2/(f[ok]+80.))))
                ids=sorted(sorted(range(len(aa)),key=lambda k:(-conditional[k],k))[:3])
                cell_axes=_n_gram_schmidt([(str(k),aa[k],bb[k],np.ones(len(dest),bool)) for k in ids],p,pd)
                if not cell_axes:continue
                for lam in (80.,240.):
                    if time.monotonic()>=deadline:return []
                    af,ad,mask=_n_macro_expert(cell_axes,p,pd,yy,c,dc,lam)
                    score=float(_g_fit_gain(base,yy,.5*af,gated).sum()/max(fit.sum(),1))
                    if score>0:ranked[('N2_expert_subspace','macro',(given_alias[label] if np.isfinite(deadline) else label),(('fit_selected',) if np.isfinite(deadline) else tuple(ids)),lam,'signed')]=(score,.5*ad,mask)
                    convex=_n_floor_direction(base,p,pd,af,ad,yy,deadline)
                    if convex is not None and convex[0]>0:ranked[('N2_expert_subspace','macro',(given_alias[label] if np.isfinite(deadline) else label),(('fit_selected',) if np.isfinite(deadline) else tuple(ids)),lam,'kkt')]=(convex[0],convex[1],mask)
        if np.isfinite(deadline) and 'N2_expert_subspace' in channels:
            # A fit-selected conditional estimator is a separate nested model:
            # each deletion fit chooses its own GIVEN direction by fit evidence.
            # Keep fixed-rank estimators too; neither can suppress the other.
            for spec,value in list(ranked.items()):
                if spec[0]=='N2_expert_subspace' and spec[1]=='macro':
                    adaptive=spec[:2]+('fit_selected_adaptive',)+spec[3:]
                    if adaptive not in ranked or value[0]>ranked[adaptive][0]:ranked[adaptive]=value
        if 'N3_entropy_skip' in channels:
            given_routes=[r for r in local if not r[0].startswith(('peer:','parent:'))]
            if given_routes:
                label,key=max(given_routes,key=lambda v:_peer_mi(v[1][rr],yy,width))
                entropy=-np.sum(p*np.log(np.maximum(p,1e-12)),1);ed=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1)
                cuts=np.quantile(entropy,[1/3,2/3]);c=key[rr]*3+np.searchsorted(cuts,entropy);dc=key[dest]*3+np.searchsorted(cuts,ed)
                basis=_g_basis(p);bd=_g_basis(pd);cells=int(max(c.max(),dc.max()))+1;n=np.bincount(c,minlength=cells);ok=n>=67
                gram=np.sum(p*basis*basis,1);rhs=basis[np.arange(len(yy)),yy]
                gs=np.bincount(c,weights=gram,minlength=cells);rs=np.bincount(c,weights=rhs,minlength=cells)
                for lam in (40.,120.):
                    theta=np.clip(rs/(gs+lam),-.65,.65);loo=np.clip((rs[c]-rhs)/(gs[c]-gram+lam),-.65,.65)
                    af=loo[:,None]*basis;ad=theta[dc,None]*bd;af[~ok[c]]=0
                    score=float(_g_fit_gain(base,yy,af,gated).sum()/max(fit.sum(),1))
                    if score>0:ranked[('N3_entropy_skip','temperature',label,lam)]=(score,ad,ok[dc])
            if gated:
                sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
                fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
                if len(rr)>=150 and given_routes:
                    label,key=max(given_routes,key=lambda v:_peer_mi(v[1][rr],yy,job['K']))
                    rho=base[:,-1];rd=db[:,-1];cuts=np.quantile(rho,[1/3,2/3]);c=key[rr]*3+np.searchsorted(cuts,rho);dc=key[dest]*3+np.searchsorted(cuts,rd)
                    cells=int(max(c.max(),dc.max()))+1;n=np.bincount(c,minlength=cells);ok=n>=67
                    r=residual[:,0];v=rho*(1-rho);rs=np.bincount(c,weights=r,minlength=cells);vs=np.bincount(c,weights=v,minlength=cells)
                    theta=np.clip(rs/(vs+40),-.65,.65);loo=np.clip((rs[c]-r)/(vs[c]-v+40),-.65,.65);loo[~ok[c]]=0
                    score=float(_h_skip_gain(base,yy,loo).sum()/max(fit.sum(),1))
                    if score>0:ranked[('N3_entropy_skip','skip',label,40.)]=(score,theta[dc,None],ok[dc]&(rd>.015)&(rd<.985))
                    response=(yy==width).astype(float);prior=(response.sum()+1)/(len(yy)+2);pl=(response.sum()+1-response)/(len(yy)+1)
                    obs=np.bincount(c,weights=response,minlength=cells)
                    q=(obs[c]-response+40*pl)/(n[c]-1+40);qd=(obs[dc]+40*prior)/(n[dc]+40)
                    q[~ok[c]]=rho[~ok[c]];qd[~ok[dc]]=rd[~ok[dc]]
                    candidate=_residual_skip_project(base,np.ones(len(base),bool),q);idx=np.arange(len(yy));scale=1-.001*job['K']
                    aa=.001+scale*base[idx,yy];dd=scale*(candidate[idx,yy]-base[idx,yy]);sol=_k_loo_kkt(aa,dd,c,cells,deadline,maximum=.4)
                    if sol is None:return []
                    alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fit.sum(),1));rho_dest=np.clip((1-alpha[dc])*rd+alpha[dc]*qd,1e-9,1-1e-9)
                    shift=np.log(rho_dest/(1-rho_dest))-np.log(np.clip(rd,1e-9,1-1e-9)/np.clip(1-rd,1e-9,1.))
                    if score>0:ranked[('N3_entropy_skip','skip_kkt',label,40.)]=(score,shift[:,None],ok[dc]&(rd>.015)&(rd<.985))
        fits.append(ranked)
    proposals=[]
    for op in channels:
        for so,sr,mo,mr,record in (_n_collect_finite if np.isfinite(deadline) else _n_collect)([{k:v for k,v in f.items() if k[0]==op} for f in fits],split,rows,rest):
            active=mo&source;hm=mr&hidden;spec=record['spec']
            skip=op=='N3_entropy_skip' and spec[1] in ('skip','skip_kkt')
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-(np.log(rho/(1-rho))+so[:,0]))))[active])
                qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-(np.log(rd/(1-rd))+sr[:,0]))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                simplex=op in ('N1_peer_pair','N2_expert_subspace') and spec[-1]=='kkt'
                project=_j_simplex_project if simplex else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            gain=np.log(.001+(1-.001*job['K'])*oo[np.arange(len(y)),y])-np.log(.001+(1-.001*job['K'])*before[np.arange(len(y)),y])
            fg=np.array([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=op,se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip)
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _orthogonal_gram_schmidt_and_battery_subspace_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Fourteenth stage: Fisher-orthogonal axes, battery PRESS, entropy strata."""
    channels=('N1_ortho_additive','N1_peer_pair','N2_battery_subspace','N2_expert_subspace','N3_entropy_skip')
    thresholds={'N1_ortho_additive':3.4,'N1_peer_pair':3.5,'N2_battery_subspace':3.8,'N2_expert_subspace':3.2,'N3_entropy_skip':3.8}
    if np.isfinite(deadline):
        return _n_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline-.30,peers,base_snapshot,thresholds)
    key_fo,key_fr=dict(fo),dict(fr)
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        if time.monotonic()>=deadline:return
        # Every earlier accepted direction owns its prediction unchanged.
        if not np.isfinite(deadline) and any(job.get(k) for k in ('pure_graft','answer_graft','residual_raking','peer_raking','ef_raking','f_raking','g_raking','h_raking','i_raking','j_raking','k_raking','l_raking','m_raking')):continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        accepted=[]
        for oo,qr,active,hm,record in _n_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,channels,key_fo,key_fr):
            kappa=thresholds[record['operator']];record['kappa']=kappa
            record['full_fit_amplification']=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15)
            passed=.25<=record['full_fit_amplification']<=(2.5 if np.isfinite(deadline) else 1.5) and _n_dominates(oo,before,y,split,active,kappa)
            probe=globals().get('_N_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted and time.monotonic()<deadline:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0])
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('n_raking',[]).append(record)
            if np.isfinite(deadline):return


def _o_m_finite_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,snapshot):
    """Complete cheap M estimators before the continuous M residual searches."""
    channels=('M1_marginal_simplex','M1_additive_eb','M1_ordinal')
    thresholds={'M1_marginal_simplex':2.4,'M1_additive_eb':9.7,'M1_ordinal':3.8}
    key_fo,key_fr=dict(fo),dict(fr)
    order=_s136_order(work,items,fo,X,given,levels,deadline)
    reserved={order[0][0]} if len(order)>1 else set()
    if any('_s138_scores' in j for j in work.values()):order=sorted(order,key=lambda v:(-v[1].get('_s138_scores',{}).get('J3',0.),v[0]))
    for name,job in order:
        if time.monotonic()>=deadline:return
        if name in reserved:continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        for oo,qr,active,hm,record in _m_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo,key_fr):
            kappa=thresholds[record['operator']];record['kappa']=kappa
            if _ef_dominates(oo,before,y,split,active,all_rows=True,kappa=kappa):accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted and time.monotonic()<deadline:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('m_raking',[]).append(record);return
    if time.monotonic()<deadline:
        return _additive_multi_effect_and_multi_arm_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,snapshot)


def _o_n_finite_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,snapshot):
    """Finish compact / bundles before expensive peer-cell KKT searches."""
    thresholds={'N1_ortho_additive':3.4,'N1_peer_pair':3.5,'N2_battery_subspace':3.8,'N2_expert_subspace':2.4,'N3_entropy_skip':3.4}
    order=_s136_order(work,items,fo,X,given,levels,deadline);signal=_o_finite_order(order,items,fo,X,given,fold,deadline)
    queue=[];seen=set();key_fo,key_fr=dict(fo),dict(fr)
    for pair in zip(order,signal):
        for entry in pair:
            if entry[0] not in seen:queue.append(entry);seen.add(entry[0])
    compact_stop=min(deadline-.15,time.monotonic()+max(1.,.55*(deadline-time.monotonic())))
    for name,job in queue[:24]:
        if time.monotonic()>=compact_stop:break
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is not None and 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        count=0 if source is None else int((source&(y<width)).sum());plans=[('N1_ortho_additive',)]
        if len(job['oof'])>2 and count*width*len(job['oof'])<150000:plans.append(('N2_expert_subspace',))
        for channels in plans:
            job['_n_finite_precise']=False if channels==('N1_ortho_additive',) else True
            local_stop=min(compact_stop,time.monotonic()+(1.7 if channels==('N2_expert_subspace',) else 1.0));accepted=[]
            for oo,qr,active,hm,record in _n_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,local_stop,channels,key_fo,key_fr):
                kappa=thresholds[record['operator']];record['kappa']=kappa;record['full_fit_amplification']=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15)
                passed=_n_dominates(oo,before,y,split,active,kappa)
                probe=globals().get('_N_PROBE')
                if probe is not None:probe(name,record,oo,qr,passed)
                if passed:
                    alpha=min(1.,1.5/max(record['full_fit_amplification'],1e-15));record['query_full_fit_shrinkage']=alpha
                    target=before_r.copy();target[hm]=before_r[hm]+alpha*(qr[hm]-before_r[hm]);qr=target
                    if record['skip']:qr=_residual_skip_project(before_r,hm,qr[hm,-1])
                    accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
            if accepted and time.monotonic()<deadline:
                _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('n_raking',[]).append(record);return
    if time.monotonic()<deadline:
        return _n_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot,thresholds)


def _o_canonical(labels):
    """Axis identity is a set; changing its enumeration changes no estimator."""
    return tuple(sorted(labels))


def _o_weighted_eb(w,wd,coarse,coarse_d,yy,p,pd,tau=60.):
    """Two-level Dirichlet moments with the entire respondent deleted.

    Both observation and expectation sufficient statistics lose w_i, including
    the coarse prior. Responsibilities are probabilities, never modal labels.
    """
    width=p.shape[1];one=np.eye(width)[yy];c=np.asarray(coarse,int);dc=np.asarray(coarse_d,int)
    cells=int(max(c.max(initial=0),dc.max(initial=0)))+1;n=np.bincount(c,minlength=cells)
    total=one.sum(0);global_p=(total+1)/(len(yy)+width)
    deleted_global=(total[None]+1-one)/(len(yy)+width-1)
    obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
    exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
    cp=(obs[c]-one+tau*deleted_global)/(n[c,None]-1+tau)
    ce=(exp[c]-p+tau*deleted_global)/(n[c,None]-1+tau)
    dp=(obs[dc]+tau*global_p)/(n[dc,None]+tau);de=(exp[dc]+tau*global_p)/(n[dc,None]+tau)
    cp[n[c]<67]=deleted_global[n[c]<67];ce[n[c]<67]=deleted_global[n[c]<67]
    dp[n[dc]<67]=global_p;de[n[dc]<67]=global_p
    support=w.sum(0);ob=w.T@one;ex=w.T@p;ok=support>=30
    if not ok.any():return None
    # Each destination belongs fractionally to all peer states; expectation and
    # observation use the same support. The mixture preserves total probability.
    q=np.zeros_like(p);e=np.zeros_like(p);qd=np.zeros_like(pd);ed=np.zeros_like(pd)
    active=np.zeros(len(p));ad=np.zeros(len(pd))
    for m in np.flatnonzero(ok):
        a=w[:,m,None];b=wd[:,m,None]
        qm=(ob[m]-a*one+tau*cp)/(support[m]-a+tau)
        em=(ex[m]-a*p+tau*ce)/(support[m]-a+tau)
        dm=(ob[m]+tau*dp)/(support[m]+tau);ee=(ex[m]+tau*de)/(support[m]+tau)
        q+=a*qm;e+=a*em;qd+=b*dm;ed+=b*ee
        active+=a[:,0];ad+=b[:,0]
    q+=(1-active[:,None])*p;e+=(1-active[:,None])*p
    qd+=(1-ad[:,None])*pd;ed+=(1-ad[:,None])*pd
    af=np.clip(np.log(np.maximum(q,1e-12)/np.maximum(e,1e-12)),-1.2,1.2)
    df=np.clip(np.log(np.maximum(qd,1e-12)/np.maximum(ed,1e-12)),-1.2,1.2)
    af-=(p*af).sum(1,keepdims=True);df-=(pd*df).sum(1,keepdims=True)
    return af,df,q,qd,active>1e-9,ad>1e-9


def _o_hierarchical_axes(routes,rr,dest,yy,p,pd,tau=60.):
    """Fine 30-row cells borrow from 67-row coarse cells, not empty cells."""
    ranked=_n_axis_pool(routes,rr,dest,yy,p,pd,tau);keys=dict(routes);out=[]
    for i in range(min(3,len(ranked))):
        for j in range(i+1,min(4,len(ranked))):
            labels=_o_canonical((ranked[i][0],ranked[j][0]));a,b=(keys[k] for k in labels)
            pair=a.astype(np.int64)*(int(b.max())+1)+b;_,codes=np.unique(pair,return_inverse=True)
            c=codes[rr];dc=codes[dest];cells=int(codes.max())+1
            n=np.bincount(c,minlength=cells);ok=n>=30
            if not ok.any() or int(ok[c].sum())<67:continue
            # The coarse axis is deterministic under permutations of the pair.
            coarse=a;cc=coarse[rr];cd=coarse[dest];nc=int(coarse.max())+1
            one=np.eye(p.shape[1])[yy];total=one.sum(0);prior=(total+1)/(len(yy)+p.shape[1]);pl=(total[None]+1-one)/(len(yy)+p.shape[1]-1)
            cn=np.bincount(cc,minlength=nc);co=np.column_stack([np.bincount(cc,weights=one[:,k],minlength=nc) for k in range(p.shape[1])]);ce=np.column_stack([np.bincount(cc,weights=p[:,k],minlength=nc) for k in range(p.shape[1])])
            cp=(co[cc]-one+tau*pl)/(cn[cc,None]-1+tau);ep=(ce[cc]-p+tau*pl)/(cn[cc,None]-1+tau)
            dp=(co[cd]+tau*prior)/(cn[cd,None]+tau);ed=(ce[cd]+tau*prior)/(cn[cd,None]+tau)
            cp[cn[cc]<67]=pl[cn[cc]<67];ep[cn[cc]<67]=pl[cn[cc]<67];dp[cn[cd]<67]=prior;ed[cn[cd]<67]=prior
            obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(p.shape[1])]);exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(p.shape[1])])
            q=(obs[c]-one+tau*cp)/(n[c,None]-1+tau);e=(exp[c]-p+tau*ep)/(n[c,None]-1+tau)
            qd=(obs[dc]+tau*dp)/(n[dc,None]+tau);ed=(exp[dc]+tau*ed)/(n[dc,None]+tau)
            af=np.clip(np.log(np.maximum(q,1e-12)/np.maximum(e,1e-12)),-1.2,1.2);ad=np.clip(np.log(np.maximum(qd,1e-12)/np.maximum(ed,1e-12)),-1.2,1.2)
            af[~ok[c]]=0;ad[~ok[dc]]=0;q[~ok[c]]=p[~ok[c]];qd[~ok[dc]]=pd[~ok[dc]]
            af-=(p*af).sum(1,keepdims=True);ad-=(pd*ad).sum(1,keepdims=True)
            out.append((labels,af,ad,q,qd,ok[c],ok[dc],c,dc,cells))
    return out


def _o_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,fit_count,gated,deadline,target=None,codes=None,dc=None,cells=1):
    if time.monotonic()>=deadline:return False
    if a is not None:
        for step in (.35,.65):
            gain=float(_g_fit_gain(base,yy,step*a,gated).sum()/fit_count)
            if gain>0:ranked[spec+('tangent',step)]=(gain,step*b,mask)
        norm=max(float(np.sqrt(np.mean(np.sum(p*a*a,axis=1)))),.02)
        aa,bb=_m_joint_fisher([(repr(spec),a/norm,b/norm,mask)],p,pd,yy,60.)
        for step in (.35,.65):
            gain=float(_g_fit_gain(base,yy,step*aa,gated).sum()/fit_count)
            if gain>0:ranked[spec+('fisher',step)]=(gain,step*bb,mask)
    if target is None and a is not None and not np.isfinite(deadline) and spec[0] in ('O2_press','O2_specialist'):
        q=p*np.exp(np.clip(a,-2.,2.));q/=q.sum(1,keepdims=True)
        qd=pd*np.exp(np.clip(b,-2.,2.));qd/=qd.sum(1,keepdims=True);target=(q,qd)
    if target is not None:
        q,qd=target;scale=1-.001*base.shape[1];idx=np.arange(len(yy));mass=base[:,:p.shape[1]].sum(1)
        aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
        z=dd/aa
        if float(z.sum())<2.*np.sqrt(float(np.sum(z*z))) or float(np.maximum(np.log1p(.4*z),0.).sum()/fit_count)<1e-5:return True
        codes=np.zeros(len(yy),int) if codes is None else codes;dc=np.zeros(len(pd),int) if dc is None else dc
        sol=_k_loo_kkt(aa,dd,codes,cells,deadline,maximum=.4)
        if sol is None:return False
        alpha,loo=sol;gain=float(np.log1p(loo*dd/aa).sum()/fit_count)
        if gain>0:ranked[spec+('kkt',)]=(gain,(1-alpha[dc,None])*pd+alpha[dc,None]*qd,mask)
    return True


def _o_features(p,pd,routes,rr,dest,residual,snapshot):
    """Endogenous polynomial/uncertainty crossed with fit-selected GIVEN."""
    def endogenous(q):
        s=np.sort(q,axis=1);margin=s[:,-1]-(s[:,-2] if q.shape[1]>1 else 0.)
        return np.column_stack((q[:,:6],q[:,:6]**2,-np.sum(q*np.log(np.maximum(q,1e-12)),1),margin))
    a=endogenous(p);b=endogenous(pd);xf=[a];xd=[b]
    for label,key in routes[:2]:
        n=np.bincount(key[rr]);top=np.argsort(-n,kind='stable')[:3]
        ia=(key[rr,None]==top).astype(float);ib=(key[dest,None]==top).astype(float)
        xf.extend([ia,(a[:,-2:,None]*ia[:,None,:]).reshape(len(rr),-1)]);xd.extend([ib,(b[:,-2:,None]*ib[:,None,:]).reshape(len(dest),-1)])
    selected=_i_residual_peers(snapshot,rr,residual,3)
    if selected:
        af=np.concatenate([snapshot[k][rr][:,:8] for k in selected],1);ad=np.concatenate([snapshot[k][dest][:,:8] for k in selected],1)
        center=af.mean(0);af=af-center;ad=ad-center
        _,s,v=np.linalg.svd(af,full_matrices=False);rank=min(3,int((s>1e-7).sum()))
        if rank:
            scale=np.maximum(s[:rank]/np.sqrt(len(rr)),.03);xf.append((af@v[:rank].T)/scale);xd.append((ad@v[:rank].T)/scale)
    a=np.column_stack(xf);b=np.column_stack(xd);scale=np.maximum(a.std(0),.05)
    return a/scale,b/scale


def _o_specialist(axes,p,pd,yy,c,dc,lam=120.):
    """Class-specific expert tangents; cell blocks shrink to scalar arm fit."""
    # A single arm can be precise for one class and imprecise elsewhere.
    aa=[];bb=[]
    for label,a,b,_ in axes[:3]:
        aa.append(a);bb.append(b)
        for k in range(min(p.shape[1],4)):
            u=np.zeros_like(a);v=np.zeros_like(b);u[:,k]=a[:,k];v[:,k]=b[:,k]
            u-=(p*u).sum(1,keepdims=True);v-=(pd*v).sum(1,keepdims=True)
            aa.append(u);bb.append(v)
    A=np.stack(aa,2);B=np.stack(bb,2);cov=np.einsum('nc,nca,ncb->nab',p,A,A,optimize=True);grad=A[np.arange(len(yy)),yy]
    G=cov.sum(0)+lam*np.eye(A.shape[2]);r=grad.sum(0);beta=np.linalg.solve(G,r)
    deleted=_s136_delete_fisher(G,r,cov,grad,A,p)
    out=np.einsum('nca,na->nc',A,deleted);dest=B@beta
    # Hierarchical shrinkage to scalar (first axis per arm) is part of each
    # deleted block, with its prior also recomputed after deleting that row.
    scalar=np.zeros(A.shape[2],bool);scalar[np.arange(0,A.shape[2],1+min(p.shape[1],4))]=True
    prior=beta*scalar;prior_loo=deleted*scalar
    for cell in np.unique(c):
        ix=np.flatnonzero(c==cell);di=np.flatnonzero(dc==cell)
        if len(ix)<30:continue
        g=cov[ix].sum(0)+lam*np.eye(A.shape[2]);rhs=grad[ix].sum(0)+lam*prior
        coef=np.linalg.solve(g,rhs)
        dl=np.linalg.solve(g[None]-cov[ix],(grad[ix].sum(0)[None]-grad[ix]+lam*prior_loo[ix])[...,None])[...,0]
        out[ix]=np.einsum('nca,na->nc',A[ix],dl);dest[di]=B[di]@coef
    return np.clip(out,-1.,1.),np.clip(dest,-1.,1.)


def _o_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers)
    snap={k:v for k,v in snapshot.items() if k!=name};fits=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_j_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=_i_residual_peers(snap,rr,residual,2 if bounded else 4)
        local=[v for v in routes if not v[0].startswith('peer:') or v[0][5:] in selected]
        raw=_n_axis_pool(local,rr,dest,yy,p,pd);axes=_n_select_axes(raw,p,pd,yy);ranked={};den=max(int(fit.sum()),1)
        if 'O1_symmetric' in channels:
            pools=[('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,60.),60.)]
            if bounded:pools=[('joint',axes,40.),('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,40.),40.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,120.),120.)]
            for kind,pool,strength in pools:
                for count in (2,3):
                    if count>len(pool):continue
                    chosen=sorted(pool[:count],key=lambda v:v[0]);ident=_o_canonical(v[0] for v in chosen)
                    if kind=='joint':chosen=sorted([v for v in raw if v[0] in ident],key=lambda v:v[0])
                    else:chosen=_n_sum_orthogonal(chosen,p,pd)
                    if len(chosen)<2:continue
                    if kind=='joint':a,b=_m_joint_fisher(chosen,p,pd,yy,strength)
                    else:a=sum(v[1] for v in chosen);b=sum(v[2] for v in chosen)
                    if bounded:ident=('fit_selected_subspace',)
                    for step in ((.65,) if bounded else (.35,.65,1.)):
                        score=float(_g_fit_gain(base,yy,step*a,gated).sum()/den)
                        if score>0:ranked[(('O1_symmetric',kind,ident,strength,step) if bounded else ('O1_symmetric',kind,ident,step))]=(score,step*b,np.ones(len(dest),bool))
        if 'O1_hierarchical' in channels:
            for labels,a,b,q,qd,mask,dm,c,dc,cells in _o_hierarchical_axes(local,rr,dest,yy,p,pd):
                if not _o_fisher_kkt(ranked,('O1_hierarchical',labels),base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd),c,dc,cells):return []
        if 'O1_soft' in channels:
            coarse_labels=[v[0] for v in raw if not v[0].startswith('peer:')][:2];coarse=dict(local)
            peer_keys=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))
            for key in peer_keys[:(1 if bounded else 2)]:
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,rank];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for label in ['global']+coarse_labels[:1]:
                    c=np.zeros(len(rr),int) if label=='global' else coarse[label][rr];dc=np.zeros(len(dest),int) if label=='global' else coarse[label][dest]
                    variants=[('soft',W[rr],W[dest])]
                    # Probability quantiles retain signal when every modal state agrees.
                    if label=='global':
                        edges=np.unique(np.quantile(q[rr,rank[0]],[1/3,2/3]));codes=np.searchsorted(edges,q[:,rank[0]]);eye=np.eye(len(edges)+1)
                        variants.append(('quantiles',eye[codes[rr]],eye[codes[dest]]))
                    else:
                        counts=np.bincount(c);top=np.argsort(-counts,kind='stable')[:3]
                        a=(c[:,None]==top).astype(float);b=(dc[:,None]==top).astype(float)
                        variants.append(('soft_given',(W[rr,:,None]*a[:,None,:]).reshape(len(rr),-1),(W[dest,:,None]*b[:,None,:]).reshape(len(dest),-1)))
                    for variant,a,b in variants:
                        eb=_o_weighted_eb(a,b,c,dc,yy,p,pd)
                        if eb is None:continue
                        af,ad,qf,qd,mask,dm=eb
                        if int(mask.sum())<67:continue
                        if not _o_fisher_kkt(ranked,('O1_soft',variant,key,label),base,p,pd,yy,af,ad,dm,den,gated,deadline,(qf,qd)):return []
        if 'O2_press' in channels:
            ordered=sorted([v for v in local if v[0] in {a[0] for a in raw[:3]} and not v[0].startswith(('peer:','parent:'))],key=lambda v:v[0]);xf,xd=_o_features(p,pd,ordered,rr,dest,residual,snap)
            for lam in ((180.,) if bounded else (60.,180.,540.)):
                a,b,_=_h_ridge_press(xf,xd,residual,lam);a=np.clip(a/np.maximum(p,.04),-1.,1.);b=np.clip(b/np.maximum(pd,.04),-1.,1.)
                if not _o_fisher_kkt(ranked,('O2_press',lam),base,p,pd,yy,a,b,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O2_specialist' in channels and len(job['oof'])>1 and len(rr)*width*len(job['oof'])<180000:
            aa=[];bb=[]
            for oo,full in zip(job['oof'][:6],job['all'][:6]):
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else oo[split==held],width)
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True);aa.append(a);bb.append(b)
            energy=[float(a[np.arange(len(yy)),yy].sum())**2/(float(np.sum(p*a*a))+120.) for a in aa];order=np.argsort(-np.asarray(energy),kind='stable')[:2];expert=[('arm'+str(k),aa[k],bb[k],np.ones(len(dest),bool)) for k in order]
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            for stratum in ('global','entropy'):
                af,ad=_o_specialist(expert,p,pd,yy,np.zeros(len(p),int) if stratum=='global' else c,np.zeros(len(pd),int) if stratum=='global' else dc)
                if not _o_fisher_kkt(ranked,('O2_specialist',stratum,_o_canonical(v[0] for v in expert)),base,p,pd,yy,af,ad,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fit_s,use_s,rr_s,ys,bs,_,dest_s,ds,_,rs=_j_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rr_s)>=67:
                rho=bs[:,-1];rd=ds[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int)
                for key in list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))[:2]:
                    v=snap[key];sk=v[:,-1] if items[key].get('gate') else v.max(1);ws=np.column_stack((1-sk,sk))
                    for label in ['global']+[v[0] for v in raw[:1] if v[0] in dict(local)]:
                        ck=np.zeros(len(X),int) if label=='global' else dict(local)[label];eb=_o_weighted_eb(ws[rr_s],ws[dest_s],ck[rr_s],ck[dest_s],yb,ps,pds)
                        if eb is None:continue
                        a,b,q,qd,mask,dm=eb;af=(a[:,1]-a[:,0]);ad=(b[:,1]-b[:,0]);fisher=rho*(1-rho)*af*af;grad=(yb-rho)*af
                        theta=np.clip((grad.sum()-grad)/(fisher.sum()-fisher+60.),-1.,1.);coef=np.clip(grad.sum()/(fisher.sum()+60.),-1.,1.)
                        for step in (.35,.65):
                            score=float(_h_skip_gain(bs,ys,step*theta*af).sum()/max(fit_s.sum(),1))
                            if score>0:ranked[('O3_skip',key,label,'fisher',step)]=(score,(step*coef*ad)[:,None],(rd>.015)&(rd<.985))
                        # The official floor includes answer mass and skip mass.
                        target=_residual_skip_project(bs,np.ones(len(bs),bool),q[:,1]);idx=np.arange(len(ys));scale=1-.001*job['K'];a0=.001+scale*bs[idx,ys];d0=scale*(target[idx,ys]-bs[idx,ys])
                        z=d0/a0
                        if float(z.sum())<2.*np.sqrt(float(np.sum(z*z))):continue
                        sol=_k_loo_kkt(a0,d0,np.zeros(len(ys),int),1,deadline,maximum=.4)
                        if sol is None:return []
                        alpha,deleted=sol;score=float(np.log1p(deleted*d0/a0).sum()/max(fit_s.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd[:,1],1e-12,1-1e-12)
                        if score>0:ranked[('O3_skip',key,label,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        fits.append(ranked)
    proposals=[]
    families=sorted({(s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1]) for s in fits[-1]},key=repr)
    for family in families:
        ff=[{s:v for s,v in f.items() if (s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1])==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];op=spec[0];skip=op=='O3_skip';active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=op,se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,canonical_axes=True)
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _o_finite_order(order,items,fo,X,given,fold,deadline):
    """Rank specialist work by fit-visible class Fisher energy, not variance."""
    ranked=[];stop=min(deadline-.1,time.monotonic()+.8)
    for name,job in order:
        if time.monotonic()>=stop:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')))
        source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        use=source&(job['y'][rows]<width);y=job['y'][rows][use]
        if len(y)<67:continue
        p=_g_conditional(fo[name][use],width);residual=np.eye(width)[y]-p;energy=0.
        for arm in job['oof'][:6]:
            q=_g_conditional(arm[use],width);a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True)
            gradient=(residual*a).sum(0);fisher=(p*(1-p)*a*a).sum(0)
            energy=max(energy,float(np.sum(gradient**2/(fisher+120.))))
        ranked.append((energy/len(rows)/max(1.,width/4.),name,job))
    included={v[1] for v in ranked}
    return [(name,job) for _,name,job in sorted(ranked,key=lambda v:(-v[0],v[1]))]+[(name,job) for name,job in order if name not in included]


def _o_press_order(order,items,fo,X,given,levels,deadline):
    """A cheap deleted PRESS screen ranks useful complete fits ahead of noise."""
    ranked=[];stop=min(deadline-.15,time.monotonic()+1.25)
    for name,job in order[:32]:
        if time.monotonic()>=stop:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')))
        source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        use=source&(job['y'][rows]<width);yy=job['y'][rows][use];rr=rows[use]
        if len(rr)<67:continue
        base=fo[name][use];p=_g_conditional(base,width);residual=np.eye(width)[yy]-p
        local=[(v,X[:,i]) for i,v in enumerate(given) if 2<=int(levels[i])<=128]
        raw=_n_axis_pool(local,rr,rr,yy,p,p);labels={v[0] for v in raw[:3]}
        xf,xd=_o_features(p,p,[v for v in local if v[0] in labels],rr,rr,residual,{})
        a,b,_=_h_ridge_press(xf,xd,residual,180.);a=np.clip(a/np.maximum(p,.04),-1.,1.)
        score=max(float(_g_fit_gain(base,yy,step*a,bool(items[name].get('gate'))).sum()/len(rows)) for step in (.35,.65))
        ranked.append((score,name,job))
    included={v[1] for v in ranked}
    return [(name,job) for _,name,job in sorted(ranked,key=lambda v:(-v[0],v[1]))]+[(name,job) for name,job in order if name not in included]


def _o_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot,thresholds):
    """Complete fold bundles are committed atomically inside the finite window."""
    thresholds=dict(thresholds,O1_symmetric=2.4,O2_press=2.8,O2_specialist=2.8)
    order=_s136_order(work,items,fo,X,given,levels,deadline);key_fo,key_fr=dict(fo),dict(fr)
    signal_order=_o_finite_order(order,items,fo,X,given,fold,deadline)
    press_order=_o_press_order(signal_order,items,fo,X,given,levels,deadline);queue=[];seen=set()
    for pair in zip(press_order,signal_order):
        for entry in pair:
            if entry[0] not in seen:queue.append(entry);seen.add(entry[0])
    for name,job in queue:
        if time.monotonic()>=deadline-.1:return
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
        plans=[('O1_symmetric','O2_press')]
        source=_wave_two_source(name,items,X,given,rows)
        if source is not None and 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        if len(job['oof'])>1 and source is not None and int((source&(y<job['K']-int(bool(items[name].get('gate'))))).sum())<=4000:plans.append(('O2_specialist',))
        for channels in plans:
            accepted=[]
            local_stop=min(deadline-.08,time.monotonic()+2.5)
            for oo,qr,active,hm,record in _o_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,local_stop,channels,key_fo,key_fr):
                passed=_n_dominates(oo,before,y,split,active,thresholds[record['operator']]);record['kappa']=thresholds[record['operator']]
                probe=globals().get('_O_PROBE')
                if probe is not None:probe(name,record,oo,qr,passed)
                if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
            if accepted and time.monotonic()<deadline:
                _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('o_raking',[]).append(record);return


def _soft_responsibility_rao_blackwell_and_subspace_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Fifteenth stage: soft responsibilities, canonical EB and subspace PRESS."""
    thresholds={'O1_symmetric':3.8,'O1_hierarchical':3.2,'O1_soft':3.8,'O2_press':3.8,'O2_specialist':3.8,'O3_skip':3.8}
    if np.isfinite(deadline):return _o_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,base_snapshot,thresholds)
    channels=tuple(thresholds);key_fo,key_fr=dict(fo),dict(fr)
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        if any(job.get(k) for k in ('pure_graft','answer_graft','residual_raking','peer_raking','ef_raking','f_raking','g_raking','h_raking','i_raking','j_raking','k_raking','l_raking','m_raking','n_raking')):continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        for oo,qr,active,hm,record in _o_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,channels,key_fo,key_fr):
            kappa=thresholds[record['operator']];record['kappa']=kappa;amplification=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15);record['full_fit_amplification']=amplification
            passed=.25<=amplification<=2.5 and _n_dominates(oo,before,y,split,active,kappa)
            if passed:
                alpha=min(1.,1.5/max(amplification,1e-15));record['query_full_fit_shrinkage']=alpha
                out=before_r.copy();out[hm]=before_r[hm]+alpha*(qr[hm]-before_r[hm]);qr=out
                if record['skip']:qr=_residual_skip_project(before_r,hm,qr[hm,-1])
            probe=globals().get('_O_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('o_raking',[]).append(record)


def _p_weighted_eb(w,wd,coarse,coarse_d,yy,p,pd,tau=60.):
    """Two-level Dirichlet moments with the entire respondent deleted.

    Both observation and expectation sufficient statistics lose w_i, including
    the coarse prior. Responsibilities are probabilities, never modal labels.
    """
    tau=tau*max(1.,p.shape[1]/2.)
    width=p.shape[1];one=np.eye(width)[yy];c=np.asarray(coarse,int);dc=np.asarray(coarse_d,int)
    cells=int(max(c.max(initial=0),dc.max(initial=0)))+1;n=np.bincount(c,minlength=cells)
    total=one.sum(0);global_p=(total+1)/(len(yy)+width)
    deleted_global=(total[None]+1-one)/(len(yy)+width-1)
    obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
    exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
    cp=(obs[c]-one+tau*deleted_global)/(n[c,None]-1+tau)
    ce=(exp[c]-p+tau*deleted_global)/(n[c,None]-1+tau)
    dp=(obs[dc]+tau*global_p)/(n[dc,None]+tau);de=(exp[dc]+tau*global_p)/(n[dc,None]+tau)
    cp[n[c]<67]=deleted_global[n[c]<67];ce[n[c]<67]=deleted_global[n[c]<67]
    dp[n[dc]<67]=global_p;de[n[dc]<67]=global_p
    support=w.sum(0);ob=w.T@one;ex=w.T@p;ok=support>=20
    if not ok.any():return None
    # Each destination belongs fractionally to all peer states; expectation and
    # observation use the same support. The mixture preserves total probability.
    q=np.zeros_like(p);e=np.zeros_like(p);qd=np.zeros_like(pd);ed=np.zeros_like(pd)
    active=np.zeros(len(p));ad=np.zeros(len(pd))
    for m in np.flatnonzero(ok):
        a=w[:,m,None];b=wd[:,m,None]
        qm=(ob[m]-a*one+tau*cp)/(support[m]-a+tau)
        em=(ex[m]-a*p+tau*ce)/(support[m]-a+tau)
        dm=(ob[m]+tau*dp)/(support[m]+tau);ee=(ex[m]+tau*de)/(support[m]+tau)
        q+=a*qm;e+=a*em;qd+=b*dm;ed+=b*ee
        active+=a[:,0];ad+=b[:,0]
    q+=(1-active[:,None])*p;e+=(1-active[:,None])*p
    qd+=(1-ad[:,None])*pd;ed+=(1-ad[:,None])*pd
    af=np.clip(np.log(np.maximum(q,1e-12)/np.maximum(e,1e-12)),-1.2,1.2)
    df=np.clip(np.log(np.maximum(qd,1e-12)/np.maximum(ed,1e-12)),-1.2,1.2)
    af-=(p*af).sum(1,keepdims=True);df-=(pd*df).sum(1,keepdims=True)
    return af,df,q,qd,active>1e-9,ad>1e-9


def _p_orthogonal(axes,p,pd,yy,count=4):
    """Fit-ranked Fisher Gram-Schmidt; the fitted transform follows each axis."""
    chosen=[];available=list(axes)
    while available and len(chosen)<count:
        scored=[]
        for i,(label,a,b,mask) in enumerate(available):
            a=a.copy();b=b.copy()
            for _,u,v,_ in chosen:
                c=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15));a-=c*u;b-=c*v
            norm=float(np.sqrt(np.mean(np.sum(p*a*a,1))))
            if norm<1e-7:continue
            a/=norm;b/=norm;f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy]
            beta=np.clip((g.sum()-g)/(f.sum()-f+60.),-1.,1.)
            q=p*np.exp(np.clip(beta[:,None]*a,-2.,2.));q/=np.maximum(q.sum(1,keepdims=True),1e-300)
            score=float(np.log(np.maximum(q[np.arange(len(yy)),yy],1e-12)/np.maximum(p[np.arange(len(yy)),yy],1e-12)).mean())
            scored.append((score,-i,i,label,a,b,mask))
        if not scored:break
        score,_,i,label,a,b,mask=max(scored,key=lambda v:v[:2])
        if score<=0:break
        chosen.append((label,a,b,mask));available.pop(i)
    return chosen


def _p_emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
    """Deleted tangent/Fisher and exact official-floor simplex KKT estimators."""
    if target is None:
        q=p*np.exp(np.clip(a,-2.,2.));q/=np.maximum(q.sum(1,keepdims=True),1e-300)
        qd=pd*np.exp(np.clip(b,-2.,2.));qd/=np.maximum(qd.sum(1,keepdims=True),1e-300);target=(q,qd)
    return _o_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target)


def _p_cells(c,dc,coarse,coarse_d,yy,p,pd,tau=60.,minimum=25):
    """Fine cells borrow a deleted 67-row coarse prior and a deleted global prior."""
    tau=tau*max(1.,p.shape[1]/2.)
    width=p.shape[1];one=np.eye(width)[yy];total=one.sum(0)
    prior=(total+1)/(len(yy)+width);pl=(total[None]+1-one)/(len(yy)+width-1)
    cells=int(max(c.max(initial=0),dc.max(initial=0)))+1;nc=int(max(coarse.max(initial=0),coarse_d.max(initial=0)))+1
    n=np.bincount(c,minlength=cells);cn=np.bincount(coarse,minlength=nc);ok=n>=minimum
    if int(ok[c].sum())<67:return None
    obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
    exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
    co=np.column_stack([np.bincount(coarse,weights=one[:,k],minlength=nc) for k in range(width)])
    ce=np.column_stack([np.bincount(coarse,weights=p[:,k],minlength=nc) for k in range(width)])
    cp=(co[coarse]-one+tau*pl)/(cn[coarse,None]-1+tau);ep=(ce[coarse]-p+tau*pl)/(cn[coarse,None]-1+tau)
    dp=(co[coarse_d]+tau*prior)/(cn[coarse_d,None]+tau);ed=(ce[coarse_d]+tau*prior)/(cn[coarse_d,None]+tau)
    cp[cn[coarse]<67]=pl[cn[coarse]<67];ep[cn[coarse]<67]=pl[cn[coarse]<67]
    dp[cn[coarse_d]<67]=prior;ed[cn[coarse_d]<67]=prior
    q=(obs[c]-one+tau*cp)/(n[c,None]-1+tau);e=(exp[c]-p+tau*ep)/(n[c,None]-1+tau)
    qd=(obs[dc]+tau*dp)/(n[dc,None]+tau);ee=(exp[dc]+tau*ed)/(n[dc,None]+tau)
    a=np.clip(np.log(np.maximum(q,1e-12)/np.maximum(e,1e-12)),-1.2,1.2);b=np.clip(np.log(np.maximum(qd,1e-12)/np.maximum(ee,1e-12)),-1.2,1.2)
    a[~ok[c]]=0;b[~ok[dc]]=0;q[~ok[c]]=p[~ok[c]];qd[~ok[dc]]=pd[~ok[dc]]
    a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
    return a,b,q,qd,ok[c],ok[dc]


def _p_peer_features(selected,snapshot,routes,rr,dest,p,pd,residual):
    """Centered peer-by-GIVEN Kronecker blocks, entropy and continuous risk."""
    af=[];ad=[]
    for key in selected[:4]:
        v=snapshot[key];center=v[rr].mean(0);a=v[rr]-center;b=v[dest]-center
        rank=np.argsort(-np.sum((a.T@residual)**2,axis=1),kind='stable')[:min(4,v.shape[1])]
        af.append(a[:,rank]);ad.append(b[:,rank])
    hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1)
    # Interactions are centered with exactly the fit mean, including Query rows.
    a0=np.column_stack(af+[p[:,:6],hf[:,None]]);b0=np.column_stack(ad+[pd[:,:6],hd[:,None]])
    center=a0.mean(0);a=a0-center;b=b0-center;xf=[a];xd=[b]
    for _,key in routes[:3]:
        counts=np.bincount(key[rr]);top=np.argsort(-counts,kind='stable')[:4]
        ia=(key[rr,None]==top).astype(float);ib=(key[dest,None]==top).astype(float)
        xf.extend([ia,(a[:,:,None]*ia[:,None,:]).reshape(len(rr),-1)])
        xd.extend([ib,(b[:,:,None]*ib[:,None,:]).reshape(len(dest),-1)])
    a=np.column_stack(xf);b=np.column_stack(xd);scale=np.maximum(a.std(0),.05)
    return a/scale,b/scale


def _p_specialist_diagonal(arms,p,pd,yy,c,dc,lam=120.):
    """All-class diagonal Fisher reliability shrunk to deleted scalar arm weights."""
    out=np.zeros_like(p);dest=np.zeros_like(pd);one=np.eye(p.shape[1])[yy]
    for _,a,b,_ in arms:
        grad=(one-p)*a;fisher=p*(1-p)*a*a;g=grad.sum(1);f=np.sum(p*a*a,1)
        scalar=g.sum()/(f.sum()+lam);sl=(g.sum()-g)/(f.sum()-f+lam)
        coef=(grad.sum(0)+lam*scalar)/(fisher.sum(0)+lam)
        deleted=(grad.sum(0)[None]-grad+lam*sl[:,None])/(fisher.sum(0)[None]-fisher+lam)
        af=a*deleted;bd=b*coef
        for cell in np.unique(c):
            ix=np.flatnonzero(c==cell);di=np.flatnonzero(dc==cell)
            if len(ix)<67:continue
            cf=(grad[ix].sum(0)+lam*coef)/(fisher[ix].sum(0)+lam)
            cl=(grad[ix].sum(0)[None]-grad[ix]+lam*deleted[ix])/(fisher[ix].sum(0)[None]-fisher[ix]+lam)
            af[ix]=a[ix]*cl;bd[di]=b[di]*cf
        out+=af/len(arms);dest+=bd/len(arms)
    out-=(p*out).sum(1,keepdims=True);dest-=(pd*dest).sum(1,keepdims=True)
    return np.clip(out,-1.,1.),np.clip(dest,-1.,1.)


def _p_hierarchical_axes(routes,rr,dest,yy,p,pd,tau=60.):
    """Fine 25-row cells borrow from 67-row coarse cells, not empty cells."""
    tau=tau*max(1.,p.shape[1]/2.)
    ranked=_n_axis_pool(routes,rr,dest,yy,p,pd,tau);keys=dict(routes);out=[]
    for i in range(min(3,len(ranked))):
        for j in range(i+1,min(4,len(ranked))):
            labels=_o_canonical((ranked[i][0],ranked[j][0]));a,b=(keys[k] for k in labels)
            pair=a.astype(np.int64)*(int(b.max())+1)+b;_,codes=np.unique(pair,return_inverse=True)
            c=codes[rr];dc=codes[dest];cells=int(codes.max())+1
            n=np.bincount(c,minlength=cells);ok=n>=25
            if not ok.any() or int(ok[c].sum())<67:continue
            # The coarse axis is deterministic under permutations of the pair.
            coarse=a;cc=coarse[rr];cd=coarse[dest];nc=int(coarse.max())+1
            one=np.eye(p.shape[1])[yy];total=one.sum(0);prior=(total+1)/(len(yy)+p.shape[1]);pl=(total[None]+1-one)/(len(yy)+p.shape[1]-1)
            cn=np.bincount(cc,minlength=nc);co=np.column_stack([np.bincount(cc,weights=one[:,k],minlength=nc) for k in range(p.shape[1])]);ce=np.column_stack([np.bincount(cc,weights=p[:,k],minlength=nc) for k in range(p.shape[1])])
            cp=(co[cc]-one+tau*pl)/(cn[cc,None]-1+tau);ep=(ce[cc]-p+tau*pl)/(cn[cc,None]-1+tau)
            dp=(co[cd]+tau*prior)/(cn[cd,None]+tau);ed=(ce[cd]+tau*prior)/(cn[cd,None]+tau)
            cp[cn[cc]<67]=pl[cn[cc]<67];ep[cn[cc]<67]=pl[cn[cc]<67];dp[cn[cd]<67]=prior;ed[cn[cd]<67]=prior
            obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(p.shape[1])]);exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(p.shape[1])])
            q=(obs[c]-one+tau*cp)/(n[c,None]-1+tau);e=(exp[c]-p+tau*ep)/(n[c,None]-1+tau)
            qd=(obs[dc]+tau*dp)/(n[dc,None]+tau);ed=(exp[dc]+tau*ed)/(n[dc,None]+tau)
            af=np.clip(np.log(np.maximum(q,1e-12)/np.maximum(e,1e-12)),-1.2,1.2);ad=np.clip(np.log(np.maximum(qd,1e-12)/np.maximum(ed,1e-12)),-1.2,1.2)
            af[~ok[c]]=0;ad[~ok[dc]]=0;q[~ok[c]]=p[~ok[c]];qd[~ok[dc]]=pd[~ok[dc]]
            af-=(p*af).sum(1,keepdims=True);ad-=(pd*ad).sum(1,keepdims=True)
            out.append((labels,af,ad,q,qd,ok[c],ok[dc],c,dc,cells))
    return out


def _p_fold_context(rows,rest,y,split,best,query,source,gated,held,skip):
    """Every outer fit predicts its held respondents and the fixed Query rows."""
    values=list(_j_fold_context(rows,rest,y,split,best,query,source,gated,held,skip))
    if held is not None:
        values[6]=np.concatenate((values[6],rest))
        values[7]=np.concatenate((values[7],query),axis=0)
        values[8]=_g_conditional(values[7],best.shape[1]-int(gated))
    if not skip:
        width=best.shape[1]-int(gated);yy=values[3];one=np.eye(width)[yy]
        p=values[5].copy();pd=values[8].copy();empty=p.sum(1)<=1e-12;empty_d=pd.sum(1)<=1e-12
        # An undefined conditional is a neutral simplex reference, independent
        # of labels. All observed responses still enter the deleted EB counts.
        p[empty]=1./width;pd[empty_d]=1./width
        values[5],values[8],values[9]=p,pd,one-p
    return tuple(values)


def _p_fit_destinations(ranked,held,split,bag=True):
    if held is None:return ranked,ranked
    if not bag:return ranked,None
    count=int((split==held).sum())
    held_map={spec:(score,a[:count],m[:count]) for spec,(score,a,m) in ranked.items()}
    query_map={spec:(score,a[count:],m[count:]) for spec,(score,a,m) in ranked.items()}
    return held_map,query_map


def _p_query_average(query_fits,spec,before,gated,skip):
    """Uniform six-fit pool; unsupported folds contribute the unchanged base."""
    spec=tuple(spec)
    simplex=spec[-1]=='kkt' and not skip
    baseline=_g_conditional(before,before.shape[1]-int(gated)) if simplex else np.zeros_like(query_fits[-1][spec][1])
    total=np.zeros_like(baseline);support=np.zeros(len(before),int)
    for fit in query_fits:
        _,value,mask=fit[spec]
        total+=np.where(mask[:,None],value,baseline);support+=mask
    return total/len(query_fits),(support>=4)&query_fits[-1][spec][2]


def _p_recovered_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers)
    snap={k:v for k,v in snapshot.items() if k!=name};fits=[];query_fits=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=(_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=_i_residual_peers(snap,rr,residual,2 if bounded else 4)
        local=[v for v in routes if not v[0].startswith('peer:') or v[0][5:] in selected]
        raw=_n_axis_pool(local,rr,dest,yy,p,pd,60.*max(1.,width/2.));axes=_n_select_axes(raw,p,pd,yy);ranked={};den=max(int(fit.sum()),1)
        if 'O1_symmetric' in channels:
            pools=[('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,60.*max(1.,width/2.)),60.)]
            if bounded:pools=[('joint',axes,40.),('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,40.),40.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,120.),120.)]
            for kind,pool,strength in pools:
                for count in (2,3):
                    if count>len(pool):continue
                    chosen=sorted(pool[:count],key=lambda v:v[0]);ident=_o_canonical(v[0] for v in chosen)
                    if kind=='joint':chosen=sorted([v for v in raw if v[0] in ident],key=lambda v:v[0])
                    else:chosen=_n_sum_orthogonal(chosen,p,pd)
                    if len(chosen)<2:continue
                    if kind=='joint':a,b=_m_joint_fisher(chosen,p,pd,yy,strength)
                    else:a=sum(v[1] for v in chosen);b=sum(v[2] for v in chosen)
                    if bounded:ident=('fit_selected_subspace',)
                    for step in ((.65,) if bounded else (.35,.65,1.)):
                        score=float(_g_fit_gain(base,yy,step*a,gated).sum()/den)
                        if score>0:ranked[(('O1_symmetric',kind,ident,strength,step) if bounded else ('O1_symmetric',kind,ident,step))]=(score,step*b,np.ones(len(dest),bool))
        if 'O1_hierarchical' in channels:
            for labels,a,b,q,qd,mask,dm,c,dc,cells in _p_hierarchical_axes(local,rr,dest,yy,p,pd):
                if not _o_fisher_kkt(ranked,('O1_hierarchical',labels),base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd),c,dc,cells):return []
        if 'O1_soft' in channels:
            coarse_labels=[v[0] for v in raw if not v[0].startswith('peer:')][:2];coarse=dict(local)
            peer_keys=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))
            for key in peer_keys[:(1 if bounded else 2)]:
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,rank];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for label in ['global']+coarse_labels[:1]:
                    c=np.zeros(len(rr),int) if label=='global' else coarse[label][rr];dc=np.zeros(len(dest),int) if label=='global' else coarse[label][dest]
                    variants=[('soft',W[rr],W[dest])]
                    # Probability quantiles retain signal when every modal state agrees.
                    if label=='global':
                        edges=np.unique(np.quantile(q[rr,rank[0]],[1/3,2/3]));codes=np.searchsorted(edges,q[:,rank[0]]);eye=np.eye(len(edges)+1)
                        variants.append(('quantiles',eye[codes[rr]],eye[codes[dest]]))
                    else:
                        counts=np.bincount(c);top=np.argsort(-counts,kind='stable')[:3]
                        a=(c[:,None]==top).astype(float);b=(dc[:,None]==top).astype(float)
                        variants.append(('soft_given',(W[rr,:,None]*a[:,None,:]).reshape(len(rr),-1),(W[dest,:,None]*b[:,None,:]).reshape(len(dest),-1)))
                    for variant,a,b in variants:
                        eb=_p_weighted_eb(a,b,c,dc,yy,p,pd)
                        if eb is None:continue
                        af,ad,qf,qd,mask,dm=eb
                        if int(mask.sum())<67:continue
                        if not _o_fisher_kkt(ranked,('O1_soft',variant,key,label),base,p,pd,yy,af,ad,dm,den,gated,deadline,(qf,qd)):return []
        if 'O2_press' in channels:
            ordered=sorted([v for v in local if v[0] in {a[0] for a in raw[:3]} and not v[0].startswith(('peer:','parent:'))],key=lambda v:v[0]);xf,xd=_o_features(p,pd,ordered,rr,dest,residual,snap)
            for lam in ((180.,) if bounded else (60.,180.,540.)):
                a,b,_=_h_ridge_press(xf,xd,residual,lam);a=np.clip(a/np.maximum(p,.04),-1.,1.);b=np.clip(b/np.maximum(pd,.04),-1.,1.)
                if not _o_fisher_kkt(ranked,('O2_press',lam),base,p,pd,yy,a,b,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O2_specialist' in channels and len(job['oof'])>1 and len(rr)*width*len(job['oof'])<180000:
            aa=[];bb=[]
            for oo,full in zip(job['oof'][:6],job['all'][:6]):
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else (oo[split==held] if bounded else np.concatenate((oo[split==held],full[rest]),axis=0)),width)
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True);aa.append(a);bb.append(b)
            energy=[float(a[np.arange(len(yy)),yy].sum())**2/(float(np.sum(p*a*a))+120.) for a in aa];order=np.argsort(-np.asarray(energy),kind='stable')[:2];expert=[('arm'+str(k),aa[k],bb[k],np.ones(len(dest),bool)) for k in order]
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            for stratum in ('global','entropy'):
                af,ad=_o_specialist(expert,p,pd,yy,np.zeros(len(p),int) if stratum=='global' else c,np.zeros(len(pd),int) if stratum=='global' else dc)
                if not _o_fisher_kkt(ranked,('O2_specialist',stratum,_o_canonical(v[0] for v in expert)),base,p,pd,yy,af,ad,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fit_s,use_s,rr_s,ys,bs,_,dest_s,ds,_,rs=(_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rr_s)>=67:
                rho=bs[:,-1];rd=ds[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int)
                for key in list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))[:2]:
                    v=snap[key];sk=v[:,-1] if items[key].get('gate') else v.max(1);ws=np.column_stack((1-sk,sk))
                    for label in ['global']+[v[0] for v in raw[:1] if v[0] in dict(local)]:
                        ck=np.zeros(len(X),int) if label=='global' else dict(local)[label];eb=_p_weighted_eb(ws[rr_s],ws[dest_s],ck[rr_s],ck[dest_s],yb,ps,pds)
                        if eb is None:continue
                        a,b,q,qd,mask,dm=eb;af=(a[:,1]-a[:,0]);ad=(b[:,1]-b[:,0]);fisher=rho*(1-rho)*af*af;grad=(yb-rho)*af
                        theta=np.clip((grad.sum()-grad)/(fisher.sum()-fisher+60.),-1.,1.);coef=np.clip(grad.sum()/(fisher.sum()+60.),-1.,1.)
                        for step in (.35,.65):
                            score=float(_h_skip_gain(bs,ys,step*theta*af).sum()/max(fit_s.sum(),1))
                            if score>0:ranked[('O3_skip',key,label,'fisher',step)]=(score,(step*coef*ad)[:,None],(rd>.015)&(rd<.985))
                        # The official floor includes answer mass and skip mass.
                        target=_residual_skip_project(bs,np.ones(len(bs),bool),q[:,1]);idx=np.arange(len(ys));scale=1-.001*job['K'];a0=.001+scale*bs[idx,ys];d0=scale*(target[idx,ys]-bs[idx,ys])
                        z=d0/a0
                        if float(z.sum())<2.*np.sqrt(float(np.sum(z*z))):continue
                        sol=_k_loo_kkt(a0,d0,np.zeros(len(ys),int),1,deadline,maximum=.4)
                        if sol is None:return []
                        alpha,deleted=sol;score=float(np.log1p(deleted*d0/a0).sum()/max(fit_s.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd[:,1],1e-12,1-1e-12)
                        if score>0:ranked[('O3_skip',key,label,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        canonical={};groups={}
        for spec,value in ranked.items():
            if spec[0]=='O1_symmetric':family=('P1_ortho_rank',spec[1],spec[-1])
            elif spec[0]=='O1_soft':family=('P1_soft_rank',spec[1])+spec[4:]
            elif spec[0]=='O1_hierarchical':family=('P1_hierarchy_rank',)+spec[2:]
            elif spec[0]=='O2_specialist':family=('P3_specialist_rank',spec[1])+spec[3:]
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                canonical[(family[0],rank)+family[1:]]=value
        held_fit,query_fit=_p_fit_destinations(canonical,held,split,not bounded);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    families=sorted({(s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1]) for s in fits[-1]},key=repr)
    for family in families:
        ff=[{s:v for s,v in f.items() if (s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1])==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];op=spec[0];skip=op=='O3_skip'
            if not bounded:sr,mr=_p_query_average(query_fits,spec,before_r,gated,skip)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=op,se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,canonical_axes=True,query_pooling=('full_fit' if bounded else 'six_fit_average'))
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _p_axis_pool(routes,rr,dest,yy,p,pd,tau=60.):
    """Fine GIVEN axes borrow a deleted global coarse prior of at least 67 rows."""
    tau=tau*max(1.,p.shape[1]/2.)
    width=p.shape[1];one=np.eye(width)[yy];residual=one-p
    total=one.sum(0);prior=(total+1)/(len(yy)+width)
    pl=(total[None]+1-one)/(len(yy)+width-1);ranked=[]
    for label,key in routes:
        c=key[rr];dc=key[dest];cells=int(key.max())+1
        n=np.bincount(c,minlength=cells);ok=n>=25
        if not ok.any() or int(ok[c].sum())<67:continue
        obs=np.column_stack([np.bincount(c,weights=one[:,k],minlength=cells) for k in range(width)])
        exp=np.column_stack([np.bincount(c,weights=p[:,k],minlength=cells) for k in range(width)])
        sums=obs-exp;energy=float((sums[ok]**2/(exp[ok]+tau*prior)).sum())
        ranked.append((energy,label,c,dc,n,obs,exp,ok))
    ranked.sort(key=lambda v:(-v[0],v[1]));axes=[]
    for energy,label,c,dc,n,obs,exp,ok in ranked[:8]:
        a=np.clip(np.log((obs[c]-one+tau*pl)/(exp[c]-p+tau*pl)),-1.2,1.2)
        b=np.clip(np.log((obs[dc]+tau*prior)/(exp[dc]+tau*prior)),-1.2,1.2)
        a[~ok[c]]=0;b[~ok[dc]]=0
        a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        norm=max(float(np.sqrt(np.mean(np.sum(p*a*a,axis=1)))),.02)
        axes.append((label,a/norm,b/norm,ok[dc]))
    return axes


def _p_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers);fits=[];query_fits=[];support_trace=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=(_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=[] if bounded and channels==('P3',) else list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        raw=(_n_axis_pool if bounded else _p_axis_pool)([v for v in routes if not v[0].startswith(('peer:','parent:'))],rr,dest,yy,p,pd)
        labels=[v[0] for v in raw[:3]];given_routes=[v for label in labels for v in routes if v[0]==label]
        ranked={};den=max(int(fit.sum()),1);allmask=np.ones(len(dest),bool);floor_pool={};soft_fine=0;hier_fine=0
        def emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
            # Rank cheap deleted targets before exact KKT roots. This selects an
            # estimator within each fit, never an identity shared between folds.
            if not _o_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline):return False
            if bounded:return True
            if target is None:
                q=p*np.exp(np.clip(a,-2.,2.));q/=np.maximum(q.sum(1,keepdims=True),1e-300)
                qd=pd*np.exp(np.clip(b,-2.,2.));qd/=np.maximum(qd.sum(1,keepdims=True),1e-300);target=(q,qd)
            q,qd=target;idx=np.arange(len(yy));scale=1-.001*base.shape[1];mass=base[:,:p.shape[1]].sum(1)
            z=scale*mass*(q[idx,yy]-p[idx,yy])/(.001+scale*base[idx,yy])
            score=float(np.log1p(.4*z).sum()/den)
            if spec[0]=='P1_axis':family=(spec[0],spec[1],60.)
            elif spec[0]=='P2_quantile':family=(spec[0],spec[3],60.)
            else:family=spec
            if score>floor_pool.get(family,(-1.,None))[0]:floor_pool[family]=(score,(base,p,pd,yy,mask,den,gated,target))
            return True
        if 'P1' in channels:
            axes=[];single=[]
            for rank,key in enumerate(selected):
                q=snap[key];order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for route in range(1+len(given_routes)):
                    ck=np.zeros(len(X),int) if route==0 else given_routes[route-1][1]
                    a,b=W[rr],W[dest]
                    if route:
                        n=np.bincount(ck[rr]);top=np.argsort(-n,kind='stable')[:4]
                        a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1)
                        b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                    support=a.sum(0);soft_fine+=int(((support>=20)&(support<67)).sum())
                    eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                    if eb is None:continue
                    af,ad,qf,qd,mask,dm=eb
                    if int(mask.sum())<67:continue
                    label=('soft',rank,route);axes.append((repr(label),af,ad,dm));single.append((label,af,ad,qf,qd,dm))
            for rank,(_,ck) in enumerate(given_routes):
                counts=np.bincount(ck[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                eb=_p_cells(ck[rr],ck[dest],np.zeros(len(rr),int),np.zeros(len(dest),int),yy,p,pd)
                if eb is not None:
                    a,b,q,qd,mask,dm=eb;axes.append(('given'+str(rank),a,b,dm));single.append((('given',rank),a,b,q,qd,dm))
                for other in range(rank+1,len(given_routes)):
                    ck2=given_routes[other][1];_,code=np.unique(np.column_stack((ck,ck2)),axis=0,return_inverse=True)
                    counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                    eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                    if eb is not None:
                        a,b,q,qd,mask,dm=eb;label=('hierarchical',rank,other);axes.append((repr(label),a,b,dm));single.append((label,a,b,q,qd,dm))
            for label,a,b,q,qd,dm in single:
                if not emit(ranked,('P1_axis',)+label,base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd)):return []
            orthogonal=_p_orthogonal(axes,p,pd,yy,4)
            for count in (2,3,4):
                if len(orthogonal)<count:continue
                for lam in (120.,):
                    a,b=_m_joint_fisher(orthogonal[:count],p,pd,yy,lam)
                    if not emit(ranked,('P1_blanket',count,lam),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'P2' in channels:
            for pr,key in enumerate(selected[:(1 if bounded else 2)]):
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:1]
                vectors=[q[:,rank[0]],-np.sum(q*np.log(np.maximum(q,1e-12)),1)]
                for vi,v in enumerate(vectors):
                    for bins in ((3,) if bounded else (3,4)):
                        edges=np.unique(np.quantile(v[rr],np.arange(1,bins)/bins));codes=np.searchsorted(edges,v)
                        for gi,(_,ck) in enumerate(given_routes[:2]):
                            _,code=np.unique(np.column_stack((codes,ck)),axis=0,return_inverse=True)
                            counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                            eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                            if eb is None:continue
                            a,b,qf,qd,mask,dm=eb
                            if not emit(ranked,('P2_quantile',pr,vi,bins,gi),base,p,pd,yy,a,b,dm,den,gated,deadline,(qf,qd)):return []
            xf,xd=_p_peer_features(selected,snap,given_routes,rr,dest,p,pd,residual);center=xf.mean(0);xf-=center;xd-=center
            for method in ('svd','pls'):
                matrix=xf.T@xf if method=='svd' else (xf.T@residual)@(residual.T@xf)
                eigen,vectors=np.linalg.eigh(matrix);vectors=vectors[:,np.argsort(-eigen)]
                for rank in (3,):
                    effective=min(rank,int((eigen>max(float(eigen.max()),1.)*1e-10).sum()))
                    if effective<1:continue
                    v=vectors[:,:effective];a=xf@v;b=xd@v;scale=np.maximum(a.std(0),.05);a/=scale;b/=scale
                    for lam in (180.,):
                        af,ad,_=_h_ridge_press(a,b,residual,lam);af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                        if not emit(ranked,('P2_press',method,rank,lam),base,p,pd,yy,af,ad,allmask,den,gated,deadline):return []
        if 'P3' in channels and len(job['oof'])>1:
            arms=[]
            for ai,(oo,full) in enumerate(zip(job['oof'],job['all'])):
                if time.monotonic()>=deadline:return []
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else (oo[split==held] if bounded else np.concatenate((oo[split==held],full[rest]),axis=0)),width)
                if not bounded:
                    q[q.sum(1)<=1e-12]=p[q.sum(1)<=1e-12];qd[qd.sum(1)<=1e-12]=pd[qd.sum(1)<=1e-12]
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                score=float(_g_fit_gain(base,yy,theta[:,None]*a,gated).sum()/den)
                idx=np.arange(len(yy));scale=1-.001*job['K'];mass=base[:,:width].sum(1);aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                solution=(np.array([0.]),np.zeros(len(yy))) if bounded else _k_loo_kkt(aa,dd,np.zeros(len(yy),int),1,deadline,maximum=.4)
                if solution is None:return []
                alpha,loo=solution;kkt_gain=float(np.log1p(loo*dd/aa).sum()/den)
                energy=float(grad.sum())**2/(float(f.sum())+120.)/den
                diag_a,_=_p_specialist_diagonal([('arm',a,b,allmask)],p,pd,yy,np.zeros(len(rr),int),np.zeros(len(dest),int))
                class_gain=float(_g_fit_gain(base,yy,.65*diag_a,gated).sum()/den)
                arms.append(((energy if bounded and job.get('_p_rank_mode')=='energy' else max(score,kkt_gain,class_gain)),ai,a,b,q,qd,loo,alpha[0]))
            arms.sort(key=lambda v:(-v[0],v[1]))
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            strata=[('global',np.zeros(len(rr),int),np.zeros(len(dest),int)),('entropy',c,dc)]+[('given'+str(i),ck[rr],ck[dest]) for i,(_,ck) in enumerate(given_routes[:2])]
            for count in ((2,) if bounded else (2,3)):
                if len(arms)<count:continue
                chosen=arms[:count];axes=[('arm_rank'+str(i),a,b,allmask) for i,(_,_,a,b,*_) in enumerate(chosen)]
                q=sum((1-v[6][:,None])*p+v[6][:,None]*v[4] for v in chosen)/count;qd=sum((1-v[7])*pd+v[7]*v[5] for v in chosen)/count
                a=np.log(np.maximum(q,1e-12)/np.maximum(p,1e-12));b=np.log(np.maximum(qd,1e-12)/np.maximum(pd,1e-12))
                if not emit(ranked,('P3_coordinate',count),base,p,pd,yy,a,b,allmask,den,gated,deadline,(q,qd)):return []
                for label,c,dc in strata:
                    a,b=_p_specialist_diagonal(axes,p,pd,yy,c,dc)
                    if not emit(ranked,('P3_diagonal',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
                    if bounded and count==2 and label in ('global','entropy'):
                        a,b=_o_specialist(axes,p,pd,yy,c,dc)
                        if not emit(ranked,('P3_joint',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'P3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fs,us,rs,ys,bs,_,ds,dbs,_,_= (_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rs)>=67:
                rho=bs[:,-1];rd=dbs[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int);axes=[]
                skip_peers=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rs,(yb-rho)[:,None],4)))[:4]
                parent_routes=[v for v in routes if v[0].startswith('parent:')]
                parent_axes=_n_axis_pool(parent_routes,rs,ds,yb,ps,pds)
                skip_routes=given_routes[:2]+[v for axis in parent_axes[:2] for v in parent_routes if v[0]==axis[0]]
                for rank,key in enumerate(skip_peers):
                    # Parent classes remain distinct. Confidence alone erases
                    # the direction of an otherwise predictive routing state.
                    q=snap[key];center=q[rs]-q[rs].mean(0)
                    order=np.argsort(-np.square(center.T@(yb-rho)),kind='stable')[:min(4,q.shape[1])]
                    W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                    for gi in range(1+len(skip_routes)):
                        ck=np.zeros(len(X),int) if gi==0 else skip_routes[gi-1][1]
                        eb=_p_weighted_eb(W[rs],W[ds],ck[rs],ck[ds],yb,ps,pds)
                        if eb is not None:a,b,q,qd,mask,dm=eb;axes.append((str((rank,gi)),a,b,dm))
                ortho=_p_orthogonal(axes,ps,pds,yb,4)
                for count in (2,3,4):
                    if len(ortho)<count:continue
                    a,b=_m_joint_fisher(ortho[:count],ps,pds,yb,120.);a=a[:,1]-a[:,0];b=b[:,1]-b[:,0]
                    for step in (.35,.65):
                        score=float(_h_skip_gain(bs,ys,step*a).sum()/max(fs.sum(),1))
                        if score>0:ranked[('P3_skip',count,'fisher',step)]=(score,(step*b)[:,None],(rd>.015)&(rd<.985))
                    q=1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+a,-30,30)));qd=1/(1+np.exp(-np.clip(np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12)))+b,-30,30)))
                    target=_residual_skip_project(bs,np.ones(len(bs),bool),q);idx=np.arange(len(ys));scale=1-.001*job['K'];aa=.001+scale*bs[idx,ys];dd=scale*(target[idx,ys]-bs[idx,ys])
                    sol=_k_loo_kkt(aa,dd,np.zeros(len(ys),int),1,deadline,maximum=.4)
                    if sol is None:return []
                    alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fs.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd,1e-12,1-1e-12)
                    if score>0:ranked[('P3_skip',count,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        for spec,(_,args) in ([] if bounded else floor_pool.items()):
            base,p,pd,yy,mask,den,gated,target=args
            if not _o_fisher_kkt(ranked,spec+('fit_selected',),base,p,pd,yy,None,None,mask,den,gated,deadline,target):return []
        # Fit-selected coordinate ranks are estimator identities. Peer names,
        # demographic labels and local coordinate enumeration never enter them.
        groups={}
        for spec,value in list(ranked.items()):
            if spec[0]=='P1_axis' and spec[-1]!='kkt':family=('P1_rank',spec[1],spec[-2],spec[-1])
            elif spec[0]=='P2_quantile' and spec[-1]!='kkt':family=('P2_risk_rank',spec[2],spec[3],spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                ranked[family[:-2]+(rank,)+family[-2:]]=value
        support_trace.append(dict(soft_cells_20_66=soft_fine,hierarchical_cells_25_66=hier_fine,available_specialist_arms=len(job['oof'])))
        held_fit,query_fit=_p_fit_destinations(ranked,held,split,not bounded);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    def family_key(s):return (s[0],s[2] if s[0] in ('P3_diagonal','P3_joint') else '',s[-2] if s[-1] in (.35,.65) else s[-1])
    for family in sorted({family_key(s) for s in fits[-1]},key=repr):
        ff=[{s:v for s,v in f.items() if family_key(s)==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];skip=spec[0]=='P3_skip'
            if not bounded:sr,mr=_p_query_average(query_fits,spec,before_r,gated,skip)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=spec[0],se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,fold_invariant_spec=True,fine_support=support_trace,query_pooling=('full_fit' if bounded else 'six_fit_average'))
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _p_finite_order(order,items,fo,X,given,fold,deadline):
    """Rank specialist work by fit-visible class Fisher energy, not variance."""
    ranked=[];stop=min(deadline-.1,time.monotonic()+.8)
    for name,job in order:
        if time.monotonic()>=stop:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')))
        source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if w is not None:source &= X[rows,given.index('wave')]!=w
        use=source&(job['y'][rows]<width);y=job['y'][rows][use]
        if len(y)<67:continue
        p=_g_conditional(fo[name][use],width);residual=np.eye(width)[y]-p;energy=0.
        for arm in job['oof'][:6]:
            q=_g_conditional(arm[use],width);a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True)
            gradient=(residual*a).sum(0);fisher=(p*(1-p)*a*a).sum(0)
            energy=max(energy,float(np.sum(gradient**2/(fisher+120.))))
        ranked.append((energy/len(rows)/max(1.,len(y)*width*len(job['oof'])),name,job))
    included={v[1] for v in ranked}
    return [(name,job) for _,name,job in sorted(ranked,key=lambda v:(-v[0],v[1]))]+[(name,job) for name,job in order if name not in included]


def _p_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot):
    """Cost-ranked, atomic six-fit bundles for scalar and class specialists."""
    if np.isfinite(deadline):return _q_p_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot)
    order=_s136_order(work,items,fo,X,given,levels,deadline)
    order=_p_finite_order(order,items,fo,X,given,fold,deadline)
    order=sorted(order,key=lambda v:not bool(v[1].get('o_raking')))
    key_fo,key_fr=dict(fo),dict(fr)
    for channels in (('P3',),('P2',),('P1',)):
        for name,job in order:
            if time.monotonic()>=deadline-.05:return
            before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
            for mode in (('energy','deleted') if channels==('P3',) else ('deleted',)):
                job['_p_rank_mode']=mode;accepted=[];local=min(deadline-.03,time.monotonic()+(2.5 if channels==('P3',) else 1.2))
                for oo,qr,active,hm,record in _p_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,local,channels,key_fo,key_fr):
                    passed=_n_dominates(oo,before,y,split,active,2.8);record.update(kappa=2.8,arm_ranking=mode)
                    probe=globals().get('_P_PROBE')
                    if probe is not None:probe(name,record,oo,qr,passed)
                    if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
                if accepted and time.monotonic()<deadline:
                    _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('p_raking',[]).append(record);return


def _multi_peer_soft_blanket_and_fold_invariant_specialist_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Sixteenth stage: multi-peer blankets, reduced-rank PRESS and specialist pooling."""
    if np.isfinite(deadline):return _p_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,base_snapshot)
    key_fo,key_fr=dict(fo),dict(fr)
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        if any(job.get(k) for k in ('pure_graft','answer_graft','residual_raking','peer_raking','ef_raking','f_raking','g_raking','h_raking','i_raking','j_raking','k_raking','l_raking','m_raking','n_raking','o_raking')):continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        proposals=_p_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,('P1','P2','P3','P3_skip'),key_fo,key_fr)
        proposals+=_p_recovered_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,('O1_symmetric','O1_soft','O1_hierarchical'),key_fo,key_fr)
        for oo,qr,active,hm,record in proposals:
            hierarchy=record['operator']=='P1_hierarchy_rank' or (record['operator']=='P1_axis' and record['spec'][1] in ('given','hierarchical'))
            kappa=5.0 if record['operator'].startswith('P3') else 3.2 if hierarchy else 3.8
            if record['operator']=='P1_blanket':kappa*=np.sqrt(float(record['spec'][1]))
            record['kappa']=kappa;amplification=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15);record['full_fit_amplification']=amplification
            passed=.25<=amplification<=1.5 and _n_dominates(oo,before,y,split,active,kappa)
            if passed:
                alpha=min(1.,1.5/max(amplification,1e-15));record['query_full_fit_shrinkage']=alpha
                out=before_r.copy();out[hm]=before_r[hm]+alpha*(qr[hm]-before_r[hm]);qr=out
                if record['skip']:qr=_residual_skip_project(before_r,hm,qr[hm,-1])
            probe=globals().get('_P_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('p_raking',[]).append(record)


def _q_deleted_t(gain):
    """Paired respondent-deleted improvement, with no destination labels."""
    gain=np.asarray(gain,float)
    if len(gain)<2:return 0.
    return float(gain.mean()/max(float(gain.std(ddof=1)/np.sqrt(len(gain))),1e-15))


def _q_orthogonal(axes,p,pd,yy,count=4):
    """Deleted t-screened Gram-Schmidt with James-Stein axis reliability."""
    chosen=[];available=list(axes)
    while available and len(chosen)<count:
        scored=[]
        for i,(label,a,b,mask) in enumerate(available):
            a=a.copy();b=b.copy()
            for _,u,v,_ in chosen:
                c=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15));a-=c*u;b-=c*v
            norm=float(np.sqrt(np.mean(np.sum(p*a*a,1))))
            if norm<1e-7:continue
            a/=norm;b/=norm;f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy]
            beta=np.clip((g.sum()-g)/(f.sum()-f+60.),-1.,1.)
            q=p*np.exp(np.clip(beta[:,None]*a,-2.,2.));q/=q.sum(1,keepdims=True)
            gain=np.log(np.maximum(q[np.arange(len(yy)),yy],1e-12)/np.maximum(p[np.arange(len(yy)),yy],1e-12))
            t_axis=_q_deleted_t(g)
            if gain.mean()<=0 or t_axis<1.8:continue
            shrink=max(0.,1.-1./t_axis**2)
            scored.append((float(gain.mean())*shrink,-i,i,label,a*shrink,b*shrink,mask))
        if not scored:break
        _,_,i,label,a,b,mask=max(scored,key=lambda v:v[:2]);chosen.append((label,a,b,mask));available.pop(i)
    return chosen


def _q_specialist_diagonal(arms,p,pd,yy,c,dc,lam=120.):
    """Rare class deviations shrink exactly to the deleted scalar arm weight."""
    out=np.zeros_like(p);dest=np.zeros_like(pd);one=np.eye(p.shape[1])[yy]
    support=np.bincount(yy,minlength=p.shape[1]);normalizer=max(len(arms),1)
    for _,a,b,_ in arms:
        grad=(one-p)*a;fisher=p*(1-p)*a*a;g=grad.sum(1);f=np.sum(p*a*a,1)
        scalar=g.sum()/(f.sum()+lam);sl=(g.sum()-g)/(f.sum()-f+lam)
        coef=(grad.sum(0)+lam*scalar)/(fisher.sum(0)+lam)
        deleted=(grad.sum(0)[None]-grad+lam*sl[:,None])/(fisher.sum(0)[None]-fisher+lam)
        coef=np.where(support>=25,coef,scalar)
        deleted=np.where(support[None]-one>=25,deleted,sl[:,None])
        af=a*deleted;bd=b*coef
        for cell in np.unique(c):
            ix=np.flatnonzero(c==cell);di=np.flatnonzero(dc==cell)
            if len(ix)<67:continue
            local=one[ix].sum(0)
            cf=(grad[ix].sum(0)+lam*coef)/(fisher[ix].sum(0)+lam)
            cl=(grad[ix].sum(0)[None]-grad[ix]+lam*deleted[ix])/(fisher[ix].sum(0)[None]-fisher[ix]+lam)
            cf=np.where(local>=25,cf,scalar);cl=np.where(local[None]-one[ix]>=25,cl,sl[ix,None])
            af[ix]=a[ix]*cl;bd[di]=b[di]*cf
        out+=af/normalizer;dest+=bd/normalizer
    out-=(p*out).sum(1,keepdims=True);dest-=(pd*dest).sum(1,keepdims=True)
    return np.clip(out,-1.,1.),np.clip(dest,-1.,1.)


def _q_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,fit_count,gated,deadline,target=None,codes=None,dc=None,cells=1):
    """Every newly proposed deleted tangent must have substantive paired gain."""
    if a is not None and spec[0].startswith(('Q3','O2_press','Q2','O2_specialist')):
        gain=_g_fit_gain(base,yy,.65*a,gated)
        if gain.mean()<=0 or _q_deleted_t(gain)<1.8:return True
    return _o_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,fit_count,gated,deadline,target,codes,dc,cells)


def _q_query_average(query_fits,spec,before,gated,skip):
    """Label-free six-fit Fisher direction coherence and full-fit cosine."""
    spec=tuple(spec);simplex=spec[-1]=='kkt' and not skip
    if skip:
        rho=np.clip(before[:,-1],1e-9,1-1e-9);pd=(rho*(1-rho))[:,None]
    else:pd=_g_conditional(before,before.shape[1]-int(gated))
    shifts=[];support=[]
    for fit in query_fits:
        _,value,mask=fit[spec]
        if simplex:
            a=np.log(np.maximum(value,1e-12)/np.maximum(pd,1e-12))
            a-=(pd*a).sum(1,keepdims=True)
        else:a=value.copy()
        shifts.append(np.where(mask[:,None],a,0.));support.append(mask)
    mean=sum(shifts)/len(shifts);energy=[float(np.sum(pd*a*a)) for a in shifts]
    rho_coherence=float(np.sum(pd*mean*mean)/max(float(np.mean(energy)),1e-30))
    full=shifts[-1];cosines=[float(np.sum(pd*a*full)/max(np.sqrt(e*energy[-1]),1e-30)) for a,e in zip(shifts[:-1],energy[:-1])]
    cosine=min(cosines,default=0.);passed=rho_coherence>=.85 and cosine>=.65
    common=max(0.,(len(shifts)*rho_coherence-1.)/(len(shifts)-1.))
    shrink=min(1.,common*max(0.,cosine))
    if simplex:
        value=sum(np.where(mask[:,None],fit[spec][1],pd) for fit,mask in zip(query_fits,support))/len(query_fits)
        value=pd+shrink*(value-pd)
    else:value=shrink*mean
    mask=(sum(support)>=4)&support[-1]
    return value,mask,dict(rho_coherence=rho_coherence,min_fold_full_fisher_cosine=cosine,coherence_query_shrinkage=shrink,passed=bool(passed))


def _q_dominates(arm,ref,y,split,mask,kappa):
    """Scale-free paired inference on the active subproblem, with five folds."""
    if int(mask.sum())<67 or len(np.unique(split[mask]))!=5:return False
    arm,ref,y,split=arm[mask],ref[mask],y[mask],split[mask]
    idx=np.arange(len(y));scale=1-.001*ref.shape[1]
    gain=np.log(.001+scale*arm[idx,y])-np.log(.001+scale*ref[idx,y])
    fg=np.asarray([gain[split==f].mean() for f in np.unique(split)])
    if gain.mean()<=1e-9 or np.any(fg<0.) or any(gain[split%2==h].mean()<=0 for h in (0,1)):return False
    se=max(float(gain.std(ddof=1)/np.sqrt(len(gain))),float(fg.std(ddof=1)/np.sqrt(5)))
    return bool(gain.mean()>=kappa*se)


def _q_recovered_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        w=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if w is not None:source &= X[rows,given.index('wave')]!=w;hidden &= X[rest,given.index('wave')]!=w
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers)
    snap={k:v for k,v in snapshot.items() if k!=name};fits=[];query_fits=[];skip_t_fits=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=_i_residual_peers(snap,rr,residual,2 if bounded else 4)
        local=[v for v in routes if not v[0].startswith('peer:') or v[0][5:] in selected]
        raw=_n_axis_pool(local,rr,dest,yy,p,pd,60.*max(1.,width/2.));axes=_q_orthogonal(raw,p,pd,yy,4);ranked={};skip_tests={};den=max(int(fit.sum()),1)
        if 'O1_symmetric' in channels:
            pools=[('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,60.*max(1.,width/2.)),60.)]
            if bounded:pools=[('joint',axes,40.),('joint',axes,120.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,40.),40.),('symmetric',_n_raw_axes(local,rr,dest,yy,p,pd,120.),120.)]
            for kind,pool,strength in pools:
                for count in (2,3):
                    if count>len(pool):continue
                    chosen=sorted(pool[:count],key=lambda v:v[0]);ident=_o_canonical(v[0] for v in chosen)
                    if kind=='joint':chosen=sorted([v for v in raw if v[0] in ident],key=lambda v:v[0])
                    else:chosen=_q_orthogonal(chosen,p,pd,yy,len(chosen))
                    if len(chosen)<2:continue
                    if kind=='joint':a,b=_m_joint_fisher(chosen,p,pd,yy,strength)
                    else:a=sum(v[1] for v in chosen);b=sum(v[2] for v in chosen)
                    if bounded:ident=('fit_selected_subspace',)
                    for step in ((.65,) if bounded else (.35,.65,1.)):
                        score=float(_g_fit_gain(base,yy,step*a,gated).sum()/den)
                        if score>0:ranked[(('O1_symmetric',kind,ident,strength,step) if bounded else ('O1_symmetric',kind,ident,step))]=(score,step*b,np.ones(len(dest),bool))
        if 'O1_hierarchical' in channels:
            for labels,a,b,q,qd,mask,dm,c,dc,cells in _p_hierarchical_axes(local,rr,dest,yy,p,pd):
                if not _q_fisher_kkt(ranked,('O1_hierarchical',labels),base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd),c,dc,cells):return []
        if 'O1_soft' in channels:
            coarse_labels=[v[0] for v in raw if not v[0].startswith('peer:')][:2];coarse=dict(local)
            peer_keys=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))
            for key in peer_keys[:(1 if bounded else 2)]:
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,rank];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for label in ['global']+coarse_labels[:1]:
                    c=np.zeros(len(rr),int) if label=='global' else coarse[label][rr];dc=np.zeros(len(dest),int) if label=='global' else coarse[label][dest]
                    variants=[('soft',W[rr],W[dest])]
                    # Probability quantiles retain signal when every modal state agrees.
                    if label=='global':
                        edges=np.unique(np.quantile(q[rr,rank[0]],[1/3,2/3]));codes=np.searchsorted(edges,q[:,rank[0]]);eye=np.eye(len(edges)+1)
                        variants.append(('quantiles',eye[codes[rr]],eye[codes[dest]]))
                    else:
                        counts=np.bincount(c);top=np.argsort(-counts,kind='stable')[:3]
                        a=(c[:,None]==top).astype(float);b=(dc[:,None]==top).astype(float)
                        variants.append(('soft_given',(W[rr,:,None]*a[:,None,:]).reshape(len(rr),-1),(W[dest,:,None]*b[:,None,:]).reshape(len(dest),-1)))
                    for variant,a,b in variants:
                        eb=_p_weighted_eb(a,b,c,dc,yy,p,pd)
                        if eb is None:continue
                        af,ad,qf,qd,mask,dm=eb
                        if int(mask.sum())<67:continue
                        if not _q_fisher_kkt(ranked,('O1_soft',variant,key,label),base,p,pd,yy,af,ad,dm,den,gated,deadline,(qf,qd)):return []
        if 'O2_press' in channels:
            ordered=sorted([v for v in local if v[0] in {a[0] for a in raw[:3]} and not v[0].startswith(('peer:','parent:'))],key=lambda v:v[0]);xf,xd=_o_features(p,pd,ordered,rr,dest,residual,snap)
            for lam in ((180.,) if bounded else (60.,180.,540.)):
                a,b,_=_h_ridge_press(xf,xd,residual,lam);a=np.clip(a/np.maximum(p,.04),-1.,1.);b=np.clip(b/np.maximum(pd,.04),-1.,1.)
                if not _q_fisher_kkt(ranked,('O2_press',lam),base,p,pd,yy,a,b,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O2_specialist' in channels and len(job['oof'])>1:
            aa=[];bb=[]
            for oo,full in zip(job['oof'][:6],job['all'][:6]):
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((oo[split==held],full[rest]),axis=0),width)
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True);aa.append(a);bb.append(b)
            qualified=[]
            for k,(a,b) in enumerate(zip(aa,bb)):
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                gain=_g_fit_gain(base,yy,theta[:,None]*a,gated);t_arm=_q_deleted_t(gain)
                if gain.mean()>0 and t_arm>=1.8:qualified.append((float(gain.mean())/t_arm,k,a,b))
            qualified.sort(key=lambda v:(-v[0],v[1]));expert=[('arm'+str(k),a,b,np.ones(len(dest),bool)) for _,k,a,b in qualified[:2]]
            if not expert:expert=[]
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            for stratum in ('global','entropy'):
                af,ad=_q_specialist_diagonal(expert,p,pd,yy,np.zeros(len(p),int) if stratum=='global' else c,np.zeros(len(pd),int) if stratum=='global' else dc)
                if not _q_fisher_kkt(ranked,('O2_specialist',stratum,_o_canonical(v[0] for v in expert)),base,p,pd,yy,af,ad,np.ones(len(dest),bool),den,gated,deadline):return []
        if 'O3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fit_s,use_s,rr_s,ys,bs,_,dest_s,ds,_,rs=_p_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rr_s)>=67:
                rho=bs[:,-1];rd=ds[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int)
                for key in list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+selected))[:2]:
                    v=snap[key];sk=v[:,-1] if items[key].get('gate') else v.max(1);ws=np.column_stack((1-sk,sk))
                    for label in ['global']+[v[0] for v in raw[:1] if v[0] in dict(local)]:
                        ck=np.zeros(len(X),int) if label=='global' else dict(local)[label];eb=_p_weighted_eb(ws[rr_s],ws[dest_s],ck[rr_s],ck[dest_s],yb,ps,pds)
                        if eb is None:continue
                        a,b,q,qd,mask,dm=eb;af=(a[:,1]-a[:,0]);ad=(b[:,1]-b[:,0]);fisher=rho*(1-rho)*af*af;grad=(yb-rho)*af
                        theta=np.clip((grad.sum()-grad)/(fisher.sum()-fisher+60.),-1.,1.);coef=np.clip(grad.sum()/(fisher.sum()+60.),-1.,1.)
                        for step in (.35,.65):
                            score=float(_h_skip_gain(bs,ys,step*theta*af).sum()/max(fit_s.sum(),1))
                            if score>0:ranked[('O3_skip',key,label,'fisher',step)]=(score,(step*coef*ad)[:,None],(rd>.015)&(rd<.985))
                            if score>0:skip_tests[('O3_skip',key,label,'fisher',step)]=_q_deleted_t(_h_skip_gain(bs,ys,step*theta*af))
                        # The official floor includes answer mass and skip mass.
                        target=_residual_skip_project(bs,np.ones(len(bs),bool),q[:,1]);idx=np.arange(len(ys));scale=1-.001*job['K'];a0=.001+scale*bs[idx,ys];d0=scale*(target[idx,ys]-bs[idx,ys])
                        z=d0/a0
                        if float(z.sum())<2.*np.sqrt(float(np.sum(z*z))):continue
                        sol=_k_loo_kkt(a0,d0,np.zeros(len(ys),int),1,deadline,maximum=.4)
                        if sol is None:return []
                        alpha,deleted=sol;score=float(np.log1p(deleted*d0/a0).sum()/max(fit_s.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd[:,1],1e-12,1-1e-12)
                        if score>0:ranked[('O3_skip',key,label,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
                        if score>0:skip_tests[('O3_skip',key,label,'kkt')]=_q_deleted_t(np.log1p(deleted*d0/a0))
        canonical={};canonical_t={};groups={}
        for spec,value in ranked.items():
            if spec[0]=='O1_symmetric':family=('Q1_ortho_rank',spec[1],spec[-1])
            elif spec[0]=='O1_soft':family=('Q1_soft_rank',spec[1])+spec[4:]
            elif spec[0]=='O1_hierarchical':family=('Q1_hierarchy_rank',)+spec[2:]
            elif spec[0]=='O2_specialist':family=('Q2_specialist_rank',spec[1])+spec[3:]
            elif spec[0]=='O3_skip':family=('Q3_skip_rank',spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(source_spec,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                canonical[(family[0],rank)+family[1:]]=value
                if source_spec in skip_tests:canonical_t[(family[0],rank)+family[1:]]=skip_tests[source_spec]
        held_fit,query_fit=_p_fit_destinations(canonical,held,split,True);fits.append(held_fit);query_fits.append(query_fit);skip_t_fits.append(canonical_t)
    proposals=[]
    families=sorted({(s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1]) for s in fits[-1]},key=repr)
    for family in families:
        ff=[{s:v for s,v in f.items() if (s[0],s[-2] if s[-1] in (.35,.65,1.) else s[-1])==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];op=spec[0];skip=op=='Q3_skip_rank'
            if skip:
                deleted_t=[v.get(tuple(spec),0.) for v in skip_t_fits]
                if min(deleted_t)<1.8:continue
                record['fit_deleted_t']=deleted_t
            sr,mr,coherence=_q_query_average(query_fits,spec,before_r,gated,skip)
            if not coherence['passed']:continue
            record.update(coherence)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=op,se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,canonical_axes=True,query_pooling='coherence_shrunk_six_fit_average')
            proposals.append((oo,qr,active,hm,record))
    return proposals

def _q_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers);fits=[];query_fits=[];support_trace=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=[] if bounded and channels==('Q2',) else list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        raw=(_n_axis_pool if bounded else _p_axis_pool)([v for v in routes if not v[0].startswith(('peer:','parent:'))],rr,dest,yy,p,pd)
        labels=[v[0] for v in raw[:3]];given_routes=[v for label in labels for v in routes if v[0]==label]
        ranked={};den=max(int(fit.sum()),1);allmask=np.ones(len(dest),bool);floor_pool={};soft_fine=0;hier_fine=0
        def emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
            # Rank cheap deleted targets before exact KKT roots. This selects an
            # estimator within each fit, never an identity shared between folds.
            if not _q_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline):return False
            if bounded:return True
            if target is None:
                q=p*np.exp(np.clip(a,-2.,2.));q/=np.maximum(q.sum(1,keepdims=True),1e-300)
                qd=pd*np.exp(np.clip(b,-2.,2.));qd/=np.maximum(qd.sum(1,keepdims=True),1e-300);target=(q,qd)
            q,qd=target;idx=np.arange(len(yy));scale=1-.001*base.shape[1];mass=base[:,:p.shape[1]].sum(1)
            z=scale*mass*(q[idx,yy]-p[idx,yy])/(.001+scale*base[idx,yy])
            score=float(np.log1p(.4*z).sum()/den)
            if spec[0]=='Q1_axis':family=(spec[0],spec[1],60.)
            elif spec[0]=='Q3_quantile':family=(spec[0],spec[3],60.)
            else:family=spec
            if score>floor_pool.get(family,(-1.,None))[0]:floor_pool[family]=(score,(base,p,pd,yy,mask,den,gated,target))
            return True
        if 'Q1' in channels:
            axes=[];single=[]
            for rank,key in enumerate(selected):
                q=snap[key];order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for route in range(1+len(given_routes)):
                    ck=np.zeros(len(X),int) if route==0 else given_routes[route-1][1]
                    a,b=W[rr],W[dest]
                    if route:
                        n=np.bincount(ck[rr]);top=np.argsort(-n,kind='stable')[:4]
                        a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1)
                        b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                    support=a.sum(0);soft_fine+=int(((support>=20)&(support<67)).sum())
                    eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                    if eb is None:continue
                    af,ad,qf,qd,mask,dm=eb
                    if int(mask.sum())<67:continue
                    label=('soft',rank,route);axes.append((repr(label),af,ad,dm));single.append((label,af,ad,qf,qd,dm))
            for rank,(_,ck) in enumerate(given_routes):
                counts=np.bincount(ck[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                eb=_p_cells(ck[rr],ck[dest],np.zeros(len(rr),int),np.zeros(len(dest),int),yy,p,pd)
                if eb is not None:
                    a,b,q,qd,mask,dm=eb;axes.append(('given'+str(rank),a,b,dm));single.append((('given',rank),a,b,q,qd,dm))
                for other in range(rank+1,len(given_routes)):
                    ck2=given_routes[other][1];_,code=np.unique(np.column_stack((ck,ck2)),axis=0,return_inverse=True)
                    counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                    eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                    if eb is not None:
                        a,b,q,qd,mask,dm=eb;label=('hierarchical',rank,other);axes.append((repr(label),a,b,dm));single.append((label,a,b,q,qd,dm))
            for label,a,b,q,qd,dm in single:
                if not emit(ranked,('Q1_axis',)+label,base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd)):return []
            orthogonal=_q_orthogonal(axes,p,pd,yy,4)
            for count in (1,2,3,4):
                if len(orthogonal)<count:continue
                for lam in (120.,):
                    a,b=_m_joint_fisher(orthogonal[:count],p,pd,yy,lam)
                    if not emit(ranked,('Q1_blanket',count,lam),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'Q3' in channels:
            for pr,key in enumerate(selected[:(1 if bounded else 2)]):
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:1]
                vectors=[q[:,rank[0]],-np.sum(q*np.log(np.maximum(q,1e-12)),1)]
                for vi,v in enumerate(vectors):
                    for bins in ((3,) if bounded else (3,4)):
                        edges=np.unique(np.quantile(v[rr],np.arange(1,bins)/bins));codes=np.searchsorted(edges,v)
                        for gi,(_,ck) in enumerate(given_routes[:2]):
                            _,code=np.unique(np.column_stack((codes,ck)),axis=0,return_inverse=True)
                            counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                            eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                            if eb is None:continue
                            a,b,qf,qd,mask,dm=eb
                            if not emit(ranked,('Q3_quantile',pr,vi,bins,gi),base,p,pd,yy,a,b,dm,den,gated,deadline,(qf,qd)):return []
            xf,xd=_p_peer_features(selected,snap,given_routes,rr,dest,p,pd,residual);center=xf.mean(0);xf-=center;xd-=center
            for method in ('svd','pls'):
                matrix=xf.T@xf if method=='svd' else (xf.T@residual)@(residual.T@xf)
                eigen,vectors=np.linalg.eigh(matrix);vectors=vectors[:,np.argsort(-eigen)]
                for rank in (3,):
                    effective=min(rank,int((eigen>max(float(eigen.max()),1.)*1e-10).sum()))
                    if effective<1:continue
                    v=vectors[:,:effective];a=xf@v;b=xd@v;scale=np.maximum(a.std(0),.05);a/=scale;b/=scale
                    for lam in (180.,):
                        af,ad,_=_h_ridge_press(a,b,residual,lam);af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                        if not emit(ranked,('Q3_press',method,rank,lam),base,p,pd,yy,af,ad,allmask,den,gated,deadline):return []
        if 'Q2' in channels and len(job['oof'])>1:
            arms=[]
            for ai,(oo,full) in enumerate(zip(job['oof'],job['all'])):
                if time.monotonic()>=deadline:return []
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((oo[split==held],full[rest]),axis=0),width)
                if not bounded:
                    q[q.sum(1)<=1e-12]=p[q.sum(1)<=1e-12];qd[qd.sum(1)<=1e-12]=pd[qd.sum(1)<=1e-12]
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                score=float(_g_fit_gain(base,yy,theta[:,None]*a,gated).sum()/den)
                idx=np.arange(len(yy));scale=1-.001*job['K'];mass=base[:,:width].sum(1);aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                solution=(np.array([0.]),np.zeros(len(yy))) if bounded else _k_loo_kkt(aa,dd,np.zeros(len(yy),int),1,deadline,maximum=.4)
                if solution is None:return []
                alpha,loo=solution;kkt_gain=float(np.log1p(loo*dd/aa).sum()/den)
                energy=float(grad.sum())**2/(float(f.sum())+120.)/den
                diag_a,_=_q_specialist_diagonal([('arm',a,b,allmask)],p,pd,yy,np.zeros(len(rr),int),np.zeros(len(dest),int))
                class_gain=float(_g_fit_gain(base,yy,.65*diag_a,gated).sum()/den)
                gains=_g_fit_gain(base,yy,theta[:,None]*a,gated)
                t_arm=_q_deleted_t(gains)
                if score<=0 or t_arm<1.8:continue
                reliability=score/max(t_arm,1.8)
                arms.append((reliability,ai,a,b,q,qd,loo,alpha[0]))
            arms.sort(key=lambda v:(-v[0],v[1]))
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            strata=[('global',np.zeros(len(rr),int),np.zeros(len(dest),int)),('entropy',c,dc)]+[('given'+str(i),ck[rr],ck[dest]) for i,(_,ck) in enumerate(given_routes[:2])]
            for count in ((1,2) if bounded else (1,2,3)):
                if len(arms)<count:continue
                chosen=arms[:count];axes=[('arm_rank'+str(i),a,b,allmask) for i,(_,_,a,b,*_) in enumerate(chosen)]
                weights=np.asarray([v[0] for v in chosen]);weights/=weights.sum()
                axes=[(label,a*np.sqrt(w),b*np.sqrt(w),mask) for (label,a,b,mask),w in zip(axes,weights)]
                q=sum(w*((1-v[6][:,None])*p+v[6][:,None]*v[4]) for w,v in zip(weights,chosen));qd=sum(w*((1-v[7])*pd+v[7]*v[5]) for w,v in zip(weights,chosen))
                a=np.log(np.maximum(q,1e-12)/np.maximum(p,1e-12));b=np.log(np.maximum(qd,1e-12)/np.maximum(pd,1e-12))
                if not emit(ranked,('Q2_coordinate',count),base,p,pd,yy,a,b,allmask,den,gated,deadline,(q,qd)):return []
                for label,c,dc in strata:
                    a,b=_q_specialist_diagonal(axes,p,pd,yy,c,dc)
                    if not emit(ranked,('Q2_diagonal',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
                    if False:
                        a,b=_o_specialist(axes,p,pd,yy,c,dc)
                        if not emit(ranked,('Q2_joint',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'Q3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fs,us,rs,ys,bs,_,ds,dbs,_,_= _p_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rs)>=67:
                rho=bs[:,-1];rd=dbs[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int);axes=[]
                skip_peers=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rs,(yb-rho)[:,None],4)))[:4]
                parent_routes=[v for v in routes if v[0].startswith('parent:')]
                parent_axes=_n_axis_pool(parent_routes,rs,ds,yb,ps,pds)
                skip_routes=given_routes[:2]+[v for axis in parent_axes[:2] for v in parent_routes if v[0]==axis[0]]
                for rank,key in enumerate(skip_peers):
                    # Parent classes remain distinct. Confidence alone erases
                    # the direction of an otherwise predictive routing state.
                    q=snap[key];center=q[rs]-q[rs].mean(0)
                    order=np.argsort(-np.square(center.T@(yb-rho)),kind='stable')[:min(4,q.shape[1])]
                    W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                    for gi in range(1+len(skip_routes)):
                        ck=np.zeros(len(X),int) if gi==0 else skip_routes[gi-1][1]
                        eb=_p_weighted_eb(W[rs],W[ds],ck[rs],ck[ds],yb,ps,pds)
                        if eb is not None:a,b,q,qd,mask,dm=eb;axes.append((str((rank,gi)),a,b,dm))
                ortho=_q_orthogonal(axes,ps,pds,yb,4)
                for count in (2,3,4):
                    if len(ortho)<count:continue
                    a,b=_m_joint_fisher(ortho[:count],ps,pds,yb,120.);a=a[:,1]-a[:,0];b=b[:,1]-b[:,0]
                    for step in (.35,.65):
                        score=float(_h_skip_gain(bs,ys,step*a).sum()/max(fs.sum(),1))
                        if score>0:ranked[('Q3_skip',count,'fisher',step)]=(score,(step*b)[:,None],(rd>.015)&(rd<.985))
                    q=1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+a,-30,30)));qd=1/(1+np.exp(-np.clip(np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12)))+b,-30,30)))
                    target=_residual_skip_project(bs,np.ones(len(bs),bool),q);idx=np.arange(len(ys));scale=1-.001*job['K'];aa=.001+scale*bs[idx,ys];dd=scale*(target[idx,ys]-bs[idx,ys])
                    sol=_k_loo_kkt(aa,dd,np.zeros(len(ys),int),1,deadline,maximum=.4)
                    if sol is None:return []
                    alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fs.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd,1e-12,1-1e-12)
                    if score>0:ranked[('Q3_skip',count,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        for spec,(_,args) in ([] if bounded else floor_pool.items()):
            base,p,pd,yy,mask,den,gated,target=args
            if not _q_fisher_kkt(ranked,spec+('fit_selected',),base,p,pd,yy,None,None,mask,den,gated,deadline,target):return []
        # Fit-selected coordinate ranks are estimator identities. Peer names,
        # demographic labels and local coordinate enumeration never enter them.
        groups={}
        for spec,value in list(ranked.items()):
            if spec[0]=='Q1_axis' and spec[-1]!='kkt':family=('Q1_rank',spec[1],spec[-2],spec[-1])
            elif spec[0]=='Q3_quantile' and spec[-1]!='kkt':family=('Q3_risk_rank',spec[2],spec[3],spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                ranked[family[:-2]+(rank,)+family[-2:]]=value
        support_trace.append(dict(soft_cells_20_66=soft_fine,hierarchical_cells_25_66=hier_fine,available_specialist_arms=len(job['oof'])))
        held_fit,query_fit=_p_fit_destinations(ranked,held,split,True);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    def family_key(s):return (s[0],s[2] if s[0] in ('Q2_diagonal','Q2_joint') else s[1] if s[0]=='Q1_blanket' else '',s[-2] if s[-1] in (.35,.65) else s[-1])
    for family in sorted({family_key(s) for s in fits[-1]},key=repr):
        ff=[{s:v for s,v in f.items() if family_key(s)==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];skip=spec[0]=='Q3_skip'
            sr,mr,coherence=_q_query_average(query_fits,spec,before_r,gated,skip)
            if not coherence['passed']:continue
            record.update(coherence)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=spec[0],se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,fold_invariant_spec=True,fine_support=support_trace,query_pooling='coherence_shrunk_six_fit_average')
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _q_stable_blankets(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,key_fo,key_fr):
    """Align actual peer/route identities before testing six-fit query directions."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo,key_fr,X,given,levels,peers);contexts=[];pquery=_g_conditional(before_r,width)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        context=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        fit,use,rr,yy,base,p,dest,db,pd,residual=context
        if len(rr)<67:return []
        selected=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        given_routes=[v for v in routes if not v[0].startswith(('peer:','parent:'))]
        raw=_p_axis_pool(given_routes,rr,dest,yy,p,pd);labels=[v[0] for v in raw[:3]];lookup=dict(given_routes);axes={}
        for label,a,b,mask in raw:
            axes[('given',label)]=(repr(('given',label)),a,b,mask)
        for key in selected:
            q=snap[key]
            # Keep the original class identities; their fit-dependent variance
            # ordering must not relabel query coordinates between partitions.
            order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])];order=np.sort(order)
            W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
            for label in ['global']+labels[:2]:
                ck=np.zeros(len(X),int) if label=='global' else lookup[label]
                a,b=W[rr],W[dest]
                if label!='global':
                    counts=np.bincount(ck[rr]);top=np.argsort(-counts,kind='stable')[:4]
                    a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1);b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                if eb is None:continue
                a,b,qf,qd,mask,dm=eb
                if mask.sum()<67:continue
                ident=('soft',key,label,tuple(int(i) for i in order));axes[ident]=(repr(ident),a,b,dm)
        contexts.append((held,context,axes))
    common=set.intersection(*(set(v[2]) for v in contexts));reliability={};axis_metrics=[]
    for ident in sorted(common,key=repr):
        values=[]
        for held,ctx,axes in contexts:
            _,_,b,m=axes[ident];count=0 if held is None else int((split==held).sum());values.append(np.where(m[count:,None],b[count:],0.))
        mean=sum(values)/6.;energy=[float(np.sum(pquery*b*b)) for b in values];rho=float(np.sum(pquery*mean*mean)/max(float(np.mean(energy)),1e-30));cos=min(float(np.sum(pquery*b*values[-1])/max(np.sqrt(e*energy[-1]),1e-30)) for b,e in zip(values[:-1],energy[:-1]))
        weight=max(0.,(6*rho-1)/5)*max(0.,cos)
        axis_metrics.append(dict(axis=repr(ident),rho_coherence=rho,min_fold_full_fisher_cosine=cos,weight=weight))
        if rho>1/6 and cos>0:reliability[ident]=weight
    fits=[];query_fits=[]
    for held,ctx,axes in contexts:
        fit,use,rr,yy,base,p,dest,db,pd,residual=ctx;pool=[]
        for ident,weight in reliability.items():
            label,a,b,m=axes[ident];pool.append((label,weight*a,weight*b,m))
        orthogonal=_q_orthogonal(pool,p,pd,yy,4);ranked={}
        if not orthogonal:return []
        for cap in (1,2,3,4):
            chosen=orthogonal[:cap]
            for lam in (60.,120.):
                a,b=_m_joint_fisher(chosen,p,pd,yy,lam)
                _o_fisher_kkt(ranked,('Q1_stable_blanket',cap,lam),base,p,pd,yy,a,b,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline)
        held_map,query_map=_p_fit_destinations(ranked,held,split,True);fits.append(held_map);query_fits.append(query_map)
    output=[]
    for cap in (1,2,3,4):
        for lam in (60.,120.):
            ff=[{s:v for s,v in f.items() if s[1:3]==(cap,lam)} for f in fits]
            for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
                sr,mr,coherence=_q_query_average(query_fits,record['spec'],before_r,gated,False)
                if not coherence['passed']:continue
                active=mo&source;hm=mr&hidden
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
                idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
                record.update(operator='Q1_stable_blanket',se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=False,axis_coherence=axis_metrics,**coherence)
                output.append((oo,qr,active,hm,record))
    return output


def _q_p_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers);fits=[];query_fits=[];support_trace=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=(_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=[] if bounded and channels==('P3',) else list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        raw=(_n_axis_pool if bounded else _p_axis_pool)([v for v in routes if not v[0].startswith(('peer:','parent:'))],rr,dest,yy,p,pd)
        labels=[v[0] for v in raw[:3]];given_routes=[v for label in labels for v in routes if v[0]==label]
        ranked={};den=max(int(fit.sum()),1);allmask=np.ones(len(dest),bool);floor_pool={};soft_fine=0;hier_fine=0
        def emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
            # Rank cheap deleted targets before exact KKT roots. This selects an
            # estimator within each fit, never an identity shared between folds.
            if not _o_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline):return False
            if bounded:return True
            if target is None:
                q=p*np.exp(np.clip(a,-2.,2.));q/=np.maximum(q.sum(1,keepdims=True),1e-300)
                qd=pd*np.exp(np.clip(b,-2.,2.));qd/=np.maximum(qd.sum(1,keepdims=True),1e-300);target=(q,qd)
            q,qd=target;idx=np.arange(len(yy));scale=1-.001*base.shape[1];mass=base[:,:p.shape[1]].sum(1)
            z=scale*mass*(q[idx,yy]-p[idx,yy])/(.001+scale*base[idx,yy])
            score=float(np.log1p(.4*z).sum()/den)
            if spec[0]=='P1_axis':family=(spec[0],spec[1],60.)
            elif spec[0]=='P2_quantile':family=(spec[0],spec[3],60.)
            else:family=spec
            if score>floor_pool.get(family,(-1.,None))[0]:floor_pool[family]=(score,(base,p,pd,yy,mask,den,gated,target))
            return True
        if 'P1' in channels:
            axes=[];single=[]
            for rank,key in enumerate(selected):
                q=snap[key];order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for route in range(1+len(given_routes)):
                    ck=np.zeros(len(X),int) if route==0 else given_routes[route-1][1]
                    a,b=W[rr],W[dest]
                    if route:
                        n=np.bincount(ck[rr]);top=np.argsort(-n,kind='stable')[:4]
                        a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1)
                        b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                    support=a.sum(0);soft_fine+=int(((support>=20)&(support<67)).sum())
                    eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                    if eb is None:continue
                    af,ad,qf,qd,mask,dm=eb
                    if int(mask.sum())<67:continue
                    label=('soft',rank,route);axes.append((repr(label),af,ad,dm));single.append((label,af,ad,qf,qd,dm))
            for rank,(_,ck) in enumerate(given_routes):
                counts=np.bincount(ck[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                eb=_p_cells(ck[rr],ck[dest],np.zeros(len(rr),int),np.zeros(len(dest),int),yy,p,pd)
                if eb is not None:
                    a,b,q,qd,mask,dm=eb;axes.append(('given'+str(rank),a,b,dm));single.append((('given',rank),a,b,q,qd,dm))
                for other in range(rank+1,len(given_routes)):
                    ck2=given_routes[other][1];_,code=np.unique(np.column_stack((ck,ck2)),axis=0,return_inverse=True)
                    counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                    eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                    if eb is not None:
                        a,b,q,qd,mask,dm=eb;label=('hierarchical',rank,other);axes.append((repr(label),a,b,dm));single.append((label,a,b,q,qd,dm))
            for label,a,b,q,qd,dm in single:
                if not emit(ranked,('P1_axis',)+label,base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd)):return []
            orthogonal=_p_orthogonal(axes,p,pd,yy,4)
            for count in (2,3,4):
                if len(orthogonal)<count:continue
                for lam in (120.,):
                    a,b=_m_joint_fisher(orthogonal[:count],p,pd,yy,lam)
                    if not emit(ranked,('P1_blanket',count,lam),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'P2' in channels:
            for pr,key in enumerate(selected[:(1 if bounded else 2)]):
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:1]
                vectors=[q[:,rank[0]],-np.sum(q*np.log(np.maximum(q,1e-12)),1)]
                for vi,v in enumerate(vectors):
                    for bins in ((3,) if bounded else (3,4)):
                        edges=np.unique(np.quantile(v[rr],np.arange(1,bins)/bins));codes=np.searchsorted(edges,v)
                        for gi,(_,ck) in enumerate(given_routes[:2]):
                            _,code=np.unique(np.column_stack((codes,ck)),axis=0,return_inverse=True)
                            counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                            eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                            if eb is None:continue
                            a,b,qf,qd,mask,dm=eb
                            if not emit(ranked,('P2_quantile',pr,vi,bins,gi),base,p,pd,yy,a,b,dm,den,gated,deadline,(qf,qd)):return []
            xf,xd=_p_peer_features(selected,snap,given_routes,rr,dest,p,pd,residual);center=xf.mean(0);xf-=center;xd-=center
            for method in ('svd','pls'):
                matrix=xf.T@xf if method=='svd' else (xf.T@residual)@(residual.T@xf)
                eigen,vectors=np.linalg.eigh(matrix);vectors=vectors[:,np.argsort(-eigen)]
                for rank in (3,):
                    effective=min(rank,int((eigen>max(float(eigen.max()),1.)*1e-10).sum()))
                    if effective<1:continue
                    v=vectors[:,:effective];a=xf@v;b=xd@v;scale=np.maximum(a.std(0),.05);a/=scale;b/=scale
                    for lam in (180.,):
                        af,ad,_=_h_ridge_press(a,b,residual,lam);af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                        if not emit(ranked,('P2_press',method,rank,lam),base,p,pd,yy,af,ad,allmask,den,gated,deadline):return []
        if 'P3' in channels and len(job['oof'])>1:
            arms=[]
            for ai,(oo,full) in enumerate(zip(job['oof'],job['all'])):
                if time.monotonic()>=deadline:return []
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else (oo[split==held] if bounded else np.concatenate((oo[split==held],full[rest]),axis=0)),width)
                if not bounded:
                    q[q.sum(1)<=1e-12]=p[q.sum(1)<=1e-12];qd[qd.sum(1)<=1e-12]=pd[qd.sum(1)<=1e-12]
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                score=float(_g_fit_gain(base,yy,theta[:,None]*a,gated).sum()/den)
                idx=np.arange(len(yy));scale=1-.001*job['K'];mass=base[:,:width].sum(1);aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                solution=(np.array([0.]),np.zeros(len(yy))) if bounded else _k_loo_kkt(aa,dd,np.zeros(len(yy),int),1,deadline,maximum=.4)
                if solution is None:return []
                alpha,loo=solution;kkt_gain=float(np.log1p(loo*dd/aa).sum()/den)
                energy=float(grad.sum())**2/(float(f.sum())+120.)/den
                diag_a,_=_p_specialist_diagonal([('arm',a,b,allmask)],p,pd,yy,np.zeros(len(rr),int),np.zeros(len(dest),int))
                class_gain=float(_g_fit_gain(base,yy,.65*diag_a,gated).sum()/den)
                arms.append(((energy if bounded and job.get('_p_rank_mode')=='energy' else max(score,kkt_gain,class_gain)),ai,a,b,q,qd,loo,alpha[0]))
            arms.sort(key=lambda v:(-v[0],v[1]))
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            strata=[('global',np.zeros(len(rr),int),np.zeros(len(dest),int)),('entropy',c,dc)]+[('given'+str(i),ck[rr],ck[dest]) for i,(_,ck) in enumerate(given_routes[:2])]
            for count in ((2,) if bounded else (2,3)):
                if len(arms)<count:continue
                chosen=arms[:count];axes=[('arm_rank'+str(i),a,b,allmask) for i,(_,_,a,b,*_) in enumerate(chosen)]
                q=sum((1-v[6][:,None])*p+v[6][:,None]*v[4] for v in chosen)/count;qd=sum((1-v[7])*pd+v[7]*v[5] for v in chosen)/count
                a=np.log(np.maximum(q,1e-12)/np.maximum(p,1e-12));b=np.log(np.maximum(qd,1e-12)/np.maximum(pd,1e-12))
                if not emit(ranked,('P3_coordinate',count),base,p,pd,yy,a,b,allmask,den,gated,deadline,(q,qd)):return []
                for label,c,dc in strata:
                    a,b=_p_specialist_diagonal(axes,p,pd,yy,c,dc)
                    if not emit(ranked,('P3_diagonal',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
                    if bounded and p.shape[1]<=2 and count==2 and label in ('global','entropy'):
                        a,b=_o_specialist(axes,p,pd,yy,c,dc)
                        if not emit(ranked,('P3_joint',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'P3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fs,us,rs,ys,bs,_,ds,dbs,_,_= (_j_fold_context if bounded else _p_fold_context)(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rs)>=67:
                rho=bs[:,-1];rd=dbs[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int);axes=[]
                skip_peers=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rs,(yb-rho)[:,None],4)))[:4]
                parent_routes=[v for v in routes if v[0].startswith('parent:')]
                parent_axes=_n_axis_pool(parent_routes,rs,ds,yb,ps,pds)
                skip_routes=given_routes[:2]+[v for axis in parent_axes[:2] for v in parent_routes if v[0]==axis[0]]
                for rank,key in enumerate(skip_peers):
                    # Parent classes remain distinct. Confidence alone erases
                    # the direction of an otherwise predictive routing state.
                    q=snap[key];center=q[rs]-q[rs].mean(0)
                    order=np.argsort(-np.square(center.T@(yb-rho)),kind='stable')[:min(4,q.shape[1])]
                    W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                    for gi in range(1+len(skip_routes)):
                        ck=np.zeros(len(X),int) if gi==0 else skip_routes[gi-1][1]
                        eb=_p_weighted_eb(W[rs],W[ds],ck[rs],ck[ds],yb,ps,pds)
                        if eb is not None:a,b,q,qd,mask,dm=eb;axes.append((str((rank,gi)),a,b,dm))
                ortho=_p_orthogonal(axes,ps,pds,yb,4)
                for count in (2,3,4):
                    if len(ortho)<count:continue
                    a,b=_m_joint_fisher(ortho[:count],ps,pds,yb,120.);a=a[:,1]-a[:,0];b=b[:,1]-b[:,0]
                    for step in (.35,.65):
                        score=float(_h_skip_gain(bs,ys,step*a).sum()/max(fs.sum(),1))
                        if score>0:ranked[('P3_skip',count,'fisher',step)]=(score,(step*b)[:,None],(rd>.015)&(rd<.985))
                    q=1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+a,-30,30)));qd=1/(1+np.exp(-np.clip(np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12)))+b,-30,30)))
                    target=_residual_skip_project(bs,np.ones(len(bs),bool),q);idx=np.arange(len(ys));scale=1-.001*job['K'];aa=.001+scale*bs[idx,ys];dd=scale*(target[idx,ys]-bs[idx,ys])
                    sol=_k_loo_kkt(aa,dd,np.zeros(len(ys),int),1,deadline,maximum=.4)
                    if sol is None:return []
                    alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fs.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd,1e-12,1-1e-12)
                    if score>0:ranked[('P3_skip',count,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        for spec,(_,args) in ([] if bounded else floor_pool.items()):
            base,p,pd,yy,mask,den,gated,target=args
            if not _o_fisher_kkt(ranked,spec+('fit_selected',),base,p,pd,yy,None,None,mask,den,gated,deadline,target):return []
        # Fit-selected coordinate ranks are estimator identities. Peer names,
        # demographic labels and local coordinate enumeration never enter them.
        groups={}
        for spec,value in list(ranked.items()):
            if spec[0]=='P1_axis' and spec[-1]!='kkt':family=('P1_rank',spec[1],spec[-2],spec[-1])
            elif spec[0]=='P2_quantile' and spec[-1]!='kkt':family=('P2_risk_rank',spec[2],spec[3],spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                ranked[family[:-2]+(rank,)+family[-2:]]=value
        support_trace.append(dict(soft_cells_20_66=soft_fine,hierarchical_cells_25_66=hier_fine,available_specialist_arms=len(job['oof'])))
        held_fit,query_fit=_p_fit_destinations(ranked,held,split,not bounded);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    def family_key(s):return (s[0],s[2] if s[0] in ('P3_diagonal','P3_joint') else '',s[-2] if s[-1] in (.35,.65) else s[-1])
    for family in sorted({family_key(s) for s in fits[-1]},key=repr):
        ff=[{s:v for s,v in f.items() if family_key(s)==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];skip=spec[0]=='P3_skip'
            if not bounded:sr,mr=_p_query_average(query_fits,spec,before_r,gated,skip)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=spec[0],se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,fold_invariant_spec=True,fine_support=support_trace,query_pooling=('full_fit' if bounded else 'six_fit_average'))
            proposals.append((oo,qr,active,hm,record))
    return proposals

def _q_p_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot):
    """Cost-ranked, atomic six-fit bundles for scalar and class specialists."""
    order=_s136_order(work,items,fo,X,given,levels,deadline)
    order=_p_finite_order(order,items,fo,X,given,fold,deadline)
    order=sorted(order,key=lambda v:not bool(v[1].get('o_raking')))
    key_fo,key_fr=dict(fo),dict(fr)
    for channels in (('P3',),('P2',),('P1',)):
        for name,job in order:
            if time.monotonic()>=deadline-.05:return
            before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
            for mode in (('energy','deleted') if channels==('P3',) else ('deleted',)):
                job['_p_rank_mode']=mode;accepted=[];local=min(deadline-.03,time.monotonic()+(2.5 if channels==('P3',) else 1.2))
                for oo,qr,active,hm,record in (_p_proposals if deadline-time.monotonic()>6.0 else _q_p_proposals)(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,local,channels,key_fo,key_fr):
                    passed=_n_dominates(oo,before,y,split,active,2.8);record.update(kappa=2.8,arm_ranking=mode)
                    probe=globals().get('_P_PROBE')
                    if probe is not None:probe(name,record,oo,qr,passed)
                    if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
                if accepted and time.monotonic()<deadline:
                    _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('p_raking',[]).append(record);return


def _q_positive_arm_proposals(name,job,arm,items,fo,fr,X,given,fold,deadline):
    """Positive deleted scalar/KKT calibration of a complete fit-only expert."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];expert,full=job['oof'][arm],job['all'][arm];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        q=_g_conditional(expert[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((expert[split==held],full[rest]),axis=0),width)
        a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.);coef=np.clip(grad.sum()/(f.sum()+120.),-1.,1.)
        gain=_g_fit_gain(base,yy,theta[:,None]*a,gated);t_arm=_q_deleted_t(gain);ranked={};trace.append(dict(deleted_gain=float(gain.mean()),t_arm=t_arm))
        if gain.mean()>0 and t_arm>=1.8:
            shrink=max(0.,1.-1./t_arm**2);af=shrink*theta[:,None]*a;ad=shrink*coef*b
            _o_fisher_kkt(ranked,('Q2_positive_arm',),base,p,pd,yy,af,ad,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline,(q,qd))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for mode in ('tangent','fisher','kkt'):
        ff=[{s:v for s,v in f.items() if (s[-1]=='kkt' if mode=='kkt' else s[-2]==mode)} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_q_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            project=_j_simplex_project if record['spec'][-1]=='kkt' else _ef_project
            oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            record.update(operator='Q2_positive_arm',skip=False,fit_only=True,arm=arm,positive_arm_trace=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out

def _q_positive_arm_order(work,items,fo,X,given,deadline):
    ranked=[]
    for name,job in sorted(work.items(),key=lambda v:(not bool(v[1].get('l_raking')),v[0])):
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use];base=fo[name][use];p=_g_conditional(base,width)
        if len(yy)<67:continue
        for arm,expert in enumerate(job['oof']):
            q=_g_conditional(expert[use],width);a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True)
            f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy];theta=np.clip((g.sum()-g)/(f.sum()-f+120.),-1.,1.)
            gain=_g_fit_gain(base,yy,theta[:,None]*a,bool(items[name].get('gate')));t_arm=_q_deleted_t(gain)
            if gain.mean()>0 and t_arm>=1.8:ranked.append((float(gain.mean())/t_arm/max(1.,len(yy)*width),name,arm))
    return [(name,arm) for _,name,arm in sorted(ranked,key=lambda v:(not bool(work[v[1]].get('l_raking')),-v[0],v[1],v[2]))]

def _q_temperature_axes(p,pd):
    a=np.log(np.maximum(p,1e-6));b=np.log(np.maximum(pd,1e-6));a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
    h=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);mu=h.mean();sd=max(float(h.std()),.05)
    h=(h-mu)/sd;hd=(hd-mu)/sd;mask=np.ones(len(pd),bool)
    return [('temperature',a,b,mask),('entropy_temperature',h[:,None]*a,hd[:,None]*b,mask),('quadratic_entropy_temperature',(h*h-1)[:,None]*a,(hd*hd-1)[:,None]*b,mask)]


def _q_nonlinear_order(work,items,fo,X,given,deadline):
    ranked=[]
    for name,job in work.items():
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use];p=_g_conditional(fo[name][use],width)
        if len(yy)<67:continue
        score=0.
        for label,a,b,mask in _q_temperature_axes(p,p[:0]):
            g=a[np.arange(len(yy)),yy];f=np.sum(p*a*a,1);score=max(score,float((g.sum()**2-np.sum(g*g))/(f.sum()+120.)))
        if score>0:ranked.append((score/max(1.,len(yy)**2*width),name))
    return [name for _,name in sorted(ranked,key=lambda v:(-v[0],v[1]))]


def _q_nonlinear_proposals(name,job,items,fo,fr,X,given,fold,deadline):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        pool=[];ts=[]
        for label,a,b,mask in _q_temperature_axes(p,pd):
            sign=1. if a[np.arange(len(yy)),yy].sum()>=0 else -1.;a=sign*a;b=sign*b;ts.append(_q_deleted_t(a[np.arange(len(yy)),yy]));pool.append((label,a,b,mask))
        axes=_q_orthogonal(pool,p,pd,yy,3);ranked={};trace.append(ts)
        for cap in (1,2,3):
            if not axes:continue
            a,b=_m_joint_fisher(axes[:cap],p,pd,yy,120.)
            for step in (.35,.65):
                score=float(_g_fit_gain(base,yy,step*a,gated).sum()/max(int(fit.sum()),1))
                if score>0:ranked[('Q3_nonlinear_temperature',cap,step)]=(score,step*b,np.ones(len(dest),bool))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for cap in (1,2,3):
        ff=[{s:v for s,v in f.items() if s[1]==cap} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_q_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
            record.update(operator='Q3_nonlinear_temperature',skip=False,fit_only=True,axis_t=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out


def _q_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot):
    order=[];q3_stop=time.monotonic()
    if deadline-time.monotonic()>=.9:
        order=_q_nonlinear_order(work,items,fo,X,given,min(deadline-.05,time.monotonic()+.2));q3_stop=min(deadline-.65,time.monotonic()+.65)
    for name in order:
        if time.monotonic()>=q3_stop-.02:break
        job=work[name];before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        for oo,qr,active,hm,record in _q_nonlinear_proposals(name,job,items,fo,fr,X,given,fold,q3_stop):
            passed=_n_dominates(oo,before,y,split,active,2.8);record['kappa']=2.8
            probe=globals().get('_Q_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted and time.monotonic()<deadline:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('q_raking',[]).append(record);return
    order=_q_positive_arm_order(work,items,fo,X,given,min(deadline-.03,time.monotonic()+.35))
    for name,arm in order:
        if time.monotonic()>=deadline-.02:return
        job=work[name];before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        for oo,qr,active,hm,record in _q_positive_arm_proposals(name,job,arm,items,fo,fr,X,given,fold,min(deadline-.01,time.monotonic()+.65)):
            passed=_n_dominates(oo,before,y,split,active,2.8);record['kappa']=2.8
            probe=globals().get('_Q_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted and time.monotonic()<deadline:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('q_raking',[]).append(record);return


def _coherence_gated_blanket_and_positive_arm_specialist_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Seventeenth stage: coherence, positive arms and supported class calibration."""
    if np.isfinite(deadline):return _q_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,base_snapshot)
    key_fo,key_fr=dict(fo),dict(fr)
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        if any(job.get(k) for k in ('pure_graft','answer_graft','residual_raking','peer_raking','ef_raking','f_raking','g_raking','h_raking','i_raking','j_raking','k_raking','l_raking','m_raking','n_raking','o_raking','p_raking')):continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        proposals=_q_stable_blankets(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,key_fo,key_fr)
        proposals+=_q_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,('Q1','Q2','Q3','Q3_skip'),key_fo,key_fr)
        proposals+=_q_recovered_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,('O1_symmetric','O1_soft','O1_hierarchical','O2_specialist','O3_skip'),key_fo,key_fr)
        for oo,qr,active,hm,record in proposals:
            stable=record['operator'].startswith('Q1') and record['rho_coherence']>=.99 and record['min_fold_full_fisher_cosine']>=.95
            kappa=2.4 if stable else 3.2;record['kappa']=kappa
            amplification=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15);record['full_fit_amplification']=amplification
            passed=.25<=amplification<=1.5 and _q_dominates(oo,before,y,split,active,kappa)
            probe=globals().get('_Q_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('q_raking',[]).append(record)


#  Stage 18. Fit-only deleted calibration; no Query response is read.
def _r_joint_fisher(axes,p,pd,yy,lam=40.):
    """Ridge in exact unit Fisher coordinates, then one post-solve reliability."""
    unit=[]
    for label,a,b,mask in axes:
        scale=float(np.sqrt(np.mean(np.sum(p*a*a,axis=1))))
        if scale>1e-12:unit.append((label,a/scale,b/scale,mask))
    if not unit:return np.zeros_like(p),np.zeros_like(pd)
    a,b=_m_joint_fisher(unit,p,pd,yy,lam)
    t=min(_q_deleted_t(v[1][np.arange(len(yy)),yy]) for v in unit)
    weight=max(0.,1.-1./max(t*t,1.))
    return weight*a,weight*b


def _r_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,fit_count,gated,deadline,target=None,codes=None,dc=None,cells=1):
    """A solved log-linear shift or an exact deleted direct simplex step."""
    if time.monotonic()>=deadline:return False
    if a is not None:
        gain=_g_fit_gain(base,yy,a,gated)
        if gain.mean()>0:
            ranked[spec+('tangent',1.)]=(float(gain.sum()/fit_count),b,mask)
    if target is not None:
        q,qd=target;scale=1-.001*base.shape[1];idx=np.arange(len(yy));mass=base[:,:p.shape[1]].sum(1)
        aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
        z=dd/aa
        if float(z.sum())<1.8*float(np.sqrt(np.sum(z*z))):return True
        codes=np.zeros(len(yy),int) if codes is None else codes;dc=np.zeros(len(pd),int) if dc is None else dc
        sol=_k_loo_kkt(aa,dd,codes,cells,deadline,maximum=1.)
        if sol is None:return False
        alpha,loo=sol;gain=np.log1p(np.maximum(loo*dd/aa,-1+1e-12))
        if gain.mean()>0 and _q_deleted_t(gain)>=1.8:
            ranked[spec+('kkt',)]=(float(gain.sum()/fit_count),(1-alpha[dc,None])*pd+alpha[dc,None]*qd,mask)
    return True


def _r_orthogonal(axes,p,pd,yy,count=4):
    """T-screened unit Fisher axes; reliability is applied only after solving."""
    chosen=[];available=list(axes)
    while available and len(chosen)<count:
        scored=[]
        for i,(label,a,b,mask) in enumerate(available):
            a=a.copy();b=b.copy()
            for _,u,v,_ in chosen:
                c=float(np.sum(p*a*u)/max(np.sum(p*u*u),1e-15));a-=c*u;b-=c*v
            norm=float(np.sqrt(np.mean(np.sum(p*a*a,1))))
            if norm<1e-7:continue
            a/=norm;b/=norm;f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy]
            beta=np.clip((g.sum()-g)/(f.sum()-f+60.),-1.,1.)
            q=p*np.exp(np.clip(beta[:,None]*a,-2.,2.));q/=q.sum(1,keepdims=True)
            gain=np.log(np.maximum(q[np.arange(len(yy)),yy],1e-12)/np.maximum(p[np.arange(len(yy)),yy],1e-12))
            t_axis=_q_deleted_t(g)
            if gain.mean()<=0 or t_axis<1.8:continue
            shrink=max(0.,1.-1./t_axis**2)
            scored.append((float(gain.mean())*shrink,-i,i,label,a,b,mask))
        if not scored:break
        _,_,i,label,a,b,mask=max(scored,key=lambda v:v[:2]);chosen.append((label,a,b,mask));available.pop(i)
    return chosen

def _r_specialist_diagonal(arms,p,pd,yy,c,dc,lam=120.):
    """Rare class deviations shrink exactly to the deleted scalar arm weight."""
    out=np.zeros_like(p);dest=np.zeros_like(pd);one=np.eye(p.shape[1])[yy]
    support=np.bincount(yy,minlength=p.shape[1]);normalizer=max(len(arms),1)
    for _,a,b,_ in arms:
        grad=(one-p)*a;fisher=p*(1-p)*a*a;g=grad.sum(1);f=np.sum(p*a*a,1)
        scalar=g.sum()/(f.sum()+lam);sl=(g.sum()-g)/(f.sum()-f+lam)
        coef=(grad.sum(0)+lam*scalar)/(fisher.sum(0)+lam)
        deleted=(grad.sum(0)[None]-grad+lam*sl[:,None])/(fisher.sum(0)[None]-fisher+lam)
        coef=np.where(support>=25,coef,scalar)
        deleted=np.where(support[None]-one>=25,deleted,sl[:,None])
        af=a*deleted;bd=b*coef
        for cell in np.unique(c):
            ix=np.flatnonzero(c==cell);di=np.flatnonzero(dc==cell)
            if len(ix)<67:continue
            local=one[ix].sum(0)
            cf=(grad[ix].sum(0)+lam*coef)/(fisher[ix].sum(0)+lam)
            cl=(grad[ix].sum(0)[None]-grad[ix]+lam*deleted[ix])/(fisher[ix].sum(0)[None]-fisher[ix]+lam)
            cf=np.where(local>=25,cf,scalar);cl=np.where(local[None]-one[ix]>=25,cl,sl[ix,None])
            af[ix]=a[ix]*cl;bd[di]=b[di]*cf
        out+=af/normalizer;dest+=bd/normalizer
    out-=(p*out).sum(1,keepdims=True);dest-=(pd*dest).sum(1,keepdims=True)
    return np.clip(out,-1.,1.),np.clip(dest,-1.,1.)

def _r_query_average(query_fits,spec,before,gated,skip):
    """Label-free six-fit Fisher direction coherence and full-fit cosine."""
    spec=tuple(spec);simplex=spec[-1]=='kkt' and not skip
    if skip:
        rho=np.clip(before[:,-1],1e-9,1-1e-9);pd=(rho*(1-rho))[:,None]
    else:pd=_g_conditional(before,before.shape[1]-int(gated))
    shifts=[];support=[]
    for fit in query_fits:
        _,value,mask=fit[spec]
        if simplex:
            a=np.log(np.maximum(value,1e-12)/np.maximum(pd,1e-12))
            a-=(pd*a).sum(1,keepdims=True)
        else:a=value.copy()
        shifts.append(np.where(mask[:,None],a,0.));support.append(mask)
    mean=sum(shifts)/len(shifts);energy=[float(np.sum(pd*a*a)) for a in shifts]
    rho_coherence=float(np.sum(pd*mean*mean)/max(float(np.mean(energy)),1e-30))
    full=shifts[-1];cosines=[float(np.sum(pd*a*full)/max(np.sqrt(e*energy[-1]),1e-30)) for a,e in zip(shifts[:-1],energy[:-1])]
    cosine=min(cosines,default=0.);passed=rho_coherence>=.85 and cosine>=.65
    common=max(0.,(len(shifts)*rho_coherence-1.)/(len(shifts)-1.))
    shrink=min(1.,common*max(0.,cosine))
    if simplex:
        value=sum(np.where(mask[:,None],fit[spec][1],pd) for fit,mask in zip(query_fits,support))/len(query_fits)
        value=value
    else:value=mean
    mask=(sum(support)>=4)&support[-1]
    return value,mask,dict(rho_coherence=rho_coherence,min_fold_full_fisher_cosine=cosine,coherence_query_shrinkage=1.,passed=bool(passed))

def _r_stable_blankets(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,key_fo,key_fr):
    """Align actual peer/route identities before testing six-fit query directions."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo,key_fr,X,given,levels,peers);contexts=[];pquery=_g_conditional(before_r,width)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        context=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        fit,use,rr,yy,base,p,dest,db,pd,residual=context
        if len(rr)<67:return []
        selected=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        given_routes=[v for v in routes if not v[0].startswith(('peer:','parent:'))]
        raw=_p_axis_pool(given_routes,rr,dest,yy,p,pd);labels=[v[0] for v in raw[:3]];lookup=dict(given_routes);axes={}
        for label,a,b,mask in raw:
            axes[('given',label)]=(repr(('given',label)),a,b,mask)
        for key in selected:
            q=snap[key]
            # Keep the original class identities; their fit-dependent variance
            # ordering must not relabel query coordinates between partitions.
            order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])];order=np.sort(order)
            W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
            for label in ['global']+labels[:2]:
                ck=np.zeros(len(X),int) if label=='global' else lookup[label]
                a,b=W[rr],W[dest]
                if label!='global':
                    counts=np.bincount(ck[rr]);top=np.argsort(-counts,kind='stable')[:4]
                    a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1);b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                if eb is None:continue
                a,b,qf,qd,mask,dm=eb
                if mask.sum()<67:continue
                ident=('soft',key,label,tuple(int(i) for i in order));axes[ident]=(repr(ident),a,b,dm)
        contexts.append((held,context,axes))
    common=set.intersection(*(set(v[2]) for v in contexts));reliability={};axis_metrics=[]
    for ident in sorted(common,key=repr):
        values=[]
        for held,ctx,axes in contexts:
            _,_,b,m=axes[ident];count=0 if held is None else int((split==held).sum());values.append(np.where(m[count:,None],b[count:],0.))
        mean=sum(values)/6.;energy=[float(np.sum(pquery*b*b)) for b in values];rho=float(np.sum(pquery*mean*mean)/max(float(np.mean(energy)),1e-30));cos=min(float(np.sum(pquery*b*values[-1])/max(np.sqrt(e*energy[-1]),1e-30)) for b,e in zip(values[:-1],energy[:-1]))
        weight=max(0.,(6*rho-1)/5)*max(0.,cos)
        axis_metrics.append(dict(axis=repr(ident),rho_coherence=rho,min_fold_full_fisher_cosine=cos,weight=weight))
        if rho>1/6 and cos>0:reliability[ident]=weight
    fits=[];query_fits=[]
    for held,ctx,axes in contexts:
        fit,use,rr,yy,base,p,dest,db,pd,residual=ctx;pool=[]
        for ident,weight in reliability.items():
            label,a,b,m=axes[ident];pool.append((label,a,b,m))
        orthogonal=_r_orthogonal(pool,p,pd,yy,4);ranked={}
        if not orthogonal:return []
        for cap in (1,2,3,4):
            chosen=orthogonal[:cap]
            for lam in (40.,90.):
                a,b=_r_joint_fisher(chosen,p,pd,yy,lam)
                _r_fisher_kkt(ranked,('R1_stable_blanket',cap,lam),base,p,pd,yy,a,b,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline)
        held_map,query_map=_p_fit_destinations(ranked,held,split,True);fits.append(held_map);query_fits.append(query_map)
    output=[]
    for cap in (1,2,3,4):
        for lam in (40.,90.):
            ff=[{s:v for s,v in f.items() if s[1:3]==(cap,lam)} for f in fits]
            for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
                sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
                if not coherence['passed']:continue
                active=mo&source;hm=mr&hidden
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
                idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
                record.update(operator='R1_stable_blanket',se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=False,axis_coherence=axis_metrics,**coherence)
                output.append((oo,qr,active,hm,record))
    return output

def _r_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers);fits=[];query_fits=[];support_trace=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=[] if bounded and channels==('R2',) else list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        raw=(_n_axis_pool if bounded else _p_axis_pool)([v for v in routes if not v[0].startswith(('peer:','parent:'))],rr,dest,yy,p,pd)
        labels=[v[0] for v in raw[:3]];given_routes=[v for label in labels for v in routes if v[0]==label]
        ranked={};den=max(int(fit.sum()),1);allmask=np.ones(len(dest),bool);floor_pool={};soft_fine=0;hier_fine=0
        def emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
            return _r_fisher_kkt(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target)
        if 'R1' in channels:
            axes=[];single=[]
            for rank,key in enumerate(selected):
                q=snap[key];order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for route in range(1+len(given_routes)):
                    ck=np.zeros(len(X),int) if route==0 else given_routes[route-1][1]
                    a,b=W[rr],W[dest]
                    if route:
                        n=np.bincount(ck[rr]);top=np.argsort(-n,kind='stable')[:4]
                        a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1)
                        b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                    support=a.sum(0);soft_fine+=int(((support>=20)&(support<67)).sum())
                    eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                    if eb is None:continue
                    af,ad,qf,qd,mask,dm=eb
                    if int(mask.sum())<67:continue
                    label=('soft',rank,route);axes.append((repr(label),af,ad,dm));single.append((label,af,ad,qf,qd,dm))
            for rank,(_,ck) in enumerate(given_routes):
                counts=np.bincount(ck[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                eb=_p_cells(ck[rr],ck[dest],np.zeros(len(rr),int),np.zeros(len(dest),int),yy,p,pd)
                if eb is not None:
                    a,b,q,qd,mask,dm=eb;axes.append(('given'+str(rank),a,b,dm));single.append((('given',rank),a,b,q,qd,dm))
                for other in range(rank+1,len(given_routes)):
                    ck2=given_routes[other][1];_,code=np.unique(np.column_stack((ck,ck2)),axis=0,return_inverse=True)
                    counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                    eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                    if eb is not None:
                        a,b,q,qd,mask,dm=eb;label=('hierarchical',rank,other);axes.append((repr(label),a,b,dm));single.append((label,a,b,q,qd,dm))
            for label,a,b,q,qd,dm in single:
                if not emit(ranked,('R1_axis',)+label,base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd)):return []
            orthogonal=_r_orthogonal(axes,p,pd,yy,4)
            for count in (1,2,3,4):
                if len(orthogonal)<count:continue
                for lam in (40.,90.):
                    a,b=_r_joint_fisher(orthogonal[:count],p,pd,yy,lam)
                    if not emit(ranked,('R1_blanket',count,lam),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'R3' in channels:
            for pr,key in enumerate(selected[:(1 if bounded else 2)]):
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:1]
                vectors=[q[:,rank[0]],-np.sum(q*np.log(np.maximum(q,1e-12)),1)]
                for vi,v in enumerate(vectors):
                    for bins in ((3,) if bounded else (3,4)):
                        edges=np.unique(np.quantile(v[rr],np.arange(1,bins)/bins));codes=np.searchsorted(edges,v)
                        for gi,(_,ck) in enumerate(given_routes[:2]):
                            _,code=np.unique(np.column_stack((codes,ck)),axis=0,return_inverse=True)
                            counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                            eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                            if eb is None:continue
                            a,b,qf,qd,mask,dm=eb
                            if not emit(ranked,('R3_quantile',pr,vi,bins,gi),base,p,pd,yy,a,b,dm,den,gated,deadline,(qf,qd)):return []
            xf,xd=_p_peer_features(selected,snap,given_routes,rr,dest,p,pd,residual);center=xf.mean(0);xf-=center;xd-=center
            for method in ('svd','pls'):
                matrix=xf.T@xf if method=='svd' else (xf.T@residual)@(residual.T@xf)
                eigen,vectors=np.linalg.eigh(matrix);vectors=vectors[:,np.argsort(-eigen)]
                for rank in (3,):
                    effective=min(rank,int((eigen>max(float(eigen.max()),1.)*1e-10).sum()))
                    if effective<1:continue
                    v=vectors[:,:effective];a=xf@v;b=xd@v;scale=np.maximum(a.std(0),.05);a/=scale;b/=scale
                    for lam in (180.,):
                        af,ad,_=_h_ridge_press(a,b,residual,lam);af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                        if not emit(ranked,('R3_press',method,rank,lam),base,p,pd,yy,af,ad,allmask,den,gated,deadline):return []
        if 'R2' in channels and len(job['oof'])>1:
            arms=[]
            for ai,(oo,full) in enumerate(zip(job['oof'],job['all'])):
                if time.monotonic()>=deadline:return []
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((oo[split==held],full[rest]),axis=0),width)
                if not bounded:
                    q[q.sum(1)<=1e-12]=p[q.sum(1)<=1e-12];qd[qd.sum(1)<=1e-12]=pd[qd.sum(1)<=1e-12]
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                score=float(_g_fit_gain(base,yy,theta[:,None]*a,gated).sum()/den)
                idx=np.arange(len(yy));scale=1-.001*job['K'];mass=base[:,:width].sum(1);aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                solution=(np.array([1.]),np.ones(len(yy)))
                if solution is None:return []
                alpha,loo=solution;kkt_gain=float(np.log1p(loo*dd/aa).sum()/den)
                energy=float(grad.sum())**2/(float(f.sum())+120.)/den
                diag_a,_=_r_specialist_diagonal([('arm',a,b,allmask)],p,pd,yy,np.zeros(len(rr),int),np.zeros(len(dest),int))
                class_gain=float(_g_fit_gain(base,yy,.65*diag_a,gated).sum()/den)
                gains=_g_fit_gain(base,yy,theta[:,None]*a,gated)
                t_arm=_q_deleted_t(gains)
                if score<=0 or t_arm<1.8:continue
                reliability=score/max(t_arm,1.8)
                arms.append((reliability,ai,a,b,q,qd,loo,alpha[0]))
            arms.sort(key=lambda v:(-v[0],v[1]))
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            strata=[('global',np.zeros(len(rr),int),np.zeros(len(dest),int)),('entropy',c,dc)]+[('given'+str(i),ck[rr],ck[dest]) for i,(_,ck) in enumerate(given_routes[:2])]
            for count in ((1,2,3)):
                if len(arms)<count:continue
                chosen=arms[:count];axes=[('arm_rank'+str(i),a,b,allmask) for i,(_,_,a,b,*_) in enumerate(chosen)]
                weights=np.asarray([v[0] for v in chosen]);weights/=weights.sum()
                axes=[(label,a,b,mask) for (label,a,b,mask),w in zip(axes,weights)]
                q=sum(w*v[4] for w,v in zip(weights,chosen));qd=sum(w*v[5] for w,v in zip(weights,chosen))
                a=np.log(np.maximum(q,1e-12)/np.maximum(p,1e-12));b=np.log(np.maximum(qd,1e-12)/np.maximum(pd,1e-12))
                if not emit(ranked,('R2_coordinate',count),base,p,pd,yy,a,b,allmask,den,gated,deadline,(q,qd)):return []
                for label,c,dc in strata:
                    a,b=_r_specialist_diagonal(axes,p,pd,yy,c,dc)
                    if not emit(ranked,('R2_diagonal',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
                    if False:
                        a,b=_o_specialist(axes,p,pd,yy,c,dc)
                        if not emit(ranked,('R2_joint',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'R3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fs,us,rs,ys,bs,_,ds,dbs,_,_= _p_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rs)>=67:
                rho=bs[:,-1];rd=dbs[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int);axes=[]
                skip_peers=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rs,(yb-rho)[:,None],4)))[:4]
                parent_routes=[v for v in routes if v[0].startswith('parent:')]
                parent_axes=_n_axis_pool(parent_routes,rs,ds,yb,ps,pds)
                skip_routes=given_routes[:2]+[v for axis in parent_axes[:2] for v in parent_routes if v[0]==axis[0]]
                for rank,key in enumerate(skip_peers):
                    # Parent classes remain distinct. Confidence alone erases
                    # the direction of an otherwise predictive routing state.
                    q=snap[key];center=q[rs]-q[rs].mean(0)
                    order=np.argsort(-np.square(center.T@(yb-rho)),kind='stable')[:min(4,q.shape[1])]
                    W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                    for gi in range(1+len(skip_routes)):
                        ck=np.zeros(len(X),int) if gi==0 else skip_routes[gi-1][1]
                        eb=_p_weighted_eb(W[rs],W[ds],ck[rs],ck[ds],yb,ps,pds)
                        if eb is not None:a,b,q,qd,mask,dm=eb;axes.append((str((rank,gi)),a,b,dm))
                ortho=_r_orthogonal(axes,ps,pds,yb,4)
                for count in (2,3,4):
                    if len(ortho)<count:continue
                    a,b=_r_joint_fisher(ortho[:count],ps,pds,yb,120.);a=a[:,1]-a[:,0];b=b[:,1]-b[:,0]
                    for step in (1.,):
                        score=float(_h_skip_gain(bs,ys,step*a).sum()/max(fs.sum(),1))
                        if score>0:ranked[('R3_skip',count,'fisher',step)]=(score,(step*b)[:,None],(rd>.015)&(rd<.985))
                    q=1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+a,-30,30)));qd=1/(1+np.exp(-np.clip(np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12)))+b,-30,30)))
                    target=_residual_skip_project(bs,np.ones(len(bs),bool),q);idx=np.arange(len(ys));scale=1-.001*job['K'];aa=.001+scale*bs[idx,ys];dd=scale*(target[idx,ys]-bs[idx,ys])
                    sol=_k_loo_kkt(aa,dd,np.zeros(len(ys),int),1,deadline,maximum=.4)
                    if sol is None:return []
                    alpha,loo=sol;score=float(np.log1p(loo*dd/aa).sum()/max(fs.sum(),1));new=np.clip((1-alpha[0])*rd+alpha[0]*qd,1e-12,1-1e-12)
                    if score>0:ranked[('R3_skip',count,'kkt')]=(score,(np.log(new/(1-new))-np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12))))[:,None],(rd>.015)&(rd<.985))
        for spec,(_,args) in []:
            base,p,pd,yy,mask,den,gated,target=args
            if not _r_fisher_kkt(ranked,spec+('fit_selected',),base,p,pd,yy,None,None,mask,den,gated,deadline,target):return []
        # Fit-selected coordinate ranks are estimator identities. Peer names,
        # demographic labels and local coordinate enumeration never enter them.
        groups={}
        for spec,value in list(ranked.items()):
            if spec[0]=='R1_axis' and spec[-1]!='kkt':family=('R1_rank',spec[1],spec[-2],spec[-1])
            elif spec[0]=='R3_quantile' and spec[-1]!='kkt':family=('R3_risk_rank',spec[2],spec[3],spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                ranked[family[:-2]+(rank,)+family[-2:]]=value
        support_trace.append(dict(soft_cells_20_66=soft_fine,hierarchical_cells_25_66=hier_fine,available_specialist_arms=len(job['oof'])))
        held_fit,query_fit=_p_fit_destinations(ranked,held,split,True);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    def family_key(s):return (s[0],s[2] if s[0] in ('R2_diagonal','R2_joint') else s[1] if s[0]=='R1_blanket' else '',s[-2] if s[-1] in (.35,.65) else s[-1])
    for family in sorted({family_key(s) for s in fits[-1]},key=repr):
        ff=[{s:v for s,v in f.items() if family_key(s)==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];skip=spec[0]=='R3_skip'
            sr,mr,coherence=_r_query_average(query_fits,spec,before_r,gated,skip)
            if not coherence['passed']:continue
            record.update(coherence)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=spec[0],se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,fold_invariant_spec=True,fine_support=support_trace,query_pooling='unattenuated_six_fit_average')
            proposals.append((oo,qr,active,hm,record))
    return proposals

def _r_positive_arm_proposals(name,job,arm,items,fo,fr,X,given,fold,deadline):
    """Positive deleted scalar/KKT calibration of a complete fit-only expert."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];expert,full=job['oof'][arm],job['all'][arm];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        q=_g_conditional(expert[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((expert[split==held],full[rest]),axis=0),width)
        a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+40.),-1.,1.);coef=np.clip(grad.sum()/(f.sum()+40.),-1.,1.)
        gain=_g_fit_gain(base,yy,theta[:,None]*a,gated);t_arm=_q_deleted_t(gain);ranked={};trace.append(dict(deleted_gain=float(gain.mean()),t_arm=t_arm))
        if gain.mean()>0 and t_arm>=1.8:
            shrink=max(0.,1.-1./t_arm**2);af=theta[:,None]*a;ad=coef*b
        else:af,ad=None,None
        _r_fisher_kkt(ranked,('R2_positive_arm',),base,p,pd,yy,af,ad,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline,(q,qd))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for mode in ('tangent','kkt'):
        ff=[{s:v for s,v in f.items() if (s[-1]=='kkt' if mode=='kkt' else s[-2]==mode)} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            project=_j_simplex_project if record['spec'][-1]=='kkt' else _ef_project
            oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            record.update(operator='R2_positive_arm',skip=False,fit_only=True,arm=arm,positive_arm_trace=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out

def _r_positive_arm_order(work,items,fo,X,given,deadline):
    ranked=[]
    for name,job in sorted(work.items(),key=lambda v:(not bool(v[1].get('l_raking')),v[0])):
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use];base=fo[name][use];p=_g_conditional(base,width)
        if len(yy)<67:continue
        for arm,expert in enumerate(job['oof']):
            q=_g_conditional(expert[use],width);a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True)
            f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy];theta=np.clip((g.sum()-g)/(f.sum()-f+120.),-1.,1.)
            gain=_g_fit_gain(base,yy,theta[:,None]*a,bool(items[name].get('gate')));t_arm=_q_deleted_t(gain)
            if gain.mean()>0 and t_arm>=1.8:ranked.append((float(gain.mean())/t_arm/max(1.,len(yy)*width),name,arm))
    return [(name,arm) for _,name,arm in sorted(ranked,key=lambda v:(not bool(work[v[1]].get('l_raking')),-v[0],v[1],v[2]))]

def _r_nonlinear_order(work,items,fo,X,given,deadline):
    ranked=[]
    for name,job in work.items():
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use];p=_g_conditional(fo[name][use],width)
        if len(yy)<67:continue
        score=0.
        for label,a,b,mask in _q_temperature_axes(p,p[:0]):
            g=a[np.arange(len(yy)),yy];f=np.sum(p*a*a,1);score=max(score,float((g.sum()**2-np.sum(g*g))/(f.sum()+120.)))
        if score>0:ranked.append((score/max(1.,len(yy)**2*width),name))
    return [name for _,name in sorted(ranked,key=lambda v:(-v[0],v[1]))]

def _r_nonlinear_proposals(name,job,items,fo,fr,X,given,fold,deadline):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        pool=[];ts=[]
        for label,a,b,mask in _q_temperature_axes(p,pd):
            sign=1. if a[np.arange(len(yy)),yy].sum()>=0 else -1.;a=sign*a;b=sign*b;ts.append(_q_deleted_t(a[np.arange(len(yy)),yy]));pool.append((label,a,b,mask))
        axes=_r_orthogonal(pool,p,pd,yy,3);ranked={};trace.append(ts)
        for cap in (1,2,3):
            if not axes:continue
            a,b=_r_joint_fisher(axes[:cap],p,pd,yy,40.)
            for step in (1.,):
                score=float(_g_fit_gain(base,yy,step*a,gated).sum()/max(int(fit.sum()),1))
                if score>0:ranked[('R3_nonlinear_temperature',cap,step)]=(score,step*b,np.ones(len(dest),bool))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for cap in (1,2,3):
        ff=[{s:v for s,v in f.items() if s[1]==cap} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
            record.update(operator='R3_nonlinear_temperature',skip=False,fit_only=True,axis_t=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out
def _r_global_axes(p,pd):
    axes=[]
    for c in range(p.shape[1]):
        a=np.zeros_like(p);b=np.zeros_like(pd);a[:,c]=1.;b[:,c]=1.
        a-=p[:,c,None];b-=pd[:,c,None]
        axes.append(('class_intercept_'+str(c),a,b,np.ones(len(pd),bool)))
    return axes


def _r_global_proposals(name,job,items,fo,fr,X,given,fold,deadline):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        pool=[];ts=[]
        for label,a,b,mask in _r_global_axes(p,pd):
            sign=1. if a[np.arange(len(yy)),yy].sum()>=0 else -1.;a=sign*a;b=sign*b;ts.append(_q_deleted_t(a[np.arange(len(yy)),yy]));pool.append((label,a,b,mask))
        axes=_r_orthogonal(pool,p,pd,yy,3);ranked={};trace.append(ts)
        for cap in (1,2,3):
            if not axes:continue
            a,b=_r_joint_fisher(axes[:cap],p,pd,yy,40.)
            for step in (1.,):
                score=float(_g_fit_gain(base,yy,step*a,gated).sum()/max(int(fit.sum()),1))
                if score>0:ranked[('R1_global_intercept',cap,step)]=(score,step*b,np.ones(len(dest),bool))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for cap in (1,2,3):
        ff=[{s:v for s,v in f.items() if s[1]==cap} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            oo=_ef_project(before,so,active,1.,gated);qr=_ef_project(before_r,sr,hm,1.,gated)
            record.update(operator='R1_global_intercept',skip=False,fit_only=True,axis_t=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out


def _r_entropy_transport(arm,before,query,y,split,active,hidden,gated,kappa):
    """Evaluate deleted gains at the unlabeled Query entropy density."""
    width=before.shape[1]-int(gated)
    if active.sum()<67 or hidden.sum()<1:return False,dict(entropy_transport_passed=False)
    p=_g_conditional(before,width);pd=_g_conditional(query,width)
    h=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1)
    edges=np.unique(np.quantile(h[active],np.arange(1,5)/5.));cells=len(edges)+1
    c=np.searchsorted(edges,h[active]);dc=np.searchsorted(edges,hd[hidden])
    n=np.bincount(c,minlength=cells);nd=np.bincount(dc,minlength=cells)
    weights=((nd+.5)/(nd.sum()+.5*cells))/((n+.5)/(n.sum()+.5*cells))
    ix=np.flatnonzero(active);scale=1-.001*before.shape[1]
    gain=(np.log(.001+scale*arm[ix,y[ix]])-np.log(.001+scale*before[ix,y[ix]]))*weights[c]
    sp=split[active];fg=np.asarray([gain[sp==f].mean() for f in np.unique(sp)])
    se=max(float(gain.std(ddof=1)/np.sqrt(len(gain))),float(fg.std(ddof=1)/np.sqrt(len(fg))),1e-15)
    passed=len(fg)==5 and gain.mean()>1e-9 and np.all(fg>=0) and all(gain[sp%2==half].mean()>0 for half in (0,1)) and gain.mean()>=kappa*se
    return bool(passed),dict(entropy_transport_passed=bool(passed),transport_gain=float(gain.mean()),transport_t=float(gain.mean()/se),transport_fold_gains=fg.tolist(),transport_fit_counts=n.tolist(),transport_query_counts=nd.tolist(),transport_weights=weights.tolist(),transport_query_labels_used=False)


def _unshrunk_gram_schmidt_and_direct_simplex_specialist_raking(items,work,final,fo,fr,X,given,levels,fold,nll_fn,deadline,peers,base_snapshot):
    """Eighteenth stage: unit Fisher blankets and direct specialist calibration."""
    if np.isfinite(deadline):return _r_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,base_snapshot)
    key_fo,key_fr=dict(fo),dict(fr)
    for name,job in _s136_order(work,items,fo,X,given,levels,deadline):
        if any(job.get(k) for k in ('pure_graft','answer_graft','residual_raking','peer_raking','ef_raking','f_raking','g_raking','h_raking','i_raking','j_raking','k_raking','l_raking','m_raking','n_raking','o_raking','p_raking','q_raking')):continue
        before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
        # Global contrasts are tested across the complete question, using
        # the existing 4-SE and substantive-gain rule before richer families.
        global_proposals=_r_global_proposals(name,job,items,fo,fr,X,given,fold,deadline)
        fixed_global=max(global_proposals,key=lambda v:v[4]['fit_mean_gain'],default=None)
        for oo,qr,active,hm,record in ([] if fixed_global is None else [fixed_global]):
            record['kappa']=4.;record['global_full_question_test']=True
            amplification=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15);record['full_fit_amplification']=amplification
            passed=.25<=amplification<=1.5 and _ef_dominates(oo,before,y,split,active,all_rows=True,kappa=4.)
            probe=globals().get('_R_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0])
            assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate'):assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('r_raking',[]).append(record);continue
        proposals=_r_stable_blankets(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,key_fo,key_fr)
        proposals+=_r_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,base_snapshot,deadline,('R1','R2','R3','R3_skip'),key_fo,key_fr)
        proposals+=_r_nonlinear_proposals(name,job,items,fo,fr,X,given,fold,deadline)
        for arm in range(len(job['oof'])):
            proposals+=_r_positive_arm_proposals(name,job,arm,items,fo,fr,X,given,fold,deadline)
        for oo,qr,active,hm,record in proposals:
            stable=record['operator'].startswith('R1') and record['rho_coherence']>=.99 and record['min_fold_full_fisher_cosine']>=.95
            kappa=2.4 if record['operator'].startswith('R1') else 3.2;record['kappa']=kappa
            amplification=record['fit_scores'][-1]/max(record['fit_mean_gain'],1e-15);record['full_fit_amplification']=amplification
            passed=.25<=amplification<=1.5 and _q_dominates(oo,before,y,split,active,kappa)
            if record['operator']=='R3_nonlinear_temperature':
                transported,transport=_r_entropy_transport(oo,before,before_r,y,split,active,hm,bool(items[name].get('gate')),kappa)
                record.update(transport);passed=passed and transported
            probe=globals().get('_R_PROBE')
            if probe is not None:probe(name,record,oo,qr,passed)
            if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
        if accepted:
            _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0]);assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
            if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
            fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('r_raking',[]).append(record)

def _r_finite_positive_order(work,items,fo,X,given,deadline):
    """A deterministic small sample orders arms; acceptance always uses all rows."""
    ranked=[]
    for name,job in sorted(work.items(),key=lambda v:(len(v[1]['oof'][0])*(v[1]['K']-int(bool(items[v[0]].get('gate'))))*len(v[1]['oof']),v[0])):
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];indices=np.flatnonzero(source&(y<width))
        if len(indices)<67:continue
        take=indices[np.linspace(0,len(indices)-1,min(512,len(indices)),dtype=int)];yy=y[take];base=fo[name][take];p=_g_conditional(base,width)
        for arm,expert in enumerate(job['oof']):
            q=_g_conditional(expert[take],width);a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True)
            f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy]
            idx=np.arange(len(yy));scale=1-.001*job['K'];aa=.001+scale*base[idx,yy];dd=scale*base[:,:width].sum(1)*(q[idx,yy]-p[idx,yy]);z=dd/aa
            if g.sum()<=0 and z.sum()<=0:continue
            theta=np.clip((g.sum()-g)/(f.sum()-f+40.),0.,1.)
            alpha=np.clip(z.sum()/max(float(np.sum(z*z)),1e-15),0.,1.)
            simplex_score=float(np.log1p(np.maximum(alpha*z,-1+1e-12)).mean())
            score=max(float(np.mean(theta*g-.5*theta*theta*f)),simplex_score)/max(1.,width*len(indices)/5000.)
            if score>0:ranked.append((score,name,arm))
    verified=[]
    for _,name,arm in sorted(ranked,key=lambda v:(-v[0],v[1],v[2]))[:48]:
        if time.monotonic()>=deadline:break
        job=work[name];rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use];base=fo[name][use];p=_g_conditional(base,width);q=_g_conditional(job['oof'][arm][use],width)
        a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True);f=np.sum(p*a*a,1);g=a[np.arange(len(yy)),yy]
        theta=np.clip((g.sum()-g)/(f.sum()-f+40.),0.,1.);gain=_g_fit_gain(base,yy,theta[:,None]*a,bool(items[name].get('gate')))
        idx=np.arange(len(yy));scale=1-.001*job['K'];aa=.001+scale*base[idx,yy];dd=scale*base[:,:width].sum(1)*(q[idx,yy]-p[idx,yy]);z=dd/aa
        direct_positive=float(z.sum())>=1.8*float(np.sqrt(np.sum(z*z)))
        alpha=np.clip(z.sum()/max(float(np.sum(z*z)),1e-15),0.,1.);simplex_score=float(np.log1p(np.maximum(alpha*z,-1+1e-12)).mean())
        if gain.mean()>0 or direct_positive:verified.append((max(float(gain.mean()),simplex_score if direct_positive else 0.)/max(1.,len(yy)*width),name,arm))
    return [(name,arm) for _,name,arm in sorted(verified,key=lambda v:(-v[0],v[1],v[2]))]


def _r_finite_tangent_emit(ranked,spec,base,p,pd,yy,a,b,mask,fit_count,gated,deadline,target=None):
    """Complete the log-linear bundle independently of exact simplex roots."""
    if time.monotonic()>=deadline:return False
    if a is not None:
        gain=_g_fit_gain(base,yy,a,gated)
        if gain.mean()>0:ranked[spec+('tangent',1.)]=(float(gain.sum()/fit_count),b,mask)
    return True

def _r_finite_loglinear_proposals(name,job,arm,items,fo,fr,X,given,fold,deadline,lam=40.):
    """Positive deleted scalar/KKT calibration of a complete fit-only expert."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];expert,full=job['oof'][arm],job['all'][arm];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        q=_g_conditional(expert[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((expert[split==held],full[rest]),axis=0),width)
        a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+lam),-1.,1.);coef=np.clip(grad.sum()/(f.sum()+lam),-1.,1.)
        gain=_g_fit_gain(base,yy,theta[:,None]*a,gated);t_arm=_q_deleted_t(gain);ranked={};trace.append(dict(deleted_gain=float(gain.mean()),t_arm=t_arm))
        if gain.mean()>0 and t_arm>=1.8:
            shrink=max(0.,1.-1./t_arm**2);af=theta[:,None]*a;ad=coef*b
            _r_finite_tangent_emit(ranked,('R2_positive_arm',),base,p,pd,yy,af,ad,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline,(q,qd))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for mode in ('tangent',):
        ff=[{s:v for s,v in f.items() if (s[-1]=='kkt' if mode=='kkt' else s[-2]==mode)} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            project=_j_simplex_project if record['spec'][-1]=='kkt' else _ef_project
            oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            record.update(operator='R2_positive_arm',ridge=lam,skip=False,fit_only=True,arm=arm,positive_arm_trace=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out



def _r_finite_direct_proposals(name,job,arm,items,fo,fr,X,given,fold,deadline,lam=40.):
    """Positive deleted scalar/KKT calibration of a complete fit-only expert."""
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<2 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if source.sum()<67:return []
    before,before_r=fo[name],fr[name];expert,full=job['oof'][arm],job['all'][arm];fits=[];query_fits=[];trace=[]
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        q=_g_conditional(expert[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((expert[split==held],full[rest]),axis=0),width)
        a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.);a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
        f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+lam),-1.,1.);coef=np.clip(grad.sum()/(f.sum()+lam),-1.,1.)
        gain=_g_fit_gain(base,yy,theta[:,None]*a,gated);t_arm=_q_deleted_t(gain);ranked={};trace.append(dict(deleted_gain=float(gain.mean()),t_arm=t_arm))
        if True:
            shrink=max(0.,1.-1./t_arm**2);af=theta[:,None]*a;ad=coef*b
            _r_fisher_kkt(ranked,('R2_direct_arm',),base,p,pd,yy,None,None,np.ones(len(dest),bool),max(int(fit.sum()),1),gated,deadline,(q,qd))
        hm,qm=_p_fit_destinations(ranked,held,split,True);fits.append(hm);query_fits.append(qm)
    out=[]
    for mode in ('kkt',):
        ff=[{s:v for s,v in f.items() if (s[-1]=='kkt' if mode=='kkt' else s[-2]==mode)} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            sr,mr,coherence=_r_query_average(query_fits,record['spec'],before_r,gated,False)
            if not coherence['passed']:continue
            active=mo&source;hm=mr&hidden
            if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
            project=_j_simplex_project if record['spec'][-1]=='kkt' else _ef_project
            oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            record.update(operator='R2_direct_simplex_arm',ridge=lam,skip=False,fit_only=True,arm=arm,positive_arm_trace=trace,**coherence);out.append((oo,qr,active,hm,record))
    return out


def _r_finite_global_order(work,items,fo,X,given,deadline):
    ranked=[]
    for name,job in work.items():
        if time.monotonic()>=deadline:break
        rows=np.flatnonzero(job['train']);width=job['K']-int(bool(items[name].get('gate')));source=_wave_two_source(name,items,X,given,rows)
        if source is None or width<2:continue
        if 'wave' in given:
            wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
            if wave is not None:source &= X[rows,given.index('wave')]!=wave
        y=job['y'][rows];use=source&(y<width);yy=y[use]
        if len(yy)<67:continue
        p=_g_conditional(fo[name][use],width);residual=np.eye(width)[yy]-p
        grad=residual.sum(0);fisher=np.sum(p*(1-p),0);noise=np.sum(residual*residual,0)
        score=float(np.sum(np.maximum(grad*grad-noise,0.)/(fisher+40.)))/max(1.,len(yy)**2*width)
        if score>0:ranked.append((score,name))
    return [name for _,name in sorted(ranked,key=lambda v:(-v[0],v[1]))]






def _r_finite_pool_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,deadline,channels,key_fo=None,key_fr=None):
    rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows]
    gated=bool(items[name].get('gate'));width=job['K']-int(gated)
    source=_wave_two_source(name,items,X,given,rows);hidden=_wave_two_source(name,items,X,given,rest)
    if source is None or hidden is None or width<1 or len(np.unique(split))!=5:return []
    if 'wave' in given:
        wave=next((i for i,v in enumerate(items['wave']['values']) if str(v)=='1'),None)
        if wave is not None:source &= X[rows,given.index('wave')]!=wave;hidden &= X[rest,given.index('wave')]!=wave
    if int(source.sum())<67:return []
    before,before_r=fo[name],fr[name];snap={k:v for k,v in snapshot.items() if k!=name}
    routes=_m_routes(name,items,work,key_fo or fo,key_fr or fr,X,given,levels,peers);fits=[];query_fits=[];support_trace=[];bounded=np.isfinite(deadline)
    for held in list(np.unique(split))+[None]:
        if time.monotonic()>=deadline:return []
        fit,use,rr,yy,base,p,dest,db,pd,residual=_p_fold_context(rows,rest,y,split,before,before_r,source,gated,held,False)
        if len(rr)<67:return []
        selected=[] if bounded and channels==('R2',) else list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rr,residual,4)))[:4]
        raw=(_n_axis_pool if bounded else _p_axis_pool)([v for v in routes if not v[0].startswith(('peer:','parent:'))],rr,dest,yy,p,pd)
        labels=[v[0] for v in raw[:3]];given_routes=[v for label in labels for v in routes if v[0]==label]
        ranked={};den=max(int(fit.sum()),1);allmask=np.ones(len(dest),bool);floor_pool={};soft_fine=0;hier_fine=0
        def emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target=None):
            return _r_finite_tangent_emit(ranked,spec,base,p,pd,yy,a,b,mask,den,gated,deadline,target)
        if 'R1' in channels:
            axes=[];single=[]
            for rank,key in enumerate(selected):
                q=snap[key];order=np.argsort(-q[rr].var(0),kind='stable')[:min(4,q.shape[1])]
                W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                for route in range(1+len(given_routes)):
                    ck=np.zeros(len(X),int) if route==0 else given_routes[route-1][1]
                    a,b=W[rr],W[dest]
                    if route:
                        n=np.bincount(ck[rr]);top=np.argsort(-n,kind='stable')[:4]
                        a=(a[:,:,None]*(ck[rr,None]==top)[:,None,:]).reshape(len(rr),-1)
                        b=(b[:,:,None]*(ck[dest,None]==top)[:,None,:]).reshape(len(dest),-1)
                    support=a.sum(0);soft_fine+=int(((support>=20)&(support<67)).sum())
                    eb=_p_weighted_eb(a,b,ck[rr],ck[dest],yy,p,pd)
                    if eb is None:continue
                    af,ad,qf,qd,mask,dm=eb
                    if int(mask.sum())<67:continue
                    label=('soft',rank,route);axes.append((repr(label),af,ad,dm));single.append((label,af,ad,qf,qd,dm))
            for rank,(_,ck) in enumerate(given_routes):
                counts=np.bincount(ck[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                eb=_p_cells(ck[rr],ck[dest],np.zeros(len(rr),int),np.zeros(len(dest),int),yy,p,pd)
                if eb is not None:
                    a,b,q,qd,mask,dm=eb;axes.append(('given'+str(rank),a,b,dm));single.append((('given',rank),a,b,q,qd,dm))
                for other in range(rank+1,len(given_routes)):
                    ck2=given_routes[other][1];_,code=np.unique(np.column_stack((ck,ck2)),axis=0,return_inverse=True)
                    counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                    eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                    if eb is not None:
                        a,b,q,qd,mask,dm=eb;label=('hierarchical',rank,other);axes.append((repr(label),a,b,dm));single.append((label,a,b,q,qd,dm))
            for label,a,b,q,qd,dm in single:
                if not emit(ranked,('R1_axis',)+label,base,p,pd,yy,a,b,dm,den,gated,deadline,(q,qd)):return []
            orthogonal=_r_orthogonal(axes,p,pd,yy,4)
            for count in (1,2,3,4):
                if len(orthogonal)<count:continue
                for lam in (40.,90.):
                    a,b=_r_joint_fisher(orthogonal[:count],p,pd,yy,lam)
                    if not emit(ranked,('R1_blanket',count,lam),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'R3' in channels:
            for pr,key in enumerate(selected[:(1 if bounded else 2)]):
                q=snap[key];rank=np.argsort(-q[rr].var(0),kind='stable')[:1]
                vectors=[q[:,rank[0]],-np.sum(q*np.log(np.maximum(q,1e-12)),1)]
                for vi,v in enumerate(vectors):
                    for bins in ((3,) if bounded else (3,4)):
                        edges=np.unique(np.quantile(v[rr],np.arange(1,bins)/bins));codes=np.searchsorted(edges,v)
                        for gi,(_,ck) in enumerate(given_routes[:2]):
                            _,code=np.unique(np.column_stack((codes,ck)),axis=0,return_inverse=True)
                            counts=np.bincount(code[rr]);hier_fine+=int(((counts>=25)&(counts<67)).sum())
                            eb=_p_cells(code[rr],code[dest],ck[rr],ck[dest],yy,p,pd)
                            if eb is None:continue
                            a,b,qf,qd,mask,dm=eb
                            if not emit(ranked,('R3_quantile',pr,vi,bins,gi),base,p,pd,yy,a,b,dm,den,gated,deadline,(qf,qd)):return []
            xf,xd=_p_peer_features(selected,snap,given_routes,rr,dest,p,pd,residual);center=xf.mean(0);xf-=center;xd-=center
            for method in ('svd','pls'):
                matrix=xf.T@xf if method=='svd' else (xf.T@residual)@(residual.T@xf)
                eigen,vectors=np.linalg.eigh(matrix);vectors=vectors[:,np.argsort(-eigen)]
                for rank in (3,):
                    effective=min(rank,int((eigen>max(float(eigen.max()),1.)*1e-10).sum()))
                    if effective<1:continue
                    v=vectors[:,:effective];a=xf@v;b=xd@v;scale=np.maximum(a.std(0),.05);a/=scale;b/=scale
                    for lam in (180.,):
                        af,ad,_=_h_ridge_press(a,b,residual,lam);af=np.clip(af/np.maximum(p,.04),-1.,1.);ad=np.clip(ad/np.maximum(pd,.04),-1.,1.)
                        if not emit(ranked,('R3_press',method,rank,lam),base,p,pd,yy,af,ad,allmask,den,gated,deadline):return []
        if 'R2' in channels and len(job['oof'])>1:
            arms=[]
            for ai,(oo,full) in enumerate(zip(job['oof'],job['all'])):
                if time.monotonic()>=deadline:return []
                q=_g_conditional(oo[use],width);qd=_g_conditional(full[rest] if held is None else np.concatenate((oo[split==held],full[rest]),axis=0),width)
                if not bounded:
                    q[q.sum(1)<=1e-12]=p[q.sum(1)<=1e-12];qd[qd.sum(1)<=1e-12]=pd[qd.sum(1)<=1e-12]
                a=np.clip(np.log(np.maximum(q,1e-6)/np.maximum(p,1e-6)),-2.,2.);b=np.clip(np.log(np.maximum(qd,1e-6)/np.maximum(pd,1e-6)),-2.,2.)
                a-=(p*a).sum(1,keepdims=True);b-=(pd*b).sum(1,keepdims=True)
                f=np.sum(p*a*a,1);grad=a[np.arange(len(yy)),yy];theta=np.clip((grad.sum()-grad)/(f.sum()-f+120.),-1.,1.)
                score=float(_g_fit_gain(base,yy,theta[:,None]*a,gated).sum()/den)
                idx=np.arange(len(yy));scale=1-.001*job['K'];mass=base[:,:width].sum(1);aa=.001+scale*base[idx,yy];dd=scale*mass*(q[idx,yy]-p[idx,yy])
                solution=(np.array([1.]),np.ones(len(yy)))
                if solution is None:return []
                alpha,loo=solution;kkt_gain=float(np.log1p(loo*dd/aa).sum()/den)
                energy=float(grad.sum())**2/(float(f.sum())+120.)/den
                diag_a,_=_r_specialist_diagonal([('arm',a,b,allmask)],p,pd,yy,np.zeros(len(rr),int),np.zeros(len(dest),int))
                class_gain=float(_g_fit_gain(base,yy,.65*diag_a,gated).sum()/den)
                gains=_g_fit_gain(base,yy,theta[:,None]*a,gated)
                t_arm=_q_deleted_t(gains)
                if score<=0 or t_arm<1.8:continue
                reliability=score/max(t_arm,1.8)
                arms.append((reliability,ai,a,b,q,qd,loo,alpha[0]))
            arms.sort(key=lambda v:(-v[0],v[1]))
            hf=-np.sum(p*np.log(np.maximum(p,1e-12)),1);hd=-np.sum(pd*np.log(np.maximum(pd,1e-12)),1);edges=np.unique(np.quantile(hf,[1/3,2/3]));c=np.searchsorted(edges,hf);dc=np.searchsorted(edges,hd)
            strata=[('global',np.zeros(len(rr),int),np.zeros(len(dest),int)),('entropy',c,dc)]+[('given'+str(i),ck[rr],ck[dest]) for i,(_,ck) in enumerate(given_routes[:2])]
            for count in ((1,2,3)):
                if len(arms)<count:continue
                chosen=arms[:count];axes=[('arm_rank'+str(i),a,b,allmask) for i,(_,_,a,b,*_) in enumerate(chosen)]
                weights=np.asarray([v[0] for v in chosen]);weights/=weights.sum()
                axes=[(label,a,b,mask) for (label,a,b,mask),w in zip(axes,weights)]
                q=sum(w*v[4] for w,v in zip(weights,chosen));qd=sum(w*v[5] for w,v in zip(weights,chosen))
                a=np.log(np.maximum(q,1e-12)/np.maximum(p,1e-12));b=np.log(np.maximum(qd,1e-12)/np.maximum(pd,1e-12))
                if not emit(ranked,('R2_coordinate',count),base,p,pd,yy,a,b,allmask,den,gated,deadline,(q,qd)):return []
                for label,c,dc in strata:
                    a,b=_r_specialist_diagonal(axes,p,pd,yy,c,dc)
                    if not emit(ranked,('R2_diagonal',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
                    if False:
                        a,b=_o_specialist(axes,p,pd,yy,c,dc)
                        if not emit(ranked,('R2_joint',count,label,120.),base,p,pd,yy,a,b,allmask,den,gated,deadline):return []
        if 'R3_skip' in channels and gated:
            sf=source&(before[:,-1]>.015)&(before[:,-1]<.985)
            fs,us,rs,ys,bs,_,ds,dbs,_,_= _p_fold_context(rows,rest,y,split,before,before_r,sf,gated,held,True)
            if len(rs)>=67:
                rho=bs[:,-1];rd=dbs[:,-1];ps=np.column_stack((1-rho,rho));pds=np.column_stack((1-rd,rd));yb=(ys==width).astype(int);axes=[]
                skip_peers=list(dict.fromkeys([k for k in _parents(name,items) if k in snap]+_i_residual_peers(snap,rs,(yb-rho)[:,None],4)))[:4]
                parent_routes=[v for v in routes if v[0].startswith('parent:')]
                parent_axes=_n_axis_pool(parent_routes,rs,ds,yb,ps,pds)
                skip_routes=given_routes[:2]+[v for axis in parent_axes[:2] for v in parent_routes if v[0]==axis[0]]
                for rank,key in enumerate(skip_peers):
                    # Parent classes remain distinct. Confidence alone erases
                    # the direction of an otherwise predictive routing state.
                    q=snap[key];center=q[rs]-q[rs].mean(0)
                    order=np.argsort(-np.square(center.T@(yb-rho)),kind='stable')[:min(4,q.shape[1])]
                    W=q[:,order];W=np.column_stack((W,np.maximum(0.,1-W.sum(1))))
                    for gi in range(1+len(skip_routes)):
                        ck=np.zeros(len(X),int) if gi==0 else skip_routes[gi-1][1]
                        eb=_p_weighted_eb(W[rs],W[ds],ck[rs],ck[ds],yb,ps,pds)
                        if eb is not None:a,b,q,qd,mask,dm=eb;axes.append((str((rank,gi)),a,b,dm))
                ortho=_r_orthogonal(axes,ps,pds,yb,4)
                for count in (2,3,4):
                    if len(ortho)<count:continue
                    a,b=_r_joint_fisher(ortho[:count],ps,pds,yb,120.);a=a[:,1]-a[:,0];b=b[:,1]-b[:,0]
                    for step in (1.,):
                        score=float(_h_skip_gain(bs,ys,step*a).sum()/max(fs.sum(),1))
                        if score>0:ranked[('R3_skip',count,'fisher',step)]=(score,(step*b)[:,None],(rd>.015)&(rd<.985))
                    q=1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+a,-30,30)));qd=1/(1+np.exp(-np.clip(np.log(np.clip(rd,1e-12,1-1e-12)/(1-np.clip(rd,1e-12,1-1e-12)))+b,-30,30)))
        for spec,(_,args) in []:
            base,p,pd,yy,mask,den,gated,target=args
            if not _r_fisher_kkt(ranked,spec+('fit_selected',),base,p,pd,yy,None,None,mask,den,gated,deadline,target):return []
        # Fit-selected coordinate ranks are estimator identities. Peer names,
        # demographic labels and local coordinate enumeration never enter them.
        groups={}
        for spec,value in list(ranked.items()):
            if spec[0]=='R1_axis' and spec[-1]!='kkt':family=('R1_rank',spec[1],spec[-2],spec[-1])
            elif spec[0]=='R3_quantile' and spec[-1]!='kkt':family=('R3_risk_rank',spec[2],spec[3],spec[-2],spec[-1])
            else:continue
            groups.setdefault(family,[]).append((spec,value))
        for family,values in groups.items():
            for rank,(_,value) in enumerate(sorted(values,key=lambda v:(-v[1][0],repr(v[0])))[:2]):
                ranked[family[:-2]+(rank,)+family[-2:]]=value
        support_trace.append(dict(soft_cells_20_66=soft_fine,hierarchical_cells_25_66=hier_fine,available_specialist_arms=len(job['oof'])))
        held_fit,query_fit=_p_fit_destinations(ranked,held,split,True);fits.append(held_fit);query_fits.append(query_fit)
    proposals=[]
    def family_key(s):return (s[0],s[2] if s[0] in ('R2_diagonal','R2_joint') else s[1] if s[0]=='R1_blanket' else '',s[-2] if s[-1] in (.35,.65) else s[-1])
    for family in sorted({family_key(s) for s in fits[-1]},key=repr):
        ff=[{s:v for s,v in f.items() if family_key(s)==family} for f in fits]
        for so,sr,mo,mr,record in _j_collect(ff,split,rows,rest):
            spec=record['spec'];skip=spec[0]=='R3_skip'
            sr,mr,coherence=_r_query_average(query_fits,spec,before_r,gated,skip)
            if not coherence['passed']:continue
            record.update(coherence)
            active=mo&source;hm=mr&hidden
            if skip:
                active &= (before[:,-1]>.015)&(before[:,-1]<.985);hm &= (before_r[:,-1]>.015)&(before_r[:,-1]<.985)
                rho=np.clip(before[:,-1],1e-12,1-1e-12);rd=np.clip(before_r[:,-1],1e-12,1-1e-12)
                oo=_residual_skip_project(before,active,(1/(1+np.exp(-np.clip(np.log(rho/(1-rho))+so[:,0],-30,30))))[active]);qr=_residual_skip_project(before_r,hm,(1/(1+np.exp(-np.clip(np.log(rd/(1-rd))+sr[:,0],-30,30))))[hm])
            else:
                if gated:active &= 1-before[:,-1]>=.2;hm &= 1-before_r[:,-1]>=.2
                project=_j_simplex_project if spec[-1]=='kkt' else _ef_project
                oo=project(before,so,active,1.,gated);qr=project(before_r,sr,hm,1.,gated)
            idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y]);fg=np.asarray([gain[split==f].mean() for f in np.unique(split)]);se=max(gain.std(ddof=1)/np.sqrt(len(y)),fg.std(ddof=1)/np.sqrt(5))
            record.update(operator=spec[0],se_ratio=float(gain.mean()/max(se,1e-300)),oof_gain=float(gain.mean()),fold_gains=fg.tolist(),n_active=int(active.sum()),fit_only=True,skip=skip,fold_invariant_spec=True,fine_support=support_trace,query_pooling='unattenuated_six_fit_average')
            proposals.append((oo,qr,active,hm,record))
    return proposals


def _r_finite_commit(name,job,candidates,items,fo,fr,final,fold,stop):
    before,before_r=fo[name],fr[name];rows=np.flatnonzero(job['train']);rest=np.flatnonzero(~job['train']);y=job['y'][rows];split=fold[rows];accepted=[]
    for oo,qr,active,hm,record in candidates:
        record['kappa']=2.8;passed=_n_dominates(oo,before,y,split,active,2.8)
        if record['operator']=='R3_nonlinear_temperature':
            transported,transport=_r_entropy_transport(oo,before,before_r,y,split,active,hm,bool(items[name].get('gate')),2.8)
            record.update(transport);passed=passed and transported
        idx=np.arange(len(y));scale=1-.001*job['K'];gain=np.log(.001+scale*oo[idx,y])-np.log(.001+scale*before[idx,y])
        ag=gain[active];sp=split[active];fg=np.asarray([ag[sp==f].mean() for f in np.unique(sp)])
        se=max(float(ag.std(ddof=1)/np.sqrt(len(ag))) if len(ag)>1 else 0.,float(fg.std(ddof=1)/np.sqrt(len(fg))) if len(fg)>1 else 0.,1e-15)
        record.update(candidate_oof_delta=-float(gain.mean()),candidate_active_gain=float(ag.mean()) if len(ag) else 0.,candidate_fold_gains=fg.tolist(),candidate_se_ratio=float(ag.mean()/se) if len(ag) else 0.)
        probe=globals().get('_R_PROBE')
        if probe is not None:probe(name,record,oo,qr,passed)
        if passed:accepted.append((record['fit_mean_gain'],oo,qr,active,hm,record))
    if accepted and time.monotonic()<stop:
        _,oo,qr,active,hm,record=max(accepted,key=lambda v:v[0])
        assert np.array_equal(oo[~active],before[~active]) and np.array_equal(qr[~hm],before_r[~hm])
        if items[name].get('gate') and not record['skip']:assert np.array_equal(oo[:,-1],before[:,-1]) and np.array_equal(qr[:,-1],before_r[:,-1])
        fo[name],fr[name],final[name][rest]=oo,qr,qr;job.setdefault('r_raking',[]).append(record);return True
    return False


def _r_finite_raking(items,work,final,fo,fr,X,given,levels,fold,deadline,peers,snapshot):
    """Bounded complete six-fit log-linear, simplex and blanket transactions."""
    started=time.monotonic();remaining=max(0.,deadline-started)
    allowance=min(8.,max(1.9,remaining-4.));stop=min(deadline,started+allowance)
    if allowance>1.9:
        pool_stop=min(stop,started+5.)
        for name,job in _s136_order(work,items,fo,X,given,levels,pool_stop):
            if time.monotonic()>=pool_stop-.02:break
            transaction_cap=min(3.,max(1.3,1.3*len(fo[name])/6000.))
            candidates=_r_finite_pool_proposals(name,job,items,work,fo,fr,X,given,levels,fold,peers,snapshot,min(pool_stop,time.monotonic()+transaction_cap),('R2',),dict(fo),dict(fr))
            if _r_finite_commit(name,job,candidates,items,fo,fr,final,fold,stop):return
    global_order=_r_finite_global_order(work,items,fo,X,given,min(stop,time.monotonic()+.12))
    global_stop=min(stop,time.monotonic()+min(3.,allowance*.4))
    for name in global_order[:16]:
        if time.monotonic()>=global_stop-.02:break
        candidates=_r_global_proposals(name,work[name],items,fo,fr,X,given,fold,min(global_stop,time.monotonic()+(.60 if allowance>1.9 else .20)))
        if candidates:candidates=[max(candidates,key=lambda v:v[4]['fit_mean_gain'])]
        if _r_finite_commit(name,work[name],candidates,items,fo,fr,final,fold,stop):return
    full_arms=_r_positive_arm_order(work,items,fo,X,given,min(stop,time.monotonic()+.25))
    sampled_arms=_r_finite_positive_order(work,items,fo,X,given,min(stop,time.monotonic()+.25))
    arms=list(dict.fromkeys(full_arms+sampled_arms))
    temperatures=_r_nonlinear_order(work,items,fo,X,given,min(stop,time.monotonic()+.18))
    pool_reserve=2. if allowance>1.9 else .45
    fast_stop=max(time.monotonic(),stop-pool_reserve)
    plans=[('arm',name,(arm,lam)) for name,arm in arms[:20] for lam in (40.,90.)]+[('temperature',name,None) for name in temperatures[:20]]
    for mode,name,arm in plans:
        if time.monotonic()>=fast_stop-.02:break
        job=work[name];args=(name,job,items,fo,fr,X,given,fold,min(fast_stop,time.monotonic()+.30))
        candidates=_r_finite_loglinear_proposals(name,job,arm[0],*args[2:],lam=arm[1]) if mode=='arm' else _r_nonlinear_proposals(*args)
        if _r_finite_commit(name,job,candidates,items,fo,fr,final,fold,stop):return
    # Keep all three general instruments reachable in finite execution. Each
    # broad family still emits only complete six-fit bundles before commit.
    broad=list(dict.fromkeys([name for name,_ in arms]+temperatures))
    key_fo,key_fr=dict(fo),dict(fr)
    channel_plan=(('R2',),('R1',),('R3',),('R3_skip',)) if allowance>1.9 else (('R1','R2','R3','R3_skip'),)
    for channels in channel_plan:
        for name in broad:
            if time.monotonic()>=stop-.02:return
            candidates=_r_proposals(name,work[name],items,work,fo,fr,X,given,levels,fold,peers,snapshot,min(stop,time.monotonic()+pool_reserve),channels,key_fo,key_fr)
            if _r_finite_commit(name,work[name],candidates,items,fo,fr,final,fold,stop):return
    # Exact direct-simplex calibration is an independent transaction so slow
    # deletion roots cannot discard an already completed log-linear bundle.
    for name,arm in arms:
        if time.monotonic()>=stop-.02:return
        candidates=_r_finite_direct_proposals(name,work[name],arm,items,fo,fr,X,given,fold,min(stop,time.monotonic()+.45))
        if _r_finite_commit(name,work[name],candidates,items,fo,fr,final,fold,stop):return


def propagate_structural_gates(items, work, stacked, X, given, levels, fold,
                               stack_fn, nll_fn, deadline=float('inf'),
                               disabled=frozenset(), row_gate_fn=None, alternate_stacked=None, oof_output=None):
    """Return hidden-row vectors and per-child fold-excluded comparison.

    ``stacked[name]`` is ``(base_oof, base_hidden, base_nll)``. The caller
    supplies its official-floor stack and loss functions. Every selected parent
    contributes its routed OOF and hidden distribution to deeper children.
    """
    _s136_full_deadline=deadline
    if np.isfinite(deadline):
        _s136_gate_start=time.monotonic()
        _s136_full_deadline=max(deadline,_s136_gate_start+(30.0 if len(work)>100 else 28.0))
        deadline=_s136_gate_start+(_s136_full_deadline-_s136_gate_start)*.16
    base_snapshot = _frozen_peer_probabilities(stacked, work, len(X))
    peers = _peer_states(stacked, work, items, len(X))
    gate_correction_codes(items)
    _IPF_CACHE.clear()
    cache = {}
    ordered = sorted(work, key=lambda n: (_depth(n, items, cache, set()),
                                          list(items).index(n)))
    final, final_oof, final_rest, reports = {}, {}, {}, []
    pending_grafts = {}
    nested_used = NESTED_TRACE
    nested_used.clear()
    # Nested arms only run while every remaining gated item still fits its
    # measured base gate cost plus nested cost before the deadline.
    gated_left = sum(bool(_declared_parents(n, items)) for n in ordered)
    item_cost, nested_cost = 0.0, 0.0
    for name in ordered:
        item_began = time.monotonic()
        nested_spent = 0.0
        gated_left -= bool(_declared_parents(name, items))
        job = work[name]
        rows = np.flatnonzero(job["train"])
        rest = np.flatnonzero(~job["train"])
        K = job["K"]
        original_job = job
        tracks = [(stacked[name], original_job)]
        if alternate_stacked and name in alternate_stacked:
            tracks.append(alternate_stacked[name])
        trials=[]
        for track_stack, job in tracks:
            graft_arms = []
            report_start=len(reports)
            base_oof, base_rest, base_loss = track_stack
            best_oof, best_rest = base_oof, base_rest
            selected_loss, selected_weight, selected_kind = base_loss, 0.0, "base"
            parents = _declared_parents(name, items)
            joint_parents = _parents(name, items)
            if parents and K > 1 and len(rows) >= 40 and time.monotonic() < deadline:
                state = _parent_state(name, parents, rows, rest, work, final_oof,
                                      final_rest, X, given, levels, items)
                candidate = _candidate(name, parents, rows, rest, work, final_oof,
                                       final_rest, X, given, levels, fold, items, state=state)
                shape = _shape_candidate(name, parents, rows, rest, work,
                                         final_oof, final_rest, X, given, levels,
                                         fold, items, base_oof, base_rest, state=state)
                emp_arms, low_arms = [], []
                y_rows, row_fold = job["y"][rows], fold[rows]
                half = (row_fold % 2) == 0

                def low_dominates(arm, ref):
                    return _low_dominates(arm, ref, y_rows, K, row_fold, nll_fn)

                if candidate is not None and time.monotonic() < deadline:
                    c_low = _candidate(name, parents, rows, rest, work, final_oof,
                        final_rest, X, given, levels, fold, items, state=state, low_shrinkage=0.05)
                    s_low = _shape_candidate(name, parents, rows, rest, work, final_oof,
                        final_rest, X, given, levels, fold, items, base_oof, base_rest,
                        state=state, low_shrinkage=0.05)
                    for arm, ref in ((c_low, candidate), (s_low, shape)):
                        if arm is not None and ref is not None:
                            masks = tuple(np.max(np.abs(a - b), axis=1) > 1e-6 for a, b in zip(arm, ref))
                            if masks[0].sum() >= 67:
                                graft_arms.append((arm, masks[0], masks[1]))
                        if (arm is not None and ref is not None
                                and np.max(np.abs(arm[0] - ref[0])) > 1e-6
                                and low_dominates(arm[0], ref[0])):
                            low_arms.append((arm, ref,
                                np.max(np.abs(arm[0] - ref[0]), axis=1) > 1e-6,
                                np.max(np.abs(arm[1] - ref[1]), axis=1) > 1e-6))
                if joint_parents != parents and "empirical_gate" not in disabled:
                    # Joint (declared, empirical) state, and the empirical parent
                    # alone: its own gate sentinel already encodes the declared
                    # parent when one routes the other (s3_12 is gated on s3_11).
                    for route in (joint_parents, joint_parents[len(parents):]):
                        emp_arms.append(_candidate(name, route, rows, rest, work,
                                                   final_oof, final_rest, X, given,
                                                   levels, fold, items))
                        emp_arms.append(_shape_candidate(name, route, rows, rest,
                                                         work, final_oof, final_rest, X,
                                                         given, levels, fold, items,
                                                         base_oof, base_rest))
                if (JOINT_IPF and "empirical_gate" not in disabled and len(joint_parents) == 2
                        and all(q in work for q in joint_parents)):
                    state_ipf = _ipf_state(joint_parents, rows, rest, work, final_oof, final_rest, fold[rows])
                    if state_ipf is not None:
                        half = (fold[rows] % 2) == 0
                        y_rows = job["y"][rows]
                        raked = (_candidate(name, joint_parents, rows, rest, work, final_oof,
                                            final_rest, X, given, levels, fold, items, state=state_ipf),
                                 _shape_candidate(name, joint_parents, rows, rest, work, final_oof,
                                                  final_rest, X, given, levels, fold, items,
                                                  base_oof, base_rest, state=state_ipf))
                        for arm, ref in zip(raked, emp_arms[:2]):
                            if arm is None or ref is None:
                                continue
                            if _half_consistent(arm[0], ref[0], y_rows, K, half, nll_fn):
                                emp_arms.append(arm)
                                nested_used["joint_ipf"] = nested_used.get("joint_ipf", 0) + 1
                if V128_K2 and "empirical_gate" not in disabled:
                    half = (fold[rows] % 2) == 0
                    y_rows = job["y"][rows]
                    s311 = "s3_11_intent_return_12m_lbl"
                    s312 = "s3_12_hope_return_someday_lbl"
                    extra_routes = []
                    if len(joint_parents) == 2 and name not in GATE_CORRECTIONS:
                        extra_routes.append((joint_parents, True))
                    if (name.startswith(("s7a_", "s7b_", "s7c_")) or parents == (s311,)) and s311 in final_oof and s312 in final_oof and name != s312:
                        if (s311, s312) != joint_parents:
                            extra_routes.append(((s311, s312), False))
                        extra_routes.append(((s311, s312), True))
                        if (s312,) != joint_parents[len(parents):]:
                            extra_routes.append(((s312,), False))
                    ref_cand = candidate[0] if candidate is not None else base_oof
                    ref_shape = shape[0] if shape is not None else base_oof
                    for eroute, use_coupled in extra_routes:
                        c_arm = _candidate(name, eroute, rows, rest, work, final_oof,
                                           final_rest, X, given, levels, fold, items,
                                           coupled=use_coupled)
                        s_arm = _shape_candidate(name, eroute, rows, rest, work, final_oof,
                                                 final_rest, X, given, levels, fold, items,
                                                 base_oof, base_rest, coupled=use_coupled)
                        for arm, ref in ((c_arm, ref_cand), (s_arm, ref_shape)):
                            if arm is not None and _half_consistent(arm[0], ref, y_rows, K, half, nll_fn):
                                emp_arms.append(arm)
                                nested_used["k2_coupled" if use_coupled else "k2_s312"] = nested_used.get("k2_coupled" if use_coupled else "k2_s312", 0) + 1
                if joint_parents != parents and "empirical_gate" not in disabled:
                    for route in (joint_parents, joint_parents[len(parents):]):
                        route_state = _parent_state(name, route, rows, rest, work,
                            final_oof, final_rest, X, given, levels, items)
                        regular = (_candidate(name, route, rows, rest, work, final_oof,
                            final_rest, X, given, levels, fold, items, state=route_state),
                            _shape_candidate(name, route, rows, rest, work, final_oof,
                            final_rest, X, given, levels, fold, items, base_oof, base_rest, state=route_state))
                        sharp = (_candidate(name, route, rows, rest, work, final_oof,
                            final_rest, X, given, levels, fold, items, state=route_state, low_shrinkage=0.05),
                            _shape_candidate(name, route, rows, rest, work, final_oof,
                            final_rest, X, given, levels, fold, items, base_oof, base_rest,
                            state=route_state, low_shrinkage=0.05))
                        for arm, ref in zip(sharp, regular):
                            if arm is not None and ref is not None:
                                masks = tuple(np.max(np.abs(a - b), axis=1) > 1e-6 for a, b in zip(arm, ref))
                                if masks[0].sum() >= 67:
                                    graft_arms.append((arm, masks[0], masks[1]))
                            if (arm is not None and ref is not None
                                    and np.max(np.abs(arm[0] - ref[0])) > 1e-6
                                    and low_dominates(arm[0], ref[0])):
                                low_arms.append((arm, ref,
                                np.max(np.abs(arm[0] - ref[0]), axis=1) > 1e-6,
                                np.max(np.abs(arm[1] - ref[1]), axis=1) > 1e-6))
                if candidate is not None and shape is not None:
                    cand_oof, cand_rest = candidate
                    shape_oof, shape_rest = shape
                    parent = parents[0]
                    legacy_loss = float("nan")
                    has_legacy = bool(job["gains"].size and not _is_zero_on_skip(name))
                    arms_oof = [base_oof, shape_oof, cand_oof]
                    arms_rest = [base_rest, shape_rest, cand_rest]
                    emp_loss = float("nan")
                    if COMPOSITE_COPULA and name in GATE_CORRECTIONS:
                        # Composite gates whose parents are predicted: the
                        # recalibrated state enters only when it beats the
                        # independence state overall and on both fold halves.
                        half = (fold[rows] % 2) == 0
                        copula = (_candidate(name, parents, rows, rest, work, final_oof, final_rest,
                                             X, given, levels, fold, items, copula=True),
                                  _shape_candidate(name, parents, rows, rest, work, final_oof, final_rest,
                                                   X, given, levels, fold, items, base_oof, base_rest, copula=True))
                        for arm, ref in zip(copula, (cand_oof, shape_oof)):
                            if arm is None:
                                continue
                            if _half_consistent(arm[0], ref, job["y"][rows], K, half, nll_fn):
                                emp_arms.append(arm)
                                nested_used["copula"] = nested_used.get("copula", 0) + 1
                            else:
                                nested_used["copula_rejected"] = nested_used.get("copula_rejected", 0) + 1
                    if ("nested_gate" not in disabled and time.monotonic()
                            + (gated_left + 1) * (item_cost + nested_cost) < deadline):
                        began = time.monotonic()
                        y_rows = job["y"][rows]
                        half = (fold[rows] % 2) == 0
                        routes = [(parents, "")]
                        if joint_parents != parents and "empirical_gate" not in disabled:
                            routes.append((joint_parents, "joint:"))
                        for route, tag in routes:
                            for depth in (1, 2):
                                found = _nested_candidate(
                                    name, route, rows, rest, work, final_oof, final_rest,
                                    X, given, levels, fold, items, base_oof, base_rest,
                                    depth=depth)
                                if found is None:
                                    continue
                                # A nested arm enters the stack only when it beats
                                # its single-parent counterpart overall and on both
                                # fold halves.
                                for arm, ref in zip(found[:2], (cand_oof, shape_oof)):
                                    if _half_consistent(arm[0], ref, y_rows, K, half, nll_fn):
                                        emp_arms.append(arm)
                                        key = tag + str(depth)
                                        nested_used[key] = nested_used.get(key, 0) + 1
                                    else:
                                        nested_used["rejected"] = nested_used.get("rejected", 0) + 1
                        nested_spent = time.monotonic() - began
                        nested_cost = max(nested_cost, nested_spent) if nested_cost == 0 else .8 * nested_cost + .2 * nested_spent
                    for extra in emp_arms:
                        if extra is not None:
                            arms_oof.append(extra[0])
                            arms_rest.append(extra[1])
                            emp_loss = min(float(nll_fn(extra[0], job["y"][rows], K)),
                                           emp_loss if np.isfinite(emp_loss) else np.inf)
                    if has_legacy:
                        if parent in work:
                            pj = work[parent]
                            parent_true = pj["y"][rows]
                            parent_oof = final_oof.get(parent, stacked[parent][0])
                            parent_rest = final_rest.get(parent, stacked[parent][1])
                        else:
                            f = given.index(parent)
                            eye = np.eye(int(levels[f]))
                            parent_true = X[rows, f]
                            parent_oof, parent_rest = eye[parent_true], eye[X[rest, f]]
                        g = int(np.argmax(job["gains"]))
                        old_oof, old_rest = _legacy_route(
                            parent_oof, parent_rest, parent_true,
                            (X[rows, g], X[rest, g]), fold[rows],
                            job["y"][rows], K, base_oof, base_rest)
                        arms_oof.append(old_oof)
                        arms_rest.append(old_rest)
                        if len(work) == 12:
                            legacy_rest, legacy_oof, legacy_weights, _, legacy_loss = stack_fn(
                                job["oof"] + [old_oof],
                                [a[rest] for a in job["all"]] + [old_rest],
                                job["y"][rows], K, name)
                    mixed, mix_oof, weights, _, loss = stack_fn(
                        arms_oof, arms_rest, job["y"][rows], K, name)
                    if low_arms:
                        low_mixed, low_oof, low_weights, _, low_loss = stack_fn(
                            arms_oof + [entry[0][0] for entry in low_arms],
                            arms_rest + [entry[0][1] for entry in low_arms], job["y"][rows], K, name)
                        if low_dominates(low_oof, mix_oof):
                            mixed, mix_oof, loss = low_mixed, low_oof, low_loss
                            # Preserve legacy-arm indexing below.
                            weights = low_weights[:len(arms_oof)]
                    if (loss + 0.00025 < selected_loss
                            and _half_consistent(mix_oof, best_oof, job['y'][rows], K,
                                                  (fold[rows] % 2) == 0, nll_fn)):
                        best_oof, best_rest = mix_oof, mixed
                        selected_loss, selected_weight, selected_kind = (
                            loss, float(np.sum(weights[1:])), "multi")
                    if (len(work) != 12 and has_legacy and weights[-1] > 0.05
                            and time.monotonic() < deadline):
                        legacy_rest, legacy_oof, legacy_weights, _, legacy_loss = stack_fn(
                            job["oof"] + [old_oof],
                            [a[rest] for a in job["all"]] + [old_rest],
                            job["y"][rows], K, name)
                    if (np.isfinite(legacy_loss) and legacy_loss + 0.00025 < selected_loss
                            and _half_consistent(legacy_oof, best_oof, job['y'][rows], K,
                                                  (fold[rows] % 2) == 0, nll_fn)):
                        best_oof, best_rest = legacy_oof, legacy_rest
                        selected_loss, selected_weight, selected_kind = (
                            legacy_loss, float(legacy_weights[-1]), "legacy")
                    for arm, ref, pure_mask_oof, pure_mask_rest in low_arms:
                        # Only the fold-fitted pure cells may change the selected model.
                        for alpha in (0.25, 0.5, 0.75, 1.0):
                            if _wave_one_index(name, items, given) is not None:
                                direct_oof = best_oof.copy()
                                direct_rest = best_rest.copy()
                                direct_oof[pure_mask_oof] = ((1.0 - alpha) * best_oof[pure_mask_oof]
                                                             + alpha * arm[0][pure_mask_oof])
                                direct_rest[pure_mask_rest] = ((1.0 - alpha) * best_rest[pure_mask_rest]
                                                               + alpha * arm[1][pure_mask_rest])
                            else:
                                pi_oof_mix = (1.0 - alpha) * best_oof[:, -1] + alpha * arm[0][:, -1]
                                pi_rest_mix = (1.0 - alpha) * best_rest[:, -1] + alpha * arm[1][:, -1]
                                direct_oof = _battery_project(best_oof, pi_oof_mix, pure_mask_oof)
                                direct_rest = _battery_project(best_rest, pi_rest_mix, pure_mask_rest)
                            if low_dominates(direct_oof, best_oof):
                                best_rest = direct_rest
                                best_oof = direct_oof
                                selected_loss = float(nll_fn(best_oof, y_rows, K))
                                selected_weight, selected_kind = alpha, "pure_gate_low"
                    reports.append((name, _depth(name, items, cache, set()),
                                    float(base_loss), float(nll_fn(shape_oof, job["y"][rows], K)),
                                    float(selected_loss), selected_weight,
                                    selected_kind, float(legacy_loss),
                                    float(nll_fn(cand_oof, job["y"][rows], K)),
                                    selected_kind != "base",
                                    EMPIRICAL_PARENTS.get(name), emp_loss,
                                    float(nll_fn(base_oof, job["y"][rows], K))))
            if row_gate_fn is not None:
                best_oof,best_rest=row_gate_fn(job,best_oof,best_rest,fold)
            trials.append((best_oof,best_rest,job.get('row_gate'),reports[report_start:],graft_arms))
            del reports[report_start:]
        from .cross_round import dominates
        chosen=0
        y=original_job['y'][rows];half=fold[rows]%2==0
        if len(trials)>1 and dominates(trials[1][0],trials[0][0],y,K,half):chosen=1
        best_oof,best_rest,gate_report,chosen_reports,grafts=trials[chosen]
        pending_grafts[name] = grafts
        reports.extend(chosen_reports)
        job=original_job
        if gate_report is not None:job['row_gate']=gate_report
        job['dual_track']={'selected': 'B' if chosen else 'A', 'offered':len(trials),
            'a_nll':float(nll_fn(trials[0][0],y,K)),
            'b_nll':float(nll_fn(trials[1][0],y,K)) if len(trials)>1 else None,
            'pre_weight':tracks[chosen][1].get('_cr_weight',0.)}
        full = np.zeros((len(X), K), dtype=float)
        full[rest] = best_rest
        final[name] = full
        final_oof[name], final_rest[name] = best_oof, best_rest
        if parents:
            spent = time.monotonic() - item_began - nested_spent
            item_cost = spent if item_cost == 0 else .8 * item_cost + .2 * spent
    _conserve_battery_skips(items, work, final, final_oof, final_rest,
                            X, given, fold, nll_fn, deadline)
    # Graft after standard selection and both battery passes. Its complement
    # is exactly the current best, so its loss difference is identically zero.
    for name, grafts in pending_grafts.items():
        if time.monotonic()>=deadline:
            break
        job = work[name]
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        y, K, split = job['y'][rows], job['K'], fold[rows]
        for arm, pure_mask_oof, pure_mask_rest in grafts:
            for alpha in (0.15, 0.30, 0.50, 0.70, 1.0):
                before, before_rest = final_oof[name], final_rest[name]
                if _wave_one_index(name, items, given) is not None:
                    oo, rr = before.copy(), before_rest.copy()
                    oo[pure_mask_oof] = ((1.0 - alpha) * before[pure_mask_oof]
                                         + alpha * arm[0][pure_mask_oof])
                    rr[pure_mask_rest] = ((1.0 - alpha) * before_rest[pure_mask_rest]
                                          + alpha * arm[1][pure_mask_rest])
                else:
                    oo = _battery_project(before, (1.0 - alpha) * before[:, -1]
                                          + alpha * arm[0][:, -1], pure_mask_oof)
                    rr = _battery_project(before_rest, (1.0 - alpha) * before_rest[:, -1]
                                          + alpha * arm[1][:, -1], pure_mask_rest)
                if _low_dominates(oo, before, y, K, split, nll_fn, pure_mask_oof):
                    assert np.array_equal(rr[~pure_mask_rest], before_rest[~pure_mask_rest])
                    final_oof[name], final_rest[name] = oo, rr
                    final[name][rest] = rr
                    job.setdefault('pure_graft', []).append({'alpha': alpha,
                        'n_active': int(pure_mask_oof.sum()),
                        'oof_gain': float(nll_fn(before, y, K) - nll_fn(oo, y, K))})
    deadline=_s136_full_deadline
    _s136_tail_start=time.monotonic()
    _s136_tail_span=max(0.,deadline-_s136_tail_start)
    _graft_conditional_answers(items, work, final, final_oof, final_rest,
                                X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.0208333333333333 if np.isfinite(deadline) else deadline))
    _residual_divergence_raking(items, work, final, final_oof, final_rest,
                                 X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.0416666666666667 if np.isfinite(deadline) else deadline))
    _peer_divergence_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.0625000000000000 if np.isfinite(deadline) else deadline), peers)
    _exponential_family_raking(items, work, final, final_oof, final_rest,
                                X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.0833333333333333 if np.isfinite(deadline) else deadline), peers)
    _fisher_peer_raking(items, work, final, final_oof, final_rest,
                         X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.1041666666666667 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _canonical_temperature_raking(items, work, final, final_oof, final_rest,
                                    X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.1250000000000000 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _loo_specialist_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.1458333333333333 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _residual_peer_and_triad_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.1666666666666667 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _consensus_interaction_and_simplex_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.5800000000000000 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _nonnegative_geodesic_and_specialist_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*0.7222222222222223 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _signed_demographic_and_consensus_raking(items, work, final, final_oof, final_rest,
                             X, given, levels, fold, nll_fn, (_s136_tail_start+_s136_tail_span*1.0000000000000000 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    (_o_m_finite_raking if np.isfinite(deadline) else _additive_multi_effect_and_multi_arm_raking)(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+4.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    (_o_n_finite_raking if np.isfinite(deadline) else _orthogonal_gram_schmidt_and_battery_subspace_raking)(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+7.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _soft_responsibility_rao_blackwell_and_subspace_raking(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+11.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _multi_peer_soft_blanket_and_fold_invariant_specialist_raking(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+14.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _coherence_gated_blanket_and_positive_arm_specialist_raking(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+17.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _unshrunk_gram_schmidt_and_direct_simplex_specialist_raking(items, work, final, final_oof, final_rest,
        X, given, levels, fold, nll_fn, (deadline+19.0 if np.isfinite(deadline) else deadline), peers, base_snapshot)
    _deterministic_given_gate_lock(items, work, final, final_oof, final_rest,
        X, given, fold, nll_fn)
    if oof_output is not None:
        oof_output.update(final_oof)
    return final, reports

def bundled_attrition_mixture(conditional, p_attrition, skip_index, randomized, gate_index):
    """An absorbing nonresponse state shared by a complete follow-up battery."""
    p_attrition=np.clip(np.asarray(p_attrition,dtype=float),0.,1.)
    p=np.asarray(conditional,dtype=float)*(1.-p_attrition[:,None])
    p[:,skip_index]+=p_attrition
    p[~randomized]=0.;p[~randomized,gate_index]=1.
    return p/np.maximum(p.sum(1,keepdims=True),1e-12)


def repropagate_post_twin_gates(items, work, final, final_oofs, post_gate_final, post_gate_oofs,
                                X, given, levels, fold, nll_fn):
    """Re-establish parent-child structural gate conservation after post-deadline modules."""
    if not V128_K3:
        return {}
    cache = {}
    ordered = sorted(work, key=lambda n: (_depth(n, items, cache, set()), list(items).index(n)))
    has_wave = "wave" in given and "wave" in items
    wave = X[:, given.index("wave")] if has_wave else None
    reverted, reconserved, zero_folded = 0, 0, 0
    gains = []
    final_rest = {}
    changed_items = set()
    for name in ordered:
        job = work[name]
        rows = np.flatnonzero(job["train"])
        rest = np.flatnonzero(~job["train"])
        K = job["K"]
        y = job["y"][rows]
        half = (fold[rows] % 2) == 0
        cur_oof = np.asarray(final_oofs.get(name), dtype=float)
        cur_rest = np.asarray(final[name][rest], dtype=float)
        ref_oof = post_gate_oofs.get(name)
        ref_full = post_gate_final.get(name)
        item_modified = False
        if ref_oof is not None and ref_full is not None:
            if not (np.allclose(cur_oof, ref_oof, atol=1e-7) and np.allclose(cur_rest, ref_full[rest], atol=1e-7)):
                item_modified = True
                # Monotonic non-degradation guard against post-gate module regressions.
                cur_loss = nll_fn(cur_oof, y, K)
                ref_loss = nll_fn(ref_oof, y, K)
                if cur_loss > ref_loss + 1e-6 or (not half.all() and half.any() and any(
                        nll_fn(cur_oof[h], y[h], K) > nll_fn(ref_oof[h], y[h], K) + 5e-5 for h in (half, ~half))):
                    cur_oof = ref_oof.copy()
                    cur_rest = ref_full[rest].copy()
                    final_oofs[name] = cur_oof
                    final[name][rest] = cur_rest
                    reverted += 1
                    item_modified = False
        if has_wave and name in ZERO_ON_SKIP:
            before_z = cur_oof.copy()
            fold_wave_one_skip(cur_oof, wave[rows], name, items)
            fold_wave_one_skip(cur_rest, wave[rest], name, items)
            if not np.allclose(cur_oof, before_z, atol=1e-7):
                final_oofs[name] = cur_oof
                final[name][rest] = cur_rest
                zero_folded += 1
                item_modified = True
        parents = _declared_parents(name, items)
        joint_parents = _parents(name, items)
        parent_modified = any(p in changed_items for p in joint_parents)
        if parents and K > 1 and len(rows) >= 40 and (item_modified or parent_modified):
            base_loss = nll_fn(cur_oof, y, K)
            best_oof, best_rest, best_loss = cur_oof, cur_rest, base_loss
            routes_to_try = [(parents, False)]
            if joint_parents != parents:
                routes_to_try.append((joint_parents, False))
                if V128_K2 and len(joint_parents) == 2 and name not in GATE_CORRECTIONS:
                    routes_to_try.append((joint_parents, True))
            for route, use_coupled in routes_to_try:
                sh = _shape_candidate(name, route, rows, rest, work, final_oofs, final_rest,
                                      X, given, levels, fold, items, cur_oof, cur_rest,
                                      coupled=use_coupled)
                if sh is None:
                    continue
                sh_oof, sh_rest = sh
                for w in (0.2, 0.4, 0.6, 0.8, 1.0):
                    cand_oof = (1.0 - w) * cur_oof + w * sh_oof
                    loss = nll_fn(cand_oof, y, K)
                    if loss < best_loss and _half_consistent(cand_oof, cur_oof, y, K, half, nll_fn, margin=1e-5):
                        best_loss = loss
                        best_oof = cand_oof
                        best_rest = (1.0 - w) * cur_rest + w * sh_rest
            if best_loss < base_loss:
                cur_oof, cur_rest = best_oof, best_rest
                final_oofs[name] = cur_oof
                final[name][rest] = cur_rest
                reconserved += 1
                gains.append(float(base_loss - best_loss))
                item_modified = True
        if item_modified:
            changed_items.add(name)
        final_rest[name] = cur_rest
    return {"reverted": reverted, "reconserved": reconserved, "zero_folded": zero_folded,
            "oof_gain": float(np.mean(gains)) if gains else 0.0}

def _given_gate_alpha(y, K, fit_rows, open_mask, closed_side):
    """Five fit-only folds; every fold must improve on alpha=1.5."""
    from chain_features import folds_for
    alphas = (0.01, 0.05, 0.20, 1.5)
    split = folds_for(open_mask[:, None], fit_rows, k=5, seed=20261118)
    losses = np.zeros((len(alphas), 5), dtype=float)
    for f in range(5):
        outer_train = fit_rows[split[fit_rows] != f]
        valid = fit_rows[split[fit_rows] == f]
        train = outer_train[(~open_mask[outer_train]) if closed_side else open_mask[outer_train]]
        valid = valid[(~open_mask[valid]) if closed_side else open_mask[valid]]
        if not len(train) or not len(valid):
            return 1.5
        prior = float(np.mean(y[outer_train] == K - 1))
        hits = np.sum(y[train] == K - 1)
        target = y[valid] == K - 1
        for i, alpha in enumerate(alphas):
            skip = (hits + alpha * prior) / (len(train) + alpha)
            mass = np.where(target, skip, 1.0 - skip)
            losses[i, f] = -np.log(np.clip(mass, 1e-15, 1.0)).mean()
    eligible = np.all(losses <= losses[-1], axis=1)
    return alphas[int(np.argmin(np.where(eligible, losses.mean(1), np.inf)))]


def _given_rescale_skip(prob, skip):
    result = prob.copy()
    sums = result[:, :-1].sum(1, keepdims=True)
    conditional = np.divide(result[:, :-1], sums,
        out=np.full_like(result[:, :-1], 1 / (prob.shape[1] - 1)), where=sums > 0)
    result[:, :-1] = conditional * (1.0 - skip)
    result[:, -1] = skip
    return result


def _deterministic_given_gate_lock(items, work, final, best_oof, best_rest,
                                  X, given, fold, nll_fn):
    """: preserve conditionals and all rows outside deterministic gates."""
    for name, job in work.items():
        gate = items[name].get('gate')
        K = job['K']
        if K < 2 or not isinstance(gate, dict) or gate.get('parent') not in given:
            continue
        values = items[gate['parent']]['values']
        observed = {str(v) for v in gate.get('observed_if', ())}
        codes = [i for i, value in enumerate(values) if str(value) in observed]
        opened_mask = np.isin(X[:, given.index(gate['parent'])], codes)
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        y = job['y']
        split = fold[rows]
        if len(np.unique(split)) != 5 or name not in best_oof or name not in best_rest:
            continue
        closed = rows[~opened_mask[rows]]
        opened = rows[opened_mask[rows]]
        lock_closed = len(closed) >= 25 and np.all(y[closed] == K - 1)
        lock_open = len(opened) >= 25 and np.all(y[opened] != K - 1)
        if not (lock_closed or lock_open):
            continue
        before, before_rest = best_oof[name], best_rest[name]
        cand_oof, cand_rest = before.copy(), before_rest.copy()
        active_oof = np.zeros(len(rows), bool)
        active_rest = np.zeros(len(rest), bool)
        for closed_side, enabled in ((True, lock_closed), (False, lock_open)):
            if not enabled:
                continue
            side = ~opened_mask if closed_side else opened_mask
            for f in np.unique(split):
                fit = rows[split != f]
                part = fit[side[fit]]
                if len(part) < 25 or not np.all((y[part] == K - 1) == closed_side):
                    continue
                alpha = _given_gate_alpha(y, K, fit, opened_mask, closed_side)
                prior = float(np.mean(y[fit] == K - 1))
                skip = (np.sum(y[part] == K - 1) + alpha * prior) / (len(part) + alpha)
                target = (split == f) & side[rows]
                cand_oof[target] = _given_rescale_skip(before[target], skip)
                active_oof |= target
            alpha = _given_gate_alpha(y, K, rows, opened_mask, closed_side)
            part = rows[side[rows]]
            prior = float(np.mean(y[rows] == K - 1))
            skip = (np.sum(y[part] == K - 1) + alpha * prior) / (len(part) + alpha)
            target = side[rest]
            cand_rest[target] = _given_rescale_skip(before_rest[target], skip)
            active_rest |= target
        yy = y[rows]
        delta = np.log((.001 + (1.0 - .001 * K) * before[np.arange(len(rows)), yy]) /
                      (.001 + (1.0 - .001 * K) * cand_oof[np.arange(len(rows)), yy]))
        gain = -float(delta.mean())
        gain_active = -float(delta[active_oof].mean()) if np.any(active_oof) else 0.0
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        fold_delta = [float(delta[split == f].mean()) for f in np.unique(split)]
        half = split % 2 == 0
        if (gain_active >= 1e-5 and gain > 0 and max(fold_delta) <= 0 and gain > 2.0 * se
                and float(delta[half].mean()) < 0 and float(delta[~half].mean()) < 0):
            assert np.array_equal(cand_oof[~active_oof], before[~active_oof])
            assert np.array_equal(cand_rest[~active_rest], before_rest[~active_rest])
            best_oof[name], best_rest[name] = cand_oof, cand_rest
            final[name][rest] = cand_rest
            job['deterministic_given_gate_lock'] = dict(gain=gain, gain_active=gain_active, se=se,
                fold_delta=fold_delta, closed=bool(lock_closed), opened=bool(lock_open))
