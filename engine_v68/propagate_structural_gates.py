"""Empirical two-stage gate propagation over visible survey answers.

Each target learns a skip probability and an answer distribution from distinct
visible subsets. Parent predictions are consumed in schema topological order.
The schema's observed_if values describe routing; observed answers, including
off-route answers, determine the transition probabilities.
"""

import time
import numpy as np


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


def _count_012(distributions):
    """Poisson-binomial 0, 1, >=2 counts under conditional independence."""
    zero = np.ones(len(distributions[0]), dtype=float)
    one = np.zeros_like(zero)
    for dist in distributions:
        p0, p1 = dist[:, 0], dist[:, 1]
        one, zero = one * p0 + zero * p1, zero * p0
    return np.column_stack((zero, one, np.clip(1.0 - zero - one, 0.0, 1.0)))


def _parent_state(name, parents, rows, rest, work, final_oof, final_rest,
                  X, given, levels, items):
    distributions_oof, distributions_rest, true_parents, widths = [], [], [], []
    for parent in parents:
        if parent in work:
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
    if name in GATE_CORRECTIONS:
        _, active_codes, mode = gate_correction_codes(items)[name]
        if mode.startswith("count"):
            if len(parents) != 10 or any(width != 2 for width in widths):
                raise ValueError("s3_23 rank gate requires ten binary parents")
            true_state = np.minimum(np.sum(true_parents, axis=0), 2).astype(np.int64)
            return (true_state, _count_012(distributions_oof),
                    _count_012(distributions_rest), 3)
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
        return (true_state, probabilities(distributions_oof),
                probabilities(distributions_rest), 2)
    true_state = np.zeros(len(rows), dtype=np.int64)
    for codes, width in zip(true_parents, widths):
        true_state = true_state * width + codes
    return (true_state, _joint(distributions_oof),
            _joint(distributions_rest), int(np.prod(widths)))


def _rates(parent_ids, group_ids, fold, y, K, P, G, F, excluded=None):
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
                  excluded=None):
    """Fit every level of the transition on eligible, fold-excluded rows."""
    fit = source if excluded is None else source & (fold != excluded)
    if not fit.any():
        return None
    return _rates(parent_ids[fit], group_ids[fit], fold[fit], y[fit],
                  K, P, G, int(fold.max()) + 1)


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
               X, given, levels, fold, items):
    state = _parent_state(name, parents, rows, rest, work, final_oof,
                          final_rest, X, given, levels, items)
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
                                  y, K, P, G, source, excluded=f)
            if rates is None:
                return None
            oof[mask] = _apply(joint_oof[mask], groups[rows][mask], rates)
    rates = _source_rates(true_joint, groups[rows], training_fold,
                          y, K, P, G, source)
    if rates is None:
        return None
    hidden = _apply(joint_rest, groups[rest], rates)
    if wave_one is not None:
        _encode_wave_one_zero(oof, X[rows, wave_feature], wave_one)
        _encode_wave_one_zero(hidden, X[rest, wave_feature], wave_one)
    return oof, hidden


def _shape_candidate(name, parents, rows, rest, work, final_oof, final_rest,
                     X, given, levels, fold, items, base_oof, base_rest):
    """Route only skip mass while preserving the child's fitted answer shape."""
    state = _parent_state(name, parents, rows, rest, work, final_oof,
                          final_rest, X, given, levels, items)
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
                                  y, K, P, G, source, excluded=f)
            if rates is None:
                return None
            wave = X[rows[mask], wave_feature] if zero_on_skip else None
            oof[mask] = _shape_apply(base_oof[mask], joint_oof[mask],
                                     groups[rows][mask], rates, wave, wave_one)
    rates = _source_rates(true_joint, groups[rows], training_fold,
                          y, K, P, G, source)
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


def propagate_structural_gates(items, work, stacked, X, given, levels, fold,
                               stack_fn, nll_fn, deadline=float('inf'),
                               disabled=frozenset(), row_gate_fn=None, alternate_stacked=None, oof_output=None):
    """Return hidden-row vectors and per-child fold-excluded comparison.

    ``stacked[name]`` is ``(base_oof, base_hidden, base_nll)``. The caller
    supplies its official-floor stack and loss functions. Every selected parent
    contributes its routed OOF and hidden distribution to deeper children.
    """
    gate_correction_codes(items)
    cache = {}
    ordered = sorted(work, key=lambda n: (_depth(n, items, cache, set()),
                                          list(items).index(n)))
    final, final_oof, final_rest, reports = {}, {}, {}, []
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
            report_start=len(reports)
            base_oof, base_rest, base_loss = track_stack
            best_oof, best_rest = base_oof, base_rest
            selected_loss, selected_weight, selected_kind = base_loss, 0.0, "base"
            parents = _declared_parents(name, items)
            joint_parents = _parents(name, items)
            if parents and K > 1 and len(rows) >= 40 and time.monotonic() < deadline:
                candidate = _candidate(name, parents, rows, rest, work, final_oof,
                                       final_rest, X, given, levels, fold, items)
                shape = _shape_candidate(name, parents, rows, rest, work,
                                         final_oof, final_rest, X, given, levels,
                                         fold, items, base_oof, base_rest)
                emp_arms = []
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
                if candidate is not None and shape is not None:
                    cand_oof, cand_rest = candidate
                    shape_oof, shape_rest = shape
                    parent = parents[0]
                    legacy_loss = float("nan")
                    has_legacy = bool(job["gains"].size and not _is_zero_on_skip(name))
                    arms_oof = [base_oof, shape_oof, cand_oof]
                    arms_rest = [base_rest, shape_rest, cand_rest]
                    emp_loss = float("nan")
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
                                job["y"][rows], K)
                    mixed, mix_oof, weights, _, loss = stack_fn(
                        arms_oof, arms_rest, job["y"][rows], K)
                    if loss + 1e-10 < selected_loss:
                        best_oof, best_rest = mix_oof, mixed
                        selected_loss, selected_weight, selected_kind = (
                            loss, float(np.sum(weights[1:])), "multi")
                    if (len(work) != 12 and has_legacy and weights[-1] > 0.05
                            and time.monotonic() < deadline):
                        legacy_rest, legacy_oof, legacy_weights, _, legacy_loss = stack_fn(
                            job["oof"] + [old_oof],
                            [a[rest] for a in job["all"]] + [old_rest],
                            job["y"][rows], K)
                    if np.isfinite(legacy_loss) and legacy_loss + 1e-10 < selected_loss:
                        best_oof, best_rest = legacy_oof, legacy_rest
                        selected_loss, selected_weight, selected_kind = (
                            legacy_loss, float(legacy_weights[-1]), "legacy")
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
            trials.append((best_oof,best_rest,job.get('row_gate'),reports[report_start:]))
            del reports[report_start:]
        from .cross_round import dominates
        chosen=0
        y=original_job['y'][rows];half=fold[rows]%2==0
        if len(trials)>1 and dominates(trials[1][0],trials[0][0],y,K,half):chosen=1
        best_oof,best_rest,gate_report,chosen_reports=trials[chosen]
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
