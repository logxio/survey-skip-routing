"""Report-conditioned empirical Bayes, nested inside five visible-row folds."""
import time
import numpy as np
from base import _nll
from webinar_prior_data import PRIORS

STRENGTHS = (0., 8., 32., 128.)
_ET_GRID = ((0., 1.), (.14, .25), (.18, .25), (.18, .35), (.22, .35))
_UNICEF_PATHS = (
    ('Country', 'Q05', 'Q13', 'Q04'),
    ('Country', 'Q04', 'Q05', 'Q13', 'Q02'),
    ('Country', 'Q02', 'Q03', 'Q04', 'Q05', 'Q13'),
    ('Country', 'Q01', 'Q02', 'Q04', 'Q05', 'Q13'),
    ('Country', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q13'),
)


def normalize(p):
    p = np.maximum(p, 1e-7)
    return p / p.sum(axis=-1, keepdims=True)


def loss(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def _ans_prior(y_tr, k, has_gate):
    if has_gate:
        ans = y_tr[y_tr < k - 1]
        cnt = np.bincount(ans, minlength=k - 1).astype(float)
        return (cnt + 1. / max(1, k - 1)) / (len(ans) + 1.)
    cnt = np.bincount(y_tr, minlength=k).astype(float)
    return (cnt + 1. / k) / (len(y_tr) + 1.)


def _given_ask(X, given, items, name, y, k, tr_rows, qu_rows):
    item = (items or {}).get(name, {})
    gate = item.get('gate')
    if not isinstance(gate, dict) or gate.get('parent') not in given:
        return None
    p_name = gate['parent']
    p_col = given.index(p_name)
    p_vals = items[p_name]['values']
    obs_codes = [i for i, v in enumerate(p_vals) if str(v) in {str(u) for u in gate.get('observed_if', ())}]
    open_tr = np.isin(X[tr_rows, p_col], obs_codes)
    open_qu = np.isin(X[qu_rows, p_col], obs_codes)
    p0 = float(np.mean(y[tr_rows] == k - 1))
    cl = tr_rows[~open_tr]
    op = tr_rows[open_tr]
    skip_cl = (np.sum(y[cl] == k - 1) + 1.5 * p0) / (len(cl) + 1.5) if len(cl) else p0
    skip_op = (np.sum(y[op] == k - 1) + 1.5 * p0) / (len(op) + 1.5) if len(op) else p0
    return np.where(open_qu, 1. - skip_op, 1. - skip_cl)


def _apply_gate_and_et(p, ask, has_gate, pr, eps, T):
    out = p.copy()
    if has_gate:
        cond = out[:, :-1] / np.maximum(out[:, :-1].sum(1, keepdims=True), 1e-12)
        if not (eps == 0. and T == 1.):
            lg = np.log(np.maximum(cond, 1e-7)) / T
            lg -= lg.max(1, keepdims=True)
            sh = np.exp(lg)
            sh /= sh.sum(1, keepdims=True)
            cond = (1. - eps) * sh + eps * pr
        if ask is None:
            ask = np.clip(out[:, :-1].sum(1), 1e-6, 1. - 1e-6)
        out[:, :-1] = cond * ask[:, None]
        out[:, -1] = 1. - ask
        return out
    if eps == 0. and T == 1.:
        return out
    lg = np.log(np.maximum(out, 1e-7)) / T
    lg -= lg.max(1, keepdims=True)
    sh = np.exp(lg)
    sh /= sh.sum(1, keepdims=True)
    return (1. - eps) * sh + eps * pr


def _report_vector(empirical, constraints, values, preserve_na=False):
    # IPF handles overlapping partial events (e.g. varies a lot, any variation).
    p = empirical.copy()
    asked = (
        len(values) - 1
        if (preserve_na and values and values[-1] == 'Not answered' and all('Not answered' not in opts for opts, _ in constraints))
        else len(values)
    )
    mass = p[:asked].sum()
    q = normalize(p[:asked])
    if preserve_na and all(len(opts) == 1 for opts, _ in constraints):
        constrained_mask = np.zeros(asked, dtype=bool)
        prob_sum = 0.0
        for opts, prob in constraints:
            opt = opts[0]
            if opt in values[:asked]:
                idx_opt = values[:asked].index(opt)
                q[idx_opt] = float(prob)
                constrained_mask[idx_opt] = True
                prob_sum += float(prob)
        unconstrained = ~constrained_mask
        if unconstrained.any():
            rem = max(0.0, 1.0 - prob_sum)
            q[unconstrained] = normalize(q[unconstrained]) * rem
        p[:asked] = normalize(q) * mass
        return normalize(p)
    compiled_idx = [np.array([v in options for v in values[:asked]]) for options, _ in constraints]
    for _ in range(1 if (preserve_na and len(constraints) <= 1) else 24):
        for idx, (_, prob) in zip(compiled_idx, constraints):
            other = ~idx
            if not idx.any() or not other.any():
                continue
            q[idx] *= prob / max(q[idx].sum(), 1e-12)
            q[other] *= (1 - prob) / max(q[other].sum(), 1e-12)
    p[:asked] = normalize(q) * mass
    return normalize(p)


def _group_codes(X, given):
    cols = [given.index(k) for k in ('Country', 'wave', 's1_2_country_asylum_lbl', 'group') if k in given]
    a = X[:, cols] if cols else np.zeros((len(X), 1), int)
    _, codes = np.unique(a, axis=0, return_inverse=True)
    return codes


def _unicef_cell_codes(X, given):
    if not set(_UNICEF_PATHS[-1]).issubset(given):
        return None
    codes = []
    for names in _UNICEF_PATHS:
        _, idx = np.unique(X[:, [given.index(n) for n in names]], axis=0, return_inverse=True)
        codes.append((idx.ravel(), int(idx.max()) + 1, len(names)))
    return codes


def _compiled(schema, X, given, work):
    dataset = 'unhcr' if 'wave' in given else 'unicef' if 'Country' in given else 'world_bank' if 'group' in given else None
    out = {}
    for name, conditions, constraints in PRIORS.get(dataset, ()):
        if name not in work:
            continue
        exact = np.ones(len(X), bool)
        event = np.ones(len(X), bool)
        parents = []
        for key, values in conditions:
            choices = schema['items'][key]['values']
            codes = [i for i, v in enumerate(choices) if v in values]
            if key in given:
                exact &= np.isin(X[:, given.index(key)], codes)
            elif key in work:
                event &= np.isin(work[key]['y'], codes)
                parents.append(key)
            else:
                exact[:] = False
        out.setdefault(name, []).append((exact, event, parents, constraints))
    return out


def _all_predictions(j, groups, work, code, train, query, values, X_raw=None, is_wb=False, is_unicef=False, cell_codes=None, alpha_pair=(12., 4.), gate_col=None):
    y = j['y']
    K = j['K']
    ng = int(code.max()) + 1
    global_count = np.bincount(y[train], minlength=K).astype(float) + .5 / K
    marginal = global_count / global_count.sum()
    if X_raw is not None:
        cnt0 = np.bincount(y[train], minlength=K).astype(float)
        pm = np.zeros(K) if (K > len(values) and cnt0[-1] == 0) else np.full(K, .5 / K)
        if K > len(values) and cnt0[-1] == 0:
            pm[:-1] = .5 / max(1, K - 1)
        p0 = (cnt0 + pm) / (len(train) + pm.sum())
        log_p0 = np.log(np.maximum(p0, 1e-12))
        ll = np.broadcast_to(log_p0, (len(query), K)).copy()
        for c in range(X_raw.shape[1]):
            col = X_raw[:, c]
            nc = int(col.max()) + 1
            ct = np.bincount(col[train] * K + y[train], minlength=nc * K).reshape(nc, K).astype(float)
            pc = (ct + 6. * p0) / (ct.sum(1, keepdims=True) + 6.)
            w_c = 1.0 if (gate_col is not None and c == gate_col) else .35
            ll += w_c * (np.log(np.maximum(pc[col[query]], 1e-12)) - log_p0)
        ll -= ll.max(1, keepdims=True)
        base_q = np.exp(ll)
        base_q /= base_q.sum(1, keepdims=True)

        if is_unicef and cell_codes is not None:
            coarse, fine = alpha_pair
            p_cell = base_q
            fifth = None
            six = []
            for i, (ids, n_cells, order) in enumerate(cell_codes):
                alpha = coarse if order <= 5 else fine
                if len(cell_codes) == 5 and i == 2:
                    fifth = p_cell
                parent = fifth if (len(cell_codes) == 5 and i in (2, 3)) else p_cell
                if len(cell_codes) == 5 and i == 4:
                    parent = (six[0] + six[1]) * 0.5
                c_tab = np.bincount(ids[train] * K + y[train], minlength=n_cells * K).reshape(n_cells, K).astype(float)
                c_q = c_tab[ids[query]]
                p_cell = (c_q + alpha * parent) / (c_q.sum(1, keepdims=True) + alpha)
                if len(cell_codes) == 5 and i in (2, 3):
                    six.append(p_cell)
            base_q = p_cell

        outputs = np.broadcast_to(base_q, (len(STRENGTHS), len(query), K)).copy()
        for exact, event, parents, constraints in groups:
            q_idx = np.flatnonzero(exact[query])
            if not len(q_idx):
                continue
            matched = train[exact[train] & event[train]]
            if not len(matched):
                continue
            cnt_m = np.bincount(y[matched], minlength=K).astype(float)
            has_unanswered_gate = K > len(values) and cnt_m[-1] == 0
            pm_m = np.zeros(K) if has_unanswered_gate else 2. * p0
            if has_unanswered_gate:
                pm_m[:-1] = .5 / max(1, K - 1)
            emp = normalize(cnt_m + pm_m)
            rep = _report_vector(emp, constraints, values, preserve_na=True)
            asked = (
                len(values) - 1
                if (values and values[-1] == 'Not answered' and all('Not answered' not in opts for opts, _ in constraints))
                else len(values)
            )
            ratio_ans = rep[:asked] / np.maximum(emp[:asked], 1e-7)
            if parents and is_unicef:
                eligible = train[exact[train]]
                eligible = eligible[np.all(np.column_stack([work[p]['y'][eligible] >= 0 for p in parents]), axis=1)]
                ev_rate = float((event[eligible].sum() + 2.0 * event[train].mean()) / (len(eligible) + 2.0)) if len(eligible) else 1.0
            else:
                ev_rate = 1.0

            for a, alpha in enumerate(STRENGTHS):
                if alpha == 0.:
                    continue
                w = alpha / (alpha + 32.)
                cur = outputs[a, q_idx]
                ans_mass = cur[:, :asked].sum(1, keepdims=True)
                tilted_ans = normalize(cur[:, :asked] * (ratio_ans ** w)[None, :]) * ans_mass
                cur_tilted = cur.copy()
                cur_tilted[:, :asked] = tilted_ans
                if K > len(values) and cnt0[-1] == 0:
                    cur_tilted[:, -1] = 0.
                    cur_tilted /= np.maximum(cur_tilted.sum(1, keepdims=True), 1e-12)
                outputs[a, q_idx] = (1.0 - ev_rate) * cur + ev_rate * cur_tilted
        return normalize(outputs)

    counts = np.zeros((ng, K))
    np.add.at(counts, (code[train], y[train]), 1.)
    baseline = (counts + 4 * marginal) / (counts.sum(1, keepdims=True) + 4)
    outputs = np.broadcast_to(baseline[code[query]], (len(STRENGTHS), len(query), K)).copy()
    for exact, event, parents, constraints in groups:
        matched = train[exact[train] & event[train]]
        if not len(matched):
            continue
        count = np.bincount(y[matched], minlength=K).astype(float)
        if (is_wb or is_unicef) and K > len(values) and count[-1] == 0:
            prior_m = np.zeros(K)
            prior_m[:-1] = .5 / max(1, K - 1)
            empirical = normalize(count + prior_m)
        else:
            prior_m = None
            empirical = normalize(count + 2 * marginal)
        report = _report_vector(empirical, constraints, values, preserve_na=(is_wb or is_unicef))
        eligible = train[exact[train]]
        if parents:
            eligible = eligible[np.all(np.column_stack([work[p]['y'][eligible] >= 0 for p in parents]), axis=1)]
            total = np.bincount(code[eligible], minlength=ng).astype(float)
            yes = np.bincount(code[eligible], weights=event[eligible], minlength=ng)
            rate = (yes + 2 * (event[eligible].mean() if len(eligible) else 0.)) / (total + 2.)
            weight = exact[query] * rate[code[query]]
        else:
            weight = exact[query].astype(float)
        for a, alpha in enumerate(STRENGTHS):
            if prior_m is not None:
                posterior = normalize(count + prior_m + alpha * report)
                posterior[-1] = 0.
                posterior /= max(posterior.sum(), 1e-12)
            else:
                posterior = normalize(count + 2 * marginal + alpha * report)
            outputs[a] = (1 - weight[:, None]) * outputs[a] + weight[:, None] * posterior
    return normalize(outputs)


def candidate(j, groups, work, code, fold, values, deadline, X_raw=None, is_wb=False, is_unicef=False, cell_codes=None, gate_col=None, name=None, items=None, given_raw=None):
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    y = j['y']
    K = j['K']
    if len(rows) < (40 if is_wb else 20) or len(np.unique(fold[rows])) != 5:
        return None
    o = np.zeros((len(rows), K), np.float32)
    full = np.zeros((len(code), K), np.float32)
    chosen = []
    global_loss = np.zeros(len(STRENGTHS))
    global_n = 0
    eval_fn = (lambda p, yy: _nll(p, yy, K)) if (is_wb or is_unicef) else loss
    fold_alphas = j.get('_eb_alphas')
    has_gate = bool((items or {}).get(name, {}).get('gate'))
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        fit = rows[fold[rows] != f]
        valid = rows[fold[rows] == f]
        al_f = tuple(fold_alphas[f]) if (fold_alphas and f < len(fold_alphas)) else (12., 4.)
        if is_unicef:
            inner = (np.arange(len(fit)) % 4) == 0
            tr, va = fit[~inner], fit[inner]
            pp_in = _all_predictions(j, groups, work, code, tr, va, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, alpha_pair=al_f, gate_col=gate_col)
            ask_va = _given_ask(X_raw, given_raw, items, name, y, K, tr, va) if (X_raw is not None and given_raw is not None) else None
            pr_in = _ans_prior(y[tr], K, has_gate)
            best_pair = (0, (0., 1.))
            best_sc = float('inf')
            for s_i, p_va in enumerate(pp_in):
                for et in _ET_GRID:
                    sc = eval_fn(_apply_gate_and_et(p_va, ask_va, has_gate, pr_in, *et), y[va])
                    if sc < best_sc:
                        best_sc = sc
                        best_pair = (s_i, et)
            best, best_et = best_pair
            chosen.append(STRENGTHS[best])
            pp_out = _all_predictions(j, groups, work, code, fit, valid, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, alpha_pair=al_f, gate_col=gate_col)
            ask_val = _given_ask(X_raw, given_raw, items, name, y, K, fit, valid) if (X_raw is not None and given_raw is not None) else None
            pr_f = _ans_prior(y[fit], K, has_gate)
            pp_cal = [_apply_gate_and_et(p_v, ask_val, has_gate, pr_f, *best_et) for p_v in pp_out]
            o[fold[rows] == f] = pp_cal[best]
            global_loss += np.array([eval_fn(p_c, y[valid]) for p_c in pp_cal]) * len(valid)
            global_n += len(valid)
        else:
            inner_loss = np.zeros(len(STRENGTHS))
            inner_n = 0
            for g in range(5):
                if g == f:
                    continue
                tr = fit[fold[fit] != g]
                va = fit[fold[fit] == g]
                if not len(tr) or not len(va):
                    continue
                pp = _all_predictions(j, groups, work, code, tr, va, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, gate_col=gate_col)
                inner_loss += np.array([eval_fn(p, y[va]) for p in pp]) * len(va)
                inner_n += len(va)
            best = int(np.argmin(inner_loss / max(1, inner_n)))
            chosen.append(STRENGTHS[best])
            pp = _all_predictions(j, groups, work, code, fit, valid, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, gate_col=gate_col)
            o[fold[rows] == f] = pp[best]
            global_loss += np.array([eval_fn(p, y[valid]) for p in pp]) * len(valid)
            global_n += len(valid)
    best = int(np.argmin(global_loss / max(1, global_n)))
    if len(rest):
        if is_unicef:
            al_full = tuple(fold_alphas[-1]) if fold_alphas else (12., 4.)
            inner = (np.arange(len(rows)) % 4) == 0
            tr, va = rows[~inner], rows[inner]
            pp_in = _all_predictions(j, groups, work, code, tr, va, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, alpha_pair=al_full, gate_col=gate_col)
            ask_va = _given_ask(X_raw, given_raw, items, name, y, K, tr, va) if (X_raw is not None and given_raw is not None) else None
            pr_in = _ans_prior(y[tr], K, has_gate)
            best_et = min(_ET_GRID, key=lambda et: eval_fn(_apply_gate_and_et(pp_in[best], ask_va, has_gate, pr_in, *et), y[va]))
            p_raw = _all_predictions(j, groups, work, code, rows, rest, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, alpha_pair=al_full, gate_col=gate_col)[best]
            ask_rest = _given_ask(X_raw, given_raw, items, name, y, K, rows, rest) if (X_raw is not None and given_raw is not None) else None
            pr_all = _ans_prior(y[rows], K, has_gate)
            full[rest] = _apply_gate_and_et(p_raw, ask_rest, has_gate, pr_all, *best_et)
        else:
            full[rest] = _all_predictions(j, groups, work, code, rows, rest, values, X_raw=X_raw, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes, gate_col=gate_col)[best]
    return o, full, {
        'fold_strengths': chosen,
        'full_strength': STRENGTHS[best],
        'calibration_folds': 5,
        'nested_calibration': True,
        'parent_query_labels_used': False,
        'oof_nll': float(eval_fn(o, y[rows])),
    }


def attach(schema, X, given, work, fold, deadline):
    from cross_round import dominates
    groups = _compiled(schema, X, given, work)
    code = _group_codes(X, given)
    trace = {}
    is_wb = 'group' in given
    is_unicef = 'Country' in given
    n_raw = sum(item.get('class') == 'GIVEN' for item in schema['items'].values()) or X.shape[1]
    X_given = X[:, :n_raw] if (is_wb or is_unicef) else None
    given_raw = list(given[:n_raw])
    cell_codes = _unicef_cell_codes(X[:, :n_raw], given_raw) if is_unicef else None
    for name, rules in groups.items():
        if time.monotonic() >= deadline:
            break
        j = work[name]
        x_in = X_given if ((is_wb and name.startswith('user.')) or is_unicef) else None
        gate = schema['items'][name].get('gate')
        g_parent = gate.get('parent') if isinstance(gate, dict) else None
        g_col = given_raw.index(g_parent) if (g_parent in given_raw) else None
        result = candidate(
            j, rules, work, code, fold, schema['items'][name]['values'], deadline,
            X_raw=x_in, is_wb=is_wb, is_unicef=is_unicef, cell_codes=cell_codes,
            gate_col=g_col, name=name, items=schema['items'], given_raw=given_raw
        )
        if result is None:
            continue
        o, p, info = result
        if not (is_wb or is_unicef):
            yy = j['y'][j['train']]
            base = j['oof'][0]
            losses = [loss((1 - a) * base + a * o, yy) for a in (0., .1, .25, .5, .75, 1.)]
            info.update(table_nll=losses[0], mixture_gain=losses[0] - min(losses), constraints=len(rules))
            info['added'] = bool(info['mixture_gain'] > 1e-8)
            if info['added']:
                j['oof'].append(o)
                j['all'].append(p)
                j['arms'].append('report_prior')
            trace[name] = info
            continue
        rows = np.flatnonzero(j['train'])
        yy = j['y'][rows]
        k = j['K']
        base = j['oof'][-1] if is_unicef else j['oof'][0]
        half = (fold[rows] % 2) == 0
        base_loss = _nll(base, yy, k)
        best_loss = base_loss
        best_alpha = 0.
        for a in (.1, .25, .5, .75, 1.):
            mix = (1. - a) * base + a * o
            l = _nll(mix, yy, k)
            if l < best_loss and dominates(mix, base, yy, k, half):
                best_loss = l
                best_alpha = a
        gain = float(base_loss - best_loss)
        info.update(table_nll=float(base_loss), mixture_gain=gain, best_alpha=best_alpha, constraints=len(rules))
        info['added'] = bool(gain > (1e-5 if is_unicef else 1e-4))
        if info['added']:
            j['oof'].append(o)
            j['all'].append(p)
            j['arms'].append('report_prior')
        trace[name] = info
    return trace
