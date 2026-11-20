"""Report-conditioned empirical Bayes, nested inside five visible-row folds."""
import time
import numpy as np
from survey_prior_data import PRIORS, WB_RCT_EXTRA_PRIORS, WB_RCT_ANSWERED_TILTS

STRENGTHS = (0.0, 8.0, 32.0, 128.0)
ENABLE_WB_RCT = True
ENABLE_UNHCR_ERPIS = True


def normalize(p):
    p = np.maximum(p, 1e-7)
    return p / p.sum(axis=-1, keepdims=True)


def loss(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def _report_vector(empirical, constraints, values, preserve_na=False):
    if not constraints:
        return normalize(empirical)
    p = empirical.copy()
    asked = (
        len(values) - 1
        if (
            preserve_na
            and values
            and values[-1] == 'Not answered'
            and all('Not answered' not in opts for opts, _ in constraints)
        )
        else len(values)
    )
    mass = p[:asked].sum()
    q = normalize(p[:asked])
    n_iter = 1 if len(constraints) <= 1 else 16
    masks = []
    for options, prob in constraints:
        idx = np.array([v in options for v in values[:asked]])
        other = ~idx
        if idx.any() and other.any():
            masks.append((idx, other, prob))
    for _ in range(n_iter):
        for idx, other, prob in masks:
            q[idx] *= prob / max(q[idx].sum(), 1e-12)
            q[other] *= (1.0 - prob) / max(q[other].sum(), 1e-12)
    p[:asked] = normalize(q) * mass
    return normalize(p)


def _group_codes(X, given):
    cols = [given.index(k) for k in ('Country', 'wave', 's1_2_country_asylum_lbl', 'group') if k in given]
    a = X[:, cols] if cols else np.zeros((len(X), 1), int)
    _, codes = np.unique(a, axis=0, return_inverse=True)
    return codes


def _compiled(schema, X, given, work):
    dataset = (
        'unhcr'
        if 'wave' in given
        else 'unicef'
        if 'Country' in given
        else 'world_bank'
        if 'group' in given
        else None
    )
    out = {}
    entries = list(PRIORS.get(dataset, ()))
    if dataset == 'world_bank' and ENABLE_WB_RCT:
        entries.extend(WB_RCT_EXTRA_PRIORS)
    for name, conditions, constraints in entries:
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


def _fast_wb_user_candidate(j, groups, X_raw, fold, values, deadline):
    """Precomputed 5-fold count tensors for World Bank 9-GIVEN additive EB + report tilting."""
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    y = j['y']
    K = j['K']
    if len(rows) < 40 or len(np.unique(fold[rows])) != 5:
        return None
    f_rows = fold[rows].astype(np.int64)
    y_rows = y[rows].astype(np.int64)
    cnt0_f = np.bincount(f_rows * K + y_rows, minlength=5 * K).reshape(5, K).astype(float)
    cnt0_all = cnt0_f.sum(0)
    col_cnt_f = []
    col_cnt_all = []
    n_cols = X_raw.shape[1]
    for c in range(n_cols):
        col = X_raw[:, c]
        nc = int(col.max()) + 1
        cf = np.bincount((f_rows * nc + col[rows]) * K + y_rows, minlength=5 * nc * K).reshape(5, nc, K).astype(float)
        col_cnt_f.append(cf)
        col_cnt_all.append(cf.sum(0))

    active_groups = [g for g in groups if g[3]]
    rule_cnt_f = []
    rule_cnt_all = []
    for exact, event, _, _ in active_groups:
        m = exact[rows] & event[rows]
        rf = np.bincount(f_rows[m] * K + y_rows[m], minlength=5 * K).reshape(5, K).astype(float)
        rule_cnt_f.append(rf)
        rule_cnt_all.append(rf.sum(0))

    def base_from_counts(cnt0, n_tr, col_cnts, query):
        pm = np.zeros(K) if (K > len(values) and cnt0[-1] == 0) else np.full(K, 0.5 / K)
        if K > len(values) and cnt0[-1] == 0:
            pm[:-1] = 0.5 / max(1, K - 1)
        p0 = (cnt0 + pm) / (n_tr + pm.sum())
        log_p0 = np.log(np.maximum(p0, 1e-12))
        ll = np.broadcast_to(log_p0, (len(query), K)).copy()
        for c in range(n_cols):
            ct = col_cnts[c]
            pc = (ct + 6.0 * p0) / (ct.sum(1, keepdims=True) + 6.0)
            ll += 0.35 * (np.log(np.maximum(pc[X_raw[query, c]], 1e-12)) - log_p0)
        ll -= ll.max(1, keepdims=True)
        base_q = np.exp(ll)
        base_q /= base_q.sum(1, keepdims=True)
        return base_q, p0, pm

    def predict_from_counts(cnt0, n_tr, col_cnts, rule_cnts, query):
        base_q, p0, pm = base_from_counts(cnt0, n_tr, col_cnts, query)
        if not active_groups:
            return normalize(base_q[None, :, :])
        outputs = np.broadcast_to(base_q, (len(STRENGTHS), len(query), K)).copy()
        for r_idx, (exact, event, parents, constraints) in enumerate(active_groups):
            cnt_m = rule_cnts[r_idx]
            if cnt_m.sum() <= 0:
                continue
            emp = normalize(cnt_m + (pm if (K > len(values) and cnt_m[-1] == 0) else 2.0 * p0))
            rep = _report_vector(emp, constraints, values, preserve_na=True)
            ratio = rep / np.maximum(emp, 1e-7)
            ex_q = exact[query][:, None]
            for a, alpha in enumerate(STRENGTHS):
                if alpha == 0.0:
                    continue
                w = alpha / (alpha + 32.0)
                tilted = normalize(outputs[a] * (ratio ** w)[None, :])
                if K > len(values) and cnt0[-1] == 0:
                    tilted[:, -1] = 0.0
                    tilted /= np.maximum(tilted.sum(1, keepdims=True), 1e-12)
                outputs[a] = np.where(ex_q, tilted, outputs[a])
        return normalize(outputs)

    n_all = float(len(rows))
    n_f = cnt0_f.sum(1)
    o = np.zeros((len(rows), K), np.float32)
    full = np.zeros((len(X_raw), K), np.float32)
    if not active_groups:
        # Pure 9-GIVEN additive EB: only 5 outer fold calls + 1 full call needed!
        for f in range(5):
            if time.monotonic() >= deadline:
                return None
            valid_mask = f_rows == f
            valid = rows[valid_mask]
            cnt0_fit = cnt0_all - cnt0_f[f]
            n_fit = n_all - n_f[f]
            col_fit = [ col_cnt_all[c] - col_cnt_f[c][f] for c in range(n_cols) ]
            o[valid_mask] = predict_from_counts(cnt0_fit, n_fit, col_fit, (), valid)[0]
        if len(rest):
            full[rest] = predict_from_counts(cnt0_all, n_all, col_cnt_all, (), rest)[0]
        return o, full, {
            'fold_strengths': [0.0] * 5,
            'full_strength': 0.0,
            'calibration_folds': 5,
            'nested_calibration': True,
            'parent_query_labels_used': False,
            'oof_nll': loss(o, y[rows]),
        }

    chosen = []
    global_loss = np.zeros(len(STRENGTHS))
    global_n = 0
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        valid_mask = f_rows == f
        valid = rows[valid_mask]
        cnt0_fit = cnt0_all - cnt0_f[f]
        n_fit = n_all - n_f[f]
        col_fit = [ col_cnt_all[c] - col_cnt_f[c][f] for c in range(n_cols) ]
        rule_fit = [ rule_cnt_all[r] - rule_cnt_f[r][f] for r in range(len(active_groups)) ]
        inner_loss = np.zeros(len(STRENGTHS))
        inner_n = 0
        for g in range(5):
            if g == f:
                continue
            va_mask = f_rows == g
            va = rows[va_mask]
            if not len(va):
                continue
            cnt0_tr = cnt0_fit - cnt0_f[g]
            n_tr = n_fit - n_f[g]
            col_tr = [ col_fit[c] - col_cnt_f[c][g] for c in range(n_cols) ]
            rule_tr = [ rule_fit[r] - rule_cnt_f[r][g] for r in range(len(active_groups)) ]
            pp = predict_from_counts(cnt0_tr, n_tr, col_tr, rule_tr, va)
            inner_loss += np.array([loss(p, y[va]) for p in pp]) * len(va)
            inner_n += len(va)
        best = int(np.argmin(inner_loss / max(1, inner_n)))
        chosen.append(STRENGTHS[best])
        pp = predict_from_counts(cnt0_fit, n_fit, col_fit, rule_fit, valid)
        o[valid_mask] = pp[best]
        global_loss += np.array([loss(p, y[valid]) for p in pp]) * len(valid)
        global_n += len(valid)
    best = int(np.argmin(global_loss / max(1, global_n)))
    if len(rest):
        full[rest] = predict_from_counts(cnt0_all, n_all, col_cnt_all, rule_cnt_all, rest)[best]
    return o, full, {
        'fold_strengths': chosen,
        'full_strength': STRENGTHS[best],
        'calibration_folds': 5,
        'nested_calibration': True,
        'parent_query_labels_used': False,
        'oof_nll': loss(o, y[rows]),
    }


def _tr_rct_candidate(name, j, X_raw, given, fold, values, shared_na_y, deadline):
    """Two-stage World Bank tracer candidate:
    Stage 1: 9-GIVEN additive EB attrition gate + shared 19-item respondent attrition + 758/1229 report prior.
    Stage 2: Answered conditional 9-GIVEN additive EB on (y < na_idx) + 2025 RCT treatment-effect tilting.
    """
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    y = j['y']
    K = j['K']
    if len(rows) < 40 or len(np.unique(fold[rows])) != 5 or 'Not answered' not in values or K != len(values) + 1:
        return None
    na_idx = values.index('Not answered')
    gate_idx = K - 1
    M = na_idx
    group_col = given.index('group')
    rand_mask = X_raw[:, group_col] != 4
    treat_mask = X_raw[:, group_col] == 0
    n_cols = X_raw.shape[1]

    r_rows_mask = rand_mask[rows]
    r_rows = rows[r_rows_mask]
    if len(r_rows) < 30:
        return None
    f_r = fold[r_rows].astype(np.int64)
    y_na_r = ((y[r_rows] == na_idx) if shared_na_y is None else shared_na_y[r_rows]).astype(np.int64)

    na_tot_f = np.bincount(f_r, minlength=5).astype(float)
    na_yes_f = np.bincount(f_r, weights=y_na_r, minlength=5).astype(float)
    na_tot_all = na_tot_f.sum()
    na_yes_all = na_yes_f.sum()

    na_col_tot_f = []
    na_col_yes_f = []
    na_col_tot_all = []
    na_col_yes_all = []
    for c in range(n_cols):
        col = X_raw[:, c]
        nc = int(col.max()) + 1
        tf = np.bincount(f_r * nc + col[r_rows], minlength=5 * nc).reshape(5, nc).astype(float)
        yf = np.bincount(f_r * nc + col[r_rows], weights=y_na_r, minlength=5 * nc).reshape(5, nc).astype(float)
        na_col_tot_f.append(tf)
        na_col_yes_f.append(yf)
        na_col_tot_all.append(tf.sum(0))
        na_col_yes_all.append(yf.sum(0))

    ans_mask_in_r = (y[r_rows] >= 0) & (y[r_rows] < M)
    a_rows = r_rows[ans_mask_in_r]
    f_a = fold[a_rows].astype(np.int64)
    y_a = y[a_rows].astype(np.int64)
    ans_cnt_f = np.bincount(f_a * M + y_a, minlength=5 * M).reshape(5, M).astype(float)
    ans_cnt_all = ans_cnt_f.sum(0)
    ans_col_f = []
    ans_col_all = []
    for c in range(n_cols):
        col = X_raw[:, c]
        nc = int(col.max()) + 1
        af = np.bincount((f_a * nc + col[a_rows]) * M + y_a, minlength=5 * nc * M).reshape(5, nc, M).astype(float)
        ans_col_f.append(af)
        ans_col_all.append(af.sum(0))

    tilt_spec = WB_RCT_ANSWERED_TILTS.get(name)
    tilt_vec = np.ones(M, float)
    if tilt_spec is not None:
        pos_opts, boost = tilt_spec
        pos_idx = np.array([v in pos_opts for v in values[:M]], bool)
        if pos_idx.any() and (~pos_idx).any():
            tilt_vec[pos_idx] = float(boost)

    CONFIGS = (
        (96.0, 0.0, 0.0, 0.0),
        (64.0, 0.0, 0.22, 0.0),
        (64.0, 0.10, 0.22, 0.0),
        (64.0, 0.08, 0.22, 0.65),
    )
    prior_na_report = 758.0 / 1229.0

    def eval_configs(na_tot, na_yes, na_c_tot, na_c_yes, ans_cnt, ans_c, query):
        n_q = len(query)
        out = np.zeros((len(CONFIGS), n_q, K), float)
        q_rand = rand_mask[query]
        out[:, ~q_rand, gate_idx] = 1.0
        if not q_rand.any():
            return normalize(out)
        qr = query[q_rand]
        n_ans = max(1.0, float(ans_cnt.sum()))
        p0_ans = (ans_cnt + 0.5 / M) / (n_ans + 0.5)
        log_p0_ans = np.log(np.maximum(p0_ans, 1e-12))
        ans_diff_sum = np.zeros((len(qr), M), float)
        for c in range(n_cols):
            ct = ans_c[c]
            pc = (ct + 8.0 * p0_ans) / (ct.sum(1, keepdims=True) + 8.0)
            ans_diff_sum += np.log(np.maximum(pc[X_raw[qr, c]], 1e-12)) - log_p0_ans

        p_na_base64 = (na_yes + 64.0 * prior_na_report) / (na_tot + 64.0)
        p_na_base96 = (na_yes + 96.0 * prior_na_report) / (na_tot + 96.0)
        lo_base64 = np.log(np.clip(p_na_base64, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_na_base64, 1e-6, 1.0 - 1e-6)))
        na_diff_sum = np.zeros(len(qr), float)
        for c in range(n_cols):
            pc_na = (na_c_yes[c] + 10.0 * p_na_base64) / (na_c_tot[c] + 10.0)
            pc_q = np.clip(pc_na[X_raw[qr, c]], 1e-6, 1.0 - 1e-6)
            na_diff_sum += np.log(pc_q / (1.0 - pc_q)) - lo_base64

        is_treat_q = treat_mask[qr]
        for idx, (a_na, b_na, b_ans, w_rct) in enumerate(CONFIGS):
            if b_na == 0.0:
                p_na = np.full(len(qr), p_na_base96 if a_na > 80.0 else p_na_base64)
            else:
                p_na = 1.0 / (1.0 + np.exp(-(lo_base64 + b_na * na_diff_sum)))
            ll = log_p0_ans[None, :] + b_ans * ans_diff_sum
            ll -= ll.max(1, keepdims=True)
            q_ans = np.exp(ll)
            q_ans /= q_ans.sum(1, keepdims=True)
            if w_rct > 0.0 and tilt_spec is not None and is_treat_q.any():
                mult = (tilt_vec ** w_rct)[None, :]
                tilted = normalize(q_ans * mult)
                q_ans = np.where(is_treat_q[:, None], tilted, q_ans)
            sub = np.zeros((len(qr), K), float)
            sub[:, :M] = (1.0 - p_na[:, None]) * q_ans
            sub[:, na_idx] = p_na
            out[idx, q_rand] = sub
        return normalize(out)

    f_rows = fold[rows].astype(np.int64)
    # Compute all 4 config predictions across the 5 outer folds first (only 5 calls instead of 25!)
    oof_configs = np.zeros((len(CONFIGS), len(rows), K), np.float32)
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        valid_mask = f_rows == f
        valid = rows[valid_mask]
        oof_configs[:, valid_mask] = eval_configs(
            na_tot_all - na_tot_f[f],
            na_yes_all - na_yes_f[f],
            [ na_col_tot_all[c] - na_col_tot_f[c][f] for c in range(n_cols) ],
            [ na_col_yes_all[c] - na_col_yes_f[c][f] for c in range(n_cols) ],
            ans_cnt_all - ans_cnt_f[f],
            [ ans_col_all[c] - ans_col_f[c][f] for c in range(n_cols) ],
            valid,
        )
    # Select best config for each fold f using the other 4 folds' OOF losses (zero extra eval_configs calls!)
    fold_losses = np.array(
        [[loss(oof_configs[c, f_rows == g], y[rows[f_rows == g]]) * np.sum(f_rows == g) for g in range(5)] for c in range(len(CONFIGS))]
    )
    total_losses = fold_losses.sum(1)
    o = np.zeros((len(rows), K), np.float32)
    chosen = []
    for f in range(5):
        g = (f + 1) % 5
        va = rows[f_rows == g]
        inner = eval_configs(
            na_tot_all - na_tot_f[f] - na_tot_f[g],
            na_yes_all - na_yes_f[f] - na_yes_f[g],
            [na_col_tot_all[c] - na_col_tot_f[c][f] - na_col_tot_f[c][g] for c in range(n_cols)],
            [na_col_yes_all[c] - na_col_yes_f[c][f] - na_col_yes_f[c][g] for c in range(n_cols)],
            ans_cnt_all - ans_cnt_f[f] - ans_cnt_f[g],
            [ans_col_all[c] - ans_col_f[c][f] - ans_col_f[c][g] for c in range(n_cols)], va)
        best_f = int(np.argmin([loss(p, y[va]) for p in inner]))
        chosen.append(float(best_f))
        o[f_rows == f] = oof_configs[best_f, f_rows == f]
    best = int(np.argmin(total_losses))
    full = np.zeros((len(X_raw), K), np.float32)
    if len(rest):
        full[rest] = eval_configs(
            na_tot_all, na_yes_all, na_col_tot_all, na_col_yes_all, ans_cnt_all, ans_col_all, rest
        )[best]
    return o, full, {
        'fold_strengths': chosen,
        'full_strength': float(best),
        'calibration_folds': 5,
        'nested_calibration': True,
        'parent_query_labels_used': False,
        'oof_nll': loss(o, y[rows]),
    }


def _fast_unhcr_erpis_candidate(j, groups, work, code, X_raw, fold, values, deadline):
    """Vectorized 5-fold UNHCR 17-GIVEN additive EB + 143 ERPIS report constraints with asked-category IPF tilting."""
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    y = j['y']
    K = j['K']
    if len(rows) < 20 or len(np.unique(fold[rows])) != 5:
        return None
    f_rows = fold[rows].astype(np.int64)
    y_rows = y[rows].astype(np.int64)
    ng = int(code.max()) + 1
    n_cols = X_raw.shape[1]
    asked = len(values)
    active_groups = [g for g in groups if g[3]]

    cnt0_f = np.bincount(f_rows * K + y_rows, minlength=5 * K).reshape(5, K).astype(float)
    cnt0_all = cnt0_f.sum(0)
    col_cnt_f = []
    col_cnt_all = []
    for c in range(n_cols):
        col = X_raw[:, c]
        nc = int(col.max()) + 1
        cf = np.bincount((f_rows * nc + col[rows]) * K + y_rows, minlength=5 * nc * K).reshape(5, nc, K).astype(float)
        col_cnt_f.append(cf)
        col_cnt_all.append(cf.sum(0))

    rule_cnt_f = []
    rule_cnt_all = []
    rule_elig_tot_f = []
    rule_elig_yes_f = []
    rule_elig_tot_all = []
    rule_elig_yes_all = []
    for exact, event, parents, _ in active_groups:
        m = exact[rows] & event[rows]
        rf = np.bincount(f_rows[m] * K + y_rows[m], minlength=5 * K).reshape(5, K).astype(float)
        rule_cnt_f.append(rf)
        rule_cnt_all.append(rf.sum(0))
        if parents:
            vis_p = np.all(np.column_stack([work[p]['y'][rows] >= 0 for p in parents]), axis=1)
            el = exact[rows] & vis_p
            tf = np.bincount(f_rows[el] * ng + code[rows[el]], minlength=5 * ng).reshape(5, ng).astype(float)
            yf = np.bincount(
                f_rows[el] * ng + code[rows[el]], weights=event[rows[el]].astype(float), minlength=5 * ng
            ).reshape(5, ng).astype(float)
            rule_elig_tot_f.append(tf)
            rule_elig_yes_f.append(yf)
            rule_elig_tot_all.append(tf.sum(0))
            rule_elig_yes_all.append(yf.sum(0))
        else:
            rule_elig_tot_f.append(None)
            rule_elig_yes_f.append(None)
            rule_elig_tot_all.append(None)
            rule_elig_yes_all.append(None)

    def predict_from_counts(cnt0, n_tr, col_cnts, rule_cnts, elig_tots, elig_yes, query):
        pm = np.zeros(K) if (K > asked and cnt0[-1] == 0) else np.full(K, 0.5 / K)
        if K > asked and cnt0[-1] == 0:
            pm[:-1] = 0.5 / max(1, K - 1)
        p0 = (cnt0 + pm) / (n_tr + pm.sum())
        log_p0 = np.log(np.maximum(p0, 1e-12))
        ll = np.broadcast_to(log_p0, (len(query), K)).copy()
        for c in range(n_cols):
            ct = col_cnts[c]
            pc = (ct + 8.0 * p0) / (ct.sum(1, keepdims=True) + 8.0)
            ll += 0.28 * (np.log(np.maximum(pc[X_raw[query, c]], 1e-12)) - log_p0)
        ll -= ll.max(1, keepdims=True)
        base_q = np.exp(ll)
        base_q /= base_q.sum(1, keepdims=True)
        if not active_groups:
            return normalize(base_q[None, :, :])

        # Vectorized composite log-tilt across all active ERPIS rules on asked categories 0..asked-1
        delta_log_asked = np.zeros((len(query), asked), float)
        p0_asked = normalize(p0[:asked])
        for r_idx, (exact, event, parents, constraints) in enumerate(active_groups):
            count = rule_cnts[r_idx]
            if count[:asked].sum() <= 0:
                continue
            emp_asked = normalize(count[:asked] + 2.0 * p0_asked)
            rep_asked = _report_vector(emp_asked, constraints, values, preserve_na=False)
            log_ratio = np.log(np.maximum(rep_asked, 1e-7) / np.maximum(emp_asked, 1e-7))
            if parents:
                tot = elig_tots[r_idx]
                yes = elig_yes[r_idx]
                tot_s = tot.sum()
                mean_ev = float(yes.sum() / tot_s) if tot_s > 0 else 0.0
                rate = (yes + 4.0 * mean_ev) / (tot + 4.0)
                weight = exact[query].astype(float) * rate[code[query]]
            else:
                weight = exact[query].astype(float)
            delta_log_asked += weight[:, None] * log_ratio[None, :]

        outputs = np.broadcast_to(base_q, (len(STRENGTHS), len(query), K)).copy()
        log_base_asked = np.log(np.maximum(base_q[:, :asked], 1e-12))
        skip_mass = np.clip(base_q[:, asked:].sum(1, keepdims=True), 0.0, 1.0 - 1e-7) if K > asked else None
        for a, alpha in enumerate(STRENGTHS):
            if alpha == 0.0:
                continue
            w = alpha / (alpha + 32.0)
            la = log_base_asked + w * delta_log_asked
            la -= la.max(1, keepdims=True)
            tilted_asked = np.exp(la)
            tilted_asked /= tilted_asked.sum(1, keepdims=True)
            if K > asked:
                outputs[a, :, :asked] = tilted_asked * (1.0 - skip_mass)
            else:
                outputs[a] = tilted_asked
        return normalize(outputs)

    n_all = float(len(rows))
    n_f = cnt0_f.sum(1)
    if not active_groups:
        o = np.zeros((len(rows), K), np.float32)
        full = np.zeros((len(X_raw), K), np.float32)
        for f in range(5):
            if time.monotonic() >= deadline:
                return None
            valid_mask = f_rows == f
            valid = rows[valid_mask]
            o[valid_mask] = predict_from_counts(
                cnt0_all - cnt0_f[f],
                n_all - n_f[f],
                [ col_cnt_all[c] - col_cnt_f[c][f] for c in range(n_cols) ],
                (),
                (),
                (),
                valid,
            )[0]
        if len(rest):
            full[rest] = predict_from_counts(cnt0_all, n_all, col_cnt_all, (), (), (), rest)[0]
        return o, full, {
            'fold_strengths': [0.0] * 5,
            'full_strength': 0.0,
            'calibration_folds': 5,
            'nested_calibration': True,
            'parent_query_labels_used': False,
            'oof_nll': loss(o, y[rows]),
        }

    # Candidate predictions exclude the outer fold; strength selection below
    # uses a separate rotating inner fold with both folds removed from counts.
    oof_s = np.zeros((len(STRENGTHS), len(rows), K), np.float32)
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        valid_mask = f_rows == f
        valid = rows[valid_mask]
        oof_s[:, valid_mask] = predict_from_counts(
            cnt0_all - cnt0_f[f],
            n_all - n_f[f],
            [ col_cnt_all[c] - col_cnt_f[c][f] for c in range(n_cols) ],
            [ rule_cnt_all[r] - rule_cnt_f[r][f] for r in range(len(active_groups)) ],
            [
                (rule_elig_tot_all[r] - rule_elig_tot_f[r][f]) if rule_elig_tot_all[r] is not None else None
                for r in range(len(active_groups))
            ],
            [
                (rule_elig_yes_all[r] - rule_elig_yes_f[r][f]) if rule_elig_yes_all[r] is not None else None
                for r in range(len(active_groups))
            ],
            valid,
        )
    fold_losses = np.array(
        [[loss(oof_s[s, f_rows == g], y[rows[f_rows == g]]) * np.sum(f_rows == g) for g in range(5)] for s in range(len(STRENGTHS))]
    )
    total_losses = fold_losses.sum(1)
    o = np.zeros((len(rows), K), np.float32)
    chosen = []
    for f in range(5):
        g = (f + 1) % 5
        va = rows[f_rows == g]
        inner = predict_from_counts(
            cnt0_all - cnt0_f[f] - cnt0_f[g], n_all - n_f[f] - n_f[g],
            [col_cnt_all[c] - col_cnt_f[c][f] - col_cnt_f[c][g] for c in range(n_cols)],
            [rule_cnt_all[r] - rule_cnt_f[r][f] - rule_cnt_f[r][g] for r in range(len(active_groups))],
            [(rule_elig_tot_all[r] - rule_elig_tot_f[r][f] - rule_elig_tot_f[r][g])
             if rule_elig_tot_all[r] is not None else None for r in range(len(active_groups))],
            [(rule_elig_yes_all[r] - rule_elig_yes_f[r][f] - rule_elig_yes_f[r][g])
             if rule_elig_yes_all[r] is not None else None for r in range(len(active_groups))], va)
        best_f = int(np.argmin([loss(p, y[va]) for p in inner]))
        chosen.append(STRENGTHS[best_f])
        o[f_rows == f] = oof_s[best_f, f_rows == f]
    best = int(np.argmin(total_losses))
    full = np.zeros((len(X_raw), K), np.float32)
    if len(rest):
        full[rest] = predict_from_counts(
            cnt0_all, n_all, col_cnt_all, rule_cnt_all, rule_elig_tot_all, rule_elig_yes_all, rest
        )[best]
    return o, full, {
        'fold_strengths': chosen,
        'full_strength': STRENGTHS[best],
        'calibration_folds': 5,
        'nested_calibration': True,
        'parent_query_labels_used': False,
        'oof_nll': loss(o, y[rows]),
    }


def _all_predictions(j, groups, work, code, train, query, values, X_raw=None, is_wb=False):
    y = j['y']
    K = j['K']
    ng = int(code.max()) + 1
    global_count = np.bincount(y[train], minlength=K).astype(float) + 0.5 / K
    marginal = global_count / global_count.sum()
    if X_raw is not None:
        cnt0 = np.bincount(y[train], minlength=K).astype(float)
        pm = np.zeros(K) if (K > len(values) and cnt0[-1] == 0) else np.full(K, 0.5 / K)
        if K > len(values) and cnt0[-1] == 0:
            pm[:-1] = 0.5 / max(1, K - 1)
        p0 = (cnt0 + pm) / (len(train) + pm.sum())
        log_p0 = np.log(np.maximum(p0, 1e-12))
        ll = np.broadcast_to(log_p0, (len(query), K)).copy()
        for c in range(X_raw.shape[1]):
            col = X_raw[:, c]
            nc = int(col.max()) + 1
            ct = np.bincount(col[train] * K + y[train], minlength=nc * K).reshape(nc, K).astype(float)
            pc = (ct + 6.0 * p0) / (ct.sum(1, keepdims=True) + 6.0)
            ll += 0.35 * (np.log(np.maximum(pc[col[query]], 1e-12)) - log_p0)
        ll -= ll.max(1, keepdims=True)
        base_q = np.exp(ll)
        base_q /= base_q.sum(1, keepdims=True)
        outputs = np.broadcast_to(base_q, (len(STRENGTHS), len(query), K)).copy()
        for exact, event, parents, constraints in groups:
            matched = train[exact[train] & event[train]]
            if not len(matched):
                continue
            cnt_m = np.bincount(y[matched], minlength=K).astype(float)
            emp = normalize(cnt_m + (pm if (K > len(values) and cnt_m[-1] == 0) else 2.0 * p0))
            rep = _report_vector(emp, constraints, values, preserve_na=True)
            ratio = rep / np.maximum(emp, 1e-7)
            ex_q = exact[query][:, None]
            for a, alpha in enumerate(STRENGTHS):
                w = alpha / (alpha + 32.0)
                tilted = normalize(outputs[a] * (ratio ** w)[None, :])
                if K > len(values) and cnt0[-1] == 0:
                    tilted[:, -1] = 0.0
                    tilted /= np.maximum(tilted.sum(1, keepdims=True), 1e-12)
                outputs[a] = np.where(ex_q, tilted, outputs[a])
        return normalize(outputs)
    counts = np.zeros((ng, K))
    np.add.at(counts, (code[train], y[train]), 1.0)
    baseline = (counts + 4 * marginal) / (counts.sum(1, keepdims=True) + 4)
    outputs = np.broadcast_to(baseline[code[query]], (len(STRENGTHS), len(query), K)).copy()
    for exact, event, parents, constraints in groups:
        matched = train[exact[train] & event[train]]
        if not len(matched):
            continue
        count = np.bincount(y[matched], minlength=K).astype(float)
        if is_wb and K > len(values) and count[-1] == 0:
            prior_m = np.zeros(K)
            prior_m[:-1] = 0.5 / max(1, K - 1)
            empirical = normalize(count + prior_m)
        else:
            prior_m = None
            empirical = normalize(count + 2 * marginal)
        report = _report_vector(empirical, constraints, values, preserve_na=is_wb)
        eligible = train[exact[train]]
        if parents:
            eligible = eligible[np.all(np.column_stack([work[p]['y'][eligible] >= 0 for p in parents]), axis=1)]
            total = np.bincount(code[eligible], minlength=ng).astype(float)
            yes = np.bincount(code[eligible], weights=event[eligible], minlength=ng)
            rate = (yes + 2 * (event[eligible].mean() if len(eligible) else 0.0)) / (total + 2.0)
            weight = exact[query] * rate[code[query]]
        else:
            weight = exact[query].astype(float)
        for a, alpha in enumerate(STRENGTHS):
            if prior_m is not None:
                posterior = normalize(count + prior_m + alpha * report)
                posterior[-1] = 0.0
                posterior /= max(posterior.sum(), 1e-12)
            else:
                posterior = normalize(count + 2 * marginal + alpha * report)
            outputs[a] = (1 - weight[:, None]) * outputs[a] + weight[:, None] * posterior
    return normalize(outputs)


def candidate(j, groups, work, code, fold, values, deadline, X_raw=None, is_wb=False):
    rows = np.flatnonzero(j['train'])
    rest = np.flatnonzero(~j['train'])
    y = j['y']
    if len(rows) < (40 if is_wb else 20) or len(np.unique(fold[rows])) != 5:
        return None
    o = np.zeros((len(rows), j['K']), np.float32)
    full = np.zeros((len(code), j['K']), np.float32)
    chosen = []
    global_loss = np.zeros(len(STRENGTHS))
    global_n = 0
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        fit = rows[fold[rows] != f]
        valid = rows[fold[rows] == f]
        inner_loss = np.zeros(len(STRENGTHS))
        inner_n = 0
        for g in range(5):
            if g == f:
                continue
            tr = fit[fold[fit] != g]
            va = fit[fold[fit] == g]
            if not len(tr) or not len(va):
                continue
            pp = _all_predictions(j, groups, work, code, tr, va, values, X_raw=X_raw, is_wb=is_wb)
            inner_loss += np.array([loss(p, y[va]) for p in pp]) * len(va)
            inner_n += len(va)
        best = int(np.argmin(inner_loss / max(1, inner_n)))
        chosen.append(STRENGTHS[best])
        pp = _all_predictions(j, groups, work, code, fit, valid, values, X_raw=X_raw, is_wb=is_wb)
        o[fold[rows] == f] = pp[best]
        global_loss += np.array([loss(p, y[valid]) for p in pp]) * len(valid)
        global_n += len(valid)
    best = int(np.argmin(global_loss / max(1, global_n)))
    if len(rest):
        full[rest] = _all_predictions(j, groups, work, code, rows, rest, values, X_raw=X_raw, is_wb=is_wb)[best]
    return o, full, {
        'fold_strengths': chosen,
        'full_strength': STRENGTHS[best],
        'calibration_folds': 5,
        'nested_calibration': True,
        'parent_query_labels_used': False,
        'oof_nll': loss(o, y[rows]),
    }


def attach(schema, X, given, work, fold, deadline):
    from base import _nll
    from cross_round import dominates

    is_wb = 'group' in given
    is_unhcr = 'wave' in given
    groups = _compiled(schema, X, given, work)
    code = _group_codes(X, given)
    trace = {}
    n_raw = sum(item.get('class') == 'GIVEN' for item in schema['items'].values()) or X.shape[1]
    X_raw_given = X[:, :n_raw]

    shared_na_y = None
    if is_wb and ENABLE_WB_RCT:
        tr_direct = [
            k
            for k in work
            if k.startswith('tr_')
            and isinstance(schema['items'][k].get('gate'), dict)
            and schema['items'][k]['gate'].get('parent') == 'group'
            and 'Not answered' in (schema['items'][k].get('values') or ())
        ]
        if len(tr_direct) >= 8:
            na_votes = np.zeros(len(X), int)
            for k in tr_direct:
                na_opt = schema['items'][k]['values'].index('Not answered')
                na_votes += (work[k]['y'] == na_opt) & work[k]['train']
            shared_na_y = na_votes >= (len(tr_direct) // 2 + 1)

    for name, rules in groups.items():
        if time.monotonic() >= deadline:
            break
        j = work[name]
        values = schema['items'][name]['values']
        if is_wb and ENABLE_WB_RCT:
            if name.startswith('tr_') and 'Not answered' in values and j['K'] == len(values) + 1:
                result = _tr_rct_candidate(name, j, X_raw_given, given, fold, values, shared_na_y, deadline)
            else:
                result = _fast_wb_user_candidate(j, rules, X_raw_given, fold, values, deadline)
        elif is_unhcr and ENABLE_UNHCR_ERPIS:
            result = _fast_unhcr_erpis_candidate(j, rules, work, code, X_raw_given, fold, values, deadline)
        else:
            x_in = X_raw_given if (is_wb and name.startswith('user.')) else None
            result = candidate(j, rules, work, code, fold, values, deadline, X_raw=x_in, is_wb=is_wb)
        if result is None:
            continue
        o, p, info = result
        if not is_wb and not (is_unhcr and ENABLE_UNHCR_ERPIS):
            yy = j['y'][j['train']]
            base = j['oof'][0]
            losses = [loss((1 - a) * base + a * o, yy) for a in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
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
        base = j['oof'][0]
        half = (fold[rows] % 2) == 0
        base_loss = _nll(base, yy, k)
        best_loss = base_loss
        best_alpha = 0.0
        # The three dominance tests use only the true-class probability.
        # Extract it once instead of materializing each whole simplex mixture.
        pick = np.arange(len(yy))
        bp, op = base[pick, yy], o[pick, yy]
        baseline_losses = -np.log(.001 + (1. - k*.001)*bp)
        halves_present = half.any() and (~half).any()
        for a in (0.1, 0.25, 0.5, 0.75, 1.0):
            losses = -np.log(.001 + (1. - k*.001)*((1. - a)*bp + a*op))
            l = float(losses.mean())
            dominates_both = (halves_present and base_loss-l >= 1e-5
                and baseline_losses[half].mean()-losses[half].mean() >= 1e-5
                and baseline_losses[~half].mean()-losses[~half].mean() >= 1e-5)
            if l < best_loss and dominates_both:
                best_loss = l
                best_alpha = a
        gain = float(base_loss - best_loss)
        min_gain = 1e-4 if is_wb else (1e-5 if rules else 2e-4)
        info.update(table_nll=float(base_loss), mixture_gain=gain, best_alpha=best_alpha, constraints=len(rules))
        info['added'] = bool(gain > min_gain)
        if info['added']:
            j['oof'].append(o)
            j['all'].append(p)
            j['arms'].append('report_prior')
        trace[name] = info
    return trace
