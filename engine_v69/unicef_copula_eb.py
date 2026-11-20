"""UNICEF : Trait-centered multi-resolution discrete cell EB, Stage-2 trait posterior conditioning, and 12-item cross-item copula EB."""
import time
import numpy as np
from .unicef_lca import _cell_tables


def _solve_simplex_kkt(comp, base_idx=0, deadline=float("inf")):
    """Fast exact simplex solver maximizing mean(log(comp @ w)) via EM warm-start + pairwise KKT bisection transfers."""
    n_rows, M = comp.shape
    if M <= 1:
        return np.ones(1, dtype=np.float64)
    losses = -np.log(np.clip(comp, 1e-12, 1.0)).mean(0)
    best_single = int(np.argmin(losses))
    w = np.full(M, 0.25 / M, dtype=np.float64)
    w[best_single] += 0.50
    w[base_idx] += 0.25
    w /= w.sum()

    for _ in range(30):
        if time.monotonic() >= deadline: break
        q = np.clip(comp @ w, 1e-12, 1.0)
        gain = (comp.T @ (1.0 / q)) / n_rows
        if float(gain.max() - w @ gain) <= 1e-7:
            break
        w *= gain
        w /= w.sum()

    q = np.clip(comp @ w, 1e-12, 1.0)
    for _ in range(28):
        if time.monotonic() >= deadline: break
        gain = (comp.T @ (1.0 / q)) / n_rows
        if float(gain.max() - w @ gain) <= 1e-7:
            break
        receiver = int(np.argmax(gain))
        donors = np.flatnonzero(w > 1e-12)
        if not len(donors):
            break
        donor = int(donors[np.argmin(gain[donors])])
        if donor == receiver or w[donor] <= 1e-12:
            break
        delta = comp[:, receiver] - comp[:, donor]
        lo, hi = 0.0, float(w[donor])
        if float(np.mean(delta / (q + hi * delta))) >= 0.0:
            amount = hi
        else:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if float(np.mean(delta / (q + mid * delta))) > 0.0:
                    lo = mid
                else:
                    hi = mid
            amount = 0.5 * (lo + hi)
        if amount <= 1e-12:
            break
        w[donor] = max(0.0, w[donor] - amount)
        w[receiver] += amount
        w /= w.sum()
        q = np.clip(comp @ w, 1e-12, 1.0)

    # Guard against numerical drift worse than best single candidate
    if float(-np.log(q).mean()) > float(losses[best_single]) - 1e-9:
        w = np.zeros(M, dtype=np.float64)
        w[best_single] = 1.0
    return w


def _mix_folds_kkt(work, fold, folds, cands, pred, n, names=None, deadline=float("inf")):
    """Per-item out-of-fold simplex mixer under the official 1e-3 floored NLL objective."""
    order = list(work if names is None else names)
    out = {}
    weights_out = {}
    M = len(cands)
    for name in order:
        if time.monotonic() >= deadline: break
        j = work[name]
        K = int(j['K'])
        rows = np.flatnonzero(j['train'])
        rf = fold[rows]
        y = np.maximum(j['y'][rows], 0)
        oof_c = np.empty((M, len(rows), K), dtype=np.float64)
        full_c = np.zeros((M, n, K), dtype=np.float64)
        for ki, c in enumerate(cands):
            for f in folds:
                P = pred(f, c, name)
                hit = rf == f
                oof_c[ki, hit] = P[rows[hit]]
                full_c[ki] += P / len(folds)
            stray = rf < 0
            if stray.any():
                oof_c[ki, stray] = full_c[ki, rows[stray]]
        picked = oof_c[:, np.arange(len(rows)), y].T
        comp = 1e-3 + (1.0 - K * 1e-3) * np.clip(picked, 0.0, 1.0)
        oof = np.empty((len(rows), K), dtype=np.float64)
        for f in folds:
            hit = rf == f
            w_f = _solve_simplex_kkt(comp[~hit], base_idx=0, deadline=deadline)
            oof[hit] = np.tensordot(w_f, oof_c[:, hit], axes=(0, 0))
        w_all = _solve_simplex_kkt(comp, base_idx=0, deadline=deadline)
        stray = rf < 0
        if stray.any():
            oof[stray] = np.tensordot(w_all, oof_c[:, stray], axes=(0, 0))
        full = np.tensordot(w_all, full_c, axes=(0, 0))
        oof = np.clip(oof, 1e-7, 1.0)
        oof /= oof.sum(1, keepdims=True)
        full = np.clip(full, 1e-7, 1.0)
        full /= full.sum(1, keepdims=True)
        out[name] = (oof.astype(np.float32), full.astype(np.float32).reshape(n, K))
        weights_out[name] = [round(float(v), 4) for v in w_all]
    return out, weights_out


def _encode_group(Xg, cols):
    sub = Xg[:, cols]
    _, inv = np.unique(sub, axis=0, return_inverse=True)
    return inv.ravel().astype(np.int64)


def _dirichlet_residual_step(P, cell_codes, n_cells, tr_idx, y_tr, K, alpha, discount=0.0):
    """Residual Dirichlet / Pitman-Yor update on top of P (n x K) using training rows tr_idx."""
    if not len(tr_idx):
        return P
    c_tr = cell_codes[tr_idx]
    n_c = np.bincount(c_tr, minlength=n_cells).astype(np.float64)
    active = n_c > 0
    if not np.any(active):
        return P
    obs = np.bincount(c_tr * K + y_tr, minlength=n_cells * K).reshape(n_cells, K).astype(np.float64)
    exp_c = np.zeros((n_cells, K), dtype=np.float64)
    for k in range(K):
        exp_c[:, k] = np.bincount(c_tr, weights=P[tr_idx, k], minlength=n_cells)
    p_bar = np.zeros((n_cells, K), dtype=np.float64)
    p_bar[active] = exp_c[active] / n_c[active, None]

    if discount > 0.0:
        m_ck = (obs > 0).astype(np.float64)
        t_c = m_ck.sum(1, keepdims=True)
        num = np.maximum(obs - discount * m_ck, 0.0) + (alpha + discount * t_c) * p_bar
    else:
        num = obs + alpha * p_bar
    den = n_c[:, None] + alpha
    post_bar = np.zeros((n_cells, K), dtype=np.float64)
    post_bar[active] = num[active] / den[active]

    delta = post_bar - p_bar
    out = np.clip(P + delta[cell_codes], 1e-7, 1.0)
    out /= out.sum(1, keepdims=True)
    return out


def _soft_dirichlet_residual_step(P, cell_codes, n_cells, tr_idx, R_tr, w_tr, K, alpha):
    """Residual Dirichlet update on top of P (n x K) using soft Rao-Blackwellized pseudo-counts R_tr (len(tr_idx) x K)."""
    if not len(tr_idx):
        return P
    c_tr = cell_codes[tr_idx]
    n_c = np.bincount(c_tr, weights=w_tr, minlength=n_cells).astype(np.float64)
    active = n_c > 1e-6
    if not np.any(active):
        return P
    obs = np.zeros((n_cells, K), dtype=np.float64)
    exp_c = np.zeros((n_cells, K), dtype=np.float64)
    for k in range(K):
        obs[:, k] = np.bincount(c_tr, weights=w_tr * R_tr[:, k], minlength=n_cells)
        exp_c[:, k] = np.bincount(c_tr, weights=w_tr * P[tr_idx, k], minlength=n_cells)
    delta = np.zeros((n_cells, K), dtype=np.float64)
    delta[active] = (obs[active] - exp_c[active]) / (n_c[active, None] + alpha)
    out = np.clip(P + delta[cell_codes], 1e-7, 1.0)
    out /= out.sum(1, keepdims=True)
    return out


def attach_trait_cell_eb(Xg, given, items, work, fold, state_keep, deadline, workers=8):
    """Multi-resolution 3-way..7-way discrete cell Pitman-Yor / Dirichlet shrinkage centered on lca_trait_tcxp."""
    start = time.monotonic()
    info = {'status': 'complete', 'items': 0, 'seconds': 0.0}
    trait_folds = (state_keep or {}).get('trait_best_folds')
    if not trait_folds or time.monotonic() >= deadline:
        info['status'] = 'no_trait_folds'
        return {}, info

    raw = list(given[:Xg.shape[1]])
    idx = {name: raw.index(name) for name in raw}
    n = len(Xg)
    folds = [int(f) for f in np.unique(fold) if f >= 0]

    hier_cols = []
    if all(k in idx for k in ('Country', 'Q05', 'Q04')):
        hier_cols.append((idx['Country'], idx['Q05'], idx['Q04']))
    if all(k in idx for k in ('Country', 'Q05', 'Q04', 'Q01')):
        hier_cols.append((idx['Country'], idx['Q05'], idx['Q04'], idx['Q01']))
    if all(k in idx for k in ('Country', 'Q05', 'Q04', 'Q01', 'Q02')):
        hier_cols.append((idx['Country'], idx['Q05'], idx['Q04'], idx['Q01'], idx['Q02']))
    if all(k in idx for k in ('Country', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05')):
        # 6-way demographic cell (all GIVEN except Q13)
        hier_cols.append((idx['Country'], idx['Q01'], idx['Q02'], idx['Q03'], idx['Q04'], idx['Q05']))
    exact_cols = tuple(range(Xg.shape[1]))

    hier_codes = []
    for cols in hier_cols:
        code = _encode_group(Xg, list(cols))
        hier_codes.append((code, int(code.max()) + 1))
    exact_code = _encode_group(Xg, list(exact_cols))
    n_exact = int(exact_code.max()) + 1

    cand_names = ('base', 'exact_30', 'exact_75', 'exact_160', 'hier_strong', 'hier_med', 'hier_py')
    done = {f: {c: {} for c in cand_names} for f in folds}

    for name, j in work.items():
        if time.monotonic() >= deadline:
            break
        K = int(j['K'])
        y = j['y']
        rows = np.flatnonzero(j['train'])
        gate = items.get(name, {}).get('gate')
        is_gated = isinstance(gate, dict)
        given_closed = np.zeros(n, dtype=bool)
        if is_gated and gate.get('parent') in idx:
            p_col = idx[gate['parent']]
            p_vals = list(items[gate['parent']]['values'])
            open_codes = [p_vals.index(v) for v in gate.get('observed_if', ()) if v in p_vals]
            given_closed = ~np.isin(Xg[:, p_col], open_codes)

        for f in folds:
            P0 = np.asarray(trait_folds[name][f], dtype=np.float64).copy()
            P0 = np.clip(P0, 1e-9, 1.0)
            P0 /= P0.sum(1, keepdims=True)
            tr = rows[fold[rows] != f]
            tr = tr[y[tr] >= 0]

            if not is_gated:
                done[f]['base'][name] = P0.astype(np.float32)
                done[f]['exact_30'][name] = _dirichlet_residual_step(P0, exact_code, n_exact, tr, y[tr], K, 30.0).astype(np.float32)
                done[f]['exact_75'][name] = _dirichlet_residual_step(P0, exact_code, n_exact, tr, y[tr], K, 75.0).astype(np.float32)
                done[f]['exact_160'][name] = _dirichlet_residual_step(P0, exact_code, n_exact, tr, y[tr], K, 160.0).astype(np.float32)

                P_hs = P0.copy()
                for (hc, nc), a_h in zip(hier_codes, (80.0, 55.0, 40.0, 32.0)):
                    P_hs = _dirichlet_residual_step(P_hs, hc, nc, tr, y[tr], K, a_h)
                P_hs = _dirichlet_residual_step(P_hs, exact_code, n_exact, tr, y[tr], K, 25.0)
                done[f]['hier_strong'][name] = P_hs.astype(np.float32)

                P_hm = P0.copy()
                for (hc, nc), a_h in zip(hier_codes, (160.0, 110.0, 80.0, 65.0)):
                    P_hm = _dirichlet_residual_step(P_hm, hc, nc, tr, y[tr], K, a_h)
                P_hm = _dirichlet_residual_step(P_hm, exact_code, n_exact, tr, y[tr], K, 55.0)
                done[f]['hier_med'][name] = P_hm.astype(np.float32)

                P_py = P0.copy()
                for (hc, nc), a_h in zip(hier_codes, (120.0, 85.0, 60.0, 48.0)):
                    P_py = _dirichlet_residual_step(P_py, hc, nc, tr, y[tr], K, a_h, discount=0.12)
                P_py = _dirichlet_residual_step(P_py, exact_code, n_exact, tr, y[tr], K, 40.0, discount=0.12)
                done[f]['hier_py'][name] = P_py.astype(np.float32)
            else:
                K_open = K - 1
                tr_open = tr[y[tr] < K_open]
                y_open = y[tr_open]
                p_skip_0 = P0[:, K_open].copy()
                p_open_0 = np.clip(1.0 - p_skip_0, 1e-9, 1.0)
                P_cond_0 = P0[:, :K_open] / np.maximum(P0[:, :K_open].sum(1, keepdims=True), 1e-12)

                if not given_closed.any():
                    y_gate = (y[tr] == K_open).astype(np.int64)
                    G0 = np.column_stack((p_open_0, p_skip_0))
                    G_30 = _dirichlet_residual_step(G0, exact_code, n_exact, tr, y_gate, 2, 40.0)
                    G_75 = _dirichlet_residual_step(G0, exact_code, n_exact, tr, y_gate, 2, 90.0)
                    G_160 = _dirichlet_residual_step(G0, exact_code, n_exact, tr, y_gate, 2, 180.0)
                else:
                    G0 = G_30 = G_75 = G_160 = np.column_stack((p_open_0, p_skip_0))

                tr_cl = tr[given_closed[tr]]
                tr_op = tr[~given_closed[tr]]
                closed_strict = len(tr_cl) >= 25 and bool(np.all(y[tr_cl] == K_open))
                open_strict = len(tr_op) >= 25 and bool(np.all(y[tr_op] < K_open))

                def _assemble(P_c, G_mat):
                    out = np.empty((n, K), dtype=np.float32)
                    out[:, :K_open] = (P_c * G_mat[:, 0:1]).astype(np.float32)
                    out[:, K_open] = G_mat[:, 1].astype(np.float32)
                    if open_strict:
                        out[~given_closed, K_open] = 0.0
                    if closed_strict:
                        out[given_closed] = 0.0
                        out[given_closed, K_open] = 1.0
                    out /= np.maximum(out.sum(1, keepdims=True), 1e-30)
                    return out

                done[f]['base'][name] = _assemble(P_cond_0, G0)
                done[f]['exact_30'][name] = _assemble(_dirichlet_residual_step(P_cond_0, exact_code, n_exact, tr_open, y_open, K_open, 25.0), G_30)
                done[f]['exact_75'][name] = _assemble(_dirichlet_residual_step(P_cond_0, exact_code, n_exact, tr_open, y_open, K_open, 65.0), G_75)
                done[f]['exact_160'][name] = _assemble(_dirichlet_residual_step(P_cond_0, exact_code, n_exact, tr_open, y_open, K_open, 140.0), G_160)

                P_hs = P_cond_0.copy()
                for (hc, nc), a_h in zip(hier_codes, (70.0, 45.0, 32.0, 26.0)):
                    P_hs = _dirichlet_residual_step(P_hs, hc, nc, tr_open, y_open, K_open, a_h)
                P_hs = _dirichlet_residual_step(P_hs, exact_code, n_exact, tr_open, y_open, K_open, 22.0)
                done[f]['hier_strong'][name] = _assemble(P_hs, G_30)

                P_hm = P_cond_0.copy()
                for (hc, nc), a_h in zip(hier_codes, (140.0, 95.0, 68.0, 54.0)):
                    P_hm = _dirichlet_residual_step(P_hm, hc, nc, tr_open, y_open, K_open, a_h)
                P_hm = _dirichlet_residual_step(P_hm, exact_code, n_exact, tr_open, y_open, K_open, 48.0)
                done[f]['hier_med'][name] = _assemble(P_hm, G_75)

                P_py = P_cond_0.copy()
                for (hc, nc), a_h in zip(hier_codes, (105.0, 72.0, 50.0, 40.0)):
                    P_py = _dirichlet_residual_step(P_py, hc, nc, tr_open, y_open, K_open, a_h, discount=0.12)
                P_py = _dirichlet_residual_step(P_py, exact_code, n_exact, tr_open, y_open, K_open, 35.0, discount=0.12)
                done[f]['hier_py'][name] = _assemble(P_py, G_75)

    valid_names = [name for name in work if all(name in done[f]['base'] for f in folds)]
    if not valid_names:
        info['status'] = 'no_valid_items'
        return {}, info

    found, info['weights'] = _mix_folds_kkt(
        work,
        fold,
        folds,
        list(cand_names),
        lambda f, c, name: done[f][c][name],
        n,
        names=valid_names, deadline=deadline,
    )
    if state_keep is not None:
        tcf = {}
        for name in valid_names:
            w_item = info['weights'].get(name)
            if w_item:
                tcf[name] = {
                    f: sum(float(w_c) * done[f][c][name] for w_c, c in zip(w_item, cand_names) if w_c > 0.0)
                    for f in folds
                }
        state_keep['trait_cell_folds'] = tcf
    info['items'] = len(found)
    info['seconds'] = round(time.monotonic() - start, 4)
    return found, info


def attach_trait_stage2(state_keep, anchor_probs, work, names, fold, deadline, workers=8):
    """Stage-2 trait posterior conditioning on top of lca_trait_tcxp/tcxv using 5-fold-isolated Pass-1 anchor probabilities."""
    start = time.monotonic()
    info = {'status': 'complete', 'items': 0, 'seconds': 0.0}
    model = (state_keep or {}).get('trait_model')
    if not model or not anchor_probs or not names or time.monotonic() >= deadline:
        info['status'] = 'no_model_or_budget'
        return {}, info

    O, blocks, n_given_cols, targets, chains, strata = model['layout']
    folds = model['folds']
    xv_items = model.get('xv_items', set())
    tcf = (state_keep or {}).get('trait_cell_folds', {})
    tbf = (state_keep or {}).get('trait_best_folds', {})
    n = len(O)
    t0 = n_given_cols
    tb = [(a - t0, b - t0) for a, b in blocks if a >= t0]
    first = {a: b for b, (a, _) in enumerate(tb)}

    gammas = (0.0, 0.25, 0.55, 0.90)
    cands = [('g', g, 0.0) for g in gammas]

    done = {f: {c: {} for c in cands} for f in folds}

    for f in folds:
        if time.monotonic() >= deadline:
            info['status'] = 'budget_expired'
            return {}, info
        for use_xv in (False, True):
            subset = [name for name in names if (name in xv_items) == use_xv]
            if not subset:
                continue
            params = model['params_xv'][f] if use_xv else model['params_xp'][f]
            log_g, lw, off, inv, stratum = params
            C = len(off)
            Qn = log_g.shape[1]
            rep = np.zeros(C, np.int64)
            rep[inv[::-1]] = np.arange(n)[::-1]
            L_cell = (O[rep, :t0] @ log_g[0, :, :t0].T + lw).astype(np.float64)
            L_cell -= L_cell.max(1, keepdims=True)
            L_base = L_cell[inv]

            Ts = _cell_tables(log_g, stratum, t0, tb)
            w = np.exp(off)

            def scaled(start_col, width):
                b = first[start_col - t0]
                a = start_col - t0
                return Ts[b][:, :, :width] * w[:, None, a:a + width]

            anchor_log_emit = {}
            for a_name in anchor_probs:
                if a_name not in targets:
                    continue
                S_a = tcf.get(a_name, {}).get(f)
                if S_a is None:
                    S_a = tbf.get(a_name, {}).get(f, anchor_probs[a_name])
                a_start, a_K, a_closed = targets[a_name]
                if a_name in chains:
                    T_a = scaled(a_start, a_K - 1)
                    P_a = T_a / np.maximum(T_a.sum(2, keepdims=True), 1e-30)
                    log_P_a = np.log(np.clip(P_a, 1e-12, 1.0))
                    S_use = np.asarray(S_a[:, :a_K - 1], dtype=np.float64).copy()
                    S_use[a_closed] = 0.0
                else:
                    T_a = scaled(a_start, a_K)
                    P_a = T_a / np.maximum(T_a.sum(2, keepdims=True), 1e-30)
                    log_P_a = np.log(np.clip(P_a, 1e-12, 1.0))
                    S_use = np.asarray(S_a, dtype=np.float64).copy()
                    S_use[a_closed] = 0.0
                ev = np.einsum('ik,iqk->iq', S_use, log_P_a[inv], optimize=True)
                anchor_log_emit[a_name] = ev

            # Precompute shared soft evidence and Q matrices across gammas for child items not in anchor_log_emit
            soft_all = np.zeros((n, Qn), dtype=np.float64)
            for ev in anchor_log_emit.values():
                soft_all += ev
            Q_shared = {}
            for g in gammas:
                L = L_base if g == 0.0 else (L_base + g * soft_all)
                L = L - L.max(1, keepdims=True)
                Q = np.exp(np.maximum(L, -80.0))
                Q /= Q.sum(1, keepdims=True)
                Q_shared[g] = Q

            for name in subset:
                t_start, K, closed = targets[name]
                y_item = work[name]['y']
                tr_item = np.flatnonzero(work[name]['train'])
                tr_f = tr_item[(fold[tr_item] != f) & (y_item[tr_item] >= 0)]
                tr_cl = tr_f[closed[tr_f]]
                tr_op = tr_f[~closed[tr_f]]
                closed_strict = len(tr_cl) >= 25 and bool(np.all(y_item[tr_cl] == K - 1))
                open_strict = len(tr_op) >= 25 and bool(np.all(y_item[tr_op] < K - 1))

                if name in anchor_log_emit:
                    soft = soft_all - anchor_log_emit[name]
                    Q_map = {}
                    for g in gammas:
                        L = L_base if g == 0.0 else (L_base + g * soft)
                        L = L - L.max(1, keepdims=True)
                        Q = np.exp(np.maximum(L, -80.0))
                        Q /= Q.sum(1, keepdims=True)
                        Q_map[g] = Q
                else:
                    Q_map = Q_shared

                if name in chains:
                    p_start, p_K, open_codes = chains[name]
                    Tp = scaled(p_start, p_K)
                    go = (Tp[:, :, open_codes].sum(2) / np.maximum(Tp.sum(2), 1e-30))[inv]
                    Tc = scaled(t_start, K - 1)
                    Tc = (Tc / np.maximum(Tc.sum(2, keepdims=True), 1e-30))[inv]
                else:
                    go = None
                    Tk = scaled(t_start, K)
                    Tk = (Tk / np.maximum(Tk.sum(2, keepdims=True), 1e-30))[inv]

                cond_cache = {}
                for g in gammas:
                    Q = Q_map[g]
                    if name in chains:
                        q_open = Q * go
                        open_mass = q_open.sum(1, keepdims=True)
                        cond = np.einsum('iq,iqk->ik', q_open / np.maximum(open_mass, 1e-30), Tc, optimize=True)
                        cond_cache[g] = (cond, open_mass[:, 0])
                    else:
                        P_unchained = np.einsum('iq,iqk->ik', Q, Tk, optimize=True)
                        cond_cache[g] = (P_unchained, None)

                for cand in cands:
                    _, g, _ = cand
                    if name in chains:
                        cond, om0 = cond_cache[g]
                        P = np.empty((n, K), dtype=np.float64)
                        eff_open = np.clip(om0, 0.0, 1.0)
                        P[:, :K - 1] = cond * eff_open[:, None]
                        P[:, K - 1] = 1.0 - eff_open
                    else:
                        P = cond_cache[g][0].copy()

                    if open_strict:
                        P[~closed, -1] = 0.0
                    if closed_strict:
                        P[closed] = 0.0
                        P[closed, -1] = 1.0
                    P /= np.maximum(P.sum(1, keepdims=True), 1e-30)
                    done[f][cand][name] = P.astype(np.float32)

    found, info['weights'] = _mix_folds_kkt(
        work,
        fold,
        folds,
        cands,
        lambda f, c, name: done[f][c][name],
        n,
        names=names, deadline=deadline,
    )
    info['items'] = len(found)
    info['seconds'] = round(time.monotonic() - start, 4)
    return found, info


def attach_copula_eb(Xg, given, items, work, fold, deadline, workers=8):
    """12-item cross-item Rao-Blackwellized cell EB & bilinear log-odds copula (Q11<->Q15, Q06<->Q17, Q07<->Q17, Q06<->Q18, Q08->Q09<->Q10, Q14..Q18)."""
    start = time.monotonic()
    info = {'status': 'complete', 'items': 0, 'seconds': 0.0}
    if not work or time.monotonic() >= deadline:
        info['status'] = 'no_budget'
        return {}, info

    n = len(Xg)
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    names = list(work.keys())
    raw = list(given[:Xg.shape[1]])
    idx = {name: raw.index(name) for name in raw}
    c_col = idx.get('Country', 0)
    country = Xg[:, c_col].astype(np.int64)
    G_country = int(country.max()) + 1

    # Cell groupings for cross-item Rao-Blackwellized pseudo-counts:
    # 4-way (Country, Q05, Q04, Q01), 6-way (all GIVEN except Q13 so Q14..Q18 borrow from Q13='No' rows!), and 7-way exact
    c4_cols = [idx[k] for k in ('Country', 'Q05', 'Q04', 'Q01') if k in idx] or [0]
    c6_cols = [idx[k] for k in ('Country', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05') if k in idx] or [0]
    cell4 = _encode_group(Xg, c4_cols)
    n_cell4 = int(cell4.max()) + 1
    cell6 = _encode_group(Xg, c6_cols)
    n_cell6 = int(cell6.max()) + 1
    cell7 = _encode_group(Xg, list(range(Xg.shape[1])))
    n_cell7 = int(cell7.max()) + 1

    preferred_base = ('trait_cell_eb', 'lca_trait_stage2', 'lca_trait_tcxp', 'lca_trait_tcx', 'lca_trait_tc')
    consensus_arms = ('trait_cell_eb', 'lca_trait_stage2', 'lca_trait_tcxp', 'lca_country', 'shallow', 'shallow_asked', 'logit', 'network')

    S_fold = {f: {} for f in folds}
    P_base = {f: {} for f in folds}

    for name, j in work.items():
        K = int(j['K'])
        y = j['y']
        rows = np.flatnonzero(j['train'])
        rest = np.flatnonzero(~j['train'])
        pk = j.get('partition_k', {})
        arm_map = {a: idx_a for idx_a, a in enumerate(j['arms']) if pk.get(a, 5) == 5}

        base_indices = [arm_map[a] for a in preferred_base if a in arm_map]
        if not base_indices:
            base_indices = list(arm_map.values())[:2]
        cand_indices = [arm_map[a] for a in consensus_arms if a in arm_map]
        if not cand_indices:
            cand_indices = base_indices

        for f in folds:
            tr_mask = fold[rows] != f
            y_tr = y[rows[tr_mask]]

            def _blend(indices):
                if len(indices) == 1:
                    i0 = indices[0]
                    full = np.asarray(j['all'][i0], dtype=np.float64).copy()
                    full[rows] = j['oof'][i0]
                    return np.clip(full, 1e-7, 1.0)
                losses = []
                for i_arm in indices:
                    o_tr = np.clip(j['oof'][i_arm][tr_mask], 1e-9, 1.0)
                    losses.append(float(-np.log(o_tr[np.arange(len(y_tr)), y_tr]).mean()))
                losses = np.asarray(losses, dtype=np.float64)
                w = np.exp(-12.0 * (losses - losses.min()))
                w /= w.sum()
                full = np.zeros((n, K), dtype=np.float64)
                for weight, i_arm in zip(w, indices):
                    full[rows] += weight * j['oof'][i_arm]
                    full[rest] += weight * j['all'][i_arm][rest]
                full = np.clip(full, 1e-7, 1.0)
                full /= full.sum(1, keepdims=True)
                return full

            P_base[f][name] = _blend(base_indices)
            S_fold[f][name] = _blend(cand_indices)

    cand_keys = ('base', 'rb_cell_strong', 'rb_cell_med', 'rb_6way', 'r300', 'r120', 'rb_plus_r180')
    done = {f: {ck: {} for ck in cand_keys} for f in folds}

    # Domain affinity bonus for structural FPCC pairs (Q11<->Q15, Q06<->Q17, Q07<->Q17, Q06<->Q18, Q08->Q09<->Q10, Q14<->Q16<->Q18)
    domain_pairs = {
        ('Q11', 'Q15'): 2.5, ('Q15', 'Q11'): 2.5,
        ('Q06', 'Q17'): 2.0, ('Q17', 'Q06'): 2.0,
        ('Q07', 'Q17'): 2.2, ('Q17', 'Q07'): 2.2,
        ('Q06', 'Q18'): 2.0, ('Q18', 'Q06'): 2.0,
        ('Q17', 'Q18'): 1.8, ('Q18', 'Q17'): 1.8,
        ('Q14', 'Q16'): 2.0, ('Q16', 'Q14'): 2.0,
        ('Q14', 'Q18'): 1.6, ('Q18', 'Q14'): 1.6,
        ('Q09', 'Q10'): 2.2, ('Q10', 'Q09'): 2.2,
        ('Q06', 'Q10'): 1.8, ('Q10', 'Q06'): 1.8,
        ('Q06', 'Q11'): 1.8, ('Q11', 'Q06'): 1.8,
        ('Q11', 'Q12'): 1.8, ('Q12', 'Q11'): 1.8,
    }

    for name, j in work.items():
        if time.monotonic() >= deadline:
            break
        K = int(j['K'])
        y = j['y']
        rows = np.flatnonzero(j['train'])
        gate = items.get(name, {}).get('gate')
        is_gated = isinstance(gate, dict)
        K_open = K - 1 if is_gated else K
        given_closed = np.zeros(n, dtype=bool)
        open_strict = False
        closed_strict = False
        if is_gated and gate.get('parent') in idx:
            p_col = idx[gate['parent']]
            p_vals = list(items[gate['parent']]['values'])
            open_codes = [p_vals.index(v) for v in gate.get('observed_if', ()) if v in p_vals]
            given_closed = ~np.isin(Xg[:, p_col], open_codes)
            closed_rows = rows[given_closed[rows]]
            open_rows = rows[~given_closed[rows]]
            closed_strict = len(closed_rows) >= 100 and bool(np.all(y[closed_rows] == K_open))
            open_strict = len(open_rows) >= 100 and bool(np.all(y[open_rows] < K_open))

        peers = [m for m in names if m != name]
        for f in folds:
            P0 = P_base[f][name].copy()
            done[f]['base'][name] = P0.astype(np.float32)

            tr = rows[fold[rows] != f]
            tr_open = tr[(y[tr] >= 0) & (y[tr] < K_open)]
            if len(tr_open) < 40 or not peers:
                for ck in cand_keys[1:]:
                    done[f][ck][name] = P0.astype(np.float32)
                continue

            y_open = y[tr_open]
            Y_onehot = np.eye(K_open, dtype=np.float64)[y_open]

            if is_gated:
                p_skip = P0[:, K_open].copy()
                p_open_blend = np.clip(1.0 - p_skip, 1e-9, 1.0)
                P_cond_0 = P0[:, :K_open] / np.maximum(P0[:, :K_open].sum(1, keepdims=True), 1e-12)
            else:
                P_cond_0 = P0
                p_open_blend = None

            def _assemble(P_c):
                if is_gated:
                    P_out = np.empty((n, K), dtype=np.float32)
                    P_out[:, :K_open] = (P_c * p_open_blend[:, None]).astype(np.float32)
                    P_out[:, K_open] = (1.0 - p_open_blend).astype(np.float32)
                    if open_strict:
                        P_out[~given_closed, K_open] = 0.0
                    if closed_strict:
                        P_out[given_closed] = 0.0
                        P_out[given_closed, K_open] = 1.0
                    P_out /= np.maximum(P_out.sum(1, keepdims=True), 1e-30)
                    return P_out
                return P_c.astype(np.float32)

            # Part A: Cross-Item Rao-Blackwellized Cell Pseudo-Counts on ALL training rows tr
            R_tr = np.zeros((len(tr), K_open), dtype=np.float64)
            w_tr = np.zeros(len(tr), dtype=np.float64)
            c_tr_all = country[tr]
            peer_ranks = []

            for m in peers:
                ym = work[m]['y']
                Km = int(work[m]['K'])
                m_gated = isinstance(items.get(m, {}).get('gate'), dict)
                Km_open = Km - 1 if m_gated else Km
                both = tr_open[(ym[tr_open] >= 0) & (ym[tr_open] < Km_open)]
                if len(both) < 50:
                    continue
                ym_both = ym[both]
                yn_both = y[both]
                joint_g = np.bincount(ym_both * K_open + yn_both, minlength=Km_open * K_open).reshape(Km_open, K_open).astype(np.float64)
                px = joint_g.sum(1, keepdims=True) / len(both)
                py = joint_g.sum(0, keepdims=True) / len(both)
                p_xy = joint_g / len(both)
                nz = p_xy > 0
                mi = float(np.sum(p_xy[nz] * np.log(p_xy[nz] / np.maximum((px @ py)[nz], 1e-15)))) - (Km_open - 1) * (K_open - 1) / (2.0 * len(both))
                if mi <= 1e-4:
                    continue
                w_m = mi * domain_pairs.get((m, name), 1.0)
                peer_ranks.append((w_m, m))
                T_glob = (joint_g + 5.0 * py) / (joint_g.sum(1, keepdims=True) + 5.0)

                c_both = country[both]
                joint_c = np.bincount((c_both * Km_open + ym_both) * K_open + yn_both, minlength=G_country * Km_open * K_open).reshape(G_country, Km_open, K_open).astype(np.float64)
                n_cm = joint_c.sum(2, keepdims=True)
                T_country = (joint_c + 18.0 * T_glob[None, :, :]) / (n_cm + 18.0)

                ym_tr = ym[tr]
                open_m = (ym_tr >= 0) & (ym_tr < Km_open)
                if np.any(open_m):
                    R_tr[open_m] += w_m * T_country[c_tr_all[open_m], ym_tr[open_m]]
                    w_tr[open_m] += w_m

            valid_rb = w_tr > 0
            if np.any(valid_rb):
                R_tr[valid_rb] /= w_tr[valid_rb, None]
                rel_w = np.clip(w_tr / np.maximum(np.median(w_tr[valid_rb]), 1e-6), 0.0, 2.0)

                P_rb_s = _soft_dirichlet_residual_step(P_cond_0, cell4, n_cell4, tr, R_tr, rel_w, K_open, 55.0)
                P_rb_s = _soft_dirichlet_residual_step(P_rb_s, cell6, n_cell6, tr, R_tr, rel_w, K_open, 30.0)
                P_rb_s = _soft_dirichlet_residual_step(P_rb_s, cell7, n_cell7, tr, R_tr, rel_w, K_open, 25.0)
                done[f]['rb_cell_strong'][name] = _assemble(P_rb_s)

                P_rb_m = _soft_dirichlet_residual_step(P_cond_0, cell4, n_cell4, tr, R_tr, rel_w, K_open, 120.0)
                P_rb_m = _soft_dirichlet_residual_step(P_rb_m, cell6, n_cell6, tr, R_tr, rel_w, K_open, 65.0)
                P_rb_m = _soft_dirichlet_residual_step(P_rb_m, cell7, n_cell7, tr, R_tr, rel_w, K_open, 55.0)
                done[f]['rb_cell_med'][name] = _assemble(P_rb_m)

                P_rb_6 = _soft_dirichlet_residual_step(P_cond_0, cell6, n_cell6, tr, R_tr, rel_w, K_open, 35.0)
                done[f]['rb_6way'][name] = _assemble(P_rb_6)
            else:
                P_rb_m = P_cond_0
                done[f]['rb_cell_strong'][name] = done[f]['base'][name]
                done[f]['rb_cell_med'][name] = done[f]['base'][name]
                done[f]['rb_6way'][name] = done[f]['base'][name]

            # Part B: Top-3 MI Peer Consensus Log-Odds Residual
            top_peers = [m for _, m in sorted(peer_ranks, reverse=True)[:3]] or peers[:3]
            peer_blocks_tr = [S_fold[f][m][tr_open] for m in top_peers]
            peer_blocks_all = [S_fold[f][m] for m in top_peers]
            Phi_tr = np.hstack(peer_blocks_tr)
            Phi_all = np.hstack(peer_blocks_all)
            mu_phi = Phi_tr.mean(0, keepdims=True)
            Phi_tr = Phi_tr - mu_phi
            Phi_all = Phi_all - mu_phi

            Gram = Phi_tr.T @ Phi_tr
            diag_idx = np.arange(Gram.shape[0])

            for r_key, lam, base_mat in (('r300', 300.0, P_cond_0), ('r120', 120.0, P_cond_0), ('rb_plus_r180', 180.0, P_rb_m)):
                log_B0 = np.log(np.clip(base_mat, 1e-9, 1.0))
                G_init = Phi_tr.T @ (Y_onehot - base_mat[tr_open])
                A1 = 0.25 * Gram.copy()
                A1[diag_idx, diag_idx] += lam
                B = np.linalg.solve(A1, G_init)
                B -= B.mean(1, keepdims=True)

                L1_tr = log_B0[tr_open] + Phi_tr @ B
                L1_tr -= L1_tr.max(1, keepdims=True)
                P1_tr = np.exp(L1_tr)
                P1_tr /= P1_tr.sum(1, keepdims=True)
                G1 = Phi_tr.T @ (Y_onehot - P1_tr) - lam * B
                B = B + np.linalg.solve(A1, G1)
                B -= B.mean(1, keepdims=True)

                L_all = log_B0 + np.clip(Phi_all @ B, -3.5, 3.5)
                L_all -= L_all.max(1, keepdims=True)
                P_cond = np.exp(L_all)
                P_cond /= P_cond.sum(1, keepdims=True)
                done[f][r_key][name] = _assemble(P_cond)

    valid_names = [name for name in work if all(name in done[f]['base'] for f in folds)]
    if not valid_names:
        info['status'] = 'no_valid_items'
        return {}, info

    raw_found, info['weights'] = _mix_folds_kkt(
        work,
        fold,
        folds,
        list(cand_keys),
        lambda f, c, name: done[f][c][name],
        n,
        names=valid_names, deadline=deadline,
    )
    found = {}
    gains = {}
    for name, (oof_mix, full_mix) in raw_found.items():
        j = work[name]
        rows = np.flatnonzero(j['train'])
        rf = fold[rows]
        y_tr = np.maximum(j['y'][rows], 0)
        K = int(j['K'])
        base_oof = np.empty((len(rows), K), dtype=np.float64)
        for f in folds:
            hit = rf == f
            base_oof[hit] = done[f]['base'][name][rows[hit]]
        stray = rf < 0
        if stray.any():
            base_oof[stray] = oof_mix[stray]

        def _fl(P_sub, y_sub):
            p = P_sub / np.maximum(P_sub.sum(1, keepdims=True), 1e-30)
            return float(-np.log(1e-3 + (1.0 - K * 1e-3) * p[np.arange(len(y_sub)), y_sub]).mean())

        half = (rf % 2) == 0
        g_all = _fl(base_oof, y_tr) - _fl(oof_mix, y_tr)
        g_even = _fl(base_oof[half], y_tr[half]) - _fl(oof_mix[half], y_tr[half])
        g_odd = _fl(base_oof[~half], y_tr[~half]) - _fl(oof_mix[~half], y_tr[~half])
        gains[name] = round(g_all, 6)
        if g_all > 1e-5 and g_even >= -1e-5 and g_odd >= -1e-5:
            found[name] = (oof_mix, full_mix)
    info['oof_gains'] = gains
    info['items'] = len(found)
    info['seconds'] = round(time.monotonic() - start, 4)
    return found, info


def apply_smooth_audit(work, stacked, final, final_oofs, fold, items, X=None, given=None, run_temp=True):
    """the configured option: Non-zeroing substantive probability audit + deterministic GIVEN-gate conservation under official 1e-3 mixing floor."""
    start = time.monotonic()
    betas = (0.92, 0.96, 1.00, 1.04, 1.08, 1.14) if run_temp else (1.00,)
    folds = [int(f) for f in np.unique(fold) if f >= 0]
    trace = {'policy': 'smooth_conditional_temperature_and_strict_given_gate', 'updated': 0, 'gate_conserved': 0, 'items': {}, 'seconds': 0.0}

    def _floored_loss(P, y_sub, K):
        if not len(y_sub):
            return 0.0
        p = P / np.maximum(P.sum(1, keepdims=True), 1e-30)
        hit = p[np.arange(len(y_sub)), y_sub]
        return float(-np.log(1e-3 + (1.0 - K * 1e-3) * hit).mean())

    def _scale_temp(P, beta, is_gated):
        if beta == 1.0:
            return P.copy()
        P = np.asarray(P, dtype=np.float64)
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-30)
        if not is_gated:
            q = np.power(np.clip(P, 1e-12, 1.0), beta)
            return q / np.maximum(q.sum(1, keepdims=True), 1e-30)
        K = P.shape[1]
        K_open = K - 1
        p_skip = np.clip(P[:, K_open:K], 0.0, 1.0)
        p_open = 1.0 - p_skip
        cond = P[:, :K_open] / np.maximum(P[:, :K_open].sum(1, keepdims=True), 1e-12)
        q_cond = np.power(np.clip(cond, 1e-12, 1.0), beta)
        q_cond /= np.maximum(q_cond.sum(1, keepdims=True), 1e-30)
        return np.hstack((q_cond * p_open, p_skip))

    for name, j in work.items():
        if name not in final:
            continue
        K = int(j['K'])
        rows = np.flatnonzero(j['train'])
        rest = np.flatnonzero(~j['train'])
        y = j['y'][rows]
        if len(y) < 40:
            continue
        oof = np.asarray(final_oofs.get(name, stacked[name][0]), dtype=np.float64)
        oof = oof / np.maximum(oof.sum(1, keepdims=True), 1e-30)
        gate = items.get(name, {}).get('gate')
        is_gated = isinstance(gate, dict)
        f_rows = fold[rows]
        half = (f_rows % 2) == 0

        # Step 1: Deterministic GIVEN-gate structural conservation (restores exact 0/1 structural gate
        # after finite_pop_raking / residual_population without zeroing any substantive option 0..K-2)
        gate_gain = 0.0
        gate_applied = False
        if is_gated and X is not None and given is not None and gate.get('parent') in given:
            p_col = given.index(gate['parent'])
            p_vals = list(items[gate['parent']]['values'])
            open_codes = [p_vals.index(v) for v in gate.get('observed_if', ()) if v in p_vals]
            open_all = np.isin(X[:, p_col], open_codes)
            open_tr = open_all[rows]
            closed_tr = ~open_tr
            strict_closed = int(closed_tr.sum()) >= 100 and bool(np.all(y[closed_tr] == K - 1))
            strict_open = int(open_tr.sum()) >= 100 and bool(np.all(y[open_tr] < K - 1))
            if strict_closed or strict_open:
                def _conserve_gate(P_mat, open_mask_sub):
                    Q = np.asarray(P_mat, dtype=np.float64).copy()
                    if strict_closed and np.any(~open_mask_sub):
                        Q[~open_mask_sub, :K - 1] = 0.0
                        Q[~open_mask_sub, K - 1] = 1.0
                    if strict_open and np.any(open_mask_sub):
                        Q[open_mask_sub, K - 1] = 0.0
                        s = np.maximum(Q[open_mask_sub, :K - 1].sum(1, keepdims=True), 1e-30)
                        Q[open_mask_sub, :K - 1] /= s
                    return Q / np.maximum(Q.sum(1, keepdims=True), 1e-30)

                oof_g = _conserve_gate(oof, open_tr)
                g_tot = _floored_loss(oof, y, K) - _floored_loss(oof_g, y, K)
                g_ev = _floored_loss(oof[half], y[half], K) - _floored_loss(oof_g[half], y[half], K)
                g_od = _floored_loss(oof[~half], y[~half], K) - _floored_loss(oof_g[~half], y[~half], K)
                if g_tot > 1e-6 and g_ev >= 0.0 and g_od >= 0.0:
                    oof = oof_g
                    final_oofs[name] = oof
                    if len(rest):
                        final[name][rest] = _conserve_gate(final[name][rest], open_all[rest])
                    gate_gain = float(g_tot)
                    gate_applied = True
                    trace['gate_conserved'] += 1

        # Step 2: Smooth open-conditional temperature scaling (never zeroes any substantive option)
        cand_oofs = {b: _scale_temp(oof, b, is_gated) for b in betas}
        adj_oof = oof.copy()
        for f in folds:
            tr_m = f_rows != f
            va_m = f_rows == f
            if not np.any(tr_m) or not np.any(va_m):
                continue
            best_b = min(betas, key=lambda b: _floored_loss(cand_oofs[b][tr_m], y[tr_m], K))
            adj_oof[va_m] = cand_oofs[best_b][va_m]

        before = _floored_loss(oof, y, K)
        after = _floored_loss(adj_oof, y, K)
        gain_even = _floored_loss(oof[half], y[half], K) - _floored_loss(adj_oof[half], y[half], K)
        gain_odd = _floored_loss(oof[~half], y[~half], K) - _floored_loss(adj_oof[~half], y[~half], K)
        full_b = min(betas, key=lambda b: _floored_loss(cand_oofs[b], y, K))

        if full_b != 1.0 and (before - after) > 2e-5 and gain_even > 1e-5 and gain_odd > 1e-5:
            final_oofs[name] = adj_oof
            if len(rest):
                final[name][rest] = _scale_temp(final[name][rest], full_b, is_gated)
            trace['updated'] += 1
            trace['items'][name] = {'beta': float(full_b), 'oof_gain': float(before - after + gate_gain), 'gate_conserved': gate_applied, 'updated': True}
        else:
            trace['items'][name] = {'beta': 1.0, 'oof_gain': float(gate_gain), 'gate_conserved': gate_applied, 'updated': gate_applied}

    trace['seconds'] = round(time.monotonic() - start, 4)
    return trace

