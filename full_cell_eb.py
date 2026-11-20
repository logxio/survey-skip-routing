"""Exact demographic cells with nested fold-local Dirichlet backoff and joint parent-branch residual calibration."""
import time
import numpy as np
from base import _nll, _FLOOR

# Orders four and five shrink coarsely; the two sixth-order tables and the
# seven-way cell keep their own finer prior so dense exact cells speak.
COARSE = (4., 12.)
FINE = (.5, 1.5, 4., 10.)
HEAVY_COARSE = (2., 4., 8., 12., 20.)
HEAVY_FINE = (.25, .5, 1., 1.5, 2.5, 4., 7., 10.)
ET_GRID = ((0., 1.), (.14, .25), (.18, .25), (.18, .35), (.22, .35))
RIDGE_ALPHAS = (16., 48.)
RIDGE_BETAS = (0., .25, .5, 1.)
_GIVEN = ('Country', 'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q13')
PATHS = (('Country', 'Q05', 'Q13', 'Q04'),
         ('Country', 'Q04', 'Q05', 'Q13', 'Q02'),
         *(tuple(x for x in _GIVEN if x != omit) for omit in ('Q01', 'Q02', 'Q03', 'Q04', 'Q05')),
         _GIVEN)



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


def _parent_ask(work, chain, items, name, y, k, fit_rows, qu_rows):
    item = (items or {}).get(name, {})
    gate = item.get('gate')
    if not isinstance(gate, dict):
        return None
    p_name = gate.get('parent')
    if p_name not in getattr(chain, 'anchors', ()):
        return None
    p_probs = chain.probabilities(p_name, fit_rows)[qu_rows]
    p_vals = items[p_name]['values']
    obs_codes = [i for i, v in enumerate(p_vals) if str(v) in {str(u) for u in gate.get('observed_if', ())}]
    if p_name in work:
        yp = work[p_name]['y'][fit_rows]
        emp_codes = [c for c in range(len(p_vals)) if (yp == c).sum() >= 10 and np.mean(y[fit_rows][yp == c] < k - 1) > 0.5]
        if emp_codes:
            obs_codes = emp_codes
    if not obs_codes:
        obs_codes = [p_vals.index('Yes')] if (name == 'Q09' and 'Yes' in p_vals) else ([p_vals.index('No')] if 'No' in p_vals else [0])
    return np.clip(p_probs[:, obs_codes].sum(1), 1e-6, 1. - 1e-6)


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


def _et_losses(p, ask, has_gate, pr, grid, y, k):
    """Score the existing grid with shared transforms and only observed entries."""
    rows = np.arange(len(y))
    if has_gate:
        cond = p[:, :-1] / np.maximum(p[:, :-1].sum(1, keepdims=True), 1e-12)
        if ask is None:
            ask = np.clip(p[:, :-1].sum(1), 1e-6, 1. - 1e-6)
        cols = np.minimum(y, cond.shape[1] - 1)
    else:
        cond, cols = p, y
    transforms = {}
    original = cond[rows, cols]
    for eps, T in grid:
        if eps == 0. and T == 1.:
            picked = original
        else:
            if T not in transforms:
                lg = np.log(np.maximum(cond, 1e-7)) / T
                lg -= lg.max(1, keepdims=True)
                sh = np.exp(lg)
                sh /= sh.sum(1, keepdims=True)
                transforms[T] = sh[rows, cols]
            picked = (1. - eps) * transforms[T] + eps * pr[cols]
        if has_gate:
            picked = np.where(y == k - 1, 1. - ask, picked * ask)
        yield -float(np.log(_FLOOR + (1.0 - k * _FLOOR) * picked).mean())


def _webinar_personas(X, given, items):
    """Three FPCC webinar latent demographic-religious persona indicators plus parent branch."""
    n = len(X)
    if not {'Country', 'Q02', 'Q03', 'Q05', 'Q13'}.issubset(given):
        return np.zeros((n, 0), np.float32)

    def _in(col_name, names):
        col = X[:, given.index(col_name)]
        vals = items.get(col_name, {}).get('values', ())
        codes = [i for i, v in enumerate(vals) if v in names]
        return np.isin(col, codes).astype(np.float32)

    male = _in('Q02', ('Male',))
    female = _in('Q02', ('Female',))
    rural = _in('Q03', ('Rural',))
    abrahamic = _in('Q05', ('Christianity', 'Islam', 'Judaism'))
    indic = _in('Q05', ('Hinduism', 'Buddhism'))
    c_p1 = _in('Country', ('Kenya', 'Mali', 'Nigeria', 'Zambia'))
    c_p2 = _in('Country', ('Nepal', 'Egypt'))
    c_p3 = _in('Country', ('Iraq', 'Guatemala'))
    parent_yes = _in('Q13', ('Yes',))

    # Persona 1: Vaccination supporters aligned with leaders (male, Abrahamic, Kenya/Mali/Nigeria/Zambia)
    p1 = male * abrahamic * c_p1
    # Persona 2: Vaccination supporters not aligned with leaders (Indic religions, Nepal/Egypt)
    p2 = indic * c_p2
    # Persona 3: Hesitant respondents aligned with opposing leaders (female, rural, Iraq/Guatemala)
    p3 = female * rural * c_p3
    return np.column_stack((p1, p2, p3, parent_yes * p1, parent_yes * p2, parent_yes * p3)).astype(np.float32)


def _vectorized_ridge_loo(z_fit, z_query, residual, alphas=RIDGE_ALPHAS):
    """Single thin eigendecomposition solves all Ridge alphas and exact LOO residuals on fit."""
    zf = np.asarray(z_fit, dtype=np.float64)
    zq = np.asarray(z_query, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)
    n_fit = len(zf)
    z_mean = zf.mean(0)
    zf_c = zf - z_mean
    zq_c = zq - z_mean
    res_mean = res.mean(0)
    res_c = res - res_mean

    G = zf_c.T @ zf_c
    evals, V = np.linalg.eigh(G)
    evals = np.maximum(evals, 0.0)
    ZV_fit = zf_c @ V
    ZV_qu = zq_c @ V
    ZV2 = ZV_fit * ZV_fit
    VtZtY = ZV_fit.T @ res_c

    out = {}
    for alpha in alphas:
        inv = 1.0 / (evals + float(alpha))
        w_proj = inv[:, None] * VtZtY
        y_hat_c = ZV_fit @ w_proj
        h_diag = ZV2 @ inv + (1.0 / max(1, n_fit))
        denom = np.maximum(1.0 - h_diag[:, None], 1e-3)
        corr_loo = (res_mean + y_hat_c - h_diag[:, None] * res) / denom
        corr_qu = res_mean + ZV_qu @ w_proj
        out[float(alpha)] = (corr_loo, corr_qu)
    return out


class Cells:
    def __init__(self, X, given, heavy=False, items=None):
        paths = PATHS if set(PATHS[-1]).issubset(given) else (
            [tuple(given[:i]) for i in range(4, min(7, len(given)) + 1)])
        self.ids = []
        self.orders = []
        for names in paths:
            _, ids = np.unique(X[:, [given.index(n) for n in names]], axis=0, return_inverse=True)
            self.ids.append(ids.ravel())
            self.orders.append(len(names))
        self.sizes = [int(ids.max()) + 1 for ids in self.ids]
        self.n = len(X)
        self.X, self.given, self.items = X, given, items or {}
        from pathlib import Path
        import json
        self.external = json.loads(Path(__file__).with_name('country_marginals.json').read_text())
        grouped = {}
        for row in self.external:
            grouped.setdefault((row['country'], row['item']), {})[row['option']] = float(row['reported_percent']) / 100.
        self.external = grouped
        self.config = {}
        self.count_totals = {}
        self.country = given.index('Country') if 'Country' in given else None
        coarse, fine = (HEAVY_COARSE, HEAVY_FINE) if heavy else (COARSE, FINE)
        self.grid = [(a, b) for a in coarse for b in fine]

    def raw_counts(self, y, k, fit, query):
        """Eight query-indexed sufficient statistics, independent of prior policy."""
        yf = y[fit]
        return [np.bincount(ids[fit] * k + yf, minlength=size * k).reshape(size, k)[ids[query]]
                for ids, size in zip(self.ids, self.sizes)]

    def sufficient_counts(self, y, k, fit):
        """Integer totals; subtraction excludes a whole inner fold exactly."""
        yf = y[fit]
        tables = [np.bincount(ids[fit] * k + yf, minlength=size * k).reshape(size, k)
                  for ids, size in zip(self.ids, self.sizes)]
        marginal = np.bincount(yf, minlength=k)
        country = None
        if self.country is not None:
            ids = self.X[:, self.country]
            nc = max(int(ids.max()) + 1, len(self.items.get('Country', {}).get('values', ())))
            country = np.bincount(ids[fit] * k + yf, minlength=nc * k).reshape(nc, k)
        return marginal, country, tables

    def counts_from_totals(self, y, k, fit, query, name, total, removed=None, answered_only=False):
        marginal, country, tables = total
        if removed is not None:
            marginal = marginal - removed[0]
            country = country - removed[1] if country is not None else None
        values = [table[ids[query]] - (removed[2][i][ids[query]] if removed is not None else 0)
                  for i, (table, ids) in enumerate(zip(tables, self.ids))]
        prior = self.country_prior(y, k, fit, query, name, answered_only,
                                   statistics=(marginal, country))
        return prior, values

    def crossfit_counts(self, y, k, fit, query, name, alphas, answered_only=False):
        total = self.sufficient_counts(y, k, fit)
        inner = np.arange(len(fit)) % 5
        base = np.zeros((len(fit), k))
        for f in range(5):
            mask = inner == f
            va = fit[mask]
            removed = self.sufficient_counts(y, k, va)
            counts = self.counts_from_totals(y, k, fit[~mask], va, name, total, removed, answered_only)
            base[mask] = self.predict(counts, alphas)
        counts = self.counts_from_totals(y, k, fit, query, name, total, answered_only=answered_only)
        return base, self.predict(counts, alphas)

    def counts(self, y, k, fit, query, name=None, answered_only=False, policy=None, strength=None):
        return (self.country_prior(y, k, fit, query, name, answered_only, policy, strength),
                self.raw_counts(y, k, fit, query))

    def country_prior(self, y, k, fit, query, name=None, answered_only=False, policy=None, strength=None,
                      statistics=None):
        default_policy, default_strength = self.config.get(name, ("full", 25.))
        policy = default_policy if policy is None else policy
        strength = default_strength if strength is None else strength
        marginal = (np.bincount(y[fit], minlength=k) if statistics is None else statistics[0]).astype(float)
        prior = (marginal + 1. / k) / (len(fit) + 1.)
        if self.country is not None and name in self.items:
            ids = self.X[:, self.country]
            nc = max(int(ids.max()) + 1, len(self.items['Country']['values']))
            cnt = (np.bincount(ids[fit] * k + y[fit], minlength=nc * k).reshape(nc, k)
                   if statistics is None else statistics[1]).astype(float)
            cp = np.tile(prior, (nc, 1))
            item = self.items[name]
            values = item['values'][:k]
            gated_prior = bool(item.get('gate')) and not answered_only
            countries = self.items['Country']['values']
            for (country, target), rates in self.external.items():
                if target != name or country not in countries:
                    continue
                selected = rates
                if policy == 'fixed' or (policy == 'children' and name not in ('Q14', 'Q15', 'Q16', 'Q17', 'Q18')):
                    keep = 'Yes' if name == 'Q08' else 'Very Important' if name == 'Q14' else None
                    selected = {opt: r for opt, r in rates.items() if opt == keep}
                known = [(values.index(opt), rate) for opt, rate in selected.items() if opt in values]
                if not known:
                    continue
                c = countries.index(country)
                asked = 1. - cp[c, -1] if gated_prior else 1.
                n_answered = k - int(gated_prior)
                idx = np.array([opt for opt, _ in known], dtype=int)
                rr = np.array([rate for _, rate in known], dtype=float)
                total = float(rr.sum())
                free = np.ones(n_answered, dtype=bool)
                free[idx] = False
                original = cp[c, :n_answered].copy()
                if not free.any() or total >= .97:
                    cp[c, :n_answered] = 0.
                    cp[c, idx] = asked * rr / max(total, 1e-12)
                else:
                    cp[c, idx] = asked * rr
                    mass = original[free].sum()
                    relative = original[free] / mass if mass > 0. else np.full(free.sum(), 1. / free.sum())
                    cp[c, np.flatnonzero(free)] = asked * (1. - total) * relative
                if gated_prior:
                    cp[c, -1] = 1. - asked
            prior = (cnt + strength * cp) / (cnt.sum(1, keepdims=True) + strength)
            prior = prior[ids[query]]
        return prior

    def counts_answered(self, y, k_ans, fit_ans, query, name=None):
        fit_ans = np.asarray(fit_ans)
        fit_ans = fit_ans[(y[fit_ans] >= 0) & (y[fit_ans] < k_ans)]
        return self.counts(y, k_ans, fit_ans, query, name=name, answered_only=True)

    def predict(self, counts, alphas):
        coarse, fine = alphas
        prior, values = counts
        p = prior
        six = []
        fifth = None
        for i, c in enumerate(values):
            alpha = coarse if self.orders[i] <= 5 else fine
            if self.orders[i] == 6:
                parent = fifth
            elif self.orders[i] == 7 and six:
                parent = sum(six) / len(six)
            else:
                parent = p
            p = (c + alpha * parent) / (c.sum(1, keepdims=True) + alpha)
            if self.orders[i] == 5:
                fifth = p
            if self.orders[i] == 6:
                six.append(p)
        return p

    def predict_grid(self, counts):
        """Evaluate alpha candidates together with the same per-element arithmetic."""
        prior, values = counts
        grid = np.asarray(self.grid)
        p = prior
        fifth, six = None, []
        for i, c in enumerate(values):
            alpha = grid[:, 0 if self.orders[i] <= 5 else 1, None, None]
            if self.orders[i] == 6:
                parent = fifth
            elif self.orders[i] == 7 and six:
                parent = sum(six) / len(six)
            else:
                parent = p
            p = (c + alpha * parent) / (c.sum(1, keepdims=True) + alpha)
            if self.orders[i] == 5:
                fifth = p
            if self.orders[i] == 6:
                six.append(p)
        return dict(zip(self.grid, p))

    def fit(self, job, fold, name=None, source_job=None):
        y, k = job['y'], job['K']
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        if not self.ids or len(rows) < 20:
            return None
        oof = np.zeros((len(rows), k))
        full = np.zeros((self.n, k))
        selected = []
        origin = job if source_job is None else source_job
        origin_rows = np.flatnonzero(origin['train'])
        if name not in self.count_totals:
            self.count_totals[name] = self.sufficient_counts(origin['y'], k, origin_rows)
        total = self.count_totals[name]
        def counts_for(fit, query):
            removed_rows = np.setdiff1d(origin_rows, fit, assume_unique=True)
            removed = self.sufficient_counts(origin['y'], k, removed_rows)
            return self.counts_from_totals(y, k, fit, query, name, total, removed)
        for f in [*np.unique(fold[rows]), None]:
            fit = rows if f is None else rows[fold[rows] != f]
            query = rest if f is None else rows[fold[rows] == f]
            # Disjoint inner split: no held-out-fold labels enter alpha selection.
            inner = (np.arange(len(fit)) % 4) == 0
            a, b = fit[~inner], fit[inner]
            counts = counts_for(a, b)
            predictions = self.predict_grid(counts)
            alphas = min(self.grid, key=lambda v: _nll(predictions[v], y[b], k))
            pred = self.predict(counts_for(fit, query), alphas)
            if f is None:
                full[rest] = pred
            else:
                oof[fold[rows] == f] = pred
            selected.append(alphas)
        return oof, full, selected


def attach(X, given, work, fold, deadline=float('inf'), heavy=False, items=None):
    from cross_round import dominates
    start = time.monotonic()
    info = {'offered': 0, 'seconds': 0., 'alphas': {}}
    if not 4 <= len(given) <= 7:
        return info
    cells = Cells(X, given, heavy=heavy, items=items)
    for name, job in work.items():
        if time.monotonic() >= deadline:
            break
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        if not cells.ids or len(rows) < 20:
            continue
        y, k = job['y'], job['K']
        has_gate = bool((items or {}).get(name, {}).get('gate'))
        et_grid = ((0., 1.),) if not has_gate and k <= 2 else ET_GRID
        o = np.zeros((len(rows), k))
        p = np.zeros((cells.n, k))
        o_cal = np.zeros((len(rows), k))
        p_cal = np.zeros((cells.n, k))
        o_ref = np.zeros_like(o)
        p_ref = np.zeros_like(p)
        o_ref_cal, p_ref_cal = np.zeros_like(o), np.zeros_like(p)
        a_ref_cal = []
        policies = []
        a_ref = []
        a_raw = []
        a_cal = []
        for f in [*np.unique(fold[rows]), None]:
            fit = rows if f is None else rows[fold[rows] != f]
            query = rest if f is None else rows[fold[rows] == f]
            inner = (np.arange(len(fit)) % 4) == 0
            tr_in, va_in = fit[~inner], fit[inner]
            raw_in = cells.raw_counts(y, k, tr_in, va_in)
            raw_query = cells.raw_counts(y, k, fit, query)
            configs = [('fixed', 25.), ('children', 25.), ('full', 25.), ('full', 128.)]
            cached = {cfg: (cells.country_prior(y, k, tr_in, va_in, name, policy=cfg[0], strength=cfg[1]), raw_in)
                      for cfg in configs}
            predictions = {}
            for cfg in configs:
                shared = next((prev for prev in predictions
                               if np.array_equal(cached[cfg][0], cached[prev][0])), None)
                predictions[cfg] = (predictions[shared] if shared is not None
                                    else cells.predict_grid(cached[cfg]))
            losses = {cfg: {al: _nll(pred, y[va_in], k) for al, pred in predictions[cfg].items()}
                      for cfg in configs}
            ref_in = cached['fixed', 25.]
            ref_preds = predictions['fixed', 25.]
            ref_al = min(cells.grid, key=losses['fixed', 25.].get)
            a_ref.append(ref_al)
            ref_query = (cells.country_prior(y, k, fit, query, name, policy='fixed', strength=25.), raw_query)
            ref_pred = cells.predict(ref_query, ref_al)
            policy, strength = min(configs, key=lambda cfg: min(losses[cfg].values()))
            policies.append((policy, strength))
            cnt_qu = ref_query if (policy, strength) == ('fixed', 25.) else (
                cells.country_prior(y, k, fit, query, name, policy=policy, strength=strength), raw_query)
            preds_in = predictions[policy, strength]
            al0 = min(cells.grid, key=losses[policy, strength].get)
            pred0 = ref_pred if (policy, strength) == ('fixed', 25.) else cells.predict(cnt_qu, al0)
            a_raw.append(al0)

            ask_va = _given_ask(X, given, items, name, y, k, tr_in, va_in)
            ask_qu = _given_ask(X, given, items, name, y, k, fit, query)
            pr_in = _ans_prior(y[tr_in], k, has_gate)
            pr_f = _ans_prior(y[fit], k, has_gate)
            ref_cal_losses = {(al, et): loss for al in cells.grid
                for et, loss in zip(et_grid, _et_losses(ref_preds[al], ask_va, has_gate, pr_in, et_grid, y[va_in], k))}
            ref_al_cal, ref_et = min(ref_cal_losses, key=ref_cal_losses.get)
            a_ref_cal.append(ref_al_cal)
            ref_cal = _apply_gate_and_et(cells.predict(ref_query, ref_al_cal), ask_qu, has_gate, pr_f, *ref_et)
            if (policy, strength) == ('fixed', 25.):
                best_pair = (ref_al_cal, ref_et)
            else:
                cal_losses = {(al, et): loss for al in cells.grid
                    for et, loss in zip(et_grid, _et_losses(preds_in[al], ask_va, has_gate, pr_in, et_grid, y[va_in], k))}
                best_pair = min(cal_losses, key=cal_losses.get)
            al_best, et_best = best_pair
            a_cal.append(al_best)
            p_qu = pred0 if al_best == al0 else cells.predict(cnt_qu, al_best)
            pred_c = _apply_gate_and_et(p_qu, ask_qu, has_gate, pr_f, *et_best)
            if f is None:
                p_ref[rest] = ref_pred
                p_ref_cal[rest] = ref_cal
                p[rest] = pred0
                p_cal[rest] = pred_c
            else:
                mask_f = fold[rows] == f
                o_ref[mask_f] = ref_pred
                o_ref_cal[mask_f] = ref_cal
                o[mask_f] = pred0
                o_cal[mask_f] = pred_c
        half = (fold[rows] % 2) == 0
        if dominates(o_cal, o, y[rows], k, half):
            o, p, a_sel = o_cal, p_cal, a_cal
        else:
            a_sel = a_raw
        if dominates(o_ref_cal, o_ref, y[rows], k, half):
            o_ref, p_ref, a_ref = o_ref_cal, p_ref_cal, a_ref_cal
        grid = (0., .25, .5, .75, 1.)
        ys = y[rows]
        split = fold[rows]
        losses = np.stack([-np.log(np.clip((1.-k*.001) *
            ((1.-lam)*o_ref + lam*o)[np.arange(len(rows)), ys] + .001,
            1e-12, 1.)) for lam in grid])
        mixed = np.empty_like(o)
        fold_weights = []
        for f in np.unique(split):
            va = split == f
            lam = grid[int(np.argmin(losses[:, ~va].mean(1)))]
            mixed[va] = (1.-lam)*o_ref[va] + lam*o[va]
            fold_weights.append(lam)
        lam = grid[int(np.argmin(losses.mean(1)))]
        o, p = mixed, (1.-lam)*p_ref + lam*p
        info.setdefault('external', {})[name] = {'fold_policies': policies,
            'blend_weight': lam, 'fold_weights': fold_weights,
            'corrected_baseline_nll': float(_nll(o_ref, ys, k)),
            'selected_nll': float(_nll(o, ys, k))}
        job['_eb_external'] = policies
        job['_eb_alphas'] = [tuple(x) for x in a_sel]
        job['arms'].append('full_cell_eb')
        job['oof'].append(o)
        job['all'].append(p)
        info['offered'] += 1
        info['alphas'][name] = list(a_sel[-1])
    info['seconds'] = time.monotonic() - start
    return info


def attach_conditioned(X, given, work, fold, bank, chain, deadline, items, alphas=None, workers=1):
    """Cross-fitted demographic EB with Country/Q05/webinar-persona joint residual layer.

    Items are independent; they run on a thread pool and attach in work order.
    """
    from concurrent.futures import ThreadPoolExecutor
    from chain_features import check_clock, BudgetExpired, priority_sources, response_block
    from cross_round import dominates
    start = time.monotonic()
    cells = Cells(X, given, items=items)
    alpha_map = alphas or {}
    info = {'offered': 0, 'targets': {}, 'seconds': 0.}

    c_col = given.index('Country') if 'Country' in given else None
    C_oh = np.eye(int(X[:, c_col].max()) + 1, dtype=np.float32)[X[:, c_col]] if c_col is not None else np.zeros((len(X), 0), np.float32)
    q5_col = given.index('Q05') if 'Q05' in given else None
    Q5_oh = np.eye(int(X[:, q5_col].max()) + 1, dtype=np.float32)[X[:, q5_col]] if q5_col is not None else np.zeros((len(X), 0), np.float32)
    personas = _webinar_personas(X, given, items or {})

    def one(name):
        job = work[name]
        rows = np.flatnonzero(job['train'])
        rest = np.flatnonzero(~job['train'])
        y, k = job['y'], job['K']
        oof_raw = np.zeros((len(rows), k))
        full_raw = np.zeros((len(X), k))
        oof_cal = np.zeros((len(rows), k))
        full_cal = np.zeros((len(X), k))
        oof_ans_cal = np.zeros((len(rows), k))
        full_ans_cal = np.zeros((len(X), k))
        fold_alphas = job.get('_eb_alphas')
        has_gate = bool((items or {}).get(name, {}).get('gate'))
        is_rel_child = name in ('Q09', 'Q10')
        et_grid = ((0., 1.),) if not has_gate and k <= 2 else ET_GRID
        chosen_ridge = []
        try:
            anchors = [a for a in priority_sources(name, items, chain.anchors) if a != name]
            if not anchors:
                anchors = [a for a in chain.anchors if a != name]
            for idx_f, f in enumerate([*np.unique(fold[rows]), None]):
                check_clock(deadline)
                external = job.get("_eb_external")
                if external:
                    cells.config[name] = external[idx_f]
                fit = rows if f is None else rows[fold[rows] != f]
                query = rest if f is None else rows[fold[rows] == f]
                alpha_pair = tuple(fold_alphas[idx_f]) if (fold_alphas and idx_f < len(fold_alphas)) else tuple(alpha_map.get(name, (12., 4.)))

                z_anc = np.column_stack([response_block(chain.probabilities(a, fit)) for a in anchors])
                anc_probs = np.column_stack([chain.probabilities(a, fit) for a in anchors]).astype(np.float32)
                anc_c = anc_probs - anc_probs[fit].mean(0)
                if C_oh.shape[1]:
                    c_c = C_oh - C_oh[fit].mean(0)
                    inter = (c_c[:, :, None] * anc_c[:, None, :]).reshape(len(X), -1)
                else:
                    inter = np.zeros((len(X), 0), np.float32)

                z_parts = [z_anc, C_oh, Q5_oh, personas, inter]
                z = np.column_stack([p for p in z_parts if p.shape[1] > 0])
                mu = z[fit].mean(0)
                sd = z[fit].std(0)
                sd[sd < .05] = 1.
                z = (z - mu) / sd

                base_fit, base_query = cells.crossfit_counts(y, k, fit, query, name, alpha_pair)

                residual = np.eye(k)[y[fit]] - base_fit
                ridge_sols = _vectorized_ridge_loo(z[fit], z[query], residual, alphas=RIDGE_ALPHAS)

                ask_fit = _parent_ask(work, chain, items, name, y, k, fit, fit) if (is_rel_child and 'Q08' in anchors) else _given_ask(X, given, items, name, y, k, fit, fit)
                ask_qu = _parent_ask(work, chain, items, name, y, k, fit, query) if (is_rel_child and 'Q08' in anchors) else _given_ask(X, given, items, name, y, k, fit, query)
                pr_f = _ans_prior(y[fit], k, has_gate)

                # Legacy baseline (alpha=32 equivalent / uncalibrated) for oof_raw
                corr_loo_48, corr_qu_48 = ridge_sols[48.]
                logits0 = np.log(np.maximum(base_query, 1e-5)) + np.clip(corr_qu_48 / np.maximum(base_query, .08), -2., 2.)
                logits0 -= logits0.max(1, keepdims=True)
                pred_raw = np.exp(logits0)
                pred_raw /= pred_raw.sum(1, keepdims=True)
                if ask_qu is not None:
                    pred_raw = pred_raw.copy()
                    pred_raw[:, :-1] *= ask_qu[:, None] / np.maximum(pred_raw[:, :-1].sum(1, keepdims=True), 1e-12)
                    pred_raw[:, -1] = 1. - ask_qu

                # Nested selection over r_alpha in (16., 48.), beta in (0., .25, .5, 1.), and ET_GRID using exact LOO residuals
                best_cfg = (48., 1., (0., 1.))
                best_score = float('inf')
                for r_alpha in RIDGE_ALPHAS:
                    corr_loo, _ = ridge_sols[r_alpha]
                    step_f = np.clip(corr_loo / np.maximum(base_fit, .08), -2., 2.)
                    log_bf = np.log(np.maximum(base_fit, 1e-5))
                    for beta in RIDGE_BETAS:
                        if beta == 0. and r_alpha != RIDGE_ALPHAS[0]:
                            continue
                        logits_f = log_bf + beta * step_f
                        logits_f -= logits_f.max(1, keepdims=True)
                        pf = np.exp(logits_f)
                        pf /= pf.sum(1, keepdims=True)
                        for et, sc in zip(et_grid, _et_losses(pf, ask_fit, has_gate, pr_f, et_grid, y[fit], k)):
                            if sc < best_score:
                                best_score = sc
                                best_cfg = (r_alpha, beta, et)

                r_alpha, beta, best_et = best_cfg
                chosen_ridge.append((r_alpha, beta))
                _, corr_qu = ridge_sols[r_alpha]
                logits_q = np.log(np.maximum(base_query, 1e-5)) + beta * np.clip(corr_qu / np.maximum(base_query, .08), -2., 2.)
                logits_q -= logits_q.max(1, keepdims=True)
                pred_best_raw = np.exp(logits_q)
                pred_best_raw /= pred_best_raw.sum(1, keepdims=True)
                if ask_qu is not None:
                    pred_best_raw[:, :-1] *= ask_qu[:, None] / np.maximum(pred_best_raw[:, :-1].sum(1, keepdims=True), 1e-12)
                    pred_best_raw[:, -1] = 1. - ask_qu
                pred_cal = _apply_gate_and_et(pred_best_raw, ask_qu, has_gate, pr_f, *best_et)

                # For Q09/Q10 (gated on Q08='No'), also compute answered-only conditional cell EB + Ridge residual
                if has_gate and ask_qu is not None and ask_fit is not None:
                    fit_ans = fit[y[fit] < k - 1]
                    if len(fit_ans) >= 20:
                        in_a = (np.arange(len(fit_ans)) % 4) == 0
                        tr_a, va_a = fit_ans[~in_a], fit_ans[in_a]
                        cnt_a = cells.counts_answered(y, k - 1, tr_a, va_a, name=name)
                        pr_in_a = _ans_prior(y[tr_a], k, True)
                        best_a_pair = (alpha_pair, best_et)
                        best_a_loss = float('inf')
                        for al_cand in cells.grid:
                            p_va_a = cells.predict(cnt_a, al_cand)
                            for et_cand, l_a in zip(ET_GRID, _et_losses(p_va_a, None, False, pr_in_a, ET_GRID, y[va_a], k - 1)):
                                if l_a < best_a_loss:
                                    best_a_loss = l_a
                                    best_a_pair = (al_cand, et_cand)
                        al_a, et_a = best_a_pair
                        base_ans, p_qu_a = cells.crossfit_counts(
                            y, k - 1, fit_ans, query, name, al_a, answered_only=True)
                        sols = _vectorized_ridge_loo(z[fit_ans], z[query],
                            np.eye(k - 1)[y[fit_ans]] - base_ans)
                        best = (float('inf'), 16., 0., et_a)
                        prior_a = _ans_prior(y[fit_ans], k, True)
                        for ra, (loo, _) in sols.items():
                            for beta_a in RIDGE_BETAS:
                                if beta_a == 0. and ra != RIDGE_ALPHAS[0]:
                                    continue
                                logits = np.log(np.maximum(base_ans, 1e-5)) + beta_a * np.clip(loo / np.maximum(base_ans, .08), -2., 2.)
                                logits -= logits.max(1, keepdims=True)
                                pa = np.exp(logits); pa /= pa.sum(1, keepdims=True)
                                for et, loss in zip(et_grid, _et_losses(pa, None, False, prior_a, et_grid, y[fit_ans], k - 1)):
                                    if loss < best[0]: best = (loss, ra, beta_a, et)
                        _, ra, beta_a, et_a = best
                        corr = sols[ra][1]
                        logits = np.log(np.maximum(p_qu_a, 1e-5)) + beta_a * np.clip(corr / np.maximum(p_qu_a, .08), -2., 2.)
                        logits -= logits.max(1, keepdims=True)
                        p_qu_a = np.exp(logits); p_qu_a /= p_qu_a.sum(1, keepdims=True)
                        cond_qu_a = _apply_gate_and_et(p_qu_a, None, False, prior_a, *et_a)
                        pred_ans = np.zeros((len(query), k))
                        pred_ans[:, :-1] = cond_qu_a * ask_qu[:, None]
                        pred_ans[:, -1] = 1. - ask_qu
                    else:
                        pred_ans = pred_cal
                else:
                    pred_ans = pred_cal

                if f is None:
                    full_raw[rest] = pred_raw
                    full_cal[rest] = pred_cal
                    full_ans_cal[rest] = pred_ans
                else:
                    mask_f = fold[rows] == f
                    oof_raw[mask_f] = pred_raw
                    oof_cal[mask_f] = pred_cal
                    oof_ans_cal[mask_f] = pred_ans

            half = (fold[rows] % 2) == 0
            oof, full = (oof_cal, full_cal) if dominates(oof_cal, oof_raw, y[rows], k, half) else (oof_raw, full_raw)
            full_state_nll = float(_nll(oof, y[rows], k))
            answered_nll = float(_nll(oof_ans_cal, y[rows], k)) if has_gate else None
            selected_arm = 'full_state'
            if has_gate and dominates(oof_ans_cal, oof, y[rows], k, half):
                oof, full = oof_ans_cal, full_ans_cal
                selected_arm = 'answered_conditioned'
            return oof, full, {
                'answered_conditioned': bool(has_gate),
                'selected_arm': selected_arm,
                'full_state_oof_nll': full_state_nll,
                'answered_oof_nll': answered_nll,
                'anchors': anchors,
                'folds': len(np.unique(fold[rows])),
                'alphas': list(alpha_map.get(name, (12., 4.))),
                'ridge_alpha': chosen_ridge[-1][0] if chosen_ridge else 48.,
                'ridge_beta': chosen_ridge[-1][1] if chosen_ridge else 1.,
                'oof_nll': float(_nll(oof, y[rows], k)),
            }
        except BudgetExpired:
            return None

    names = [name for name in work if not (name in chain.anchors and name != 'Q09')]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(names) or 1))) as executor:
        results = list(executor.map(one, names))
    for name, found in zip(names, results):
        if found is None:
            continue
        oof, full, target = found
        job = work[name]
        job['arms'].append('conditional_cell_eb')
        job['oof'].append(oof)
        job['all'].append(full)
        info['offered'] += 1
        info['targets'][name] = target
    info['seconds'] = time.monotonic() - start
    return info
