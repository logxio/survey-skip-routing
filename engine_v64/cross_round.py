"""Fold-isolated cross-round response kernel using only supplied covariates.

This is a probabilistic neighbourhood, not a household identity assignment.
All neighbourhoods and answer transitions live inside this predict invocation.
No identifiers, associations, or respondent-level diagnostics are exported.
"""
import re
import time
import numpy as np
from .base import _nll
from .chain_features import BudgetExpired, check_clock
BLOCK = ('s1_2_country_asylum_lbl', 'governorate_col', 's2_4a_origin_governorate_lbl')
COUNTS = (('s2_3_boys_0_5', 's2_3_boys_6_17', 's2_3_men_18_59', 's2_3_men_60plus'), ('s2_3_girls_0_5', 's2_3_girls_6_17', 's2_3_women_18_59', 's2_3_women_60plus'))
SOFT = ('s2_1_hh_relationship_lbl', 's2_1c_education_level_lbl', 's2_1d_marital_status_lbl', 's2_2_single_parent_lbl')
ROOT_CONTEXT = ('s3_11_intent_return_12m_lbl', 's4_0_hh_anyone_working_lbl', 's6_5_assistance_received_lbl')
CONTEXT_ENABLED = True
IDF_ENABLED = False
MI_ENABLED = True
BIDIR_MI_ENABLED = True
FACTORED_MI_ENABLED = True
TRIPLE_ENABLED = True

def _bounds(values):
    lo, hi = ([], [])
    for value in values:
        match = re.fullmatch('(\\d+)(?:-(\\d+)|(\\+))?', str(value).strip())
        if not match:
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            a = float(match[1])
            lo.append(a)
            hi.append(float(match[2]) if match[2] else np.inf if match[3] else a)
    return (np.array(lo + [np.nan]), np.array(hi + [np.nan]))

def _gap(al, ah, bl, bh):
    gap = np.maximum(0.0, np.maximum(al - bh, bl - ah))
    return np.where(np.isfinite(gap), gap, 1.0)

class Kernel:

    def __init__(self, schema, X, given, deadline=float('inf')):
        self.X, self.given = (X, given)
        self.wave = X[:, given.index('wave')]
        self.n = len(X)
        self.blocks = {}
        columns = [given.index(name) for name in BLOCK[:2]]
        valid = np.ones(self.n, bool)
        for name, col in zip(BLOCK[:2], columns):
            values = schema['items'][name]['values']
            bad = [i for i, v in enumerate(values) if 'want to answer' in str(v)]
            valid &= (X[:, col] < len(values)) & ~np.isin(X[:, col], bad)
        for i in np.flatnonzero(valid & (self.wave < 2)):
            key = tuple(X[i, columns])
            self.blocks.setdefault(key, []).append(i)
        self.intervals = {}
        self.ranks = {}
        for name in [n for sex in COUNTS for n in sex] + ['s2_1b_age']:
            lo, hi = _bounds(schema['items'][name]['values'])
            code = X[:, given.index(name)]
            self.intervals[name] = (lo[code], hi[code])
            valid_codes = np.flatnonzero(np.isfinite(lo[:-1]))
            rank = np.full(len(lo), -100, int)
            rank[valid_codes[np.argsort(lo[valid_codes], kind='stable')]] = np.arange(len(valid_codes))
            self.ranks[name] = rank[code]
        # Information weights use supplied covariates only, within geography.
        # Missing/generalised/refusal levels supply no identity evidence.
        self.idf_columns = [given.index(n) for n in
                            tuple(n for sex in COUNTS for n in sex)
                            + ('s2_1b_age',) + SOFT + BLOCK[2:]]
        self.idf = np.zeros((self.n, len(self.idf_columns)), np.float32)
        if IDF_ENABLED:
            for members in self.blocks.values():
                members = np.asarray(members, dtype=np.int64)
                for j, col in enumerate(self.idf_columns):
                    values = schema['items'][given[col]]['values']
                    code = X[members, col]
                    counts = np.bincount(code, minlength=len(values) + 1)
                    weight = np.log((len(members) + 1.) / (counts[code] + 1.))
                    valid_codes = np.array([not any(t in str(v).lower() for t in
                        ('generalised', 'prefer not', 'want to answer', 'not recorded',
                         "don't know", 'don’t know')) for v in values] + [False])
                    weight *= valid_codes[code]
                    self.idf[members, j] = weight
        from .finite_pop_raking import Capacity, CAPACITY_ENABLED
        self.capacity = Capacity(X, given) if CAPACITY_ENABLED else None
        self.edges = []
        for members in self.blocks.values():
            check_clock(deadline)
            members = np.array(members, dtype=np.int64)
            for w in (0, 1):
                left = members[self.wave[members] == w]
                right = members[self.wave[members] != w]
                if not len(left) or not len(right):
                    continue
                for start in range(0, len(left), 64):
                    query = left[start:start + 64]
                    check_clock(deadline)
                    d = self.distance(query, right)
                    self.edges.append((query, right, d.astype(np.float32)))

    def distance(self, query, donor):
        d = np.zeros((len(query), len(donor)), float)
        cross = self.wave[query, None] != self.wave[donor][None, :]
        for sex in COUNTS:
            adjacent = np.zeros_like(d, dtype=bool)
            for name in sex:
                ranks = self.ranks[name]
                adjacent |= np.abs(ranks[query, None] - ranks[donor][None, :]) == 1
            discount = np.where(cross & adjacent, 0.5, 1.0)
            lo = np.column_stack([self.intervals[n][0] for n in sex])
            hi = np.column_stack([self.intervals[n][1] for n in sex])
            al, ah = (lo[query], hi[query])
            bl, bh = (lo[donor], hi[donor])
            d += discount * 0.75 * _gap(al.sum(1)[:, None], ah.sum(1)[:, None], bl.sum(1)[None, :], bh.sum(1)[None, :])
            for b in range(3):
                d += discount * 0.25 * _gap(al[:, :b + 1].sum(1)[:, None], ah[:, :b + 1].sum(1)[:, None], bl[:, :b + 1].sum(1)[None, :], bh[:, :b + 1].sum(1)[None, :])
        lo, hi = self.intervals['s2_1b_age']
        age_gap = _gap(lo[query, None], hi[query, None], lo[donor][None, :], hi[donor][None, :])
        col = self.given.index('s2_1b_age')
        age_gap[self.X[query, col, None] == self.X[donor, col][None, :]] = 0.0
        ranks = self.ranks['s2_1b_age']
        age_gap *= np.where(cross & (np.abs(ranks[query, None] - ranks[donor][None, :]) == 1), 0.5, 1.0)
        d += 0.15 * age_gap
        for name in SOFT + BLOCK[2:]:
            col = self.given.index(name)
            d += 0.5 * (self.X[query, col, None] != self.X[donor, col][None, :])
        if IDF_ENABLED:
            # Weighted Hamming correction reorders ambiguous near neighbours.
            # Exact matches keep distance zero; empty-information rows retain d.
            weighted = np.zeros_like(d)
            information = np.zeros_like(d)
            mismatch = np.zeros_like(d)
            usable = np.zeros_like(d)
            for j, col in enumerate(self.idf_columns):
                a = self.idf[query, j, None]
                b = self.idf[donor, j][None, :]
                valid = (a > 0.) & (b > 0.)
                weight = np.minimum(a, b) * valid
                diff = self.X[query, col, None] != self.X[donor, col][None, :]
                weighted += weight * diff
                information += weight
                mismatch += valid * diff
                usable += valid
            correction = np.divide(weighted, information, out=np.zeros_like(d),
                                   where=information > 0.)
            correction -= np.divide(mismatch, usable, out=np.zeros_like(d),
                                    where=usable > 0.)
            d = np.maximum(0., d + correction)
        mismatch = np.zeros_like(d, dtype=np.int16)
        adjacent = np.zeros_like(d, dtype=np.int16)
        for name in [n for sex in COUNTS for n in sex] + ['s2_1b_age']:
            col = self.given.index(name)
            different = self.X[query, col, None] != self.X[donor, col][None, :]
            mismatch += different
            rank = self.ranks[name]
            adjacent += different & (rank[query, None] >= 0) & (rank[donor][None, :] >= 0) & (np.abs(rank[query, None]-rank[donor][None, :]) == 1)
        fixed = np.ones_like(d, dtype=bool)
        for name in SOFT + BLOCK:
            col = self.given.index(name)
            fixed &= self.X[query, col, None] == self.X[donor, col][None, :]
        exact = fixed & (mismatch == 0)
        near = fixed & (mismatch == 1) & (adjacent == 1)
        d[exact] = 0.
        d[near] = np.log(4.) / 3.
        return d

    def graph(self, eligible, deadline=float('inf')):
        ids = np.zeros((self.n, 8), np.int64)
        weights = np.zeros((self.n, 8), np.float64)
        reverse = {}
        for query, donor, distance in self.edges:
            key = donor.tobytes()
            mins = distance.min(axis=0)
            reverse[key] = np.minimum(reverse.get(key, mins), mins)
        for query, donor, distance in self.edges:
            check_clock(deadline)
            pos = np.flatnonzero(eligible[donor])
            if not len(pos):
                continue
            d = distance[:, pos]
            d = d + 0.35 * np.maximum(0.0, d - reverse[donor.tobytes()][pos][None, :])
            k = min(8, len(pos))
            order = np.argsort(d, axis=1, kind='stable')[:, :k]
            nearest = np.take_along_axis(d, order, axis=1)
            raw = np.where(nearest <= 1.25, np.exp(-3.0 * nearest), 0.0)
            twin = nearest[:, :1] <= 0.25
            raw[twin & (nearest > np.maximum(nearest[:, :1] + 0.25, np.log(4.) / 3. + 1e-7))] = 0.0
            raw /= np.maximum(1.0, raw.sum(1, keepdims=True))
            ids[query, :k] = donor[pos[order]]
            weights[query, :k] = raw
        load = np.bincount(ids.ravel(), weights=weights.ravel(), minlength=self.n)
        weights /= np.maximum(1.0, load[ids])
        if self.capacity is not None:
            weights = self.capacity.apply(ids, weights, eligible)
        return (ids, weights)

class _Plan:
    """Per-fold donor edges, compressed to the non-zero fit-donor weights.

    Every target sharing a train mask shares fit, query and graph, so the edge
    lists are built once per fold. Zero-weight slots never contributed to any
    count or prediction; dropping them keeps every sum in the same order."""

    def __init__(self, wave, fit, val, query, ids, weights, codes):
        self.fit, self.val, self.query, self.ids, self.weights = fit, val, query, ids, weights
        self.codes = codes
        fit_mask = np.zeros(len(wave), bool); fit_mask[fit] = True
        self.side = {}
        for w in (0, 1):
            tr = fit[wave[fit] == w]
            qpos = np.flatnonzero(wave[query] == w); q = query[qpos]
            wt = weights[tr] * fit_mask[ids[tr]]
            t_row, t_slot = np.nonzero(wt > 0)
            wq = weights[q] * fit_mask[ids[q]]
            q_row, q_slot = np.nonzero(wq > 0)
            q_w = wq[q_row, q_slot]
            self.side[w] = dict(tr=tr, qpos=qpos, n_q=len(q), donor_rows=fit[wave[fit] == 1 - w],
                t_y=tr[t_row], t_donor=ids[tr][t_row, t_slot], t_w=wt[t_row, t_slot],
                q_row=q_row, q_donor=ids[q][q_row, q_slot], q_w=q_w,
                mass=np.bincount(q_row, weights=q_w, minlength=len(q))[:, None])

    def code(self, ctx):
        key = id(ctx)
        if key not in self.codes:
            # Edges reach fit donors only, so the fold mask is always true here.
            self.codes[key] = np.where(ctx['train'] & (ctx['y'] >= 0), ctx['y'], ctx['K'])
        return self.codes[key]


def _distribution(job, plan, reserve_q, *, return_evidence=False, context=(), panel_joint=None, factored=False, return_pair=False, return_base=False, limit=None, zero_gate=None):
    """Fit-only directional tables; joint/factored arms share base statistics."""
    y, k = job['y'], job['K']
    result = reserve_q.copy(); numerator = np.zeros_like(result)
    masses = np.zeros((len(plan.query), 1), float)
    if return_base:
        base_result = reserve_q.copy(); base_num = np.zeros_like(result)
    if return_pair:
        factor_result = reserve_q.copy(); factor_num = np.zeros_like(result)
    for w in (0, 1):
        side = plan.side[w]
        tr, qpos = side['tr'], side['qpos']
        if not side['n_q']:
            continue
        counts = np.bincount(y[tr], minlength=k).astype(float)
        prior = (counts + 1. / k) / (len(tr) + 1.)
        transition = 3. * np.eye(k) + prior[None, :]
        if panel_joint is not None:
            joint = panel_joint if w == 1 else panel_joint.T
            support = joint.sum(1, keepdims=True)
            transition += np.divide(8. * joint, support, out=np.zeros_like(joint), where=support > 0)
        wt, yt = side['t_w'], y[side['t_y']]
        q_row, wq, n_q = side['q_row'], side['q_w'], side['n_q']
        st0, sq0 = y[side['t_donor']], y[side['q_donor']]
        transition += np.bincount(st0*k+yt, weights=wt, minlength=k*k).reshape(k,k)
        transition /= transition.sum(1, keepdims=True)
        base_transition = transition
        contexts = context.get(w, ()) if isinstance(context, dict) else context
        encoded = []
        for ctx in contexts:
            codes = plan.code(ctx)
            encoded.append((ctx['K'] + 1, codes[side['t_donor']], codes[side['q_donor']]))
        def conditional(base, st, sq, encoded_context):
            width, ct, cq = encoded_context
            st, sq = st * width + ct, sq * width + cq
            hits = np.bincount(st*k+yt, weights=wt,
                minlength=len(base)*width*k).reshape(len(base)*width,k)
            table = (hits + 8.*np.repeat(base,width,axis=0))/(hits.sum(1,keepdims=True)+8.)
            return table, st, sq
        flat = (q_row[:, None] * k + np.arange(k)).ravel()
        def predict_table(table, sq, st=None):
            edge = table[sq]
            if zero_gate is not None and w == 1 and st is not None:
                z, opened, observed, target_open = zero_gate
                td, qd = side['t_donor'], side['q_donor']
                fit_zero = (st0 == z) & observed[td]
                query_zero = (sq0 == z) & observed[qd]
                if fit_zero.any() and query_zero.any():
                    # Each direction has fit donors only. Split wave-one zero
                    # into parent-open and parent-closed response states.
                    state = st[fit_zero] * 2 + opened[td[fit_zero]].astype(int)
                    query_state = sq[query_zero] * 2 + opened[qd[query_zero]].astype(int)
                    prior_table = np.repeat(table, 2, axis=0)
                    value = np.zeros((int(query_zero.sum()), k))
                    for gate_state in (0, 1):
                        pt = target_open[side['t_y'][fit_zero]]
                        pq = target_open[plan.query[qpos[q_row[query_zero]]]]
                        pt = pt if gate_state else 1. - pt
                        pq = pq if gate_state else 1. - pq
                        hits = np.bincount(state*k + yt[fit_zero],
                            weights=wt[fit_zero]*pt, minlength=len(prior_table)*k).reshape(-1,k)
                        conditional_table = (hits + 8.*prior_table)/(hits.sum(1,keepdims=True)+8.)
                        value += pq[:,None]*conditional_table[query_state]
                    edge = edge.copy(); edge[query_zero] = value
            return np.bincount(flat, weights=(wq[:, None] * edge).ravel(),
                minlength=n_q*k).reshape(n_q, k)
        if return_base:
            plain_p = predict_table(base_transition,sq0,st0)
        st, sq = st0, sq0
        first = None
        if not factored or return_pair:
            cap = limit if limit is not None else 2 if isinstance(context,dict) else len(encoded)
            for index, enc in enumerate(encoded[:cap]):
                transition, st, sq = conditional(transition,st,sq,enc)
                if index == 0:
                    first = (transition, sq, st)
            p = predict_table(transition,sq,st)
        # Only the proposed low-entropy and high-cardinality target regimes
        # spend extra work in the paired path. Eligibility uses fit labels.
        collapse = False
        do_factor = bool(encoded) and (factored or return_pair)
        if do_factor:
            donor_rows = side['donor_rows']
            count_d = np.bincount(y[donor_rows],minlength=k).astype(float)
            pd = count_d[count_d>0]/max(1.,count_d.sum())
            collapse = w == 1 and -np.sum(pd*np.log(pd)) < .05
            do_factor = factored or k >= 5 or collapse
        if do_factor:
            if collapse:
                base = ((count_d/max(1.,count_d.sum())) @ base_transition)[None,:]
                ft, fq = np.zeros_like(st0),np.zeros_like(sq0)
            else:
                base,ft,fq = base_transition,st0,sq0
            pf = np.zeros((n_q,k),float)
            for index, enc in enumerate(encoded[:3]):
                if index == 0 and first is not None and not collapse:
                    table,sqf,stf = first
                else:
                    table,stf,sqf = conditional(base,ft,fq,enc)
                pf += predict_table(table,sqf,stf)
            pf /= len(encoded[:3])
        elif return_pair:
            pf = p
        if factored:
            if not do_factor:
                p = predict_table(base_transition,sq0,st0)
            else:
                p = pf
        mass = side['mass']
        result[qpos] = (p+.25*reserve_q[qpos])/(mass+.25)
        numerator[qpos] = p; masses[qpos] = mass
        if return_base:
            base_result[qpos] = (plain_p+.25*reserve_q[qpos])/(mass+.25)
            base_num[qpos] = plain_p
        if return_pair:
            factor_result[qpos] = (pf+.25*reserve_q[qpos])/(mass+.25)
            factor_num[qpos] = pf
    if return_base:
        return (base_result,base_num,masses),(result,numerator,masses)
    if return_pair:
        return (result,numerator,masses),(factor_result,factor_num,masses)
    return (result,numerator,masses) if return_evidence else result


def _k3_gate(name, work, items, X, given):
    """Observed donor routing and answer-free target routing probabilities."""
    from .twin_base import _zero_code
    from .twin_posterior import source_open
    zero = _zero_code(items, name)
    gate = items[name].get('gate')
    if zero < 0 or not gate or gate['parent'] == name:
        return None
    parent = gate['parent']; values = items[parent]['values']
    allowed = [values.index(v) for v in gate['observed_if'] if v in values]
    if parent in given:
        opened = source_open(name, work, items, X, given)
        observed = X[:,given.index(parent)] < len(values)
        target_open = np.isin(X[:,given.index(parent)], allowed).astype(float)
    elif parent in work and work[parent].get('oof') and work[parent].get('all'):
        p = work[parent]; rows = np.flatnonzero(p['train']); rest = np.flatnonzero(~p['train'])
        opened = source_open(name, work, items, X, given)
        observed = p['train'] & (p['y'] >= 0) & (p['y'] < len(values))
        target_open = np.zeros(len(X), float)
        for ix, pred in ((rows,p['oof'][0]), (rest,p['all'][0][rest])):
            pred = np.asarray(pred,float)
            target_open[ix] = pred[:,allowed].sum(1)/np.maximum(pred.sum(1),1e-300)
    else:
        return None
    return zero, opened, observed, np.clip(target_open,0.,1.)


def _mm_ami(tables):
    """Vectorized second-order Miller-Madow adjusted MI, natural logs.

    Zero marginal categories contribute zero degrees of freedom. The expansion
    is asymptotic, so sparse tables are evaluated as approximate rankings.
    """
    t = np.asarray(tables, dtype=np.float64)
    r, c = t.sum(axis=-1), t.sum(axis=-2)
    n = r.sum(axis=-1)
    safe_n = np.maximum(n, 1.)
    rr, cc = (r > 0).sum(-1), (c > 0).sum(-1)
    def xlogx(x):
        return x * np.log(np.maximum(x, 1.))
    sr, sc = xlogx(r).sum(-1), xlogx(c).sum(-1)
    mi = (xlogx(t).sum(axis=(-2, -1)) - sr - sc) / safe_n + np.log(safe_n)
    hx, hy = np.log(safe_n) - sr / safe_n, np.log(safe_n) - sc / safe_n
    ir = np.divide(safe_n[..., None], r, out=np.zeros_like(r), where=r > 0).sum(-1)
    ic = np.divide(safe_n[..., None], c, out=np.zeros_like(c), where=c > 0).sum(-1)
    emi = (rr-1)*(cc-1)/(2.*safe_n) + (ir-1)*(ic-1)/(12.*safe_n**2)
    ami = (mi-emi)/np.maximum(.5*(hx+hy)-emi, 1e-12)
    return np.where((n >= 24) & (rr > 1) & (cc > 1), ami, 0.)


def _fast_mi_rankings(work, schema, wave, fit, ids, weights, deadline):
    """One fit-only encoded pool and dense contingency product per direction."""
    fit_mask = np.zeros(len(wave), bool); fit_mask[fit] = True
    result = {}
    wave_fit = {w: fit[wave[fit] == w] for w in (0,1)}
    names = list(work)
    widths = np.array([work[n]['K'] for n in names], int)
    offsets = np.r_[0, np.cumsum(widths)]
    for w in ((0, 1) if BIDIR_MI_ENABLED else (1,)):
        check_clock(deadline)
        tr = fit[(wave[fit] == w) & (weights[fit, 0] >= .25)]
        donor = ids[tr, 0]
        ok = fit_mask[donor] & (wave[donor] == 1-w)
        tr, donor = tr[ok], donor[ok]
        pool = []
        for name in names:
            ctx = work[name]; item = schema['items'][name]
            if ctx['K'] > 8 or (not BIDIR_MI_ENABLED and item.get('gate')):
                continue
            valid = ctx['train'] & (ctx['y'] >= 0) & (ctx['y'] < len(item.get('values', ())))
            checks = (1-w,) if BIDIR_MI_ENABLED else (0, 1)
            if any(not len(wave_fit[dw]) or valid[wave_fit[dw]].mean() <= .8 for dw in checks):
                continue
            good = valid[donor]
            if good.sum() >= 24 and np.unique(ctx['y'][donor[good]]).size > 1:
                pool.append(name)
        pool.sort()
        if len(tr) < 24 or not pool:
            result[w] = {name: [] for name in names}; continue
        # Float32 counts are exact for these sample sizes; probability algebra
        # below uses float64. Missing anchor/target observations remain zero.
        poff = np.r_[0, np.cumsum([work[n]['K'] for n in pool])]
        left = np.zeros((poff[-1], len(tr)), np.float32)
        right = np.zeros((len(tr), offsets[-1]), np.float32)
        for a, name in enumerate(pool):
            j = work[name]; y = j['y'][donor]
            ok = j['train'][donor] & (y >= 0) & (y < len(schema['items'][name]['values']))
            row = np.flatnonzero(ok); left[poff[a]+y[ok], row] = 1.
        for b, name in enumerate(names):
            j = work[name]; y = j['y'][tr]
            ok = j['train'][tr] & (y >= 0) & (y < j['K'])
            row = np.flatnonzero(ok); right[row, offsets[b]+y[ok]] = 1.
        all_counts = left @ right
        ranked = {}
        # A padded anchor axis evaluates the whole pool in one NumPy reduction.
        max_width = max(work[n]['K'] for n in pool)
        take = np.zeros((len(pool), max_width), int)
        valid = np.zeros_like(take, bool)
        for a, name in enumerate(pool):
            width = work[name]['K']; take[a,:width] = poff[a] + np.arange(width); valid[a,:width] = True
        # Equal target widths share one vectorized entropy reduction.
        for width in np.unique(widths):
            group = np.flatnonzero(widths == width)
            target_columns = offsets[group, None] + np.arange(width)
            table = all_counts[take[None, :, :, None], target_columns[:, None, None, :]]
            table *= valid[None, :, :, None]
            scores = _mm_ami(table)
            order = np.argsort(-scores, axis=1, kind='stable')
            for row, b in enumerate(group):
                name = names[b]
                ranked[name] = [pool[a] for a in order[row] if pool[a] != name and scores[row, a] > 0.][:3]
        result[w] = ranked
    if not BIDIR_MI_ENABLED:
        result[0] = result[1]
    return result


def attach(schema, X, given, fold, work, deadline=float('inf')):
    """Complete existing evidence first, then refine it within the same window."""
    start = time.monotonic()
    required = set(BLOCK + SOFT + ('wave', 's2_1b_age') + tuple(n for s in COUNTS for n in s))
    info = {'mi_evaluated': 0, 'mi_ranking_seconds': 0., 'factored_evaluated': 0, 'factored_accepted': 0, 'triple_evaluated': 0, 'triple_accepted': 0, 'mi_accepted': 0, 'mi_offered': 0, 'context_accepted': 0,
            'offered': 0, 'seconds': 0., 'reason': 'not_unhcr'}
    if not required.issubset(given):
        return info
    info['reason'] = 'complete'
    stop = min(deadline, start + 13.8 if CONTEXT_ENABLED else start + 11.7)
    graph_cache, sample_cache, codes = {}, {}, {}

    def empty(rows, rest, job):
        k = job['K']
        return (np.zeros((len(rows), k)), np.zeros((len(X), k)),
                np.zeros((len(rows), k)), np.zeros((len(rows), 1)),
                np.zeros((len(rest), k)), np.zeros((len(rest), 1)))

    def store(candidate, f, val, rest, result):
        p, num, mass = result
        co, cf, cn, cm, rn, rm = candidate
        if f == 5:
            cf[rest] = p; rn[:] = num; rm[:] = mass
        else:
            co[val] = p; cn[val] = num; cm[val] = mass

    def fold_input(job, rows, rest, f):
        check_clock(stop)
        key = (job['train'].tobytes(), f)
        if key not in graph_cache:
            fit = rows if f == 5 else rows[fold[rows] != f]
            val = fold[rows] == f
            query = rest if f == 5 else rows[val]
            eligible = np.zeros(len(X), bool); eligible[fit] = True
            graph_cache[key] = _Plan(kernel.wave, fit, val, query, *kernel.graph(eligible, stop), codes)
        plan = graph_cache[key]
        reserve = job['all'][0][plan.query] if f == 5 else job['oof'][0][plan.val]
        return plan.val, key, plan, reserve

    try:
        kernel = Kernel(schema, X, given, stop)
        # Complete the existing two candidates for all items before spending
        # the residual budget on MI. Every completed item remains available.
        for name, job in work.items():
            check_clock(stop)
            rows = np.flatnonzero(job['train']); rest = np.flatnonzero(~job['train'])
            if len(rows) < 20:
                continue
            plain, conditioned = empty(rows, rest, job), empty(rows, rest, job)
            context = [work[a] for a in ROOT_CONTEXT if a in work and a != name] if CONTEXT_ENABLED else []
            report_joint = None
            if name == 's3_11_intent_return_12m_lbl':
                from .survey_prior_data import PANEL_JOINT
                cells = {row['indicator']: row['percent'] / 100. for row in PANEL_JOINT}
                vals = schema['items'][name]['values']; no, yes = vals.index('0. No'), vals.index('1. Yes')
                report_joint = np.zeros((job['K'], job['K']))
                report_joint[no, no] = cells['no return intention in wave1, still no wave2']
                report_joint[no, yes] = cells['no return intention wave1, now yes wave2']
                report_joint[yes, no] = cells['return intention wave1, now no wave2']
                report_joint[yes, yes] = cells['return intention wave1, still yes wave2']
                report_o = np.zeros_like(plain[0]); report_p = np.zeros_like(plain[1])
            for f in range(6):
                val, key, plan, reserve = fold_input(job, rows, rest, f)
                if context:
                    plain_result, found = _distribution(job, plan,
                        reserve, context=context, return_base=True)
                else:
                    plain_result = _distribution(job, plan, reserve, return_evidence=True)
                    found = plain_result
                store(plain, f, val, rest, plain_result)
                store(conditioned, f, val, rest, found)
                if report_joint is not None:
                    rp = _distribution(job, plan, reserve, panel_joint=report_joint)
                    if f == 5:report_p[rest] = rp
                    else:report_o[val] = rp
            accepted = bool(context) and dominates(conditioned[0], plain[0],
                job['y'][rows], job['K'], fold[rows] % 2 == 0, margin=1e-5)
            chosen = conditioned if accepted else plain
            job['_cr_arm'] = chosen[:2]; job['_cr_state'] = chosen[2:]
            job['_cr_context_accepted'] = accepted
            info['context_accepted'] += int(accepted); info['offered'] += 1
            if report_joint is not None:
                job['oof'].append(report_o); job['all'].append(report_p)
                job['arms'].append('panel_report_transition')
                info['report_joint_cells'] = len(PANEL_JOINT)
                info['report_retained_mass'] = float(report_joint.sum())
                info['report_returned_mass'] = float(sum(row['percent'] / 100. for row in PANEL_JOINT if row['indicator'].endswith('returned')))
        info['base_seconds'] = time.monotonic() - start
        if MI_ENABLED:
            # Give first use of the residual window to targets whose visible
            # first-wave answer has the least information, then gated targets.
            def priority(name):
                j = work[name]; y = j['y'][j['train'] & (kernel.wave == 0)]
                counts = np.bincount(y, minlength=j['K']).astype(float)
                p = counts[counts > 0] / max(1., counts.sum())
                entropy = -float(np.sum(p * np.log(p)))
                return (entropy > .05, not j.get('has_gate', bool(schema['items'][name].get('gate'))), entropy, name)
            for name in sorted((n for n,j in work.items() if '_cr_arm' in j), key=priority):
                check_clock(stop)
                job = work[name]; rows = np.flatnonzero(job['train']); rest = np.flatnonzero(~job['train'])
                candidate = empty(rows, rest, job); mi_names = []
                for f in range(6):
                    val, key, plan, reserve = fold_input(job, rows, rest, f)
                    if key not in sample_cache:
                        ranked_at = time.monotonic()
                        sample_cache[key] = _fast_mi_rankings(work, schema, kernel.wave, plan.fit, plan.ids, plan.weights, stop)
                        info['mi_ranking_seconds'] += time.monotonic() - ranked_at
                    names = {w: sample_cache[key][w][name] for w in (0, 1)}
                    contexts = {w: [work[a] for a in names[w]] for w in (0, 1)}
                    mi_names.append(names)
                    found = _distribution(job, plan, reserve, return_evidence=True, context=contexts)
                    store(candidate, f, val, rest, found)
                info['mi_evaluated'] += 1
                offered = any(any(n.values()) for n in mi_names)
                base = job['_cr_arm'][0]
                accepted = offered and dominates(candidate[0], base, job['y'][rows],
                    job['K'], fold[rows] % 2 == 0, margin=1e-5)
                job['_cr_mi'] = {'accepted': bool(accepted), 'offered': bool(offered),
                    'anchors_by_fold': mi_names, 'oof_gain': float(_nll(base,
                        job['y'][rows], job['K']) - _nll(candidate[0], job['y'][rows], job['K']))}
                info['mi_offered'] += int(offered); info['mi_accepted'] += int(accepted)
                if accepted:
                    job['_cr_arm'] = candidate[:2]; job['_cr_state'] = candidate[2:]
        # Complete joint MI for every target before spending the remaining
        # window on factored alternatives. A partial alternative never replaces
        # an already complete six-fold candidate.
        info['joint_seconds'] = time.monotonic()-start
        if MI_ENABLED and FACTORED_MI_ENABLED:
            for name in sorted((n for n,j in work.items() if '_cr_mi' in j),key=priority):
                check_clock(stop)
                job=work[name]; rows=np.flatnonzero(job['train']); rest=np.flatnonzero(~job['train'])
                factor=empty(rows,rest,job)
                for f in range(6):
                    val,key,plan,reserve=fold_input(job,rows,rest,f)
                    contexts={w:[work[a] for a in sample_cache[key][w][name]] for w in (0,1)}
                    _, found=_distribution(job,plan,reserve,context=contexts,return_pair=True)
                    store(factor,f,val,rest,found)
                info['factored_evaluated']+=1
                accepted=dominates(factor[0],job['_cr_arm'][0],job['y'][rows],
                    job['K'],fold[rows]%2==0,margin=1e-5)
                job['_cr_mi']['factored_accepted']=bool(accepted)
                if accepted:
                    job['_cr_arm']=factor[:2];job['_cr_state']=factor[2:]
                    info['factored_accepted']+=1
        info['factored_seconds'] = time.monotonic()-start
        if MI_ENABLED and TRIPLE_ENABLED:
            # Third MI anchor: the 3-way cell shrinks to the 2-way cell, which
            # shrinks to the 1-way cell and the base transition (Dirichlet, 8).
            for name in sorted((n for n,j in work.items() if '_cr_mi' in j),key=priority):
                check_clock(stop)
                job=work[name]
                if not any(len(sample_cache[k][w][name]) >= 3 for k in sample_cache for w in (0,1)):
                    continue
                rows=np.flatnonzero(job['train']); rest=np.flatnonzero(~job['train'])
                triple=empty(rows,rest,job)
                for f in range(6):
                    val,key,plan,reserve=fold_input(job,rows,rest,f)
                    contexts={w:[work[a] for a in sample_cache[key][w][name]] for w in (0,1)}
                    found=_distribution(job,plan,reserve,return_evidence=True,context=contexts,limit=3)
                    store(triple,f,val,rest,found)
                info['triple_evaluated']+=1
                accepted=dominates(triple[0],job['_cr_arm'][0],job['y'][rows],
                    job['K'],fold[rows]%2==0,margin=1e-5)
                job['_cr_mi']['triple_accepted']=bool(accepted)
                if accepted:
                    job['_cr_arm']=triple[:2];job['_cr_state']=triple[2:]
                    info['triple_accepted']+=1
        import ortho_config as _ortho
        if getattr(_ortho, 'V114_K3', False):
            info.update(k3_evaluated=0, k3_accepted=0, k3_gains={})
            for name, job in work.items():
                check_clock(stop)
                if '_cr_arm' not in job:
                    continue
                gate = _k3_gate(name, work, schema['items'], X, given)
                if gate is None:
                    continue
                rows=np.flatnonzero(job['train']); rest=np.flatnonzero(~job['train'])
                candidate=empty(rows,rest,job)
                mi=job.get('_cr_mi', {})
                triple=mi.get('triple_accepted', False)
                factored=mi.get('factored_accepted', False) and not triple
                use_mi=mi.get('accepted', False) or factored or triple
                for f in range(6):
                    val,key,plan,reserve=fold_input(job,rows,rest,f)
                    if use_mi:
                        context={w:[work[a] for a in sample_cache[key][w][name]] for w in (0,1)}
                    else:
                        context=[work[a] for a in ROOT_CONTEXT if a in work and a != name] if job.get('_cr_context_accepted') else ()
                    found=_distribution(job,plan,reserve,return_evidence=True,
                        context=context,return_pair=factored,limit=3 if triple else None,zero_gate=gate)
                    if factored:
                        found=found[1]
                    store(candidate,f,val,rest,found)
                check_clock(stop)
                accepted=dominates(candidate[0],job['_cr_arm'][0],job['y'][rows],
                    job['K'],fold[rows]%2==0,margin=1e-5)
                gain=float(_nll(job['_cr_arm'][0],job['y'][rows],job['K'])-
                    _nll(candidate[0],job['y'][rows],job['K']))
                info['k3_evaluated']+=1
                job['_cr_k3']={'accepted':bool(accepted),'oof_gain':gain}
                if accepted:
                    job['_cr_arm']=candidate[:2];job['_cr_state']=candidate[2:]
                    info['k3_accepted']+=1;info['k3_gains'][name]=gain
    except BudgetExpired:
        info['reason'] = ('k3_budget' if 'k3_evaluated' in info else
            'joint_complete_factor_budget' if info['mi_evaluated'] == len(work) else 'budget_expired')
    info['context_roots'] = [a for a in ROOT_CONTEXT if a in work] if CONTEXT_ENABLED else []
    info['seconds'] = time.monotonic() - start
    return info

ROW_PRIORS = (0.05, 0.1, 0.18, 0.3, 0.6, 1.2, 2.5)
ROW_THRESHOLDS = (1e-09, 0.25, 0.45)

def _mix_rows(base, numerator, m, c, ask_only, thresh):
    """Twin evidence mixed on rows with mass >= thresh; other rows keep their bytes.

    ask_only keeps the routed skip probability rho in column k-1 and moves
    only the asked-conditional answers; otherwise the full k x k transition
    evidence, skip included, is mixed."""
    out = base.copy()
    hit = m[:, 0] >= thresh
    if not hit.any():
        return (out, hit)
    k = base.shape[1]
    if ask_only:
        b = base[hit]
        rho = b[:, k - 1:]
        asked = 1.0 - rho
        num = numerator[hit, :k - 1]
        m_asked = num.sum(1, keepdims=True)
        cond = b[:, :k - 1] / np.maximum(asked, 1e-12)
        effective_c = np.maximum(c, m_asked * (0.15 / 0.85))
        mixed = (num + effective_c * cond) / (m_asked + effective_c)
        live = asked[:, 0] > 1e-09
        b[live, :k - 1] = asked[live] * mixed[live]
        out[hit] = b
    else:
        effective_c = np.maximum(c, m[hit] * (0.15 / 0.85))
        out[hit] = (numerator[hit] + effective_c * base[hit]) / (m[hit] + effective_c)
    return (out, hit)

def row_gate(job, strong_oof, strong_rest, fold):
    """Select (prior, mode, mass threshold) on full OOF loss of visible rows.

    The grid contains the  setting (full mix, every supported row) and
    the  setting (asked-only, near twins), so the selected candidate is
    never worse on OOF than either. Rows below the chosen threshold keep
    the routed prediction byte for byte."""
    state = job.get('_cr_state')
    if state is None:
        return (strong_oof, strong_rest)
    num, mass, full_num, full_mass = state
    y, k = (job['y'][job['train']], job['K'])
    modes = (False, True) if bool(job.get('has_gate', False)) and k > 2 else (False,)
    if not len(y) or not np.any(mass[:, 0] > 0):
        job['row_gate'] = {'accepted': False, 'twin_rows': 0}
        return (strong_oof, strong_rest)
    before = _nll(strong_oof, y, k)
    best = None
    for thresh in ROW_THRESHOLDS:
        if not np.any(mass[:, 0] >= thresh):
            continue
        for ask_only in modes:
            for c in ROW_PRIORS:
                cand = _mix_rows(strong_oof, num, mass, c, ask_only, thresh)[0]
                loss = _nll(cand, y, k)
                if best is None or loss < best[0]:
                    best = (loss, cand, c, ask_only, thresh)
    after, candidate, c, ask_only, thresh = best
    half = fold[job['train']] % 2 == 0
    accepted = after < before - 1e-05 and all((_nll(candidate[h], y[h], k) < _nll(strong_oof[h], y[h], k) for h in (half, ~half) if h.any()))
    job['row_gate'] = {'c': c, 'ask_only': bool(ask_only), 'thresh': float(thresh), 'oof_gain': float(before - after), 'accepted': bool(accepted), 'twin_rows': int((mass[:, 0] >= thresh).sum()), 'zero_mass_unchanged': True}
    if not accepted:
        return (strong_oof, strong_rest)
    return (candidate, _mix_rows(strong_rest, full_num, full_mass, c, ask_only, thresh)[0])

def refine(work, fold, deadline=float('inf')):
    """Reuse paired evidence with base arms ranked outside each held-out fold."""
    start = time.monotonic()
    info = {'mi_accepted': 0, 'mi_offered': 0, 'context_accepted': 0, 'offered': 0, 'upgraded': 0, 'added': 0, 'priors': {}}
    try:
        for job in work.values():
            state = job.get('_cr_state')
            if state is None:
                continue
            if time.monotonic() >= deadline:
                break
            rows = np.flatnonzero(job['train'])
            rest = np.flatnonzero(~job['train'])
            split = fold[rows]
            half = split % 2 == 0
            if not half.any() or half.all():
                continue
            arms = [i for i, arm in enumerate(job['arms']) if arm != 'cross_round']
            if not arms or any((not (split != f).any() for f in range(5))):
                continue
            y, k = (job['y'][rows], job['K'])
            strong_oof = np.zeros((len(rows), k))
            strong_rest = np.zeros((len(rest), k))
            idx = np.arange(len(rows))
            row_losses = np.stack([-np.log(0.001 + (1 - 0.001 * k) * job['oof'][i][idx, y]) for i in arms])
            total_losses = row_losses.sum(axis=1)
            rest_arms = [job['all'][i][rest] for i in arms]
            for f in range(5):
                fit = split != f
                val = ~fit
                loo = (total_losses - row_losses[:, val].sum(axis=1)) / max(1, int(fit.sum()))
                selected = np.argsort(loo, kind='stable')[:3]
                best = [arms[i] for i in selected]
                strong_oof[val] = sum((job['oof'][i][val] for i in best)) / len(best)
                strong_rest += sum((rest_arms[i] for i in selected)) / (5 * len(best))
            num_oof, mass_oof, num_rest, mass_rest = state
            trials = [((num_oof + c * strong_oof) / (mass_oof + c), c) for c in (0.15, 0.25, 0.5)]
            cand, c = min(trials, key=lambda t: _nll(t[0], y, k))
            info['offered'] += 1
            old = job['arms'].index('cross_round') if 'cross_round' in job['arms'] else None
            references = [strong_oof, job['oof'][old] if old is not None else job['oof'][0]]
            if _nll(cand, y, k) >= min((_nll(ref, y, k) for ref in references)) - 0.0002:
                continue
            if not all((_nll(cand[h], y[h], k) < _nll(ref[h], y[h], k) for ref in references for h in (half, ~half))):
                continue
            full = np.zeros((len(job['y']), k))
            full[rest] = (num_rest + c * strong_rest) / (mass_rest + c)
            if old is None:
                job['arms'].append('cross_round')
                job['oof'].append(cand)
                job['all'].append(full)
                info['added'] += 1
            else:
                job['oof'][old] = cand
                job['all'][old] = full
                info['upgraded'] += 1
            info['priors'][str(c)] = info['priors'].get(str(c), 0) + 1
    except BudgetExpired:
        info['reason'] = 'complete'
    info['seconds'] = time.monotonic() - start
    return info

def pre_track(job, fold, calibrate, deadline):
    """Build a separate stack; leave every no-pre-arm input untouched."""
    if '_cr_arm' not in job:
        return None
    o, full = job['_cr_arm']
    y = job['y'][job['train']]
    k = job['K']
    rows = np.flatnonzero(job['train'])
    half = fold[rows] % 2 == 0
    threshold = 0.0015 * np.sqrt(1200 / max(len(rows), 1200))
    if len(rows) < 2500 and (not dominates(o, job['oof'][0], y, k, half, threshold)):
        return None
    alternate = dict(job)
    for key in ('arms', 'oof', 'all'):
        alternate[key] = list(job[key])
    alternate['arms'].append('cross_round')
    alternate['oof'].append(o)
    alternate['all'].append(full)
    refine({0: alternate}, fold, deadline)
    rest = np.flatnonzero(~job['train'])
    p, o, w, t, loss = calibrate(alternate['oof'], [a[rest] for a in alternate['all']], y, k, deadline=deadline)
    alternate['_cr_weight'] = float(w[alternate['arms'].index('cross_round')])
    return ((o, p, loss), alternate)

def dominates(candidate, baseline, y, k, half, margin=1e-05):
    masks = (np.ones(len(y), bool), half, ~half)
    return all((m.any() and _nll(baseline[m], y[m], k) - _nll(candidate[m], y[m], k) >= margin for m in masks))
