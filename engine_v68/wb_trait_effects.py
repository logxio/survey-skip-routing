"""Direct covariate target offsets and fold-isolated trait transport."""
import time
import numpy as np
from scipy.sparse import csr_matrix, vstack
from .wb_lca import _Expired
from . import v68_config

EFFECT_RIDGE = 20.0
EFFECT_ITERS = 15
PAIR_RIDGE = 20.0
PAIR_RANK = True
PAIR_EDGE = 1.1


def _cell_sums(R, idx, size):
    """sum of R rows into flat bins idx: (size, nodes)."""
    return np.stack([np.bincount(idx, weights=R[:, q], minlength=size) for q in range(R.shape[1])], 1)


def _cell_tables(log_g, stratum, t0, tb):
    """One compact emission table per stratum and target block."""
    T = np.exp(log_g[:, :, t0:])
    return [[np.ascontiguousarray(T[s, :, a:b]) for a, b in tb]
            for s in range(len(T))]


def _pair_levels(cells, pairs):
    """Joint level codes of GIVEN column pairs on the distinct cells: cells x pairs."""
    cols = []
    for a, b in pairs:
        _, code = np.unique(cells[:, a] * (int(cells[:, b].max()) + 1) + cells[:, b], return_inverse=True)
        cols.append(code.ravel())
    return np.column_stack(cols)


def _direct_effects_reference(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata, deadline, pairs=(), rank=1, refresh=False):
    """Returns (target offsets per cell: cells x target columns, row -> cell, cell stratum).

    P(answer | node, cell) = T * w / sum(T * w) per block with w = exp(offset), so
    an iteration rescales the fixed tables instead of recomputing a softmax.
    pairs: GIVEN column index pairs that also take a joint-level offset ( U1).
    rank > 1 keeps every noise-cleared component of a pair table ( U2);
    refresh re-reads the training posterior under the fitted offsets and refits once."""
    Ot = O[rows]
    m, D = Ot.shape
    G, Qn = log_g.shape[0], log_g.shape[1]
    parts = [np.arange(m)] if G == 1 else [np.flatnonzero(strata[rows] == s) for s in range(G)]
    L = np.empty((m, Qn), np.float32)
    for s, idx in enumerate(parts):
        if len(idx):
            L[idx] = Ot[idx] @ log_g[s].T
    L += lw
    L -= L.max(1, keepdims=True)
    R = np.exp(np.maximum(L, -80.0)).astype(np.float64)
    R /= R.sum(1, keepdims=True)
    cells, inv = np.unique(Xg, axis=0, return_inverse=True)
    inv = inv.ravel()
    C = len(cells)
    t0 = n_given_cols
    tb = [(a - t0, b - t0) for a, b in blocks if a >= t0]
    Dt = D - t0
    starts = np.array([a for a, _ in tb])
    # A cell's stratum is the stratum of any of its rows (strata are GIVEN codes).
    stratum = np.zeros(C, np.int64)
    if G > 1:
        stratum[inv] = strata
    c_rows = inv[rows]
    seen = np.flatnonzero(np.bincount(c_rows, minlength=C) > 0)
    where = np.full(C, -1, np.int64)
    where[seen] = np.arange(len(seen))
    cr = where[c_rows]
    Cs = len(seen)
    Yt = Ot[:, t0:]
    ii, dd = np.nonzero(Yt)
    Ns = np.bincount(cr[ii] * Dt + dd, minlength=Cs * Dt).reshape(Cs, Dt)
    bo = np.add.reduceat(Yt, starts, axis=1) > 0
    ib, bb = np.nonzero(bo)
    n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
    Ts = _cell_tables(log_g, stratum[seen], t0, tb)
    s_cells = [slice(None)] if G == 1 else [np.flatnonzero(stratum[seen] == s) for s in range(G)]
    ridge = [EFFECT_RIDGE] * cells.shape[1]
    if pairs:
        cells = np.column_stack([cells, _pair_levels(cells, pairs)])
        ridge += [PAIR_RIDGE] * len(pairs)
    cs = cells[seen]
    J = cs.shape[1]
    # Sparse one-hot sufficient statistics: each respondent cell contributes once per level.
    E = [csr_matrix((np.ones(Cs), (cs[:, j], np.arange(Cs))), shape=(int(cells[:, j].max()) + 1, Cs)) for j in range(J)]
    beta = [np.zeros((E[j].shape[0], Dt)) for j in range(J)]
    E_all = vstack(E, format='csr')
    design = E_all.T.tocsr()
    h = np.empty((Cs, Dt))
    def node_counts(n):
        return [[np.ascontiguousarray(n[sc, b, :].T) for b in range(len(tb))]
                for sc in s_cells]
    ns = node_counts(n)

    def sweep(update):
        update = list(update)
        K = len(update) if v68_config.V115_W2 else J
        Eu = vstack([E[j] for j in update], format='csr')
        counts = Eu @ Ns
        for _ in range(EFFECT_ITERS):
            if time.monotonic() >= deadline:
                raise _Expired()
            w = np.exp(design @ np.concatenate(beta)).astype(np.float32)
            for s, sc in enumerate(s_cells):
                ws = np.ascontiguousarray(w[sc].T)
                for b, (a, e) in enumerate(tb):
                    wb = ws[a:e]
                    T = Ts[s][b]
                    den = T @ wb
                    h[sc, a:e] = (wb * (T.T @ (ns[s][b] / den).astype(np.float32))).T
            eh = Eu @ h
            eg = counts - eh
            pos = 0
            for j in update:
                # Jensen over the columns this sweep moves: curvature K h keeps the joint step a majorizer.
                end = pos + len(beta[j])
                beta[j] += np.clip((eg[pos:end] - ridge[j] * beta[j]) / (K * eh[pos:end] + ridge[j]), -1.0, 1.0)
                pos = end

    def fit():
        if pairs and PAIR_RANK and rank > 1:
            #  U2: main effects first, the free pair tables on what they leave,
            # so a pair with no signal gates to zero without moving the main effects.
            main = range(J - len(pairs))
            sweep(main)
            sweep(range(J - len(pairs), J))
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(main)
            return
        sweep(range(J))
        if pairs and PAIR_RANK:
            # A pair's free table is mostly noise; keep its shrunk rank-one part (one
            # score per joint level times one direction over every target option),
            # then refit the main effects around it.
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_one(beta[j], E[j] @ rows_per_cell) if rank <= 1 else _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(range(J - len(pairs)))

    fit()
    if refresh:
        # EM's next E step: n came from the trait-only posterior; re-read it with
        # the offsets inside the target emissions, then refit from where we are.
        w = np.exp(sum(beta[j][cs[:, j]] for j in range(J)))
        L = np.empty((m, Qn), np.float64)
        for s, idx in enumerate(parts):
            if len(idx):
                L[idx] = Ot[idx, :t0] @ log_g[s, :, :t0].T
        L += lw
        # log(w[cell, answer]) is constant across quadrature nodes and cancels
        # in the posterior. Preserve the original float32 emission rounding,
        # but accumulate the refresh in float64, as in the cell-expanded form.
        for s, sc in enumerate(s_cells):
            for b, (a, e) in enumerate(tb):
                hit = (dd >= a) & (dd < e)
                if G > 1:
                    hit &= stratum[c_rows[ii]] == s
                rr, answer = ii[hit], dd[hit] - a
                T = Ts[s][b]
                den = np.empty((Cs, Qn), np.float64)
                den[sc] = np.log(np.maximum(w[sc, a:e] @ T.T, 1e-300))
                lt = np.log(T.astype(np.float64))[:, answer].T - den[cr[rr]]
                np.add.at(L, rr, lt * Yt[rr, dd[hit]][:, None])
        L -= L.max(1, keepdims=True)
        R = np.exp(np.maximum(L, -80.0))
        R /= R.sum(1, keepdims=True)
        n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
        ns = node_counts(n)
        fit()
    off = sum(beta[j][cells[:, j]] for j in range(J))
    return off, inv, stratum


def _rank_gated(B, count, rank):
    """_rank_one generalized ( U2): a singular component is kept only above
    PAIR_EDGE times the Marchenko-Pastur edge of the noise, where pure noise's top
    component sits, then takes _rank_one's James-Stein share. The noise is
    re-read from the components left out; a table with no signal returns zero."""
    keep = count > 0
    r = np.sqrt(count[keep])[:, None]
    M = r * B[keep]
    L, D = M.shape
    out = np.zeros_like(B)
    if L < 3 or D < 3:
        return out
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = 0
    for _ in range(4):
        noise = float((s[k:] ** 2).sum()) / max((L - k) * (D - k), 1)
        k_new = int(min(rank, len(s) - 1, np.count_nonzero(s ** 2 > PAIR_EDGE * noise * (np.sqrt(L) + np.sqrt(D)) ** 2)))
        if k_new == k:
            break
        k = k_new
    if not k:
        return out
    shrink = np.maximum(0.0, 1.0 - noise * (L + D) / np.maximum(s[:k] ** 2, 1e-30))
    out[keep] = ((U[:, :k] * (s[:k] * shrink)) @ Vt[:k]) / r
    return out


def _rank_one(B, count):
    """James-Stein shrunk rank-one part of a levels x columns offset table, rows
    scaled by sqrt(row count) so every entry carries about the same noise."""
    keep = count > 0
    r = np.sqrt(count[keep])[:, None]
    M = r * B[keep]
    L, D = M.shape
    out = np.zeros_like(B)
    if L < 3 or D < 3:
        return out
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    noise = float((s[1:] ** 2).sum()) / ((L - 1) * (D - 1))
    shrink = max(0.0, 1.0 - noise * (L + D) / max(float(s[0]) ** 2, 1e-30))
    out[keep] = (s[0] * shrink) * np.outer(U[:, 0], Vt[0]) / r
    return out


def predict_effects(O, t0, log_g, lw, off, inv, stratum, blocks, targets, nest):
    """Compose target gates after direct logits, preserving conditional mass."""
    n, C = len(O), len(off)
    rep = np.zeros(C, np.int64)
    rep[inv[::-1]] = np.arange(n)[::-1]
    L = O[rep, :t0] @ log_g[0, :, :t0].T + lw
    L -= L.max(1, keepdims=True)
    Q = np.exp(np.maximum(L, -80.0)); Q /= Q.sum(1, keepdims=True)
    # Integrate each two-dimensional emission table directly. Parent gates need
    # only their selected answer columns, never a cells x nodes x answers tensor.
    tb = [(a, b) for a, b in blocks if a >= t0]
    parts = [slice(None)] if len(log_g) == 1 else [np.flatnonzero(stratum == s) for s in range(len(log_g))]
    weights = np.exp(off)
    tables = {}
    for a, b in tb:
        Ts = [np.ascontiguousarray(np.exp(log_g[s, :, a:b])) for s in range(len(log_g))]
        den = np.empty((C, log_g.shape[1]))
        for s, sc in enumerate(parts):
            den[sc] = weights[sc, a-t0:b-t0] @ Ts[s].T
        tables[a] = [Ts, den, None, None]
    def column(col):
        a = next(a for a, b in tb if a <= col < b)
        Ts, den, gate, fill = tables[a]
        value = np.empty_like(den)
        for s, sc in enumerate(parts):
            value[sc] = weights[sc, col-t0, None] * Ts[s][:, col-a] / den[sc]
        if gate is not None:
            value *= gate
            if col-a == fill:
                value += 1-gate
        return value
    for a, K, parent_cols, fill, renorm in nest:
        gate = sum(column(col) for col in parent_cols)
        Ts, den, _, _ = tables[a]
        if renorm:
            for s, sc in enumerate(parts):
                Ts[s] = Ts[s].copy()
                Ts[s][:, -1] = 0
                den[sc] = np.maximum(weights[sc, a-t0:a-t0+K] @ Ts[s].T, 1e-30)
        tables[a] = [Ts, den, gate, fill]
    out = np.zeros((n, O.shape[1]-t0), np.float32)
    for a, b in tb:
        Ts, den, gate, fill = tables[a]
        weighted = Q if gate is None else Q * gate
        P = np.empty((C, b-a))
        for s, sc in enumerate(parts):
            P[sc] = weights[sc, a-t0:b-t0] * ((weighted[sc] / den[sc]) @ Ts[s])
        if gate is not None:
            P[:, fill] += (Q * (1-gate)).sum(1)
        out[:, a-t0:b-t0] = P[inv]
    return out


def guarded(base, candidates, work, fold, margin=0.001):
    """Two half comparisons outside the evaluation fold; hidden uses visible only."""
    result = {}; decisions = {}
    for name, (bo, bp) in base.items():
        j=work[name]; rows=np.flatnonzero(j['train']); rf=fold[rows]; y=j['y'][rows]; K=j['K']
        choices=[(key, value[name]) for key,value in candidates.items() if name in value]
        def losses(p):
            p=np.asarray(p,float); p=p/np.maximum(p.sum(1,keepdims=True),1e-30)
            return -np.log(0.001+(1-K*0.001)*p[np.arange(len(y)),y])
        bl=losses(bo); gains=[bl-losses(v[0]) for _,v in choices]
        o=bo.copy(); p=bp.copy(); chosen={}
        for f in [*np.unique(rf),None]:
            train=np.ones(len(rows),bool) if f is None else rf!=f
            halves=[train & (rf%2==v) for v in (0,1)]
            admissible=[i for i,d in enumerate(gains) if all(m.any() and d[m].mean()>=margin for m in halves)]
            if not admissible: continue
            i=max(admissible,key=lambda i:gains[i][train].mean())
            co,cp=choices[i][1]
            if f is None:p[~j['train']]=cp[~j['train']]
            else:o[rf==f]=co[rf==f]
            chosen[str(f)]=choices[i][0]
        p[rows]=o;result[name]=(o,p);decisions[name]=chosen
    return result,decisions


def stage2(keep, anchor_probs, work, fold, deadline, gammas=(0.15,0.3,0.5), mode='ratio'):
    """Jeffrey likelihood-ratio transport, or tempered soft likelihood control.

    Each retained emission model excludes f. Only anchor predictions for that
    model's held rows are read. No observed target answers enter this posterior.
    Identity anchor evidence produces an exactly unchanged ratio posterior.
    """
    if not keep:return {},{'status':'no_models'}
    from .wb_lca import _emit, MIX
    models,targets,t0,strata,base=keep
    n=len(fold); folds=np.unique(fold[fold>=0]); rows=np.flatnonzero(fold>=0); rest=np.flatnonzero(fold<0)
    follow=[k for k in work if k.startswith('tr_')]
    anchors=[k for k in anchor_probs if k in targets and not k.startswith('tr_')]
    if not anchors:return {},{'status':'no_anchors'}
    cands={str(g):{k:(np.zeros((len(rows),work[k]['K']),np.float32),np.zeros((n,work[k]['K']),np.float32)) for k in follow} for g in gammas}
    started=time.monotonic()
    for f in folds:
        if time.monotonic()>=deadline:return {},{'status':'window_closed'}
        Q,T=models[int(f)]
        query=np.r_[rows[fold[rows]==f],rest]; hit=fold[rows]==f
        st=np.zeros(len(query),int) if len(T)==1 else strata[query]
        evidence=np.zeros((len(query),Q.shape[1]),float)
        for name in anchors:
            a,K,closed=targets[name]
            tab=T[st,:,a:a+K].astype(float)
            p=anchor_probs[name][query].astype(float)
            p/=np.maximum(p.sum(1,keepdims=True),1e-30)
            if mode=='ratio':
                old=np.einsum('nq,nqk->nk',Q[query],tab)
                # Soft evidence is a probability forecast, not an observed draw.
                ratio=p/np.maximum(old,1e-12)
                term=np.log(np.maximum(np.einsum('nqk,nk->nq',tab,ratio),1e-12))
            else:term=np.einsum('nk,nqk->nq',p,np.log(np.maximum(tab,1e-12)))
            term[closed[query]]=0; evidence+=term
        for gamma in gammas:
            L=np.log(np.maximum(Q[query],1e-30))+gamma*evidence
            L-=L.max(1,keepdims=True);P=np.exp(L);P/=P.sum(1,keepdims=True)
            for name in follow:
                a,K,closed=targets[name]
                tab=T[st,:,a:a+K]
                pred=np.einsum('nq,nqk->nk',P,tab)
                original=np.einsum('nq,nqk->nk',Q[query],tab)
                # Transport the trait update onto the completed four-kind mixture.
                # At gamma=0 this is the unchanged mixture, including class arms.
                target_base=base[name][1][query].astype(float)
                target_base=np.maximum((target_base-MIX/K)/(1-MIX),0)
                pred=target_base*pred/np.maximum(original,1e-12)
                pred/=np.maximum(pred.sum(1,keepdims=True),1e-30)
                pred[closed[query]]=0;pred[closed[query],-1]=1
                pred=(1-MIX)*pred+MIX/K
                o,p=cands[str(gamma)][name];o[hit]=pred[:hit.sum()];p[rest]+=pred[hit.sum():]/len(folds)
    for v in cands.values():
        for o,p in v.values():p[rows]=o
    gated,chosen=guarded({k:base[k] for k in follow},cands,work,fold)
    return gated,{'seconds':time.monotonic()-started,'anchors':len(anchors),'mode':mode,'chosen':chosen,'raw':cands}


def attach_chain_lca(keep, stacked_anchor, chain, work, fold, items, anchors, post_items, children, deadline, event=None):
    """Stage-conditioned cascaded LCA posterior transport over upstream StackedChain parents."""
    if not keep or not stacked_anchor:
        return {}, {'status': 'no_models'}
    from .wb_lca import MIX
    from .rao_blackwell import adjusted_mi
    from .chain_features import priority_sources
    from .paired_transition import PAIRS, EXTRA_PAIRS, V127_PAIRS
    models_by_kind, kind_order, targets, t0, strata, base_found, w_by_item = keep
    if not models_by_kind or not kind_order:
        return {}, {'status': 'no_kinds'}
    started = time.monotonic()
    n = len(fold)
    folds = np.unique(fold[fold >= 0])
    rows = np.flatnonzero(fold >= 0)
    rest = np.flatnonzero(fold < 0)
    rf = fold[rows]
    anchor_probs = {}
    for a, (oa, pa) in stacked_anchor.items():
        if a not in work or a not in targets:
            continue
        ja = work[a]
        qa = np.zeros((n, ja['K']), np.float64)
        qa[ja['train']] = oa
        qa[~ja['train']] = pa
        qa = np.maximum(qa, 1e-12)
        qa /= qa.sum(1, keepdims=True)
        anchor_probs[a] = qa
    post_set = set(post_items)
    cascade_names = [name for name in list(post_items) + list(children) if name in work and name in targets and name in base_found]
    if not cascade_names or not anchor_probs:
        return {}, {'status': 'no_cascade_items'}
    gammas = (0.30, 0.65, 1.10)
    cand_keys = [str(g) for g in gammas] + (['ev_' + str(g) for g in gammas] if event is not None else [])
    cands = {
        ck: {name: (np.zeros((len(rows), work[name]['K']), np.float32), np.zeros((n, work[name]['K']), np.float32)) for name in cascade_names}
        for ck in cand_keys
    }
    ev_cache = {}
    top_by_item = {}
    for fi, f in enumerate(folds):
        if time.monotonic() >= deadline:
            return {}, {'status': 'window_closed', 'seconds': time.monotonic() - started}
        hit = rf == f
        fit = rows[~hit]
        query = np.r_[rows[hit], rest]
        n_val = int(hit.sum())

        def get_ev(mk, a):
            key = (mk, int(f), a)
            if key not in ev_cache:
                Q, T = models_by_kind[mk][int(f)]
                st = np.zeros(len(query), int) if len(T) == 1 else strata[query]
                a_start, Ka, closed_a = targets[a]
                tab_a = T[st, :, a_start - t0 : a_start - t0 + Ka].astype(np.float64)
                old_a = np.einsum('nq,nqk->nk', Q[query], tab_a)
                ratio_a = anchor_probs[a][query] / np.maximum(old_a, 1e-12)
                term = np.log(np.maximum(np.einsum('nqk,nk->nq', tab_a, ratio_a), 1e-12))
                term[closed_a[query]] = 0.0
                ev_cache[key] = term
            return ev_cache[key]

        for name in cascade_names:
            if time.monotonic() >= deadline:
                return {}, {'status': 'window_closed', 'seconds': time.monotonic() - started}
            j = work[name]
            K = j['K']
            y_full = j['y']
            pool = [a for a in (anchors if name in post_set else list(anchors) + list(post_items)) if a in anchor_probs and a != name]
            if not pool:
                continue
            prio = set(priority_sources(name, items, work))
            for d in (PAIRS, EXTRA_PAIRS, V127_PAIRS):
                if name in d:
                    prio.add(d[name])
            shortlist = [a for a in pool if a in prio]
            if chain is not None:
                for a in chain.informative(name, fit, count=6):
                    if a in pool and a not in shortlist:
                        shortlist.append(a)
                for a in chain.switches(name, fit, count=2):
                    if a in pool and a not in shortlist:
                        shortlist.append(a)
            if len(shortlist) < 6:
                for a in pool:
                    if a not in shortlist:
                        shortlist.append(a)
                    if len(shortlist) >= 10:
                        break
            tracer = name.startswith('tr_') and 'Not answered' in items[name]['values']
            na_idx = items[name]['values'].index('Not answered') if tracer else -1
            fit_ans = fit[(y_full[fit] >= 0) & (y_full[fit] != na_idx) & (y_full[fit] != K - 1)] if tracer else fit
            scored = []
            for a in shortlist:
                ya = work[a]['y']
                Ka = work[a]['K']
                mi_all = adjusted_mi(ya[fit], Ka, y_full[fit], K) if np.all(ya[fit] >= 0) else 0.0
                mi_ans = adjusted_mi(ya[fit_ans], Ka, y_full[fit_ans], K) if (tracer and len(fit_ans) >= 25 and np.all(ya[fit_ans] >= 0)) else 0.0
                score = max(mi_all, mi_ans, 0.002 if a in prio else 0.0) * (1.5 if a in prio else 1.0)
                if score > 0:
                    scored.append((score, a))
            scored.sort(key=lambda x: (-x[0], x[1]))
            top = scored[:4]
            if not top:
                continue
            w_sum = sum(s for s, _ in top)
            weights_a = [(a, s / w_sum) for s, a in top]
            if fi == 0:
                top_by_item[name] = [a for _, a in top]
            a_t, Kt, closed_t = targets[name]
            w_models = w_by_item.get(name)
            wf = w_models[fi] if (w_models is not None and fi < len(w_models) and len(w_models[fi]) == len(kind_order)) else np.full(len(kind_order), 1.0 / len(kind_order))
            orig_mix = np.zeros((len(query), Kt), np.float64)
            cond_by_gamma = {g: np.zeros((len(query), Kt), np.float64) for g in gammas}
            for ki, mk in enumerate(kind_order):
                w_mk = float(wf[ki])
                if w_mk <= 1e-8:
                    continue
                Q, T = models_by_kind[mk][int(f)]
                st = np.zeros(len(query), int) if len(T) == 1 else strata[query]
                tab_t = T[st, :, a_t - t0 : a_t - t0 + Kt].astype(np.float64)
                orig_mix += w_mk * np.einsum('nq,nqk->nk', Q[query], tab_t)
                ev = np.zeros((len(query), Q.shape[1]), np.float64)
                for a, wa in weights_a:
                    ev += wa * get_ev(mk, a)
                log_q = np.log(np.maximum(Q[query], 1e-30))
                for g in gammas:
                    L = log_q + g * ev
                    L -= L.max(1, keepdims=True)
                    P = np.exp(L)
                    P /= P.sum(1, keepdims=True)
                    cond_by_gamma[g] += w_mk * np.einsum('nq,nqk->nk', P, tab_t)
            target_base = np.maximum((base_found[name][1][query].astype(np.float64) - MIX / Kt) / (1.0 - MIX), 0.0)
            for g in gammas:
                pred = target_base * cond_by_gamma[g] / np.maximum(orig_mix, 1e-12)
                pred /= np.maximum(pred.sum(1, keepdims=True), 1e-30)
                pred[closed_t[query]] = 0.0
                pred[closed_t[query], -1] = 1.0
                pred_m = ((1.0 - MIX) * pred + MIX / Kt).astype(np.float32)
                o, p = cands[str(g)][name]
                o[hit] = pred_m[:n_val]
                p[rest] += pred_m[n_val:] / len(folds)
                if tracer and event is not None:
                    ev_oof, ev_full, rand_mask = event
                    e_q = np.r_[ev_oof[rows[hit]], ev_full[rest]]
                    r_q = rand_mask[query]
                    cond = pred.copy()
                    cond[:, [na_idx, Kt - 1]] = 0.0
                    s_c = cond.sum(1, keepdims=True)
                    cond = np.divide(cond, s_c, out=np.full_like(cond, 1.0 / max(1, Kt - 2)), where=s_c > 1e-12)
                    cond[:, [na_idx, Kt - 1]] = 0.0
                    e_clip = np.clip(e_q, 0.0, 1.0)
                    coupled = cond * e_clip[:, None]
                    coupled[:, na_idx] = 1.0 - e_clip
                    coupled[~r_q] = 0.0
                    coupled[~r_q, Kt - 1] = 1.0
                    coupled_m = ((1.0 - MIX) * coupled + MIX / Kt).astype(np.float32)
                    oe, pe = cands['ev_' + str(g)][name]
                    oe[hit] = coupled_m[:n_val]
                    pe[rest] += coupled_m[n_val:] / len(folds)
    for ck, mp in list(cands.items()):
        if ck.startswith('ev_'):
            for name in list(mp):
                if not (name.startswith('tr_') and 'Not answered' in items[name]['values']):
                    mp.pop(name, None)
        for o, p in mp.values():
            p[rows] = o
    gated, chosen = guarded({k: base_found[k] for k in cascade_names}, cands, work, fold, margin=0.0002)
    admitted = {}
    for name, (o, p) in gated.items():
        if chosen.get(name):
            admitted[name] = (o, p)
    return admitted, {
        'seconds': round(time.monotonic() - started, 3),
        'offered': len(cascade_names),
        'admitted': len(admitted),
        'chosen': {k: v for k, v in chosen.items() if v},
        'top_anchors': top_by_item,
    }


def _direct_effects_fast(O, blocks, n_given_cols, Xg, rows, log_g, lw, strata, deadline, pairs=(), rank=1, refresh=False):
    """Returns (target offsets per cell: cells x target columns, row -> cell, cell stratum).

    P(answer | node, cell) = T * w / sum(T * w) per block with w = exp(offset), so
    an iteration rescales the fixed tables instead of recomputing a softmax.
    pairs: GIVEN column index pairs that also take a joint-level offset ( U1).
    rank > 1 keeps every noise-cleared component of a pair table ( U2);
    refresh re-reads the training posterior under the fitted offsets and refits once."""
    Ot = O[rows]
    m, D = Ot.shape
    G, Qn = log_g.shape[0], log_g.shape[1]
    parts = [np.arange(m)] if G == 1 else [np.flatnonzero(strata[rows] == s) for s in range(G)]
    L = np.empty((m, Qn), np.float32)
    for s, idx in enumerate(parts):
        if len(idx):
            L[idx] = Ot[idx] @ log_g[s].T
    L += lw
    L -= L.max(1, keepdims=True)
    R = np.exp(np.maximum(L, -80.0)).astype(np.float64)
    R /= R.sum(1, keepdims=True)
    cells, inv = np.unique(Xg, axis=0, return_inverse=True)
    inv = inv.ravel()
    C = len(cells)
    t0 = n_given_cols
    tb = [(a - t0, b - t0) for a, b in blocks if a >= t0]
    Dt = D - t0
    starts = np.array([a for a, _ in tb])
    # A cell's stratum is the stratum of any of its rows (strata are GIVEN codes).
    stratum = np.zeros(C, np.int64)
    if G > 1:
        stratum[inv] = strata
    c_rows = inv[rows]
    seen = np.flatnonzero(np.bincount(c_rows, minlength=C) > 0)
    where = np.full(C, -1, np.int64)
    where[seen] = np.arange(len(seen))
    cr = where[c_rows]
    Cs = len(seen)
    Yt = Ot[:, t0:]
    ii, dd = np.nonzero(Yt)
    Ns = np.bincount(cr[ii] * Dt + dd, minlength=Cs * Dt).reshape(Cs, Dt)
    bo = np.add.reduceat(Yt, starts, axis=1) > 0
    ib, bb = np.nonzero(bo)
    n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
    Ts = _cell_tables(log_g, stratum[seen], t0, tb)
    s_cells = [slice(None)] if G == 1 else [np.flatnonzero(stratum[seen] == s) for s in range(G)]
    ridge = [EFFECT_RIDGE] * cells.shape[1]
    if pairs:
        cells = np.column_stack([cells, _pair_levels(cells, pairs)])
        ridge += [PAIR_RIDGE] * len(pairs)
    cs = cells[seen]
    J = cs.shape[1]
    # Sparse one-hot sufficient statistics: each respondent cell contributes once per level.
    E = [csr_matrix((np.ones(Cs, np.float32), (cs[:, j], np.arange(Cs))), shape=(int(cells[:, j].max()) + 1, Cs)) for j in range(J)]
    beta = [np.zeros((E[j].shape[0], Dt)) for j in range(J)]
    E_all = vstack(E, format='csr')
    design = E_all.T.tocsr()
    h = np.empty((Cs, Dt), np.float32)
    def node_counts(n):
        return [[np.ascontiguousarray(n[sc, b, :].T) for b in range(len(tb))]
                for sc in s_cells]
    ns = node_counts(n)

    sweep_cache = {}
    def sweep(update):
        update = tuple(update)
        K = len(update) if v68_config.V115_W2 else J
        if update not in sweep_cache:
            Eu = vstack([E[j] for j in update], format='csr')
            sweep_cache[update] = (Eu, Eu @ Ns)
        Eu, counts = sweep_cache[update]
        for _ in range(EFFECT_ITERS):
            if time.monotonic() >= deadline:
                raise _Expired()
            w = np.exp(design @ np.concatenate(beta).astype(np.float32)).astype(np.float32)
            for s, sc in enumerate(s_cells):
                ws = np.ascontiguousarray(w[sc].T)
                for b, (a, e) in enumerate(tb):
                    wb = ws[a:e]
                    T = Ts[s][b]
                    den = T @ wb
                    h[sc, a:e] = (wb * (T.T @ (ns[s][b] / den).astype(np.float32))).T
            eh = Eu @ h
            eg = counts - eh
            pos = 0
            for j in update:
                # Jensen over the columns this sweep moves: curvature K h keeps the joint step a majorizer.
                end = pos + len(beta[j])
                beta[j] += np.clip((eg[pos:end] - ridge[j] * beta[j]) / (K * eh[pos:end] + ridge[j]), -1.0, 1.0)
                pos = end

    def fit():
        if pairs and PAIR_RANK and rank > 1:
            #  U2: main effects first, the free pair tables on what they leave,
            # so a pair with no signal gates to zero without moving the main effects.
            main = range(J - len(pairs))
            sweep(main)
            sweep(range(J - len(pairs), J))
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(main)
            return
        sweep(range(J))
        if pairs and PAIR_RANK:
            # A pair's free table is mostly noise; keep its shrunk rank-one part (one
            # score per joint level times one direction over every target option),
            # then refit the main effects around it.
            rows_per_cell = np.bincount(cr, minlength=Cs).astype(np.float64)
            for j in range(J - len(pairs), J):
                beta[j] = _rank_one(beta[j], E[j] @ rows_per_cell) if rank <= 1 else _rank_gated(beta[j], E[j] @ rows_per_cell, rank)
            sweep(range(J - len(pairs)))

    fit()
    if refresh:
        # EM's next E step: n came from the trait-only posterior; re-read it with
        # the offsets inside the target emissions, then refit from where we are.
        w = np.exp(sum(beta[j][cs[:, j]] for j in range(J)))
        L = np.empty((m, Qn), np.float64)
        for s, idx in enumerate(parts):
            if len(idx):
                L[idx] = Ot[idx, :t0] @ log_g[s, :, :t0].T
        L += lw
        # log(w[cell, answer]) is constant across quadrature nodes and cancels
        # in the posterior. Preserve the original float32 emission rounding,
        # but accumulate the refresh in float64, as in the cell-expanded form.
        for s, sc in enumerate(s_cells):
            for b, (a, e) in enumerate(tb):
                hit = (dd >= a) & (dd < e)
                if G > 1:
                    hit &= stratum[c_rows[ii]] == s
                rr, answer = ii[hit], dd[hit] - a
                T = Ts[s][b]
                den = np.empty((Cs, Qn), np.float64)
                den[sc] = np.log(np.maximum(w[sc, a:e] @ T.T, 1e-300))
                lt = np.log(T.astype(np.float64))[:, answer].T - den[cr[rr]]
                np.add.at(L, rr, lt * Yt[rr, dd[hit]][:, None])
        L -= L.max(1, keepdims=True)
        R = np.exp(np.maximum(L, -80.0))
        R /= R.sum(1, keepdims=True)
        n = _cell_sums(R[ib], cr[ib] * len(tb) + bb, Cs * len(tb)).reshape(Cs, len(tb), Qn)
        ns = node_counts(n)
        fit()
    off = sum(beta[j][cells[:, j]] for j in range(J))
    return off, inv, stratum



def _direct_effects(*args, **kwargs):
    impl = _direct_effects_fast if v68_config.V113_W1 or v68_config.V115_W1 or v68_config.V115_W3 else _direct_effects_reference
    return impl(*args, **kwargs)
