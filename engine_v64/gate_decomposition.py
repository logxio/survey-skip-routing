"""Fold-isolated asked and conditional-answer candidates for gated items."""

import numpy as np

from .base import _fit_table, _mi_drivers, _safe_table_config
from .chain_features import BudgetExpired, check_clock
from .propagate_structural_gates import _wave_two_source, _wave_one_index, _encode_wave_one_zero
from .v64_config import V128_K1


def _stage(tables, X, singles, levels, y, rows, query, width, deadline,
           n_raw_given, is_deep_hierarchy=True, extra=None, pre_design=None):
    """Select this stage's GIVEN features and EB prior on its own labels."""
    check_clock(deadline)
    if not len(rows):
        return np.full((len(query), width), 1.0 / width), None, ()
    train = np.zeros(len(X), bool)
    train[rows] = True
    if is_deep_hierarchy:
        order = _mi_drivers(X[:, :n_raw_given], y, rows,
                            levels[:n_raw_given], count=3)
        config = (order, float(np.clip(np.sqrt(len(rows)) / 10., 1., 30.)))
    else:
        config = _safe_table_config(X, rows, n_raw_given)
    _, full, _ = _fit_table(tables, y, train, width, n_raw_given,
                            check_clock=lambda: check_clock(deadline),
                            fixed_config=config)
    selected = np.arange(n_raw_given,dtype=np.int64) if is_deep_hierarchy else np.asarray(config[0],dtype=np.int64)
    if pre_design is not None:
        design = pre_design
    else:
        offsets = np.r_[0, np.cumsum(levels)]
        columns = (np.concatenate([np.arange(offsets[f], offsets[f + 1])
                                   for f in selected]) if len(selected)
                   else np.empty(0, dtype=np.int64))
        has_extra = extra is not None and extra.shape[1] > 0
        if not len(columns) and not has_extra:
            return full[query], None, ()
        design = singles[:, columns]
        if has_extra:
            from scipy import sparse
            design = sparse.hstack((design, sparse.csr_matrix(extra))).tocsr()
    classes = np.unique(y[rows])
    if len(classes) < 2:
        return full[query], None, tuple(int(f) for f in selected)
    from sklearn.linear_model import LogisticRegression
    max_it = (35 if len(rows) > 2400 else 45) if (V128_K1 and is_deep_hierarchy) else 60
    lr_tol = (2e-3 if len(rows) > 2400 else 1e-3) if (V128_K1 and is_deep_hierarchy) else 1e-3
    model = LogisticRegression(C=.3, max_iter=max_it, tol=lr_tol)
    check_clock(deadline)
    model.fit(design[rows], y[rows])
    check_clock(deadline)
    logits = np.zeros((len(query), width), float)
    logits[:, model.classes_.astype(int)] = model.predict_proba(design[query])
    check_clock(deadline)
    return full[query], logits, tuple(int(f) for f in selected)


def _predict(tables, X, singles, levels, y, fit, query, k, deadline,
             n_raw_given, is_deep_hierarchy=True, extra=None, raw_singles=None):
    check_clock(deadline)
    pre_design = None
    if V128_K1 and is_deep_hierarchy and raw_singles is not None:
        has_extra = extra is not None and extra.shape[1] > 0
        if has_extra:
            from scipy import sparse
            pre_design = sparse.hstack((raw_singles, sparse.csr_matrix(extra))).tocsr()
        else:
            pre_design = raw_singles
    asked = (y != k - 1).astype(np.int64)
    gate_eb, gate_logit, gate_features = _stage(
        tables, X, singles, levels, asked, fit, query, 2, deadline,
        n_raw_given, is_deep_hierarchy, extra, pre_design=pre_design)
    answers = fit[asked[fit] == 1]
    answer_eb, answer_logit, answer_features = _stage(
        tables, X, singles, levels, y, answers, query, k - 1, deadline,
        n_raw_given, is_deep_hierarchy, extra, pre_design=pre_design)

    table = _compose(gate_eb, answer_eb)
    gate = gate_eb if gate_logit is None else .8 * gate_logit + .2 * gate_eb
    answer = answer_eb if answer_logit is None else .8 * answer_logit + .2 * answer_eb
    logit = _compose(gate, answer)
    return table, logit, (gate_features, answer_features)


def _compose(gate, answer):
    out = np.empty((len(gate), answer.shape[1] + 1), float)
    out[:, :-1] = gate[:, 1, None] * answer
    out[:, -1] = gate[:, 0]
    return out


def _direct_table(tables, X, y, rows, source, rest, k, deadline,
                  n_raw_given):
    """Shallow-survey tables with the s4b fixed-config fold exclusion."""
    fit_source = rows[source]
    asked_rows = fit_source[y[fit_source] != k - 1]
    asked = (y != k - 1).astype(np.int64)
    gate_train = np.zeros(len(X), bool)
    gate_train[fit_source] = True
    answer_train = np.zeros(len(X), bool)
    answer_train[asked_rows] = True
    gate_oof, gate_full, _ = _fit_table(
        tables, asked, gate_train, 2, n_raw_given,
        check_clock=lambda: check_clock(deadline),
        fixed_config=_safe_table_config(X, fit_source, n_raw_given),
        oof_query=rows)
    answer_oof, answer_full, _ = _fit_table(
        tables, y, answer_train, k - 1, n_raw_given,
        check_clock=lambda: check_clock(deadline),
        fixed_config=_safe_table_config(X, asked_rows, n_raw_given),
        oof_query=rows)
    full = np.zeros((len(X), k), np.float32)
    full[rest] = _compose(gate_full[rest], answer_full[rest])
    return _compose(gate_oof, answer_oof).astype(np.float32), full


def candidate(name, item, items, given, X, levels, singles, tables,
              job, fold, deadline, n_raw_given=None,
              is_deep_hierarchy=True, chain=None, raw_singles=None):
    """Return both complete five-fold OOF arms and full-fit hidden predictions."""
    if not item.get('gate'):
        return None
    rows = np.flatnonzero(job['train'])
    rest = np.flatnonzero(~job['train'])
    source = _wave_two_source(name, items, X, given, rows)
    if source is None or not source.any():
        return None
    fit_source = rows[source]
    y, k = job['y'], job['K']
    if k < 2:
        return None
    if n_raw_given is None:n_raw_given=X.shape[1]
    if V128_K1 and is_deep_hierarchy and raw_singles is None and n_raw_given > 0:
        raw_width = int(np.sum(levels[:n_raw_given]))
        raw_singles = singles if singles.shape[1] == raw_width else singles[:, :raw_width]
    fold_rows = fold[rows]
    if is_deep_hierarchy:
        oof_table=np.zeros((len(rows),k),np.float32)
        full_table=np.zeros((len(X),k),np.float32)
    else:
        oof_table,full_table=_direct_table(
            tables,X,y,rows,source,rest,k,deadline,n_raw_given)
    oof_logit=np.zeros_like(oof_table)
    full_logit = np.zeros_like(full_table)
    if V128_K1 and is_deep_hierarchy:
        n_folds = int(fold_rows.max()) + 1
        used_folds = 0
        for f in range(n_folds):
            check_clock(deadline)
            val = rows[fold_rows == f]
            if not len(val):
                continue
            fit = fit_source[fold[fit_source] != f]
            if not len(fit):
                return None
            extra = (chain.extras(name, rows[fold_rows != f])
                     if chain is not None and name in chain.children else None)
            q = np.concatenate((val, rest)) if len(rest) else val
            table, logit, _ = _predict(tables, X, singles, levels, y,
                                     fit, q, k, deadline, n_raw_given,
                                     is_deep_hierarchy, extra, raw_singles=raw_singles)
            mask = fold_rows == f
            oof_table[mask] = table[:len(val)]
            oof_logit[mask] = logit[:len(val)]
            if len(rest):
                full_table[rest] += table[len(val):]
                full_logit[rest] += logit[len(val):]
            used_folds += 1
        if used_folds > 1 and len(rest):
            full_table[rest] /= float(used_folds)
            full_logit[rest] /= float(used_folds)
    else:
        for f in range(int(fold_rows.max()) + 1):
            check_clock(deadline)
            val = rows[fold_rows == f]
            if not len(val):
                continue
            fit = fit_source[fold[fit_source] != f]
            if not len(fit):
                return None
            extra = (chain.extras(name, rows[fold_rows != f])
                     if chain is not None and name in chain.children else None)
            table, logit, _ = _predict(tables, X, singles, levels, y,
                                     fit, val, k, deadline,n_raw_given,
                                     is_deep_hierarchy, extra)
            mask = fold_rows == f
            if is_deep_hierarchy:oof_table[mask] = table
            oof_logit[mask] = logit
        check_clock(deadline)
        if len(rest):
            extra = (chain.extras(name, rows)
                     if chain is not None and name in chain.children else None)
            table, logit, _ = _predict(tables, X, singles, levels, y,
                                     fit_source, rest, k, deadline,n_raw_given,
                                     is_deep_hierarchy, extra)
            if is_deep_hierarchy:full_table[rest] = table
            full_logit[rest] = logit
    wave_one = _wave_one_index(name, items, given)
    if wave_one is not None:
        wave = given.index('wave')
        _encode_wave_one_zero(oof_table, X[rows, wave], wave_one)
        _encode_wave_one_zero(oof_logit, X[rows, wave], wave_one)
        full_table[rest] = _encode_wave_one_zero(full_table[rest], X[rest, wave], wave_one)
        full_logit[rest] = _encode_wave_one_zero(full_logit[rest], X[rest, wave], wave_one)
    check_clock(deadline)
    return [('gate_table', oof_table, full_table),
            ('gate_logit', oof_logit, full_logit)]
