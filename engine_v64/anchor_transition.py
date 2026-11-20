"""Pass-2 transition arm: (declared parent, top Pass-1 anchor) -> child answer.

For each outer fold the anchor is chosen by MI on the fit rows only, the
count table uses the fit rows' true parent and anchor answers, and it is
applied to the parent's and anchor's stacked Pass-1 probabilities on the
held-out rows.  The joint table shrinks to the anchor-only table, which
shrinks to the fit-row marginal.
"""

import numpy as np
from .chain_features import check_clock
from .propagate_structural_gates import _parents


def _table(keys, width, labels, k, alpha, prior):
    counts = np.bincount(keys * k + labels, minlength=width * k).reshape(width, k).astype(float)
    return (counts + alpha * prior) / (counts.sum(1, keepdims=True) + alpha)


def _transition(chain, work, name, parent, anchor, fit, query, k, labels, alpha=8.0):
    fit_labels = labels[fit]
    marginal = np.bincount(fit_labels, minlength=k).astype(float)
    prior = (marginal + 1.0 / k) / (marginal.sum() + 1.0)
    pa = chain.probabilities(anchor)[query].astype(float) if anchor else None
    if anchor:
        ka = int(work[anchor]['K'])
        a = work[anchor]['y'][fit]
        T_a = _table(a, ka, fit_labels, k, alpha, prior)
        if parent is None:
            out = np.clip(pa, 0., None) @ T_a
        else:
            kp = int(work[parent]['K'])
            p = work[parent]['y'][fit]
            hits = np.bincount((p * ka + a) * k + fit_labels,
                               minlength=kp * ka * k).reshape(kp * ka, k).astype(float)
            # Joint cells shrink toward their anchor row of T_a.
            shrunk = (hits + alpha * np.tile(T_a, (kp, 1))) / (hits.sum(1, keepdims=True) + alpha)
            pp = np.clip(chain.probabilities(parent)[query].astype(float), 0., None)
            weights = (pp[:, :, None] * np.clip(pa, 0., None)[:, None, :]).reshape(len(query), -1)
            out = weights @ shrunk
    else:
        kp = int(work[parent]['K'])
        T_p = _table(work[parent]['y'][fit], kp, fit_labels, k, alpha, prior)
        out = np.clip(chain.probabilities(parent)[query].astype(float), 0., None) @ T_p
    total = out.sum(1, keepdims=True)
    return np.divide(out, total, out=np.tile(prior, (len(out), 1)), where=total > 0)


def candidate(name, items, work, chain, fold, deadline):
    """Five-fold OOF and full-fit hidden prediction, or None without a link."""
    job = work[name]
    y, k = job['y'], int(job['K'])
    rows = np.flatnonzero(job['train'])
    rest = np.flatnonzero(~job['train'])
    if len(rows) < 50 or len(np.unique(y[rows])) < 2:
        return None
    parent = next((p for p in _parents(name, items) if p in chain.block), None)
    oof = np.zeros((len(rows), k), np.float32)
    full = np.zeros((len(y), k), np.float32)
    row_fold = fold[rows]
    for f in np.unique(row_fold):
        check_clock(deadline)
        tr = rows[row_fold != f]
        val = rows[row_fold == f]
        top = chain.informative(name, tr, count=1)
        anchor = top[0] if top else None
        if anchor is None and parent is None:
            return None
        oof[row_fold == f] = _transition(chain, work, name, parent, anchor, tr, val, k, y)
    check_clock(deadline)
    top = chain.informative(name, rows, count=1)
    anchor = top[0] if top else None
    if anchor is None and parent is None:
        return None
    if len(rest):
        full[rest] = _transition(chain, work, name, parent, anchor, rows, rest, k, y)
    return oof, full, anchor, parent
