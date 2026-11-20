"""Fourth- through seventh-order tables with fold-isolated EB priors.

The same visible-row folds used by the base table exclude each OOF row from
every count.  Only the full visible set is used for hidden-row predictions.
"""

import time

import numpy as np


def _loss(p, y, floor=0.001):
    k = p.shape[1]
    return -float(np.log(floor + (1.0 - k * floor) *
                         p[np.arange(len(y)), y]).mean())


def _shrink(tables, features, y, train, k, alpha, prior_oof, prior_all):
    counts_oof, counts_all = tables.counts(tables.ids(features), y, train, k)
    oof = (counts_oof + alpha * prior_oof) / (
        counts_oof.sum(axis=1, keepdims=True) + alpha)
    full = (counts_all + alpha * prior_all) / (
        counts_all.sum(axis=1, keepdims=True) + alpha)
    return oof, full


def fit(tables, job, n_given, deadline=float('inf')):
    """Return (OOF, full-frame, diagnostics), or None if folds do not agree.

    The selected fine table replaces the old table branch before the later
    convex stack.  The old table is its empirical Bayes prior, so a fine cell
    only gets weight proportional to its support.
    """
    if n_given < 4:
        return None
    train, y, k = job["train"], job["y"], job["K"]
    target = y[train]
    base_oof, base_all = job["oof"][0], job["all"][0]
    base_loss = _loss(base_oof, target)
    ranking = list(np.argsort(-job["gains"]))
    core = ranking[:3]
    halves = (tables.fold[train] % 2 == 0,
              tables.fold[train] % 2 == 1)

    def accepted(candidate):
        gain = base_loss - _loss(candidate, target)
        return (gain > 0.0002 and all(
            _loss(base_oof[h], target[h]) > _loss(candidate[h], target[h])
            for h in halves if h.any()))

    best = None
    best_loss = base_loss
    path = (base_oof, base_all, tuple(core), None)
    four_alpha = None
    # The lowest-loss table at each depth supplies the next depth's prior.
    # Acceptance is independent: an unhelpful lower order can still carry a
    # useful interaction at six or seven features, while a lower-order winner
    # remains available if deeper candidates do not pass the fold checks.
    for depth in range(4, min(n_given, 7) + 1):
        prior_oof, prior_all, prior_features, _ = path
        chosen = None
        chosen_loss = float('inf')
        for extra in ranking:
            if extra in prior_features:
                continue
            features = tuple(sorted((*prior_features, extra)))
            for alpha in (5.0, 20.0, 80.0):
                if time.monotonic() >= deadline:
                    break
                oof, full = _shrink(tables, features, y, train, k,
                                    alpha, prior_oof, prior_all)
                loss = _loss(oof, target)
                if loss < chosen_loss:
                    chosen_loss = loss
                    chosen = (oof, full, features, alpha)
                if loss < best_loss and accepted(oof):
                    best_loss = loss
                    best = (oof, full, features, alpha)
            if time.monotonic() >= deadline:
                break
        if chosen is None or time.monotonic() >= deadline:
            break
        path = chosen
        if depth == 4:
            four_alpha = chosen[3]

    if best is None:
        return None

    oof, full, features, alpha = best
    return oof, full, {"depth": len(features), "features": features,
                       "alpha": alpha, "oof_gain": base_loss - best_loss,
                       "depth4_alpha": four_alpha}
