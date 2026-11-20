import ortho_config as _ortho
"""Parent-child skip-mass coupling for gated items whose parent is a target.

A gated child's never-asked mass equals the parent's closed-route mass under
the schema routing. The tail calibration stages (finite-population raking,
residual projections) move each item separately, so the two drift apart. The
coupled child keeps its answered distribution and replaces the skip mass by a
convex blend of its own estimate and the parent's closed mass; the blend is
picked on one fold half and scored on the other, then admitted per item only
if it beats the incumbent on the full OOF set and on both halves.
"""
import time
import numpy as np
from .base import _nll
from .cross_round import dominates

ENABLED = _ortho.V72_K2
GRID = (0.25, 0.5, 0.75, 0.9, 1.0)


def couple(child, parent, open_cols, w):
    closed = 1.0 - parent[:, open_cols].sum(1)
    na = np.clip((1.0 - w) * child[:, -1] + w * closed, 1e-6, 1.0 - 1e-6)
    answered = child[:, :-1]
    total = answered.sum(1, keepdims=True)
    shape = np.where(total > 1e-12, answered / np.maximum(total, 1e-300), 1.0 / answered.shape[1])
    out = np.empty_like(child)
    out[:, :-1] = shape * (1.0 - na)[:, None]
    out[:, -1] = na
    return out


def _pick(child, parent, open_cols, y, K, mask):
    return min(GRID, key=lambda w: _nll(couple(child[mask], parent[mask], open_cols, w), y[mask], K))


def apply(schema, work, fold, final, oofs, deadline):
    info = {}
    if not ENABLED or len(work) != 12:
        return info
    items = schema['items']
    for name, j in work.items():
        gate = items[name].get('gate')
        if not isinstance(gate, dict) or gate.get('parent') not in work or time.monotonic() >= deadline:
            continue
        parent = gate['parent']
        if name not in oofs or parent not in oofs or not np.array_equal(work[parent]['train'], j['train']):
            continue
        values = items[parent]['values']
        open_cols = [values.index(v) for v in gate.get('observed_if', ()) if v in values]
        if not open_cols:
            continue
        rows = np.flatnonzero(j['train'])
        rest = np.flatnonzero(~j['train'])
        K = j['K']
        y = j['y'][rows]
        half = fold[rows] % 2 == 0
        old, par = oofs[name], oofs[parent]
        if old.shape != (len(rows), K) or par.shape[0] != len(rows):
            continue
        # Cross-half selection: each half is coupled with the weight the other half chose.
        w_even, w_odd = _pick(old, par, open_cols, y, K, ~half), _pick(old, par, open_cols, y, K, half)
        trial = old.copy()
        trial[half] = couple(old[half], par[half], open_cols, w_even)
        trial[~half] = couple(old[~half], par[~half], open_cols, w_odd)
        w = _pick(old, par, open_cols, y, K, np.ones(len(rows), bool))
        gain = float(_nll(old, y, K) - _nll(trial, y, K))
        accepted = bool(dominates(trial, old, y, K, half))
        if accepted:
            final[name][rest] = couple(final[name][rest], final[parent][rest], open_cols, w)
            oofs[name] = couple(old, par, open_cols, w)
        info[name] = {'parent': parent, 'weight': float(w), 'half_weights': [float(w_even), float(w_odd)],
                      'crossfit_gain': gain, 'accepted': accepted}
    return info
