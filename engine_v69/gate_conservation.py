"""Restore questionnaire routing after all item-wise posterior updates."""
import numpy as np


def project(child, opened):
    out = np.asarray(child, dtype=float).copy()
    total = out[:, :-1].sum(1, keepdims=True)
    conditional = np.divide(out[:, :-1], total, out=np.full_like(out[:, :-1], 1. / (out.shape[1]-1)), where=total > 0.)
    opened = np.clip(opened, 0., 1.)
    out[:, :-1] = conditional * opened[:, None]
    out[:, -1] = 1. - opened
    return out


def apply(items, given, X, work, final, oofs):
    trace = {}
    pending = {name for name in work if isinstance(items[name].get('gate'), dict)}
    while pending:
        ready = [name for name in pending if items[name]['gate']['parent'] not in pending]
        if not ready:
            raise ValueError('cyclic questionnaire gate')
        for name in ready:
            pending.remove(name)
            gate = items[name]['gate']; parent = gate['parent']
            cols = [items[parent]['values'].index(v) for v in gate['observed_if']]
            j = work[name]; rows = np.flatnonzero(j['train']); rest = np.flatnonzero(~j['train'])
            if parent in given:
                opened = np.isin(X[:, given.index(parent)], cols).astype(float)
            elif parent in final and parent in oofs:
                pj = work[parent]
                full_parent = np.asarray(final[parent]).copy()
                full_parent[pj['train']] = oofs[parent]
                opened = full_parent[:, cols].sum(1)
            else:
                raise ValueError('missing parent probabilities: ' + parent)
            old = np.asarray(final[name][rest])
            before = float(np.max(np.abs(old[:, -1] - (1.-opened[rest])))) if len(rest) else 0.
            final[name][rest] = project(old, opened[rest])
            oofs[name] = project(oofs[name], opened[rows])
            after = float(np.max(np.abs(final[name][rest, -1] - (1.-opened[rest])))) if len(rest) else 0.
            trace[name] = {'parent': parent, 'max_difference_before': before, 'max_difference_after': after}
    return trace
