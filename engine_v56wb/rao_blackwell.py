"""Response-state Dirichlet transitions with fold-local risk estimation."""
import numpy as np


def normalize(p):
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=-1, keepdims=True)


def transition(states, cardinality, y, groups, fit, query, probability, k, alpha=8.):
    fit = fit[(states[fit] >= 0) & (y[fit] >= 0)]
    ng = int(groups.max()) + 1
    count = np.bincount(states[fit] * k + y[fit], minlength=cardinality*k).reshape(cardinality, k)
    marginal = (count.sum(0) + 1./k) / (len(fit) + 1.)
    global_t = (count + alpha*marginal) / (count.sum(1, keepdims=True) + alpha)
    grouped = np.bincount((groups[fit]*cardinality + states[fit])*k + y[fit],
                          minlength=ng*cardinality*k).reshape(ng, cardinality, k)
    table = (grouped + alpha*global_t[None]) / (grouped.sum(2, keepdims=True) + alpha)
    # A per-row copy of the joint-state table grows as N***Kchild.
    # Grouped matrix multiplication keeps only the small shared table.
    out = np.empty((len(query), k), float)
    for group in np.unique(groups[query]):
        mask = groups[query] == group
        out[mask] = probability[mask] @ table[group]
    return normalize(out)


def adjusted_mi(a, ka, b, kb):
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if not len(a):
        return 0.
    joint = np.bincount(a*kb+b, minlength=ka*kb).reshape(ka, kb)/len(a)
    independent = joint.sum(0)[None, :] * joint.sum(1)[:, None]
    ok = joint > 0
    return float((joint[ok]*np.log(joint[ok]/independent[ok])).sum() - (ka-1)*(kb-1)/(2.*len(a)))


def convex_weights(predictions, y):
    """Simplex maximum likelihood with the official probability floor."""
    a = np.asarray(predictions)
    if len(y) == 0:
        return np.full(len(a), 1./len(a))
    truth = .001 + (1.-a.shape[2]*.001)*a[:, np.arange(len(y)), y]
    w = np.full(len(a), 1./len(a))
    for _ in range(32):
        w *= np.mean(truth / np.maximum(w @ truth, 1e-12), axis=1)
        w /= w.sum()
    return w


def marginal_pair(a, b):
    return (a[:, :, None]*b[:, None, :]).reshape(len(a), -1)
