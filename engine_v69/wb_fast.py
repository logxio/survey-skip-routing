import ortho_config as _ortho
"""Sparse damped Newton softmax; all sufficient statistics use fitting rows."""
import numpy as np
from scipy import sparse
from scipy.special import expit, logsumexp
from .chain_features import check_clock

ENABLE_SOLVER = _ortho.V65_K1_FAST_STAGES
ENABLE_STAGES = False
ENABLE_RCT = _ortho.V65_K2_RCT
ENABLE_MULTITASK = _ortho.V65_K3_MULTITASK


def fit_predict(X, y, Q, C=.3, deadline=float('inf'), tol=5e-4):
    X = sparse.csr_matrix(X, dtype=np.float64)
    Q = sparse.csr_matrix(Q, dtype=np.float64)
    classes, y = np.unique(y, return_inverse=True)
    k = len(classes)
    if k == 1:
        return np.ones((len(Q.indptr)-1, 1)), classes
    n, d = X.shape
    binary = k == 2
    m = 1 if binary else k
    reg = 1. / (float(C) * n)
    B = np.zeros((d+1, m))
    freq = (np.bincount(y, minlength=k) + .5) / (n + .5*k)
    B[-1] = np.log(freq[1]/freq[0]) if binary else np.log(freq) - np.log(freq).mean()
    Y = y[:, None].astype(float) if binary else np.eye(k)[y]
    X2 = X.multiply(X)

    def forward(b):
        z = X @ b[:-1] + b[-1]
        if binary:
            p = expit(z)
            loss = np.mean(np.logaddexp(0., z) - Y*z)
        else:
            lse = logsumexp(z, axis=1, keepdims=True)
            p = np.exp(z-lse)
            loss = np.mean(lse[:, 0] - z[np.arange(n), y])
        return p, loss + .5*reg*np.sum(b[:-1]**2)

    p, loss = forward(B)
    for outer in range(20):
        check_clock(deadline)
        residual = (p-Y)/n
        grad = np.vstack((X.T @ residual + reg*B[:-1], residual.sum(0)))
        if np.max(np.abs(grad)) <= tol:
            break
        curvature = p*(1.-p)
        diag = np.vstack((X2.T @ curvature / n + reg, curvature.mean(0))) + 1e-8
        def hv(v):
            z = X @ v[:-1] + v[-1]
            r = curvature*z if binary else p*(z - np.sum(p*z, axis=1, keepdims=True))
            return np.vstack((X.T @ r / n + reg*v[:-1], r.mean(0))) + 1e-8*v
        delta = np.zeros_like(B)
        r = -grad.copy(); z = r/diag; direction = z.copy()
        rz = np.sum(r*z)
        threshold = max(1e-12, .01*rz)
        for inner in range(16):
            h = hv(direction); curvature_d = np.sum(direction*h)
            if curvature_d <= 0: break
            step = rz/curvature_d
            delta += step*direction; r -= step*h
            z = r/diag; new_rz = np.sum(r*z)
            if new_rz <= threshold: break
            direction = z + (new_rz/rz)*direction; rz = new_rz
        slope = np.sum(grad*delta)
        if slope >= 0: delta = -grad/diag; slope = np.sum(grad*delta)
        step = 1.
        for line in range(16):
            next_p, next_loss = forward(B+step*delta)
            if next_loss <= loss + 1e-4*step*slope: break
            step *= .5
        B += step*delta; p, loss = next_p, next_loss
    z = Q @ B[:-1] + B[-1]
    if binary:
        p = expit(z[:, 0]); return np.column_stack((1-p,p)), classes
    return np.exp(z-logsumexp(z,axis=1,keepdims=True)), classes
