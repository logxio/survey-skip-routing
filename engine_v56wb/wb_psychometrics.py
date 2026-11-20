"""Nested ordinal scale coupling from visible questionnaire responses."""
import time
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from .wb_scale_data import SCALE_KEYS
from .chain_features import BudgetExpired

RIASEC = ('realistic', 'invest', 'artistic', 'social', 'enterp', 'convent')
LIKERT = ('1', '2', '3', '4', '5')


def normalize(p):
    p = np.maximum(p, 1e-8)
    return p / p.sum(axis=1, keepdims=True)


def loss(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())


def raw_scale(scores, scale):
    """Score raw responses only when the caller actually supplies raw items."""
    ids = SCALE_KEYS.get(scale + '_ADD_MAP', ())
    reverse = SCALE_KEYS.get(scale + '_REVERSE_SCALE_MAP', ())
    if not ids or any(i not in scores for i in ids):
        return None
    return np.mean([6 - np.asarray(scores[i]) if i in reverse else np.asarray(scores[i])
                    for i in ids], axis=0)


def selected(items, work):
    return [k for k in work if (k.startswith('sc_nstd_') and
             items[k]['values'][:2] == ['<2.5', '2.5-3.0']) or
            tuple(items[k]['values'][:5]) == LIKERT or
            k in ('user.preferredSector', 'user.targetSector', 'tr_target_sector',
                  'tr_riasec_target_sector', 'tr_self_efficacy', 'tr_feel_work')]


def factor_matrix(bank, train, names):
    """Rebuild all factors with outer-training labels and inner-OOF fit rows."""
    cache = bank.partition(train)
    cols = []
    expected = {}
    for name in names:
        p, _ = bank.table_probability(name, train, cache)
        values = bank.items[name]['values']
        if name.endswith('Sector') or name in ('tr_target_sector', 'tr_riasec_target_sector'):
            cols.append(p)
            continue
        valid = np.array([v != 'Not answered' for v in values] + [False])
        indices = np.flatnonzero(valid)
        ordinal = np.linspace(-1., 1., len(indices))
        mass = p[:, indices].sum(1)
        mean = (p[:, indices] @ ordinal) / np.maximum(mass, 1e-8)
        variance = (p[:, indices] @ (ordinal * ordinal)) / np.maximum(mass, 1e-8) - mean * mean
        expected[name] = mean
        cols.append(np.column_stack((mean, variance, 1 - mass)))
    # Six independent axes and the two circular contrasts encode the public
    # R-I-A-S-E-C ordering; their regression coefficients are learned per fold.
    ring = [expected.get('sc_nstd_' + k) for k in RIASEC]
    if all(v is not None for v in ring):
        six = np.column_stack(ring)
        angles = np.arange(6) * np.pi / 3
        cols.append(np.column_stack((six @ np.cos(angles), six @ np.sin(angles),
                                     six.mean(1), six.var(1))))
    for c in range(bank.n_raw_given):
        width = int(bank.levels[c])
        cols.append(np.eye(width, dtype=np.float32)[np.minimum(bank.X[:, c], width-1)])
    return np.column_stack(cols).astype(np.float64)


def _nominal(A, y, fit, query, K):
    out = np.full((len(query), K), 1e-8)
    if not len(fit):
        return normalize(out)
    classes = np.unique(y[fit])
    if len(classes) == 1:
        out[:, classes[0]] = 1.
    else:
        model = LogisticRegression(C=.1, max_iter=70, tol=2e-3)
        model.fit(A[fit], y[fit])
        out[:, model.classes_] = model.predict_proba(A[query])
    return normalize(out)


def ordinal(A, y, fit, query, values, K):
    """Proportional-odds latent score with a separate unanswered/skip model."""
    ordered = [i for i, v in enumerate(values) if v != 'Not answered']
    if len(ordered) < 3:
        return _nominal(A, y, fit, query, K)
    mapping = np.full(K, -1, dtype=int)
    mapping[ordered] = np.arange(len(ordered))
    observed = fit[mapping[y[fit]] >= 0]
    if len(observed) < 25:
        return _nominal(A, y, fit, query, K)
    # The categorical gate retains each special category, rather than imposing
    # an ordinal order on missingness or skip semantics.
    gate_y = np.where(mapping[np.maximum(y, 0)] >= 0, 0, y + 1)
    gate = _nominal(A, gate_y, fit, query, K + 1)
    B = A[observed]; z = mapping[y[observed]]; cuts = len(ordered)-1
    targets = (z[:, None] <= np.arange(cuts)).astype(float)
    empirical = np.clip((targets.sum(0)+.5)/(len(z)+1), 1e-4, 1-1e-4)
    start = np.r_[np.log(empirical/(1-empirical)), np.zeros(B.shape[1])]
    # Composite cumulative log likelihood estimates a single discrimination
    # vector. Ordered empirical thresholds initialize the shared latent model.
    def objective(theta):
        logits = theta[:cuts][None, :] - (B @ theta[cuts:])[:, None]
        err = (expit(logits)-targets)/len(z)
        value = np.sum(np.logaddexp(0, logits)-targets*logits)/len(z)
        value += .5*.03*np.dot(theta[cuts:], theta[cuts:])
        grad = np.r_[err.sum(0), -B.T @ err.sum(1)+.03*theta[cuts:]]
        return value, grad
    fitted = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options={'maxiter':55, 'ftol':1e-7})
    theta = fitted.x
    cumulative = expit(np.sort(theta[:cuts])[None, :] - (A[query] @ theta[cuts:])[:, None])
    probabilities = np.diff(np.column_stack((np.zeros(len(query)), cumulative, np.ones(len(query)))), axis=1)
    out = gate[:, 1:].copy()
    out[:, ordered] = probabilities * gate[:, :1]
    return normalize(out)


def candidate(name, bank, fold, factors, deadline):
    j = bank.work[name]; rows = np.flatnonzero(j['train']); rest = np.flatnonzero(~j['train'])
    if len(rows) < 40 or len(np.unique(fold[rows])) != 5:
        return None
    n = len(bank.X); K = j['K']; y = j['y']; values = bank.items[name]['values']
    o = np.zeros((len(rows), K), np.float32); full = np.zeros((n, K), np.float32)
    nominal = name.endswith('Sector') or name in ('tr_target_sector', 'tr_riasec_target_sector')
    for f in range(5):
        if time.monotonic() >= deadline:
            return None
        tr = rows[fold[rows] != f]; va = rows[fold[rows] == f]; query = np.r_[va, rest]
        key = (f, tr.tobytes())
        if key not in factors:
            A = factor_matrix(bank, tr, selected(bank.items, bank.work))
            mean = A[tr].mean(0); std = np.maximum(A[tr].std(0), .05)
            factors[key] = np.clip((A-mean)/std, -12, 12)
        A = factors[key]
        p = _nominal(A, y, tr, query, K) if nominal else ordinal(A, y, tr, query, values, K)
        o[fold[rows] == f] = p[:len(va)]
        full[rest] += p[len(va):]/5
    return o, full


def attach(schema, X, given, work, fold, deadline, bank):
    if 'sc_nstd_extra' not in work:
        return {}
    bank.deadline = deadline
    factors = {}; trace = {}
    targets = selected(schema['items'], work)
    for name in targets:
        if time.monotonic() >= deadline:
            break
        try:
            found = candidate(name, bank, fold, factors, deadline)
        except BudgetExpired:
            break
        if found is None:
            continue
        o, p = found; j = work[name]; y = j['y'][j['train']]; base = j['oof'][0]
        half = np.arange(len(y)) % 2
        gains = [loss(base[h], y[h])-loss(o[h], y[h]) for h in (half == 0, half == 1)]
        gain = loss(base, y)-loss(o, y)
        added = bool(gain > 1e-8 and min(gains) > 1e-8)
        trace[name] = {'oof_nll':loss(o,y), 'table_nll':loss(base,y), 'gain':gain,
                       'half_gains':gains, 'added':added, 'nested_crossfit':True}
        if added:
            j['oof'].append(o); j['all'].append(p); j['arms'].append('psychometric_irt')
    return trace
