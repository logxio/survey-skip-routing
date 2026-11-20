"""Partition-local response profiles, distilled into GIVEN-only predictions.

``fit_latent_profiles`` returns one arm per question: (OOF in job.train order,
full refit predictions for ALL rows). ``deadline`` uses time.monotonic().
X contains categorical GIVEN codes (including deterministic GIVEN crosses).
given=None uses all columns; names select the leading len(given) columns;
integer indices select those columns explicitly. No target-derived X is safe.

For an outer-fold second-stage tree, construct a fresh LatentProfileModel and
call fit(X, work, outer_train, deadline, given), then predict(X) or transform(X).
Both its training-row and validation-row features exclude outer validation
answers. Never use global OOF as the training features of an outer-fold tree.
An inner cross-fit, if desired, must be entirely inside outer_train.
"""
import time

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from given_crosses import interaction_columns


class ProfileBudgetExpired(TimeoutError):
    """A partition/scale did not finish; it must not contribute predictions."""


def _clock(deadline):
    if time.monotonic() >= deadline:
        raise ProfileBudgetExpired("latent profile deadline reached")


def _rows(train, n):
    values = np.asarray(train)
    if values.dtype == bool:
        if values.shape != (n,):
            raise ValueError("train mask must have one entry per row")
        return np.flatnonzero(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError("train must be a boolean mask or integer row indices")
    values = values.astype(np.int64, copy=False)
    if np.any((values < 0) | (values >= n)) or len(np.unique(values)) != len(values):
        raise ValueError("train row indices must be distinct and in range")
    return values


def _jobs(work, n):
    jobs = {}
    for name, job in work.items():
        rows = _rows(job['train'], n)
        y = np.asarray(job['y'])
        k = int(job['K'])
        if y.shape != (n,) or k < 1:
            raise ValueError("invalid y shape or K for " + name)
        mask = np.zeros(n, bool)
        mask[rows] = True
        jobs[name] = (y, rows, mask, k)
    return jobs


def _columns(X, given):
    if given is None:
        return np.arange(X.shape[1])
    values = list(given)
    if all(isinstance(v, (int, np.integer)) for v in values):
        cols = np.asarray(values, dtype=int)
    elif all(isinstance(v, str) for v in values):
        cols = np.arange(len(values))
    else:
        raise ValueError("given must contain column names or integer indices")
    if np.any((cols < 0) | (cols >= X.shape[1])) or len(np.unique(cols)) != len(cols):
        raise ValueError("given columns must be distinct and in range")
    return cols


def _normalize(p):
    p = np.maximum(np.asarray(p, dtype=np.float32), 1e-8)
    return p / p.sum(axis=1, keepdims=True)


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    return _normalize(np.exp(logits))


def _responsibility(z, centers, temperature):
    d2 = np.maximum((z*z).sum(1)[:, None] + (centers*centers).sum(1)[None, :]
                    - 2*z @ centers.T, 0)
    return _softmax(-d2 / temperature)


class LatentProfileModel:
    """A fitted partition, retaining GIVEN encoders, mappings and P(Y|C) only.

    Every cost knob is explicit. Single-thread fits and chunked prediction are
    defaults. The working-memory estimate is an allocation bound, not process
    RSS; caller-owned X/work and loaded runtime libraries are additional.
    Deadlines are cooperative between SVD/k-means/ridge operations and at each
    boosting iteration. Any over-budget fit raises without returning a model.
    """
    def __init__(self, scales=(8, 16, 32), *, svd_dim=24, svd_iter=4,
                 cluster_iter=40, soft_iter=5, soft_temperature=0.5,
                 dirichlet=5.0, ridge_alpha=5.0, ridge_iter=150,
                 ridge_tol=1e-3, tree_rounds=0, tree_leaves=15,
                 tree_min_samples=20, tree_learning_rate=0.08,
                 tree_l2=2.0, tree_weight=0.5, batch_size=512,
                 seed=20261031, threads=1, max_working_bytes=1024**3):
        self.scales = tuple(dict.fromkeys(int(s) for s in scales))
        if not self.scales or min(self.scales) < 1:
            raise ValueError("scales must contain positive cluster counts")
        self.svd_dim, self.svd_iter = int(svd_dim), int(svd_iter)
        self.cluster_iter, self.soft_iter = int(cluster_iter), int(soft_iter)
        self.soft_temperature, self.dirichlet = float(soft_temperature), float(dirichlet)
        self.ridge_alpha, self.ridge_iter = float(ridge_alpha), int(ridge_iter)
        self.ridge_tol = float(ridge_tol)
        self.tree_rounds, self.tree_leaves = int(tree_rounds), int(tree_leaves)
        self.tree_min_samples, self.tree_learning_rate = int(tree_min_samples), float(tree_learning_rate)
        self.tree_l2, self.tree_weight = float(tree_l2), float(tree_weight)
        self.batch_size, self.seed, self.threads = int(batch_size), int(seed), int(threads)
        self.max_working_bytes = int(max_working_bytes)
        if (min(self.svd_dim, self.svd_iter, self.cluster_iter, self.ridge_iter,
                self.tree_min_samples, self.batch_size, self.threads) < 1
                or self.soft_iter < 0 or self.tree_rounds < 0 or self.tree_leaves < 2
                or min(self.soft_temperature, self.dirichlet, self.ridge_alpha,
                       self.ridge_tol, self.tree_learning_rate) <= 0
                or self.tree_l2 < 0 or not 0 <= self.tree_weight <= 1
                or self.max_working_bytes < 1):
            raise ValueError("invalid latent profile cost/smoothing parameters")
        self.fitted_ = False

    def _memory(self, amount):
        if amount > self.max_working_bytes:
            raise MemoryError("latent profile allocation estimate exceeds max_working_bytes")

    def fit(self, X, work, train, deadline=float('inf'), given=None):
        """Fit only train intersect job.train; all other answers remain unread."""
        self.fitted_ = False
        self.arms_ = []
        start = time.monotonic()
        _clock(deadline)
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D categorical GIVEN matrix")
        rows = _rows(train, len(X))
        jobs = _jobs(work, len(X))
        self.columns_ = _columns(X, given)
        self.n_features_in_ = X.shape[1]
        names=list(given) if given is not None and all(isinstance(v,str) for v in given) else None
        self.crosses_=interaction_columns(len(self.columns_),names)
        self.radix_=np.maximum(1,np.max(X[np.ix_(rows,self.columns_)],axis=0)+1) if len(rows) else np.ones(len(self.columns_))
        self.widths_ = {name: job[3] for name, job in jobs.items()}
        width = sum(self.widths_.values())
        # CSR answers, randomized SVD workspace, responsibilities and encodings.
        estimate = (len(rows)*max(1, len(jobs))*20
                    + (len(rows)+width)*(self.svd_dim+10)*24
                    + len(rows)*(max(self.scales)+len(self.columns_))*32
                    + width*max(self.scales)*16)
        self._memory(estimate)
        with threadpool_limits(limits=self.threads):
            self._fit(X, jobs, rows, deadline)
        _clock(deadline)
        self.fitted_ = True
        self.report_['seconds'] = time.monotonic() - start
        self.report_['estimated_working_bytes'] = estimate
        return self

    def _fit(self, X, jobs, rows, deadline):
        # Inspect labels only AFTER intersecting the explicitly allowed rows.
        labels, rr, cc, offset = {}, [], [], 0
        for name, (y, _, mask, k) in jobs.items():
            _clock(deadline)
            local = np.flatnonzero(mask[rows])
            values = y[rows[local]]
            if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
                raise ValueError("noninteger observed labels for " + name)
            values = values.astype(np.int64)
            if np.any((values < 0) | (values >= k)):
                raise ValueError("observed labels outside K for " + name)
            labels[name] = (local, values, k)
            rr.append(local)
            cc.append(values + offset)
            offset += k
        if rr:
            row_index, col_index = np.concatenate(rr), np.concatenate(cc)
        else:
            row_index = col_index = np.empty(0, np.int64)
        answers = sparse.csr_matrix((np.ones(len(row_index), np.float32),
                                    (row_index, col_index)), shape=(len(rows), offset))
        active = np.asarray(answers.getnnz(axis=1) > 0)
        # Rows with no observed answer supply no supervised profile evidence.
        keep = np.flatnonzero(active)
        remap = np.full(len(rows), -1, np.int64)
        remap[keep] = np.arange(len(keep))
        labels = {name: (remap[local], values, k)
                  for name, (local, values, k) in labels.items()}
        rows, answers = rows[keep], answers[keep]
        self.report_ = {'fit_rows': len(rows), 'answer_nnz': int(answers.nnz),
                        'answer_width': offset, 'svd_dim': 0, 'scales': []}
        self.priors_ = {}
        for name, (_, values, k) in labels.items():
            self.priors_[name] = ((np.bincount(values, minlength=k) + 1.0/k)
                                  / (len(values)+1)).astype(np.float32)
        identical = all(len(values) in (0, len(rows)) and
                        (not len(values) or np.all(values == values[0]))
                        for _, values, _ in labels.values())
        if len(rows) < 2 or offset < 2 or identical:
            # Empty/single-response partitions have a defined smoothed prior.
            self.report_['constant'] = True
            return
        answer_counts=answers
        norm = np.sqrt(np.maximum(1, answers.getnnz(axis=1))).astype(np.float32)
        answers = answers.multiply((1/norm)[:, None]).tocsr()
        _clock(deadline)
        dim = min(self.svd_dim, len(rows)-1, offset-1)
        svd = TruncatedSVD(n_components=dim, n_iter=self.svd_iter, random_state=self.seed)
        z = svd.fit_transform(answers).astype(np.float32)
        _clock(deadline)
        self.report_['svd_dim'] = dim
        del answers, svd
        raw = self._raw_features(X[rows])
        if not np.isfinite(raw).all():
            raise ValueError("GIVEN codes must be finite; encode missing as a category")
        self.encoder_ = None
        design = None
        if raw.shape[1]:
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', dtype=np.float32, min_frequency=max(10,int(.005*len(rows))))
            design = self.encoder_.fit_transform(raw)
            tree_x = self._tree_matrix(raw) if self.tree_rounds else None
        for scale in self.scales:
            _clock(deadline)
            clusters = min(scale, len(rows))
            km = MiniBatchKMeans(n_clusters=clusters, n_init=1,
                                max_iter=self.cluster_iter, random_state=self.seed+scale,
                                batch_size=min(self.batch_size, len(rows)),
                                reassignment_ratio=0.0)
            km.fit(z)
            _clock(deadline)
            dmin = ((z-km.cluster_centers_[km.predict(z)])**2).sum(1)
            temperature = max(float(np.median(dmin))*self.soft_temperature, 1e-6)
            centers = km.cluster_centers_.astype(np.float32)
            for _ in range(self.soft_iter):
                _clock(deadline)
                membership = _responsibility(z, centers, temperature)
                centers = (membership.T @ z) / np.maximum(membership.sum(0)[:, None], 1e-8)
            membership = _responsibility(z, centers, temperature)
            # One sparse-dense multiplication covers every target and class.
            counts=np.asarray(answer_counts.T @ membership).T
            conditional={};offset=0
            for name, (_, _, k) in labels.items():
                block=counts[:,offset:offset+k];offset+=k
                conditional[name]=((block+self.dirichlet*self.priors_[name]) /
                    (block.sum(1,keepdims=True)+self.dirichlet)).astype(np.float32)
            arm = {'scale': scale, 'clusters': clusters, 'conditional': conditional,
                   'prior': membership.mean(0), 'ridge': None, 'tree': None}
            if design is not None and clusters > 1:
                ridge = Ridge(alpha=self.ridge_alpha, solver='lsqr', tol=self.ridge_tol,
                              max_iter=self.ridge_iter)
                ridge.fit(design, membership)
                _clock(deadline)
                arm['ridge'] = ridge
                if (self.tree_rounds and self.tree_weight
                        and len(rows) >= 2*self.tree_min_samples
                        and np.any(np.ptp(tree_x, axis=0) > 0)):
                    arm['tree'] = self._fit_tree(tree_x, membership, deadline)
            self.arms_.append(arm)
            self.report_['scales'].append({'requested': scale, 'clusters': clusters,
                                           'temperature': temperature,
                                           'tree_rounds': (arm['tree'].current_iteration()
                                                           if arm['tree'] is not None else 0)})

    def _raw_features(self,X):
        raw=np.asarray(X[:,self.columns_],dtype=np.float32)
        parts=[raw]
        for group in self.crosses_:
            key=np.zeros(len(raw),np.float64);valid=np.ones(len(raw),bool)
            for c in group:
                valid &= (raw[:,c]>=0)&(raw[:,c]<self.radix_[c])
                key=key*self.radix_[c]+raw[:,c]
            parts.append(np.where(valid,key,-1)[:,None])
        return np.column_stack(parts)

    def _tree_matrix(self, raw):
        result = np.empty(raw.shape, dtype=np.int32)
        for c, values in enumerate(self.encoder_.categories_):
            indices = np.searchsorted(values, raw[:, c])
            valid = indices < len(values)
            valid &= values[np.minimum(indices, len(values)-1)] == raw[:, c]
            result[:, c] = np.where(valid, indices, -1)
        return result

    def _fit_tree(self, raw, membership, deadline):
        try:
            import lightgbm as lgb
            clusters = membership.shape[1]
            log_prior = np.log(np.maximum(membership.mean(0), 1e-8))

            def objective(pred, data):
                _clock(deadline)
                probabilities = _softmax(pred + log_prior)
                return probabilities-membership, np.maximum(2*probabilities*(1-probabilities), 1e-5)

            def checkpoint(env):
                _clock(deadline)

            dataset = lgb.Dataset(raw, label=membership.argmax(1),
                                  categorical_feature=list(range(raw.shape[1])), free_raw_data=True)
            model = lgb.train({'objective': objective, 'num_class': clusters, 'metric': 'None',
                               'num_leaves': self.tree_leaves, 'learning_rate': self.tree_learning_rate,
                               'min_data_in_leaf': self.tree_min_samples, 'lambda_l2': self.tree_l2,
                               'verbosity': -1, 'num_threads': self.threads, 'deterministic': True,
                               'force_col_wise': True, 'feature_pre_filter': False, 'seed': self.seed},
                              dataset, num_boost_round=self.tree_rounds, callbacks=[checkpoint])
            _clock(deadline)
            return model
        except (ImportError, OSError):
            return None

    def _memberships(self, raw):
        design = self.encoder_.transform(raw) if self.encoder_ is not None else None
        tree_x = self._tree_matrix(raw) if self.encoder_ is not None and self.tree_rounds else None
        for arm in self.arms_:
            if arm['ridge'] is None:
                yield np.broadcast_to(arm['prior'], (len(raw), arm['clusters']))
                continue
            probability = _normalize(arm['ridge'].predict(design))
            if arm['tree'] is not None:
                try:
                    logits = arm['tree'].predict(tree_x, raw_score=True, num_threads=self.threads)
                    tree = _softmax(logits + np.log(np.maximum(arm['prior'], 1e-8)))
                    probability = _normalize((1-self.tree_weight)*probability + self.tree_weight*tree)
                except (ImportError, OSError):
                    pass
            yield probability

    def _query(self, X):
        if not self.fitted_:
            raise RuntimeError("fit a complete partition before predicting")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("prediction X must have the original GIVEN columns")
        return X

    def predict(self, X, deadline=float('inf')):
        """All question probabilities from GIVEN only, on arbitrary new rows."""
        X = self._query(X)
        self._memory(len(X)*sum(self.widths_.values())*4)
        result = {name: np.empty((len(X), k), np.float32) for name, k in self.widths_.items()}
        with threadpool_limits(limits=self.threads):
            for start in range(0, len(X), self.batch_size):
                _clock(deadline)
                stop = min(len(X), start+self.batch_size)
                raw = self._raw_features(X[start:stop])
                if not np.isfinite(raw).all():
                    raise ValueError("GIVEN codes must be finite")
                if not self.arms_:
                    for name in result:
                        result[name][start:stop] = self.priors_[name]
                    continue
                memberships = list(self._memberships(raw))
                for name in result:
                    p = sum(m @ arm['conditional'][name] for m, arm in zip(memberships, self.arms_))
                    result[name][start:stop] = _normalize(p)
        _clock(deadline)
        return result

    def transform(self, X, deadline=float('inf')):
        """Concatenated P(C|GIVEN) for a second-stage model in this partition."""
        X = self._query(X)
        width = sum(arm['clusters'] for arm in self.arms_) or 1
        self._memory(len(X)*width*4)
        result = np.empty((len(X), width), np.float32)
        with threadpool_limits(limits=self.threads):
            for start in range(0, len(X), self.batch_size):
                _clock(deadline)
                stop = min(len(X), start+self.batch_size)
                raw = self._raw_features(X[start:stop])
                if not np.isfinite(raw).all():
                    raise ValueError("GIVEN codes must be finite")
                result[start:stop] = (np.column_stack(list(self._memberships(raw)))
                                      if self.arms_ else 1.0)
        _clock(deadline)
        return result


def fit_latent_profiles(X, work, fold, deadline, given=None, scales=(8, 16, 32), **costs):
    """Complete scales only: all outer folds PLUS a full visible-row refit.

    Incomplete scales are discarded atomically, preserving any earlier complete
    scales. Returned OOF follows job.train (boolean row order or supplied index
    order). full is an independent all-visible refit, not a half-data ensemble.
    Costs are the explicit keyword parameters of LatentProfileModel.__init__.
    Validation/programming differences propagate; only an expired budget is handled.
    """
    start = time.monotonic()
    X, fold = np.asarray(X), np.asarray(fold)
    if X.ndim != 2 or fold.shape != (len(X),) or fold.dtype.kind not in 'iu':
        raise ValueError("X/fold shape or fold integer dtype is invalid")
    if np.any(fold < -1):
        raise ValueError("fold uses -1 for hidden and nonnegative visible ids")
    jobs = _jobs(work, len(X))
    for name, (_, rows, _, _) in jobs.items():
        if np.any(fold[rows] < 0):
            raise ValueError("job.train contains hidden fold rows for " + name)
    visible = np.flatnonzero(fold >= 0)
    folds = np.unique(fold[visible])
    report = {'status': 'empty', 'completed_scales': [], 'discarded_scales': [],
              'folds': folds.tolist(), 'partitions': [], 'items': 0}
    results = {}
    scales = tuple(dict.fromkeys(int(s) for s in scales))
    if not work or not len(visible):
        report['seconds'] = time.monotonic()-start
        return results, report
    # Validate costs and reserve sums + staging + predictions before allocating.
    config = LatentProfileModel(scales=scales, **costs)
    config._memory(sum((len(rows)+3*len(X))*k*4 for _, rows, _, k in jobs.values()))
    for scale in scales:
        stage = 'start'
        try:
            _clock(deadline)
            pending = {name: np.full((len(rows), k), np.nan, np.float32)
                       for name, (_, rows, _, k) in jobs.items()}
            for f in folds:
                stage = 'fold:' + str(int(f))
                train, query = visible[fold[visible] != f], visible[fold[visible] == f]
                model = LatentProfileModel(scales=(scale,), **costs).fit(X, work, train, deadline, given)
                pred = model.predict(X[query], deadline)
                positions = np.full(len(X), -1, np.int64)
                positions[query] = np.arange(len(query))
                for name, (_, rows, _, _) in jobs.items():
                    take = fold[rows] == f
                    pending[name][take] = pred[name][positions[rows[take]]]
                report['partitions'].append({'scale': scale, 'stage': stage, **model.report_})
                del model, pred
            stage = 'full_refit'
            model = LatentProfileModel(scales=(scale,), **costs).fit(X, work, visible, deadline, given)
            full = model.predict(X, deadline)
            report['partitions'].append({'scale': scale, 'stage': stage, **model.report_})
            del model
            for name in pending:
                if not np.isfinite(pending[name]).all():
                    raise RuntimeError("incomplete OOF coverage for " + name)
            _clock(deadline)
        except ProfileBudgetExpired:
            report['discarded_scales'].append({'scale': scale, 'stage': stage})
            break
        # Commit only complete OOF + full-refit scales.
        if not results:
            results = {name: (pending[name], full[name]) for name in pending}
        else:
            for name, (oof, all_rows) in results.items():
                oof += pending[name]
                all_rows += full[name]
        report['completed_scales'].append(scale)
        del pending, full
    count = len(report['completed_scales'])
    if count:
        for oof, full in results.values():
            oof /= count
            full /= count
    report.update(items=len(results), seconds=time.monotonic()-start,
                  status=('partial' if count else 'budget_expired') if report['discarded_scales'] else 'complete')
    return results, report
