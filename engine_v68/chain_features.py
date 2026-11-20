"""Categorical GIVEN crosses and partition-isolated EB probabilities."""
import time
import threading
from collections import OrderedDict
import numpy as np
from .propagate_structural_gates import _parents, _wave_two_source, _wave_one_index, _encode_wave_one_zero
from .latent_profiles import fit_latent_profiles
from . import v68_config

class BudgetExpired(BaseException):
    pass

def check_clock(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired()

def folds_for(X, rows, k=5, seed=20261020):
    out = np.full(len(X), -1, np.int16)
    rng = np.random.default_rng(seed)
    strata = X[rows, 0] if X.shape[1] else np.zeros(len(rows), int)
    offset = 0
    for value in np.unique(strata):
        group = rows[strata == value].copy()
        rng.shuffle(group)
        out[group] = (np.arange(len(group)) + offset) % k
        offset = (offset + len(group)) % k
    return out

def topological(names, items):
    ordered = []
    visiting = set()
    done = set()

    def visit(name):
        if name in done:
            return
        if name in visiting:
            raise ValueError('cyclic gate: ' + name)
            return None
        visiting.add(name)
        for parent in _parents(name, items):
            if parent in names:
                visit(parent)
        visiting.remove(name)
        done.add(name)
        ordered.append(name)
    for name in names:
        visit(name)
    return ordered
from .given_crosses import categorical_matrix

def _prob(logits, classes):
    if len(classes) == 2:
        p = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        return np.column_stack((1 - p, p))
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)

def _fit_hgb(X, y, categorical=0, max_iter=80, compact=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    max_bins = min(64 if compact else 254, max(16, int(len(X) // 20)))
    cats = [c for c in range(min(int(categorical), X.shape[1]))
            if np.isfinite(X[:, c]).all() and np.min(X[:, c]) >= 0
            and np.max(X[:, c]) < (32 if compact else 254)
            and np.equal(X[:, c], np.floor(X[:, c])).all()]
    if cats:
        max_bins = min(64 if compact else 255, max(max_bins, int(np.max(X[:, cats])) + 2))
    model = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=15,
        learning_rate=0.1, l2_regularization=2., early_stopping=False,
        categorical_features=cats or None, max_bins=max_bins, random_state=20261020)
    model.fit(X, y)
    return model

def hgb_stages(X, categorical, y, train, query, K, deadline, max_iter=80):
    check_clock(deadline)
    classes = np.unique(y[train])
    stages = tuple(range(10, max_iter + 1, 10))
    if max_iter not in stages:
        stages += (max_iter,)
    if len(classes) < 2:
        p = np.zeros((len(query), K), np.float32)
        if len(classes): p[:, int(classes[0])] = 1.
        else: p[:] = 1. / K
        return {s: p.copy() for s in stages}
    model = _fit_hgb(X[train], y[train], categorical, max_iter, compact=not v68_config.V127_W1)
    predictions = {}
    for step, p in enumerate(model.staged_predict_proba(X[query]), 1):
        if step in stages:
            check_clock(deadline)
            q = np.zeros((len(query), K), np.float32)
            q[:, model.classes_.astype(int)] = p
            predictions[step] = q
    return predictions

def hgb_crossfit_stages(feature_factory, y, rows, rest, split, k, K, deadline, max_iter=60):
    """Publish only complete cross-fitted stages, progressing all folds together."""
    partitions = []
    for f in range(k):
        check_clock(deadline)
        tr, val = rows[split[rows] != f], rows[split[rows] == f]
        query = np.concatenate((val, rest))
        features, categorical = feature_factory(tr)
        partitions.append((features, categorical, tr, query, split[rows] == f, len(val), np.unique(y[tr])))
    models = [None] * k
    stage_o, stage_p = {}, {}
    for step in range(10, max_iter + 1, 10):
        o = np.zeros((len(rows), K), np.float32)
        full = np.zeros((len(y), K), np.float32)
        try:
            for f, (features, categorical, tr, query, mask, n_val, classes) in enumerate(partitions):
                check_clock(deadline)
                pred = np.zeros((len(query), K), np.float32)
                if len(classes) < 2:
                    if len(classes): pred[:, int(classes[0])] = 1.
                    else: pred[:] = 1. / K
                else:
                    model = models[f]
                    if model is None:
                        model = _fit_hgb(features[tr], y[tr], categorical, step, compact=not v68_config.V127_W1)
                        model.set_params(warm_start=True)
                        models[f] = model
                    else:
                        model.set_params(max_iter=step)
                        model.fit(features[tr], y[tr])
                    pred[:, model.classes_.astype(int)] = model.predict_proba(features[query])
                o[mask] = pred[:n_val]
                full[rest] += pred[n_val:] / k
        except BudgetExpired:
            if not stage_o:
                raise
            break
        stage_o[step], stage_p[step] = o, full
    return stage_o, stage_p

def tree_predict(X, categorical, y, train, query, K, profile, deadline, on_fit=None, is_deep_hierarchy=True, hgb_iter=60):
    try:
        check_clock(deadline)
        classes = np.unique(y[train])
        out = np.zeros((len(query), K), np.float32)
        if len(classes) < 2:
            if len(classes):
                out[:, int(classes[0])] = 1
            else:
                out[:] = 1 / K
            return out
        if profile == 'catboost':
            import pandas as pd
            from catboost import CatBoostClassifier, Pool
            data = pd.DataFrame(X)
            for c in range(categorical):
                data[c] = data[c].astype(np.int32)
            cats = list(range(categorical))
            fit_pool = Pool(data.iloc[train], y[train], cat_features=cats, thread_count=1)
            query_pool = Pool(data.iloc[query], cat_features=cats, thread_count=1)
            model = CatBoostClassifier(iterations=60, depth=4, learning_rate=0.1, l2_leaf_reg=2.0, thread_count=1, verbose=False, allow_writing_files=False, one_hot_max_size=65535, random_seed=20261020, task_type='CPU')

            class Clock:

                def after_iteration(self, info):
                    return time.monotonic() < deadline
            model.fit(fit_pool, callbacks=[Clock()])
            check_clock(deadline)
            out[:, model.classes_.astype(int)] = model.predict_proba(query_pool, thread_count=1)
            return out
        if profile == 'hgb':
            model = _fit_hgb(X[train], y[train], categorical, hgb_iter, compact=not is_deep_hierarchy)
            check_clock(deadline)
            out[:, model.classes_.astype(int)] = model.predict_proba(X[query])
            return out
        from lightgbm import LGBMClassifier
        if profile == 'strong':
            return _strong_predict(X, categorical, y, train, query, K, deadline, out)
        leaves, rate = (31, 0.06) if profile == 'deep' else (15, 0.1)
        rounds = ({'parent': 50, 'shallow': 50, 'deep': 55} if is_deep_hierarchy else {'parent': 35, 'shallow': 35, 'deep': 45}).get(profile, 60)
        model = LGBMClassifier(n_estimators=rounds, num_leaves=leaves, learning_rate=rate, min_child_samples=20, reg_lambda=2.0, n_jobs=1, verbosity=-1, metric='None', deterministic=True, force_col_wise=True, random_state=20261020)

        def checkpoint(env):
            check_clock(deadline)
        if on_fit:
            on_fit('start')
        model.fit(X[train], y[train], categorical_feature=list(range(categorical)), callbacks=[checkpoint])
        if on_fit:
            on_fit('complete')
        check_clock(deadline)
        pred = model.booster_.predict(X[query], num_iteration=model.booster_.current_iteration(), num_threads=1)
        if pred.ndim == 1:
            pred = np.column_stack((1 - pred, pred))
        out[:, model.classes_.astype(int)] = pred
        return out
    except (ImportError, OSError):
        if profile == 'hgb':
            raise
        return tree_predict(X, categorical, y, train, query, K, 'hgb', deadline, on_fit=on_fit, is_deep_hierarchy=is_deep_hierarchy, hgb_iter=hgb_iter)

def _strong_predict(X, categorical, y, train, query, K, deadline, out):
    """Early-termination boosting for anchors whose loss repeats in descendants.

    The stop round is read on a fixed 15% slice of the fit rows only.
    """
    try:
        from lightgbm import LGBMClassifier, early_stopping
        train = np.asarray(train, dtype=np.int64)
        held = np.arange(len(train)) * 2654435761 % 100 < 15
        fit, stop_rows = (train[~held], train[held])
        if len(np.unique(y[fit])) < len(np.unique(y[train])) or len(stop_rows) < 30:
            fit, stop_rows = (train, None)
        model = LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.04, min_child_samples=30, reg_lambda=5.0, colsample_bytree=0.8, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True, random_state=20261027)
        callbacks = [lambda env: check_clock(deadline)]
        kwargs = {}
        if stop_rows is not None:
            multi = len(np.unique(y[fit])) > 2
            kwargs = dict(eval_set=[(X[stop_rows], y[stop_rows])], eval_metric='multi_logloss' if multi else 'binary_logloss')
            callbacks.append(early_stopping(40, verbose=False))
        model.fit(X[fit], y[fit], categorical_feature=list(range(categorical)), callbacks=callbacks, **kwargs)
        check_clock(deadline)
        best = model.best_iteration_ or model.booster_.current_iteration()
        pred = model.booster_.predict(X[query], num_iteration=best, num_threads=1)
        if pred.ndim == 1:
            pred = np.column_stack((1 - pred, pred))
        out[:, model.classes_.astype(int)] = pred
        return out
    except (ImportError, OSError):
        return tree_predict(X, categorical, y, train, query, K, 'hgb', deadline)

def gated_tree_predict(X, categorical, y, train, query, K, profile, deadline, *, name=None, items=None, given=None, raw=None, is_deep_hierarchy=True):
    wave_one = None
    if name is not None:
        source = _wave_two_source(name, items, raw, given, train)
        if source is None or not source.any():
            return None
        train = train[source]
        wave_one = _wave_one_index(name, items, given)
    asked = y[train] != K - 1
    gate_y = (y == K - 1).astype(np.int64)
    gate = tree_predict(X, categorical, gate_y, train, query, 2, profile, deadline, is_deep_hierarchy=is_deep_hierarchy)
    conditional = tree_predict(X, categorical, y, train[asked], query, K - 1, profile, deadline, is_deep_hierarchy=is_deep_hierarchy)
    result = np.column_stack((conditional * (1 - gate[:, 1, None]), gate[:, 1]))
    if wave_one is not None:
        result = _encode_wave_one_zero(result, raw[query, given.index('wave')], wave_one)
    return result
PRE_POST_MIRRORS = {'user.howManyHoursWillSearchPerWeek': 'user.howManyHoursSearchedPerWeek', 'user.howManyApplicationsNextMonth': 'user.howManyApplicationsLastMonth', 'user.expectsToFindJobInSixMonths': 'user.expectedToFindJobInSixMonths', 'user.targetSector': 'user.preferredSector'}
FOLLOWUP_MIRRORS = {'tr_target_sector': ('user.targetSector', 'user.preferredSector'), 'tr_riasec_target_sector': ('tr_target_sector', 'user.targetSector', 'user.preferredSector'), 'tr_search_hours_after': ('user.howManyHoursWillSearchPerWeek', 'user.howManyHoursSearchedPerWeek')}
UNICEF_ANCHORS = ('Q06', 'Q07', 'Q08', 'Q09')

def priority_sources(name, items, available):
    """Questionnaire-declared mirrors precede generic response summaries."""
    wanted = list(FOLLOWUP_MIRRORS.get(name, ()))
    if name in PRE_POST_MIRRORS:
        wanted.insert(0, PRE_POST_MIRRORS[name])
    wanted += list(_parents(name, items))
    if 'Country' in items and 'Q08' in items:
        wanted += list(UNICEF_ANCHORS)
    return list(dict.fromkeys((a for a in wanted if a != name and a in available)))

def response_block(probability):
    """Complete class simplex plus its categorical mode, without truncation."""
    return np.column_stack((probability, probability.argmax(1))).astype(np.float32)

class FoldFeatureBank:

    def __init__(self, X, items, work, fold, deadline, tables=None, n_raw_given=None, levels=None, is_deep_hierarchy=True, given=None):
        self.X, self.items, self.work, self.fold, self.deadline = (X, items, work, fold, deadline)
        self.given = given
        self.latent_features = False
        self.self_table_features = not is_deep_hierarchy
        self.is_deep_hierarchy = is_deep_hierarchy
        if tables is None:
            from .base import _Tables
            tables = _Tables(X, np.maximum(fold, 0), 5)
        self.tables = tables
        self.cells = None
        if len(work) == 12 and given is not None and 'Country' in given:
            from .full_cell_eb import Cells
            width = X.shape[1] if n_raw_given is None else n_raw_given
            self.cells = Cells(X[:, :width], given[:width], items=items)
        self.n_raw_given = X.shape[1] if n_raw_given is None else n_raw_given
        self.levels = np.max(X[:, :self.n_raw_given], axis=0) + 1 if levels is None else np.asarray(levels[:self.n_raw_given])
        self.cache = {}
        self.config_cache = {}
        self.matrix_cache = OrderedDict()
        self.matrix_cache_bytes = 0
        self.matrix_cache_hits = 0
        self.matrix_cache_limit = 384 * 1024 ** 2
        self.lock = threading.RLock()
        self.parent_fits = 0
        self.parent_fit_calls = 0
        self.cache_hits = 0
        self.model_cache_hits = 0
        self.parent_order = topological(list(work), items)
        self.parent_index = {name: i for i, name in enumerate(self.parent_order)}

    @staticmethod
    def _train_key(train):
        return np.asarray(train, dtype=np.int64).tobytes()

    def table_probability(self, name, train, cache):
        """Rebuild an EB table within this outer training partition.

        Global work[name]['oof'][0] was fitted with labels from the outer
        validation fold, so it cannot be a feature of an outer-fold model;
        the shallow-survey config is chosen once per outer partition.
        """
        with self.lock:
            guard = cache['locks'].setdefault(('table', name), threading.Lock())
        with guard:
            if name in cache['tables']:
                return cache['tables'][name]
            check_clock(self.deadline)
            if len(self.work) == 12 and self.given is not None and ('Country' in self.given):
                from .full_cell_eb import Cells
                job = self.work[name]
                mask = np.zeros(len(self.X), bool)
                mask[train] = job['y'][train] >= 0
                local = {'y': np.where(mask, job['y'], -1), 'K': job['K'], 'train': mask}
                cells = self.cells
                found = cells.fit(local, self.fold, name, source_job=job)
                if found is not None:
                    oof, p, _ = found
                    p[mask] = oof
                    p = p.astype(np.float32)
                    value = (p, float(np.var(p[mask], axis=0).sum()))
                    cache['tables'][name] = value
                    return value
            from .base import _fit_table, _mi_drivers, _safe_table_config
            job = self.work[name]
            fit = train[job['y'][train] >= 0]
            mask = np.zeros(len(self.X), bool)
            mask[fit] = True
            config_key = (name, self._train_key(train))
            config = self.config_cache.get(config_key)
            if config is None:
                if self.is_deep_hierarchy:
                    order = _mi_drivers(self.X[:, :self.n_raw_given], job['y'], fit, self.levels, count=3)
                    config = (order, float(np.clip(np.sqrt(len(fit)) / 10.0, 1.0, 30.0)))
                else:
                    config = cache.get('safe_config')
                    if config is None:
                        config = _safe_table_config(self.X, train, self.n_raw_given)
                        cache['safe_config'] = config
                with self.lock:
                    self.config_cache[config_key] = config
            oof, full, gains = _fit_table(self.tables, np.maximum(job['y'], 0), mask, job['K'], self.n_raw_given, check_clock=lambda: check_clock(self.deadline), fixed_config=config)
            p = np.asarray(full, dtype=np.float32).copy()
            p[fit] = oof
            strength = float(np.var(p[fit], axis=0).sum()) if len(fit) else 0.0
            value = (p, strength)
            cache['tables'][name] = value
            return value

    def latent_probability(self, name, train, cache):
        with self.lock:
            guard = cache['locks'].setdefault(('latent',), threading.Lock())
        with guard:
            if 'latent' not in cache:
                local = {}
                for target, job in self.work.items():
                    mask = np.zeros(len(self.X), bool)
                    mask[train] = job['train'][train]
                    local[target] = {'K': job['K'], 'train': mask, 'y': np.where(mask, job['y'], -1)}
                inner = folds_for(self.X, train, k=5, seed=20261031)
                found, trace = fit_latent_profiles(self.X, local, inner, self.deadline, given=self.given, scales=(8,), tree_rounds=16, cluster_iter=15, soft_iter=2, svd_iter=2, svd_dim=16)
                if not found:
                    raise BudgetExpired()
                    return None
                cache['latent'] = {}
                for target, (o, p) in found.items():
                    p[local[target]['train']] = o
                    cache['latent'][target] = p
                cache['latent_trace'] = trace
            return cache['latent'][name]

    def partition(self, train):
        key = self._train_key(train)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = {'raw': categorical_matrix(self.X[:, :self.n_raw_given], train, self.given[:self.n_raw_given] if self.given else None), 'parents': {}, 'tables': {}, 'locks': {}}
            return self.cache[key]

    def prewarm(self, names, folds):
        """Fill every fold partition's raw crosses and parent tables up front.

        Worker threads then only slice cached arrays instead of contending on
        the per-partition locks for the same table fits.
        """
        rows = np.flatnonzero(self.fold >= 0)
        needed = []
        for name in names:
            for parent in _parents(name, self.items):
                if parent in self.work and parent not in needed:
                    needed.append(parent)
            if not self.is_deep_hierarchy and name not in needed:
                needed.append(name)
        for fold in folds:
            for f in np.unique(fold[rows]):
                tr = rows[fold[rows] != f]
                cache = self.partition(tr)
                for parent in needed:
                    check_clock(self.deadline)
                    self.table_probability(parent, tr, cache)

    def prewarm_partitions(self, folds):
        """Build raw crosses and the shared table config for each partition.

        partition() builds under the bank lock, so a worker reaching a new
        3- or 2-fold partition would otherwise stall every other worker.
        """
        from .base import _safe_table_config
        rows = np.flatnonzero(self.fold >= 0)
        for fold in folds:
            for f in np.unique(fold[rows]):
                check_clock(self.deadline)
                tr = rows[fold[rows] != f]
                cache = self.partition(tr)
                if not self.is_deep_hierarchy and 'safe_config' not in cache:
                    cache['safe_config'] = _safe_table_config(self.X, tr, self.n_raw_given)

    def matrix(self, name, train):
        chain = getattr(self, 'chain', None)
        predictions = getattr(chain, 'S', None)
        key = (name, self._train_key(train), id(predictions), len(predictions) if predictions is not None else 0)
        with self.lock:
            if key in self.matrix_cache:
                self.matrix_cache_hits += 1
                return self.matrix_cache[key]
        value = self._matrix(name, train)
        with self.lock:
            if key in self.matrix_cache:
                return self.matrix_cache[key]
            size = value[0].nbytes
            if size <= self.matrix_cache_limit:
                while self.matrix_cache and self.matrix_cache_bytes + size > self.matrix_cache_limit:
                    _, old = self.matrix_cache.popitem(last=False)
                    self.matrix_cache_bytes -= old[0].nbytes
                self.matrix_cache[key] = value
                self.matrix_cache_bytes += size
        return value

    def clear_matrices(self):
        with self.lock:
            self.matrix_cache.clear()
            self.matrix_cache_bytes = 0

    def _matrix(self, name, train):
        cache = self.partition(train)
        raw = cache['raw']
        parts = [raw, self.X[:, self.n_raw_given:]]
        parents = set(_parents(name, self.items))
        chain = getattr(self, 'chain', None)
        if len(self.work) in (12, 76) and not (chain is not None and name in chain.children):
            for anchor in priority_sources(name, self.items, self.work):
                parts.append(response_block(self.table_probability(anchor, train, cache)[0]))
        if self.self_table_features:
            parts.append(self.table_probability(name, train, cache)[0])
        if self.latent_features:
            parts.append(self.latent_probability(name, train, cache))
        if not self.is_deep_hierarchy:
            before = self.parent_order[max(0, self.parent_index[name] - 8):self.parent_index[name]]
            ranked = []
            for previous in before:
                if previous in parents:
                    continue
                p, gain = self.table_probability(previous, train, cache)
                ranked.append((gain, self.parent_index[previous], p))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            parts.extend((p for gain, _, p in ranked[:2] if gain > 0))
        chain = getattr(self, 'chain', None)
        if chain is not None and name in chain.children:
            parts.append(chain.extras(name, train))
            return (np.column_stack(parts), raw.shape[1])
        for parent in self.parent_order:
            if parent not in parents:
                continue
            parts.append(self.table_probability(parent, train, cache)[0])
        return (np.column_stack(parts), raw.shape[1])

def split_anchors(names, items):
    """Pass 1 anchors: ungated items and every item another item routes on."""
    names = list(names)
    inside = set(names)
    parents = set()
    for name in names:
        parents.update((p for p in _parents(name, items) if p in inside))
    anchors = [n for n in names if not items[n].get('gate') or n in parents]
    chosen = set(anchors)
    return (anchors, [n for n in names if n not in chosen])

class StackedChain:
    """Pass-1 stacked anchor probabilities as Pass-2 features.

    ``stacked[name]`` holds the anchor's fused OOF probabilities on visible
    rows and its fused hidden-row prediction, so every visible row's feature
    came from models that never saw that row's own anchor label.  Anchor
    choice by mutual information and the latent basis are both refit on each
    outer training partition only.
    """

    def __init__(self, items, work, stacked, anchors, children, n, latent=12, bank=None):
        self.items = items
        self.work = work
        self.children = set(children)
        self.bank = bank
        self.anchors = [a for a in anchors if a in stacked]
        self.latent = latent
        self.lock = threading.RLock()
        self.cache = {}
        blocks = []
        self.block = {}
        start = 0
        for name in self.anchors:
            rows = np.flatnonzero(work[name]['train'])
            rest = np.flatnonzero(~work[name]['train'])
            o, p = stacked[name]
            full = np.zeros((n, work[name]['K']), np.float32)
            full[rows] = o
            full[rest] = p
            blocks.append(full)
            self.block[name] = (start, start + full.shape[1])
            start += full.shape[1]
        self.S = np.column_stack(blocks) if blocks else np.zeros((n, 0), np.float32)
        self.labels = {a: work[a]['y'] for a in self.anchors}

    def probabilities(self, name, train=None):
        if name in self.block:
            a, b = self.block[name]
            return self.S[:, a:b]
        return self.bank.table_probability(name, train, self.bank.partition(train))[0]

    def _latent(self, train, key):
        with self.lock:
            if ('z', key) in self.cache:
                return self.cache['z', key]
        S = np.column_stack([self.probabilities(a, train) for a in self.anchors]) if self.bank is not None and self.anchors else self.S
        k = min(self.latent, S.shape[1] - 1, len(train) - 1)
        if k < 1:
            Z = np.zeros((len(S), 0), np.float32)
        else:
            mu = S[train].mean(0)
            Xc = (S[train] - mu).astype(np.float64)
            if len(train) > S.shape[1]:
                _, evecs = np.linalg.eigh(Xc.T @ Xc)
                V = evecs[:, -1:-k - 1:-1]
            else:
                _, _, vt = np.linalg.svd(Xc, full_matrices=False)
                V = vt[:k].T
            signs = np.sign(V[np.abs(V).argmax(0), np.arange(k)])
            V = V * np.where(signs == 0, 1.0, signs)
            Z = (S - mu) @ V
            sd = Z[train].std(0)
            sd[sd <= 0] = 1.0
            Z = (Z / sd).astype(np.float32)
        with self.lock:
            self.cache['z', key] = Z
        return Z

    def informative(self, name, train, count=3):
        """Anchors with the highest MI against this item on the fit rows."""
        y = self.work[name]['y']
        fit = train[y[train] >= 0]
        if not len(fit):
            return []
        target = y[fit]
        K = int(self.work[name]['K'])
        parents = set(_parents(name, self.items))
        H, offsets = self._onehot(fit, np.asarray(fit, dtype=np.int64).tobytes())
        desired = np.eye(K, dtype=np.float32)[target]
        joint = (H.T @ desired).astype(np.float64) / len(fit)
        py = np.bincount(target, minlength=K) / len(fit)
        px = joint.sum(1)
        independent = px[:, None] * py
        information = np.zeros_like(joint)
        valid = joint > 0
        information[valid] = joint[valid] * np.log(joint[valid] / independent[valid])
        columns = information.sum(1)
        scores = []
        for anchor, (a, b) in zip(self.anchors, offsets):
            if anchor == name or anchor in parents or np.any(self.labels[anchor][fit] < 0):
                continue
            mi = float(columns[a:b].sum()) - (b - a - 1) * (K - 1) / (2.0 * len(fit))
            if mi > 0:
                scores.append((mi, anchor))
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [a for _, a in scores[:count]]

    def _onehot(self, fit, key):
        """Fit-row one-hot of every anchor's true answer, one per partition."""
        with self.lock:
            if ('h', key) in self.cache:
                return self.cache['h', key]
        offsets = []
        blocks = []
        start = 0
        for anchor in self.anchors:
            K = int(self.work[anchor]['K'])
            a = self.labels[anchor][fit]
            block = np.zeros((len(fit), K), np.float32)
            ok = a >= 0
            block[np.flatnonzero(ok), a[ok]] = 1.0
            blocks.append(block)
            offsets.append((start, start + K))
            start += K
        value = (np.concatenate(blocks, 1) if blocks else np.zeros((len(fit), 0), np.float32), offsets)
        with self.lock:
            self.cache['h', key] = value
        return value

    def switches(self, name, train, count=2):
        """Anchors whose answers best explain who skipped this item (fit rows)."""
        job = self.work[name]
        K = int(job['K'])
        if not self.items[name].get('gate') or not self.anchors:
            return []
        y = job['y']
        fit = train[y[train] >= 0]
        if len(fit) < 50:
            return []
        target = (y[fit] == K - 1).astype(np.float32)
        if target.min() == target.max():
            return []
        H, offsets = self._onehot(fit, np.asarray(fit, dtype=np.int64).tobytes())
        from .propagate_structural_gates import _mi_columns
        scores = _mi_columns(H, offsets, target)
        out = []
        for i in np.argsort(-scores):
            if scores[i] <= 0 or len(out) >= count:
                break
            if self.anchors[i] != name:
                out.append(self.anchors[i])
        return out

    def extras(self, name, train):
        if 'group' not in self.items and 'Country' not in self.items:
            train = np.asarray(train, dtype=np.int64)
            key = train.tobytes()
            with self.lock:
                chosen = self.cache.get(('mi', name, key))
            if chosen is None:
                chosen = self.informative(name, train)
                for anchor in self.switches(name, train):
                    if anchor not in chosen:
                        chosen = chosen + [anchor]
                with self.lock:
                    self.cache['mi', name, key] = chosen
            parents = list(_parents(name, self.items))
            if self.items[name].get('gate') and name.startswith('s'):
                for anchor in ('s3_11_intent_return_12m_lbl', 's3_12_hope_return_someday_lbl'):
                    if anchor != name and anchor in self.block and (anchor not in parents):
                        parents.append(anchor)
            parts = [self.probabilities(p, train) for p in parents if p in self.block]
            parts.extend((self.probabilities(a, train) for a in chosen if a not in parents))
            parts.append(self._latent(train, key))
            return np.column_stack(parts).astype(np.float32)
        train = np.asarray(train, dtype=np.int64)
        key = train.tobytes()
        with self.lock:
            chosen = self.cache.get(('mi', name, key))
        if chosen is None:
            chosen = self.informative(name, train)
            for anchor in self.switches(name, train):
                if anchor not in chosen:
                    chosen = chosen + [anchor]
            with self.lock:
                self.cache['mi', name, key] = chosen
        priority = priority_sources(name, self.items, self.work)
        chosen = list(dict.fromkeys(priority + chosen))
        chosen = [a for a in chosen if a != name and a in self.work]
        parts = [response_block(self.probabilities(a, train)) for a in chosen]
        if name == 'tr_riasec_target_sector' and 'tr_target_sector' in self.work:
            from .paired_transition import project
            parent = self.work['tr_target_sector']
            child = self.work[name]
            groups = self.bank.X[:, self.bank.given.index('group')]
            parent_p = self.probabilities('tr_target_sector', train)
            projected = project(parent['y'], child['y'], groups, train, np.arange(len(groups)), parent_p, parent['K'], child['K'], deterministic=True)
            for f in np.unique(self.bank.fold[train]):
                fit = train[self.bank.fold[train] != f]
                query = train[self.bank.fold[train] == f]
                projected[query] = project(parent['y'], child['y'], groups, fit, query, parent_p[query], parent['K'], child['K'], deterministic=True)
            parts.insert(0, response_block(projected))
        parts.append(self._latent(train, key))
        return np.column_stack(parts).astype(np.float32) if parts else np.empty((len(self.S), 0), np.float32)

class FoldPlan:
    """Pick 5, 3 or 2 outer folds per item so every queued item still fits.

    Cost is measured per fold on completed items of the same profile; an
    item starts with the largest fold count whose projected cost for all
    remaining items fits the profile's remaining wall clock.
    """

    def __init__(self, folds, total, stop, threads, max_high_folds=None, prior=()):
        self.folds = folds
        self.total = total
        self.stop = stop
        self.threads = max(1, threads)
        self.max_high_folds = max_high_folds
        self.started = 0
        self.costs = list(prior)
        self.lock = threading.Lock()
        self.used = {k: 0 for k in folds}
        self.completed = {k: 0 for k in folds}
    adaptive = True

    def choose(self):
        with self.lock:
            self.started += 1
            remaining = self.total - self.started + 1
            options = sorted(self.folds, reverse=True)
            if self.max_high_folds is not None and self.started > self.max_high_folds and (not (self.adaptive and self.costs)):
                k = options[-1]
            elif not self.costs:
                k = options[0]
            else:
                per = float(np.median(self.costs))
                room = max(0.0, self.stop - time.monotonic()) * self.threads * 0.9
                k = next((k for k in options if k * per * remaining <= room), options[-1])
            self.used[k] += 1
            return k

    def record(self, folds, seconds):
        with self.lock:
            self.costs.append(seconds / max(1, folds))
            self.completed[folds] = self.completed.get(folds, 0) + 1
