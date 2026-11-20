"""Visible-answer neighbor features for the three SimulacraBench instruments.

All row indices in this module refer to the frame passed to ``predict``.  The
index is built from GIVEN columns only; answers enter only ``distributions``.
For training rows, supply a fold vector: every lookup then excludes that row's
fold.  Held-out rows use every visible row, including DEV in phase 2.
"""

from __future__ import annotations

import numpy as np
import time
from chain_features import BudgetExpired


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:raise BudgetExpired()


def _given(frame, schema):
    return [name for name, item in schema["items"].items()
            if item["class"] == "GIVEN" and name in frame]


def _visible(frame, schema):
    targets = [name for name, item in schema["items"].items()
               if item["class"] == "PREDICT"]
    return np.flatnonzero(frame[targets].notna().any(axis=1).to_numpy())


def _ordered(name):
    s = name.lower()
    return any(word in s for word in ("age", "boys_", "girls_", "men_", "women_",
                                      "year", "wage", "pay"))


def _encode(frame, schema, given, deadline=None):
    """One-hot nominal fields and scaled ordered bands; no target encoding."""
    columns = []
    codes = []
    for name in given:
        _check(deadline)
        levels = list(schema["items"][name]["values"])
        lookup = {str(value): i for i, value in enumerate(levels)}
        raw = frame[name].to_numpy(dtype=object)
        code = np.fromiter((lookup.get(str(value), len(levels)) for value in raw),
                           dtype=np.int32, count=len(raw))
        codes.append(code)
        if _ordered(name):
            # Bands are already in their declared order.  Unknown is distinct.
            scale = max(1, len(levels) - 1)
            columns.append(np.minimum(code, len(levels) - 1)[:, None] /
                           scale * 0.8)
            columns.append((code == len(levels))[:, None].astype(float))
        else:
            weight = 2.0 if any(t in name.lower() for t in
                                ("country", "governorate", "province")) else 1.0
            columns.append(np.eye(len(levels) + 1, dtype=np.float32)[code] * weight)
    return np.column_stack(columns).astype(np.float32), np.column_stack(codes)


def _keys(codes, selected):
    matrix = np.ascontiguousarray(codes[:, selected])
    return [tuple(row) for row in matrix]


class NeighborIndex:
    """Shared nearest-row index. Build once, reuse across every target item."""

    def __init__(self, frame, schema, *, folds=5, seed=20261018,
                 candidates=96, leaf_trees=24, deadline=None):

        _check(deadline)
        self.frame = frame
        self.schema = schema
        self.deadline = deadline
        self.given = _given(frame, schema)
        self.visible = _visible(frame, schema)
        self.n = len(frame)
        self.fold = np.full(self.n, -1, dtype=np.int64)
        rng = np.random.default_rng(seed)
        if "wave" in self.given and "s1_2_country_asylum_lbl" in self.given:
            strata = frame.iloc[self.visible][
                ["wave", "s1_2_country_asylum_lbl"]].astype(str).agg("|".join, axis=1)
        elif "Country" in self.given:
            strata = frame.iloc[self.visible]["Country"].astype(str)
        else:
            strata = None
        if strata is None:
            self.fold[self.visible] = rng.permutation(len(self.visible)) % folds
        else:
            for value in strata.unique():
                rows = self.visible[(strata == value).to_numpy()]
                self.fold[rng.permutation(rows)] = np.arange(len(rows)) % folds
        self.X, self.codes = _encode(frame, schema, self.given, deadline)
        # Value-IDF categorical Hamming: code magnitudes have no metric meaning.
        self.idf=[]
        for j in range(self.codes.shape[1]):
            counts=np.bincount(self.codes[self.visible,j], minlength=int(self.codes[:,j].max())+1)
            self.idf.append(1.+np.log((len(self.visible)+1.)/(counts+1.)))
        self.exact = {}
        exact_sets = {"full": list(range(len(self.given)))}
        if "wave" in self.given:
            exact_sets["no_wave"] = [i for i, name in enumerate(self.given)
                                      if name != "wave"]
        for label, selection in exact_sets.items():
            _check(deadline)
            groups = {}
            keys = _keys(self.codes, selection)
            for offset, row in enumerate(self.visible):
                if offset % 256 == 0:_check(deadline)
                groups.setdefault(keys[row], []).append(int(row))
            self.exact[label] = [np.asarray(groups.get(key, ()), dtype=np.int32)
                                 for key in keys]

        # Country is a physical blocking field.  This cuts the distance work
        # by an order of magnitude and keeps neighbors in the same location.
        block_name = next((n for n in self.given if "country" in n.lower()),
                          self.given[0])
        block = self.codes[:, self.given.index(block_name)]
        self.knn = np.full((self.n, candidates), -1, dtype=np.int32)
        self.distance = np.full((self.n, candidates), np.inf, dtype=np.float32)
        for value in np.unique(block):
            library = self.visible[block[self.visible] == value]
            queries = np.flatnonzero(block == value)
            if not len(library):
                continue
            width = min(candidates + 1, len(library))
            for start in range(0, len(queries), 256):
                _check(deadline)
                q = queries[start:start + 256]
                dist=np.zeros((len(q),len(library)),np.float32)
                denom=np.zeros(len(q),np.float32)
                for j,weights in enumerate(self.idf):
                    w=weights[self.codes[q,j]].astype(np.float32)
                    dist += (self.codes[q,j,None]!=self.codes[library,j][None,:])*w[:,None]
                    denom+=w
                dist/=denom[:,None]
                # A visible query may be inside the library.
                near = np.argpartition(dist, width - 1, axis=1)[:, :width]
                ordered = np.argsort(np.take_along_axis(dist, near, axis=1), axis=1)
                near = np.take_along_axis(near, ordered, axis=1)
                for j, row in enumerate(q):
                    hits = library[near[j]]
                    hits = hits[hits != row][:candidates]
                    count = len(hits)
                    self.knn[row, :count] = hits
                    self.distance[row, :count] = dist[j, np.searchsorted(library, hits)]
        self.leaf_trees = leaf_trees
        self._leaf_rank = None
        self._leaf_partial = None
        self._candidate_cache = {}
        self._target_cache = {}
        self._plan_cache = {}

    def _allowed(self, row, candidates, fold_out):
        candidates = candidates[candidates >= 0]
        candidates = candidates[candidates != row]
        if fold_out and self.fold[row] >= 0:
            candidates = candidates[self.fold[candidates] != self.fold[row]]
        return candidates

    def context(self, query_rows, limit=64, *, fold_out=False):
        """TabPFN context: exact fingerprints first, then nearby visible rows."""
        out = np.full((len(query_rows), limit), -1, dtype=np.int32)
        for j, row in enumerate(query_rows):
            row = int(row)
            seen = set()
            chosen = []
            for source in (*self.exact.values(), self.knn):
                for hit in self._allowed(row, source[row], fold_out):
                    hit = int(hit)
                    if hit not in seen:
                        chosen.append(hit)
                        seen.add(hit)
                    if len(chosen) == limit:
                        break
                if len(chosen) == limit:
                    break
            out[j, :len(chosen)] = chosen
        return out

    def _joint_response(self):
        """Target summary for one multi-target GBDT; no hidden answers enter."""
        cols = []
        for name, item in self.schema["items"].items():
            if item["class"] != "PREDICT":
                continue
            options = list(item["values"])
            if item.get("gate"):
                options.append(self.schema.get("gated_value", "NA_GATED"))
            lookup = {str(v): i for i, v in enumerate(options)}
            raw = self.frame[name].to_numpy(dtype=object)[self.visible]
            code = np.fromiter((lookup.get(str(v), -1) for v in raw),
                               dtype=np.int32, count=len(raw))
            if np.any(code < 0):
                continue
            if len(np.unique(code)) > 1:
                cols.append((code / max(1, len(options) - 1)).astype(np.float32))
        if not cols:
            return np.zeros(len(self.visible), np.float32)
        return np.mean(np.column_stack(cols), axis=1)

    def leaf_candidates(self, deadline=None):
        """Rank k-NN candidates by cross-fit GBDT leaf co-occurrence."""
        _check(deadline)
        if self._leaf_rank is not None:
            return self._leaf_rank
        from sklearn.ensemble import GradientBoostingRegressor

        if self._leaf_partial is None:
            self._leaf_partial = (self._joint_response(), self.knn.copy(), set())
        label, rank, done_folds = self._leaf_partial
        fold_values = np.unique(self.fold[self.visible])
        for held_fold in (*fold_values, -1):
            if int(held_fold) in done_folds:
                continue
            _check(deadline)
            train_rows = self.visible if held_fold == -1 else self.visible[
                self.fold[self.visible] != held_fold]
            query_rows = np.flatnonzero(self.fold == held_fold)
            if held_fold == -1:
                query_rows = np.flatnonzero(self.fold < 0)
            if not len(query_rows):
                done_folds.add(int(held_fold))
                continue
            y = label[np.searchsorted(self.visible, train_rows)]
            model = GradientBoostingRegressor(n_estimators=self.leaf_trees,
                                              max_depth=3, learning_rate=0.08,
                                              random_state=20261018)
            model.fit(self.X[train_rows], y)
            qleaf = model.apply(self.X[query_rows]).reshape(len(query_rows), -1)
            lleaf = model.apply(self.X[self.visible]).reshape(len(self.visible), -1)
            for offset, row in enumerate(query_rows):
                if offset % 128 == 0:_check(deadline)
                hits = self._allowed(int(row), self.knn[row], held_fold != -1)
                if not len(hits):
                    continue
                pos = np.searchsorted(self.visible, hits)
                similarity = (lleaf[pos] == qleaf[offset]).sum(axis=1)
                # Stable order keeps the distance order within equal leaf score.
                ordered = hits[np.argsort(-similarity, kind="stable")]
                rank[row, :len(ordered)] = ordered
            done_folds.add(int(held_fold))
        self._leaf_rank = rank
        self._leaf_partial = None
        return rank

    def candidates_for(self, path, row, *, fold_out=False, deadline=None):
        if path in self.exact:
            return self._allowed(row, self.exact[path][row], fold_out)
        if path == "knn":
            return self._allowed(row, self.knn[row], fold_out)
        if path == "leaf":
            return self._allowed(row, self.leaf_candidates(deadline)[row], fold_out)
        raise ValueError(path)

    def distributions(self, name, rows, *, path="full", k=16,
                      fold_out=False, alpha=0.4, deadline=None):
        """Return (probabilities, support). Smoothing uses other-fold prior."""
        _check(deadline)
        if name not in self._target_cache:
            item = self.schema["items"][name]
            options = list(item["values"])
            if item.get("gate"):
                options.append(self.schema.get("gated_value", "NA_GATED"))
            lookup = {str(v): i for i, v in enumerate(options)}
            all_values = self.frame[name].to_numpy(dtype=object)
            encoded = np.fromiter((lookup.get(str(v), -1) for v in all_values),
                                  dtype=np.int32, count=self.n)
            width = len(options)
            priors = {}
            for fold in (-1, *np.unique(self.fold[self.visible])):
                lib = self.visible if fold == -1 else self.visible[
                    self.fold[self.visible] != fold]
                lib = lib[encoded[lib] >= 0]
                prior = np.bincount(encoded[lib], minlength=width).astype(float)
                prior += 1.0 / width
                priors[int(fold)] = prior / prior.sum()
            self._target_cache[name] = encoded, width, priors
        encoded, width, priors = self._target_cache[name]
        # A fixed 32-neighbor cache is shared by all fully observed targets.
        # Partially observed targets retain the exact label-filtering path below.
        if k <= 32 and np.all(encoded[self.visible] >= 0):
            if not hasattr(self, '_candidate_cache'):
                self._candidate_cache = {}
            key = (path, bool(fold_out))
            if key not in self._candidate_cache:
                self._candidate_cache[key] = np.full((self.n, 32), -2, np.int32)
            cache = self._candidate_cache[key]
            missing = np.asarray(rows)[cache[rows, 0] == -2]
            for offset, row in enumerate(missing):
                if offset % 128 == 0: _check(deadline)
                hits = self.candidates_for(path, int(row), fold_out=fold_out, deadline=deadline)[:32]
                cache[row] = -1
                cache[row, :len(hits)] = hits
            rows_arr = np.asarray(rows, dtype=np.int64)
            # Include every row: equal endpoints need not describe equal queries.
            key = (rows_arr.tobytes(), path, int(k), bool(fold_out))
            if key not in self._plan_cache:
                hits = cache[rows_arr, :k]
                valid = hits >= 0
                support = valid.sum(axis=1).astype(np.int32)
                rank_weights = (1. / np.sqrt(1. + np.arange(k))
                                if path == "knn" else np.ones(k))
                rr, cc = np.nonzero(valid)
                weight_sums = np.array([rank_weights[:count].sum() for count in range(k+1)])
                fold_values = np.unique(self.fold[self.visible])
                fold_ids = np.searchsorted(fold_values, self.fold[rows_arr])
                if not fold_out:
                    fold_ids[:] = len(fold_values)
                else:
                    fold_ids[self.fold[rows_arr] < 0] = len(fold_values)
                self._plan_cache[key] = (hits[rr, cc], support, weight_sums[support, None],
                                         rr, rank_weights[cc], fold_ids, fold_values)
            hits, support, weight_sums, rr, weights, fold_ids, fold_values = self._plan_cache[key]
            p_mat = np.stack([priors.get(int(f), priors[-1]) for f in fold_values] + [priors[-1]])
            prior = p_mat[fold_ids]
            counts = np.bincount(rr * width + encoded[hits], weights=weights,
                                 minlength=len(rows_arr) * width).reshape(len(rows_arr), width)
            probabilities = (counts + alpha * prior) / (weight_sums + alpha)
            probabilities[support == 0] = prior[support == 0]
            result = probabilities.astype(np.float32)
            _check(deadline)
            return result, support
        result = np.zeros((len(rows), width), np.float32)
        support = np.zeros(len(rows), np.int32)
        for j, row in enumerate(rows):
            if j % 128 == 0:_check(deadline)
            row = int(row)
            prior = priors[int(self.fold[row]) if fold_out else -1]
            hits = self.candidates_for(path, row, fold_out=fold_out, deadline=deadline)
            hits = hits[encoded[hits] >= 0][:k]
            support[j] = len(hits)
            if not len(hits):
                result[j] = prior
                continue
            if path == "knn":
                # Distance ranks approximate the metric when fold filtering
                # removes nearer rows.  Exact paths and leaf paths use counts.
                weights = 1.0 / np.sqrt(1.0 + np.arange(len(hits)))
            else:
                weights = np.ones(len(hits))
            counts = np.bincount(encoded[hits], weights=weights, minlength=width)
            result[j] = (counts + alpha * prior) / (weights.sum() + alpha)
        _check(deadline)
        return result, support


def build(frame, schema, **kwargs):
    """Construct a reusable index from the current predict() frame."""
    return NeighborIndex(frame, schema, **kwargs)


def select(index, name, baseline_oof, baseline_query, query_rows,
           *, threshold_nats=0.003, deadline=None):
    """Select a neighbor distribution by visible-row OOF log loss.

    ``baseline_oof`` is aligned to ``index.visible`` and must itself be OOF.
    ``baseline_query`` is aligned to ``query_rows``.  Returns query probabilities
    and a report with baseline/selected NLL, support and the chosen path.
    The official 0.001 floor is applied only when measuring loss.
    """
    unchanged=(np.asarray(baseline_query,dtype=float).copy(),
               {'path':None,'Contextual':float('nan')},
               np.asarray(baseline_oof,dtype=float).copy())
    if deadline is not None and time.monotonic() >= deadline:return unchanged
    try:
        visible = index.visible
        encoded, width, _ = index._target_cache.get(name, (None, None, None))
        if encoded is None:
            index.distributions(name, visible[:1], deadline=deadline)
            encoded, width, _ = index._target_cache[name]
        valid = encoded[visible] >= 0
        rows = visible[valid]
        baseline = np.asarray(baseline_oof)[valid]
        y = encoded[rows]
        if len(rows) == 0:
            raise ValueError(f"{name} has no visible answers")

        def row_loss(chosen):
            return -np.log(0.001 + (1.0 - 0.001 * width) * chosen)

        row_idx = np.arange(len(rows))
        base_chosen = baseline[row_idx, y]
        base_losses = row_loss(base_chosen)
        initial = float(base_losses.mean())
        best = (initial, None, None, 0.0, 0, None)
        paths = [("full", 16)]
        if "no_wave" in index.exact:
            paths.append(("no_wave", 16))
        paths += [("knn", 8), ("knn", 32), ("leaf", 16)]
        half = index.fold[rows] % 2 == 0
        half_losses = [float(base_losses[h].mean()) for h in (half, ~half)]
        large = len(index.schema['items']) > 30
        weights = (0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0) if large else (0.25, 0.5, 0.75, 1.0)
        for path, k in paths:
            if deadline is not None and time.monotonic() >= deadline:
                break
            probs, support = index.distributions(name, rows, path=path, k=k,
                                                  fold_out=True, deadline=deadline)
            active = support > 0
            if active.sum() < (40 if path in index.exact else 200):
                continue
            prob_chosen = probs[row_idx, y]
            base_active_loss = float(base_losses[active].mean())
            for weight in weights:
                trial_chosen = base_chosen.copy()
                trial_chosen[active] = base_chosen[active] * (1.0 - weight) + prob_chosen[active] * weight
                losses = row_loss(trial_chosen)
                loss = float(losses.mean())
                threshold = (0.0003 if path in index.exact else 0.0015) if large else threshold_nats
                if loss >= best[0] or initial - loss < threshold:
                    continue
                if large and path in index.exact and base_active_loss - float(losses[active].mean()) < 0.008:
                    continue
                # Each half of the OOF rows must point in the same direction.
                if float(losses[half].mean()) >= half_losses[0]:
                    continue
                if float(losses[~half].mean()) >= half_losses[1]:
                    continue
                best = (loss, path, k, weight, int(active.sum()), (probs, active))

        loss, path, k, weight, count, selected_valid = best
        selected_oof = np.asarray(baseline_oof).copy()
        if selected_valid is not None:
            probs, active = selected_valid
            selected_valid = baseline.copy()
            selected_valid[active] = baseline[active] * (1.0 - weight) + probs[active] * weight
            selected_oof[valid] = selected_valid
        query = np.asarray(baseline_query).copy()
        if path is not None:
            probabilities, support = index.distributions(name, query_rows, path=path,
                                                          k=k, fold_out=False, deadline=deadline)
            active = support > 0
            query[active] = query[active] * (1.0 - weight) + probabilities[active] * weight
        report = {"baseline_nll": initial, "selected_nll": loss,
                  "delta_nll": initial - loss, "path": path, "k": k,
                  "weight": weight, "oof_supported": count,
                  "n_visible": len(rows)}
        return query, report, selected_oof
    except BudgetExpired:
        return unchanged
