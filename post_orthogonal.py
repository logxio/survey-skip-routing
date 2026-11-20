"""Post-calibration GIVEN trees and nominal IRT, isolated from the base chain."""
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from chain_features import BudgetExpired, check_clock, topological
from propagate_structural_gates import ZERO_ON_SKIP


def _prior(y, train, width):
    counts = np.bincount(y[train], minlength=width)
    return (counts + .5) / (counts.sum() + .5 * width)


def _map_proba(model, query, width, frame):
    out = np.zeros((len(query), width), dtype=float)
    if model is None or len(query) == 0:
        return out
    pred = model.predict_proba(frame.iloc[query])
    classes = np.asarray(getattr(model, 'classes_', np.arange(pred.shape[1]))).astype(int)
    out[:, classes] = pred
    return out


def _item_values(schema, name):
    values = list(schema['items'][name]['values'])
    gated = schema.get('gated_value', 'NA_GATED')
    if schema['items'][name].get('gate') and gated not in values:
        values.append(gated)
    return values


def wave_one_rows(frame):
    mask = np.zeros(len(frame), dtype=bool)
    if 'panel_hh_key' in frame:
        mask |= frame['panel_hh_key'].astype(str).str.startswith('round1_').to_numpy()
    if 'wave' in frame:
        mask |= frame['wave'].astype(str).isin(('1', '1.0')).to_numpy()
    return mask


def rescale_skip(prob, base, schema, name, closed=None, wave_one_mask=None):
    """Keep a gated item's conditional open-category ratios while locking skip mass to base."""
    out = np.asarray(prob, dtype=float).copy()
    ref = np.asarray(base, dtype=float)
    item = schema['items'][name]
    if not item.get('gate') or out.ndim != 2 or out.shape[1] <= 2:
        return out
    if ref.shape != out.shape:
        return out
    skip = np.clip(ref[:, -1], 1e-9, 1.0 - 1e-9)
    open_part = np.clip(out[:, :-1], 1e-12, None)
    open_part /= np.maximum(open_part.sum(axis=1, keepdims=True), 1e-12)
    out[:, :-1] = open_part * (1.0 - skip[:, None])
    out[:, -1] = skip
    classes = _item_values(schema, name)
    if name in ZERO_ON_SKIP and str(classes[0]) == '0' and wave_one_mask is not None:
        first = np.asarray(wave_one_mask, dtype=bool) & (ref[:, -1] < 1e-6) & (ref[:, 0] >= 0.995) & (out[:, 0] < ref[:, 0])
        answers = np.clip(out[first, 1:-1], 0.0, None)
        if answers.shape[1]:
            total = answers.sum(1, keepdims=True)
            answers = np.divide(answers, total, out=np.full_like(answers, 1.0 / answers.shape[1]), where=total > 0)
            out[first, 1:-1] = answers * np.maximum(0.0, 1.0 - ref[first, 0:1] - ref[first, -1:])
        out[first, 0] = ref[first, 0]
        out[first, -1] = ref[first, -1]
    if closed is not None:
        closed = np.asarray(closed, dtype=bool)
        out[closed] = ref[closed]
    return out


def tree_pair(frame, schema, name, train, query, seed, deadline, given, x, cat,
              fit_stop=None, skip_refit=False, rem_wave=None, base_stop=None,
              base_oof=None, base_rows=None):
    """Fit CatBoost and LightGBM, with an optional soft per-item deadline."""
    try:
        from catboost import CatBoostClassifier, Pool
        from lightgbm import LGBMClassifier, early_stopping
        from lightgbm.callback import EarlyStopException

        values = _item_values(schema, name)
        y = pd.Categorical(frame[name], categories=values).codes.astype(np.int64)
        width = len(values)
        prior = _prior(y, train, width)
        if len(train) < 10 or np.count_nonzero(np.bincount(y[train], minlength=width)) < 2:
            return np.broadcast_to(prior, (len(query), width)).copy(), {'cb_best': 0, 'best_lg': 0}

        counts = np.bincount(y[train], minlength=width)
        strata = y[train] if counts[counts > 0].min() >= 2 else None
        fit, stop = train_test_split(train, test_size=.15, random_state=seed, stratify=strata)
        for label in np.unique(y[train]):
            if not np.any(y[fit] == label):
                moved = stop[np.flatnonzero(y[stop] == label)[0]]
                fit = np.append(fit, moved)
                stop = stop[stop != moved]
        if base_stop is None and base_oof is not None:
            base_oof = np.asarray(base_oof)
            if base_oof.ndim == 2 and base_oof.shape[0] == len(frame):
                base_stop = base_oof[stop]
            elif base_oof.ndim == 2 and base_rows is not None and base_oof.shape[0] == len(base_rows):
                base_stop = base_oof[np.searchsorted(np.asarray(base_rows), stop)]

        soft = fit_stop is not None and fit_stop < deadline - 1e-9
        stop_clock = deadline if fit_stop is None else fit_stop
        item_began = time.monotonic()
        item_span = max(0.0, stop_clock - item_began)
        bounded = (not soft) and np.isfinite(deadline) and (item_span < 120.0)
        cb_fit_stop = (item_began + (0.42 if skip_refit else 0.24) * item_span) if soft else deadline
        cb_refit_stop = (item_began + 0.45 * item_span) if soft else deadline
        if bounded:
            cb_fit_stop = min(deadline - 3.2, item_began + 0.36 * item_span)
            cb_refit_stop = min(deadline - 2.0, item_began + 0.68 * item_span)

        cb_params = dict(iterations=350 if width <= 10 else 160, depth=6 if width <= 10 else 5, learning_rate=.06,
            one_hot_max_size=2, boosting_type='Plain', max_ctr_complexity=1,
            l2_leaf_reg=3, thread_count=1, verbose=False,
            random_seed=seed, allow_writing_files=False)
        params = dict(n_estimators=500, learning_rate=.04, num_leaves=15,
            min_child_samples=30, cat_smooth=20, cat_l2=10, min_data_per_group=50,
            reg_lambda=3, n_jobs=1, verbosity=-1, deterministic=True,
            force_col_wise=True, random_state=seed)
        if soft:
            wave_seconds = max(0.0, float(rem_wave if rem_wave is not None else item_span))
            cb_params['iterations'] = min(200 if width <= 10 else 110, max(40, int(wave_seconds * 35)))
            params['n_estimators'] = min(260, max(50, int(wave_seconds * 55)))

        class Clock:
            def __init__(self, limit):
                self.limit = limit

            def after_iteration(self, info):
                return time.monotonic() < self.limit

        def make_lgb_clock(limit):
            def _cb(env):
                if time.monotonic() >= limit and env.iteration > 0:
                    raise EarlyStopException(env.iteration, env.evaluation_result_list)
            return _cb

        def marginal_prior():
            return np.broadcast_to(prior, (len(query), width)).copy()

        obs = np.unique(y[fit])
        p_obs = np.clip(prior[obs], 1e-6, 1.0)
        p_obs = p_obs / np.maximum(p_obs.sum(), 1e-12)
        cat_cols = list(range(x.shape[1]))

        def _cb_pool(idx, labels=None, use_baseline=True):
            if len(obs) == 2:
                baseline = np.full(len(idx), float(np.log(p_obs[1] / p_obs[0])), dtype=float)
            else:
                baseline = np.broadcast_to(np.log(p_obs).astype(float), (len(idx), len(obs))).copy()
            kwargs = dict(cat_features=cat_cols)
            if use_baseline:
                kwargs['baseline'] = baseline
            return Pool(cat.iloc[idx], label=None if labels is None else labels, **kwargs)

        def _cb_predict(model, idx, use_baseline=True):
            pred = model.predict_proba(_cb_pool(idx, use_baseline=use_baseline))
            classes = np.asarray(getattr(model, 'classes_', np.arange(pred.shape[1]))).astype(int)
            out = np.zeros((len(idx), width), dtype=float)
            out[:, classes] = pred
            return out

        def _nll(prob, labels):
            if len(labels) == 0:
                return float('inf')
            p = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0)
            p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
            return float(-np.log(p[np.arange(len(labels)), labels]).mean())

        try:
            check_clock(deadline)
            baseline_mode = os.environ.get('SIMULACRA_CAT_BASELINE', 'dynamic')
            use_baseline = baseline_mode == 'all' or (
                baseline_mode == 'dynamic' and (soft or cb_params['iterations'] <= 120))
            cb_fit = CatBoostClassifier(**cb_params)
            cb_fit.fit(_cb_pool(fit, y[fit], use_baseline=use_baseline),
                eval_set=_cb_pool(stop, y[stop], use_baseline=use_baseline),
                early_stopping_rounds=30, callbacks=[Clock(cb_fit_stop)])
            cb_best = max(1, cb_fit.get_best_iteration() + 1)
            cb_params['iterations'] = cb_best
            cb = cb_fit if (soft and skip_refit) else CatBoostClassifier(**cb_params)
            if cb is not cb_fit:
                cb.fit(_cb_pool(train, y[train], use_baseline=use_baseline), callbacks=[Clock(cb_refit_stop)])
            pc = _cb_predict(cb, query, use_baseline=use_baseline)
            pc_stop = _cb_predict(cb_fit, stop, use_baseline=use_baseline)

            lg_began = time.monotonic()
            lg_span = max(0.12, stop_clock - lg_began)
            lg_fit_stop = (lg_began + (0.95 if skip_refit else 0.55) * lg_span) if soft else deadline
            lg_refit_stop = max(lg_began + 0.12, stop_clock) if soft else deadline
            if bounded:
                lg_fit_stop = min(deadline - 1.0, lg_began + 0.65 * lg_span)
                lg_refit_stop = max(lg_began + 0.12, deadline - 0.3)

            check_clock(deadline)
            lg_fit = LGBMClassifier(**params)
            try:
                lg_fit.fit(x.iloc[fit], y[fit], eval_set=[(x.iloc[stop], y[stop])],
                    callbacks=[early_stopping(30, verbose=False), make_lgb_clock(lg_fit_stop) if (soft or bounded) else lambda env: check_clock(deadline)])
            except EarlyStopException:
                if not hasattr(lg_fit, '_Booster'):
                    params['n_estimators'] = 1
                    lg_fit = LGBMClassifier(**params)
                    lg_fit.fit(x.iloc[fit], y[fit], callbacks=[])
            best_lg = max(1, int(getattr(lg_fit, 'best_iteration_', 0) or getattr(getattr(lg_fit, '_Booster', None), 'current_iteration', lambda: 1)() or 1))
            params['n_estimators'] = best_lg
            lg = lg_fit if (soft and skip_refit) else LGBMClassifier(**params)
            if lg is not lg_fit:
                try:
                    lg.fit(x.iloc[train], y[train], callbacks=[make_lgb_clock(lg_refit_stop) if (soft or bounded) else lambda env: check_clock(deadline)])
                except EarlyStopException:
                    lg = lg_fit
            pl = _map_proba(lg, query, width, x)
            pl_stop = _map_proba(lg_fit, stop, width, x)
            if base_stop is not None and np.shape(base_stop) == np.shape(pc_stop):
                pc_stop = rescale_skip(pc_stop, base_stop, schema, name, wave_one_mask=wave_one_rows(frame.iloc[stop]))
                pl_stop = rescale_skip(pl_stop, base_stop, schema, name, wave_one_mask=wave_one_rows(frame.iloc[stop]))
            alpha_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
            if cb_best >= 5 and best_lg >= 5 and abs(_nll(pc_stop, y[stop]) - _nll(pl_stop, y[stop])) <= 0.02:
                alpha_grid = (0.25, 0.5, 0.75)
            alpha = min(alpha_grid, key=lambda a: _nll(a * pc_stop + (1.0 - a) * pl_stop, y[stop]))
            tree_stop = alpha * pc_stop + (1.0 - alpha) * pl_stop
            tree_rest = alpha * pc + (1.0 - alpha) * pl
            stats = {'cb_best': int(cb_best), 'best_lg': int(best_lg),
                'alpha_star': float(alpha), 'pc_stop_nll': _nll(pc_stop, y[stop]),
                'pl_stop_nll': _nll(pl_stop, y[stop]), 'tree_stop_nll': _nll(tree_stop, y[stop])}
            if base_stop is not None and np.shape(base_stop) == np.shape(tree_stop):
                stats['base_stop_nll'] = _nll(base_stop, y[stop])
                stats['delta_stop'] = float(stats['tree_stop_nll'] - stats['base_stop_nll'])
                stats['stop_grid_nll'] = {str(w): _nll((1.0 - w) * base_stop + w * tree_stop, y[stop])
                                          for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)}
            return tree_rest, stats
        except BudgetExpired:
            return None, {'cb_best': 0, 'best_lg': 0}
    except (ImportError, OSError):
        return None, {'cb_best': 0, 'best_lg': 0}


def closed_given(frame, schema, name):
    gate = schema['items'][name].get('gate')
    if not gate or schema['items'][gate['parent']]['class'] != 'GIVEN':
        return np.zeros(len(frame), dtype=bool)
    values = frame[gate['parent']]
    unknown = values.isna() | values.astype(str).isin(['No response', 'NA', 'nan', 'None']) | values.astype(str).str.startswith('-')
    return (~unknown & ~values.isin(gate['observed_if'])).to_numpy()


def augment(frame, schema, work, allowed, fold, final, final_oofs, n_raw_given, deadline):
    began = time.monotonic()
    report = dict(completed=False, fixed_weights=[.70, .25, .05],
        selected_strategy='S5_stop_convex_opt', stop_grid=(0.0, .05, .10, .15, .20, .25),
        dynamic_items=0, skip_refit_items=0, items={},
        irt_query_rows=0, irt_epochs=None, irt_seconds=0.0, irt_completed=False)
    if len(allowed) < 10:
        report['seconds'] = time.monotonic() - began
        return final, final_oofs, report
    updated = {k: v.copy() for k, v in final.items()}
    updated_oofs = {k: v.copy() for k, v in final_oofs.items()}
    try:
        try:
            from irt_gh81 import predict as irt_predict
            import torch
        except (ImportError, OSError):
            irt_predict = torch = None
        targets = list(work)
        given = [k for k, v in schema['items'].items() if v['class'] == 'GIVEN']
        x = pd.DataFrame(index=np.arange(len(frame)))
        for key in given:
            x[key] = pd.Categorical(frame[key].astype(str), categories=schema['items'][key]['values'])
        cat = x.astype(object).where(x.notna(), 'NA').astype(str)
        local_schema = dict(schema, items={k: v for k, v in schema['items'].items()
            if v['class'] == 'GIVEN' or k in work})
        visible = frame.copy()
        for name in targets:
            visible.loc[~work[name]['train'], name] = np.nan
        need_rows = np.flatnonzero(np.any(
            np.column_stack([~work[k]['train'] for k in targets]), axis=1))
        query_frame = visible.iloc[need_rows].copy()
        query_frame[targets] = np.nan
        irt_budget = .22 if len(work) > 30 else .45
        irt_deadline = began + irt_budget * max(0.0, deadline - began)
        irt_epochs = 45 if len(targets) > 30 else 100
        report['irt_query_rows'] = int(len(need_rows))
        report['irt_epochs'] = irt_epochs
        irt_probs = None
        irt_started = time.monotonic()
        try:
            if torch is not None and irt_predict is not None:
                with torch.random.fork_rng(devices=[]):
                    irt_probs, _ = irt_predict(
                        visible.iloc[np.asarray(allowed)], query_frame, local_schema,
                        seed=20261114, epochs=irt_epochs, deadline=irt_deadline)
                report['irt_completed'] = True
        except (BudgetExpired, ImportError, OSError):
            irt_probs = None
        finally:
            report['irt_seconds'] = time.monotonic() - irt_started

        order = topological(list(work), schema['items'])
        workers = max(1, min(8, os.cpu_count() or 1, int(os.environ.get('SIMULACRA_THREADS', '8'))))
        completed_count = 0
        next_index = 0
        pending = {}

        def run_tree(index, name, fit_stop, skip_refit, rem_wave):
            job = work[name]
            rows = np.flatnonzero(job['train'])
            rest = np.flatnonzero(~job['train'])
            if not len(rest):
                return index, name, None, {'cb_best': 0, 'best_lg': 0}
            tree_rest, tree_stats = tree_pair(frame, schema, name, rows, rest, 20261114 + index,
                deadline, given, x, cat, fit_stop=fit_stop,
                skip_refit=skip_refit, rem_wave=rem_wave,
                base_oof=final_oofs.get(name), base_rows=rows)
            return index, name, tree_rest, tree_stats

        def submit_one(executor, index, name):
            waves_left = max(1, (len(order) - completed_count + workers - 1) // workers)
            rem_wave = max(0.35, (deadline - time.monotonic() - 1.5) / waves_left)
            dynamic = len(work) > 30 or rem_wave < 15.0
            fit_stop = deadline
            skip_refit = False
            if dynamic:
                fit_stop = min(deadline - 1.2, time.monotonic() + max(0.45, 0.92 * rem_wave))
                skip_refit = (len(work) > 30 and rem_wave < 8.0) or rem_wave < 4.0
                report['dynamic_items'] += 1
                if skip_refit:
                    report['skip_refit_items'] += 1
            future = executor.submit(run_tree, index, name, fit_stop, skip_refit, rem_wave)
            pending[future] = (index, name)

        with ThreadPoolExecutor(
            max_workers=workers,
            initializer=lambda: threadpool_limits(limits=1, user_api='openmp'),
        ) as executor:
            while next_index < len(order) and len(pending) < workers:
                submit_one(executor, next_index, order[next_index])
                next_index += 1
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future, None)
                    index, name, tree_rest, tree_stats = future.result()
                    completed_count += 1
                    if tree_rest is None or tree_stats.get('cb_best', 0) <= 0 or tree_stats.get('best_lg', 0) <= 0:
                        while next_index < len(order) and len(pending) < workers:
                            submit_one(executor, next_index, order[next_index])
                            next_index += 1
                        continue
                    job = work[name]
                    rest = np.flatnonzero(~job['train'])
                    base_full = final[name].copy()
                    closed = closed_given(frame, schema, name)[rest]
                    stop_grid = tree_stats.get('stop_grid_nll') or {}
                    tree_w = float(min((0.0, .05, .10, .15, .20, .25),
                                       key=lambda w: float(stop_grid.get(str(w), 1e99))) if stop_grid else .25)
                    if irt_probs is None:
                        base_w, irt_w = 1.0 - tree_w, 0.0
                    else:
                        base_w, irt_w = .95 - tree_w, .05
                    irt_take = np.searchsorted(need_rows, rest)
                    irt_rest = (np.zeros_like(tree_rest) if irt_probs is None
                                else irt_probs[name][irt_take])
                    tree_rest = rescale_skip(tree_rest, base_full[rest], schema, name, closed, wave_one_rows(frame.iloc[rest]))
                    if irt_probs is not None:
                        irt_rest = rescale_skip(irt_rest, base_full[rest], schema, name, closed, wave_one_rows(frame.iloc[rest]))
                    mixed_rest = base_w * base_full[rest] + tree_w * tree_rest + irt_w * irt_rest
                    mixed_rest[closed] = base_full[rest][closed]
                    base_full[rest] = mixed_rest
                    updated[name] = base_full
                    report['items'][name] = dict(accepted=True, **tree_stats)
                    while next_index < len(order) and len(pending) < workers:
                        submit_one(executor, next_index, order[next_index])
                        next_index += 1
        report.update(completed=True, selected=len(report['items']))
        cb_counts = [int(v['cb_best']) for v in report['items'].values() if int(v.get('cb_best', 0)) > 0]
        lg_counts = [int(v['best_lg']) for v in report['items'].values() if int(v.get('best_lg', 0)) > 0]
        report['tree_stats'] = {
            'catboost': {'count': len(cb_counts),
                'mean': float(np.mean(cb_counts)) if cb_counts else None,
                'median': float(np.median(cb_counts)) if cb_counts else None,
                'min': int(min(cb_counts)) if cb_counts else None},
            'lightgbm': {'count': len(lg_counts),
                'mean': float(np.mean(lg_counts)) if lg_counts else None,
                'median': float(np.median(lg_counts)) if lg_counts else None,
                'min': int(min(lg_counts)) if lg_counts else None},
        }
        return updated, updated_oofs, report
    except BudgetExpired:
        report.update(completed=False, selected=len(report['items']))
        return updated, updated_oofs, report
    finally:
        report['seconds'] = time.monotonic() - began

