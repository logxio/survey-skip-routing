"""Simulator-agnostic skip-logic KL routing and cross-fitted PPI++ fieldwork planning.

Drop-in NumPy utilities for multi-module household and longitudinal surveys
(XLSForm / KoBoToolbox `relevant` filters, MICS modules, and panel follow-ups).
Works with probability tables from any LLM persona simulator, tabular foundation
model, or empirical-Bayes / gradient-boosted survey predictor.
"""
from __future__ import annotations

import numpy as np


def project_skip_probability(
    probs: np.ndarray,
    rho: np.ndarray | float,
    skip_idx: int = -1,
    floor: float = 0.0,
) -> np.ndarray:
    """Simultaneous forward/reverse KL projection onto {q in Delta^{K-1} : q[skip_idx] = rho}.

    Adjusts non-ask or wave-attrition mass to `rho` while leaving asked-conditional
    substantive answer odds strictly invariant. Pass `floor=1e-3` when scoring under
    epsilon-floored log loss.
    """
    q = np.asarray(probs, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError("probs must be a 2D array of shape (n_respondents, n_categories >= 2).")
    r = np.broadcast_to(np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0).reshape(-1), (q.shape[0],))
    k = q.shape[1]
    s = skip_idx % k
    asked = [j for j in range(k) if j != s]
    denom = q[:, asked].sum(axis=1, keepdims=True)
    cond = np.where(denom > 1e-15, q[:, asked] / np.maximum(denom, 1e-15), 1.0 / (k - 1))
    out = np.empty_like(q)
    out[:, asked] = (1.0 - r[:, None]) * cond
    out[:, s] = r
    return floor + (1.0 - k * floor) * out if floor > 0.0 else out


def ppi_categorical_ci(
    y_labeled: np.ndarray,
    probs_labeled_oof: np.ndarray,
    probs_unlabeled: np.ndarray,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fitted clipped PPI++ category shares, confidence intervals, and sample multipliers.

    Parameters
    ----------
    y_labeled : integer array of shape (n,) with primary interview category codes in 0..K-1.
    probs_labeled_oof : cross-fitted (out-of-fold) simulator probabilities of shape (n, K).
    probs_unlabeled : simulator probabilities on augmented / unlabeled respondents of shape (N, K).
    z : normal critical value (default 1.96 for 95% confidence intervals).

    Returns
    -------
    est, ci_low, ci_high, n_eff_mult : each of shape (K,), where
        `n_eff_mult = (SE_human / SE_PPI)^2 >= 1` is the effective primary-interview
        sample-size multiplier achieved by combining `n` interviews with `N` predictions.
    """
    p_lab = np.asarray(probs_labeled_oof, dtype=np.float64)
    p_unl = np.asarray(probs_unlabeled, dtype=np.float64)
    n, n_unl = len(p_lab), len(p_unl)
    y_oh = np.eye(p_lab.shape[1])[np.asarray(y_labeled, dtype=np.int64)]
    cov_yp = ((y_oh - y_oh.mean(axis=0)) * (p_lab - p_lab.mean(axis=0))).sum(axis=0) / (n - 1)
    denom = p_lab.var(axis=0, ddof=1) + (n / n_unl) * p_unl.var(axis=0, ddof=1)
    lam = np.where(denom > 1e-15, np.clip(cov_yp / np.maximum(denom, 1e-15), 0.0, 1.0), 0.0)
    est = y_oh.mean(axis=0) + lam * (p_unl.mean(axis=0) - p_lab.mean(axis=0))
    se = np.sqrt((y_oh - lam * p_lab).var(axis=0, ddof=1) / n + (lam ** 2) * p_unl.var(axis=0, ddof=1) / n_unl)
    se_human = np.sqrt(y_oh.var(axis=0, ddof=1) / n)
    n_eff_mult = np.where(se > 1e-15, np.maximum((se_human / np.maximum(se, 1e-15)) ** 2, 1.0), 1.0)
    return est, est - z * se, est + z * se, n_eff_mult


def interview_budget_plan(
    target_effective_n: int,
    n_eff_multiplier: float,
    cost_per_interview: float = 1.0,
) -> dict[str, float]:
    """Translate an empirical PPI++ sample multiplier into primary interview budget savings."""
    mult = max(float(n_eff_multiplier), 1.0)
    required_n = int(np.ceil(target_effective_n / mult))
    saved_n = max(target_effective_n - required_n, 0)
    return {
        "target_effective_interviews": float(target_effective_n),
        "required_primary_interviews": float(required_n),
        "interviews_saved": float(saved_n),
        "budget_reduction_pct": 100.0 * saved_n / max(target_effective_n, 1),
        "estimated_cost_saved": saved_n * float(cost_per_interview),
    }


if __name__ == "__main__":
    base = np.array([[0.36, 0.24, 0.20, 0.20], [0.10, 0.15, 0.25, 0.50], [0.05, 0.05, 0.10, 0.80]])
    routed = project_skip_probability(base, rho=np.array([0.04, 0.62, 0.96]), skip_idx=-1)
    assert np.allclose(routed[0], [0.432, 0.288, 0.240, 0.040])
    assert np.allclose(routed[:, :-1] / routed[:, :-1].sum(axis=1, keepdims=True), base[:, :-1] / base[:, :-1].sum(axis=1, keepdims=True))
    plan = interview_budget_plan(target_effective_n=1000, n_eff_multiplier=1.36, cost_per_interview=45.0)
    print("Routed row 0:", np.round(routed[0], 4).tolist(), "| RECS peak budget plan:", plan)
