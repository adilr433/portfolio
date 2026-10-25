"""Mean-variance portfolio optimization: minimum variance, maximum Sharpe,
target-return portfolios and the efficient frontier (long-only by default)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Portfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def annualize(returns: pd.DataFrame, periods: int = 252):
    """Annualized mean vector and covariance matrix from periodic returns."""
    return returns.mean() * periods, returns.cov() * periods


def _stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def _solve(objective, mu, cov, long_only, max_weight, extra=()):
    n = len(mu)
    lo = 0.0 if long_only else -1.0
    bounds = [(lo, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"optimizer failed: {res.message}")
    return res.x


def _wrap(w, mu, cov, rf):
    names = mu.index if isinstance(mu, pd.Series) else range(len(mu))
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    return Portfolio(pd.Series(w, index=names, name="weight"), *_stats(w, np.asarray(mu), np.asarray(cov), rf))


def min_variance(mu, cov, rf=0.0, long_only=True, max_weight=1.0) -> Portfolio:
    C = np.asarray(cov)
    w = _solve(lambda w: w @ C @ w, mu, C, long_only, max_weight)
    return _wrap(w, mu, cov, rf)


def max_sharpe(mu, cov, rf=0.0, long_only=True, max_weight=1.0) -> Portfolio:
    m, C = np.asarray(mu), np.asarray(cov)
    w = _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ C @ w), mu, C, long_only, max_weight)
    return _wrap(w, mu, cov, rf)


def target_return(mu, cov, target, rf=0.0, long_only=True, max_weight=1.0) -> Portfolio:
    m, C = np.asarray(mu), np.asarray(cov)
    con = ({"type": "eq", "fun": lambda w: w @ m - target},)
    w = _solve(lambda w: w @ C @ w, mu, C, long_only, max_weight, con)
    return _wrap(w, mu, cov, rf)


def efficient_frontier(mu, cov, n_points=30, rf=0.0, long_only=True, max_weight=1.0) -> pd.DataFrame:
    """Return/volatility/Sharpe along the frontier from the min-variance point
    up to the highest attainable return."""
    start = min_variance(mu, cov, rf, long_only, max_weight).expected_return
    top = float(np.max(mu)) if long_only and max_weight >= 1 else float(np.max(mu)) * 1.5
    rows = []
    for t in np.linspace(start, top, n_points):
        try:
            p = target_return(mu, cov, t, rf, long_only, max_weight)
        except RuntimeError:
            continue
        rows.append({"return": p.expected_return, "volatility": p.volatility, "sharpe": p.sharpe})
    return pd.DataFrame(rows)
