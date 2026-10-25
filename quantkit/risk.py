"""Portfolio risk: Value-at-Risk and Expected Shortfall (parametric, historical,
Monte Carlo), plus drawdown and scenario stress tests.

Conventions: `returns` are simple periodic returns (rows = periods, columns =
assets). VaR and ES are reported as positive loss fractions of portfolio value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _portfolio_returns(returns, weights=None) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        return r
    w = np.full(r.shape[1], 1 / r.shape[1]) if weights is None else np.asarray(weights, float)
    return r @ w


def parametric_var(returns, weights=None, alpha=0.99, horizon=1) -> float:
    """Variance-covariance (Gaussian) VaR, scaled by sqrt(horizon)."""
    p = _portfolio_returns(returns, weights)
    mu, sd = p.mean() * horizon, p.std(ddof=1) * np.sqrt(horizon)
    return float(-(mu + norm.ppf(1 - alpha) * sd))


def parametric_es(returns, weights=None, alpha=0.99, horizon=1) -> float:
    p = _portfolio_returns(returns, weights)
    mu, sd = p.mean() * horizon, p.std(ddof=1) * np.sqrt(horizon)
    return float(-mu + sd * norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))


def historical_var(returns, weights=None, alpha=0.99) -> float:
    p = _portfolio_returns(returns, weights)
    return float(-np.quantile(p, 1 - alpha))


def historical_es(returns, weights=None, alpha=0.99) -> float:
    p = _portfolio_returns(returns, weights)
    cutoff = np.quantile(p, 1 - alpha)
    return float(-p[p <= cutoff].mean())


def monte_carlo_var(returns, weights=None, alpha=0.99, horizon=1, n_sims=100_000, seed=None):
    """Simulate correlated multivariate-normal asset returns from the sample
    mean and covariance. Returns (VaR, ES)."""
    r = np.atleast_2d(np.asarray(returns, float))
    if r.shape[0] == 1:
        r = r.T
    w = np.full(r.shape[1], 1 / r.shape[1]) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(r.mean(0) * horizon, np.cov(r, rowvar=False, ddof=1).reshape(len(w), len(w)) * horizon, n_sims)
    p = sims @ w
    cutoff = np.quantile(p, 1 - alpha)
    return float(-cutoff), float(-p[p <= cutoff].mean())


def max_drawdown(returns) -> float:
    """Largest peak-to-trough decline of the compounded equity curve (positive fraction)."""
    equity = np.cumprod(1 + np.asarray(returns, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(-(equity / peak - 1).min())


def stress_test(weights, shocks: dict[str, np.ndarray] | pd.DataFrame) -> pd.Series:
    """Portfolio P&L (fraction of value) under named per-asset return shocks."""
    w = np.asarray(weights, float)
    items = shocks.items() if isinstance(shocks, dict) else shocks.iterrows()
    return pd.Series({name: float(np.asarray(s, float) @ w) for name, s in items}, name="pnl")


def risk_report(returns, weights=None, alpha=0.99, seed=0) -> pd.Series:
    mc_var, mc_es = monte_carlo_var(returns, weights, alpha, seed=seed)
    p = _portfolio_returns(returns, weights)
    return pd.Series({
        f"Parametric VaR {alpha:.0%}": parametric_var(returns, weights, alpha),
        f"Historical VaR {alpha:.0%}": historical_var(returns, weights, alpha),
        f"Monte Carlo VaR {alpha:.0%}": mc_var,
        f"Parametric ES {alpha:.0%}": parametric_es(returns, weights, alpha),
        f"Historical ES {alpha:.0%}": historical_es(returns, weights, alpha),
        f"Monte Carlo ES {alpha:.0%}": mc_es,
        "Max drawdown": max_drawdown(p),
    })
