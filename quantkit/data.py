"""Reproducible synthetic market data so every example and test runs offline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_gbm(s0=100.0, mu=0.08, sigma=0.2, days=252 * 5, seed=0, start="2020-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days)
    idx = pd.bdate_range(start, periods=days)
    return pd.Series(s0 * np.exp(np.cumsum(shocks)), index=idx, name="price")


def simulate_returns(n_assets=5, days=252 * 5, seed=0, start="2020-01-01") -> pd.DataFrame:
    """Correlated daily returns with fat tails (Student-t, df=5) for n assets."""
    rng = np.random.default_rng(seed)
    ann_mu = rng.uniform(0.03, 0.14, n_assets)
    ann_vol = rng.uniform(0.12, 0.35, n_assets)
    a = rng.normal(size=(n_assets, n_assets))
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = 0.5 * corr / np.outer(d, d) + 0.5 * np.eye(n_assets)
    cov = np.outer(ann_vol, ann_vol) * corr / 252
    L = np.linalg.cholesky(cov)
    df = 5
    z = rng.standard_t(df, size=(days, n_assets)) / np.sqrt(df / (df - 2))
    r = ann_mu / 252 + z @ L.T
    cols = [f"Asset{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(r, index=pd.bdate_range(start, periods=days), columns=cols)
