"""Vectorized single-asset backtester with transaction costs, plus two sample
signals (moving-average crossover and time-series momentum).

Signals are positions in [-1, 1] decided at the close of day t and held over
day t+1, so there is no look-ahead: position is shifted one bar before it
earns that bar's return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100, allow_short: bool = False) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    f, s = prices.rolling(fast).mean(), prices.rolling(slow).mean()
    pos = pd.Series(np.where(f > s, 1.0, -1.0 if allow_short else 0.0), index=prices.index)
    return pos.where(s.notna(), 0.0)


def momentum(prices: pd.Series, lookback: int = 126, allow_short: bool = True) -> pd.Series:
    past = prices.pct_change(lookback)
    pos = np.sign(past)
    if not allow_short:
        pos = pos.clip(lower=0)
    return pos.fillna(0.0)


@dataclass
class BacktestResult:
    returns: pd.Series      # strategy net returns per bar
    equity: pd.Series       # compounded equity curve starting at 1
    positions: pd.Series
    stats: pd.Series


def performance_stats(returns: pd.Series, periods: int = 252, rf: float = 0.0) -> pd.Series:
    r = returns.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / periods
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods)
    excess = r - rf / periods
    sharpe = excess.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std() > 0 else np.nan
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods)
    dd = equity / equity.cummax() - 1
    return pd.Series({
        "Total return": equity.iloc[-1] - 1,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": excess.mean() * periods / downside if downside > 0 else np.nan,
        "Max drawdown": -dd.min(),
        "Calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
        "Hit rate": (r[r != 0] > 0).mean(),
    })


def run_backtest(prices: pd.Series, positions: pd.Series, cost_bps: float = 5.0,
                 periods: int = 252) -> BacktestResult:
    """cost_bps is charged on each unit of turnover (|change in position|)."""
    asset_ret = prices.pct_change().fillna(0.0)
    held = positions.shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    net = held * asset_ret - turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    equity = (1 + net).cumprod()
    stats = performance_stats(net, periods)
    stats["Turnover / yr"] = turnover.sum() / (len(prices) / periods)
    return BacktestResult(net, equity, positions, stats)
