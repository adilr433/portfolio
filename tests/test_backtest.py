import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import ma_crossover, momentum, performance_stats, run_backtest
from quantkit.data import simulate_gbm


def test_always_long_matches_buy_and_hold_without_costs():
    prices = simulate_gbm(seed=4)
    pos = pd.Series(1.0, index=prices.index)
    res = run_backtest(prices, pos, cost_bps=0)
    # position is taken at the first close, so it earns every return after day 0
    assert res.equity.iloc[-1] == pytest.approx(prices.iloc[-1] / prices.iloc[0], rel=1e-12)


def test_no_lookahead():
    # A position set from today's return must earn tomorrow's return, not today's.
    # If the engine forgot to lag positions, this would equal the perfect-foresight curve.
    prices = simulate_gbm(seed=5)
    rets = prices.pct_change().fillna(0)
    res = run_backtest(prices, np.sign(rets), cost_bps=0)
    perfect = (1 + rets.abs()).prod()
    assert res.equity.iloc[-1] < perfect / 1e3
    np.testing.assert_allclose(res.returns.values[1:], np.sign(rets).values[:-1] * rets.values[1:])


def test_costs_reduce_returns():
    prices = simulate_gbm(seed=6)
    pos = ma_crossover(prices, 10, 50)
    assert run_backtest(prices, pos, 20).equity.iloc[-1] < run_backtest(prices, pos, 0).equity.iloc[-1]


def test_signals_bounded_and_no_position_before_warmup():
    prices = simulate_gbm(seed=7)
    ma = ma_crossover(prices, 20, 100, allow_short=True)
    assert ma.iloc[:99].eq(0).all() and ma.isin([-1, 0, 1]).all()
    assert momentum(prices).isin([-1, 0, 1]).all()


def test_performance_stats_known_series():
    # equity 1 -> 1.1 -> 0.55 -> 0.605: peak 1.1, trough 0.55 => 50% drawdown
    s = performance_stats(pd.Series([0.10, -0.50, 0.10]))
    assert s["Hit rate"] == pytest.approx(2 / 3)
    assert s["Max drawdown"] == pytest.approx(0.5)
    assert s["Total return"] == pytest.approx(-0.395)
