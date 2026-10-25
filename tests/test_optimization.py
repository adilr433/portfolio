import numpy as np
import pandas as pd
import pytest

from quantkit.optimization import efficient_frontier, max_sharpe, min_variance, target_return

MU = pd.Series([0.08, 0.12, 0.15], index=["A", "B", "C"])
VOL = np.array([0.15, 0.20, 0.30])
CORR = np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]])
COV = pd.DataFrame(np.outer(VOL, VOL) * CORR, index=MU.index, columns=MU.index)


def test_weights_sum_to_one_and_long_only():
    for p in (min_variance(MU, COV), max_sharpe(MU, COV, rf=0.02)):
        assert p.weights.sum() == pytest.approx(1.0)
        assert (p.weights >= -1e-9).all()


def test_unconstrained_min_variance_matches_closed_form():
    inv = np.linalg.inv(COV.values)
    ones = np.ones(3)
    w_star = inv @ ones / (ones @ inv @ ones)
    p = min_variance(MU, COV, long_only=False)
    np.testing.assert_allclose(p.weights.values, w_star, atol=1e-5)


def test_unconstrained_tangency_matches_closed_form():
    rf = 0.02
    inv = np.linalg.inv(COV.values)
    w = inv @ (MU.values - rf)
    w /= w.sum()
    np.testing.assert_allclose(max_sharpe(MU, COV, rf, long_only=False).weights.values, w, atol=1e-4)


def test_max_sharpe_beats_frontier_points():
    best = max_sharpe(MU, COV, rf=0.02)
    frontier = efficient_frontier(MU, COV, 20, rf=0.02)
    assert best.sharpe >= frontier["sharpe"].max() - 1e-6


def test_target_return_hit_and_weight_cap():
    p = target_return(MU, COV, 0.11, max_weight=0.5)
    assert p.expected_return == pytest.approx(0.11, abs=1e-8)
    assert p.weights.max() <= 0.5 + 1e-9


def test_frontier_is_increasing_in_volatility():
    f = efficient_frontier(MU, COV, 15)
    assert f["volatility"].is_monotonic_increasing
