import numpy as np
import pytest
from scipy.stats import norm

from quantkit.data import simulate_returns
from quantkit.risk import (historical_es, historical_var, max_drawdown, monte_carlo_var,
                           parametric_es, parametric_var, risk_report, stress_test)


@pytest.fixture
def gaussian():
    return np.random.default_rng(1).normal(0.0, 0.01, 200_000)


def test_parametric_var_closed_form(gaussian):
    assert parametric_var(gaussian, alpha=0.99) == pytest.approx(0.01 * norm.ppf(0.99), rel=1e-2)


def test_methods_agree_on_gaussian_data(gaussian):
    pv = parametric_var(gaussian)
    assert historical_var(gaussian) == pytest.approx(pv, rel=2e-2)
    mc_var, mc_es = monte_carlo_var(gaussian, seed=0)
    assert mc_var == pytest.approx(pv, rel=2e-2)
    assert mc_es == pytest.approx(parametric_es(gaussian), rel=2e-2)


def test_es_exceeds_var():
    r = simulate_returns(4, seed=3)
    assert historical_es(r) > historical_var(r)
    assert parametric_es(r) > parametric_var(r)


def test_horizon_scaling():
    r = np.random.default_rng(0).normal(0, 0.01, 10_000)
    r = r - r.mean()
    assert parametric_var(r, horizon=10) == pytest.approx(parametric_var(r) * np.sqrt(10), rel=1e-9)


def test_max_drawdown_simple():
    # 1 -> 1.1 -> 0.55 -> 0.605: peak 1.1, trough 0.55 => 50%
    assert max_drawdown([0.10, -0.50, 0.10]) == pytest.approx(0.5)


def test_stress_test():
    out = stress_test([0.5, 0.5], {"crash": [-0.2, -0.1], "rally": [0.1, 0.1]})
    assert out["crash"] == pytest.approx(-0.15)
    assert out["rally"] == pytest.approx(0.10)


def test_risk_report_runs():
    rep = risk_report(simulate_returns(3, seed=2))
    assert (rep > 0).all() and len(rep) == 7
