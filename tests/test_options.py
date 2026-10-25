import numpy as np
import pytest

from quantkit.options import binomial_price, bsm_greeks, bsm_price, implied_vol

S, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


def test_bsm_reference_values():
    # Hull, Options Futures & Other Derivatives: ATM 1y, r=5%, vol=20%
    assert bsm_price(S, K, T, r, SIG, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bsm_price(S, K, T, r, SIG, "put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_with_dividends():
    q = 0.03
    c = bsm_price(S, 110, T, r, SIG, "call", q)
    p = bsm_price(S, 110, T, r, SIG, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - 110 * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = bsm_greeks(S, K, T, r, SIG, kind)
    h = 1e-4
    f = lambda **kw: bsm_price(**{"S": S, "K": K, "T": T, "r": r, "sigma": SIG, "kind": kind, **kw})
    assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-5)
    assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-3)
    assert g.vega == pytest.approx((f(sigma=SIG + h) - f(sigma=SIG - h)) / (2 * h), rel=1e-5)
    assert g.rho == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-5)
    assert g.theta == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(kind):
    price = bsm_price(S, 95, 0.5, r, 0.37, kind)
    assert implied_vol(price, S, 95, 0.5, r, kind) == pytest.approx(0.37, abs=1e-8)


def test_implied_vol_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        implied_vol(0.01, S, 50, T, r, "call")


def test_binomial_converges_to_bsm():
    assert binomial_price(S, K, T, r, SIG, "call", steps=2000) == pytest.approx(10.4506, abs=5e-3)


def test_american_put_has_early_exercise_premium():
    euro = binomial_price(S, K, T, r, SIG, "put", steps=1000)
    amer = binomial_price(S, K, T, r, SIG, "put", steps=1000, american=True)
    assert amer > euro + 0.2
    # American call on non-dividend stock is never exercised early
    assert binomial_price(S, K, T, r, SIG, "call", 1000, american=True) == pytest.approx(
        binomial_price(S, K, T, r, SIG, "call", 1000), abs=1e-10)
